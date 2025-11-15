from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import random
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass(eq=False)
class MemoryItem:
    key: torch.Tensor   # [D]
    val: Dict[str, Any] # sketch: {'z_slow': Tensor[D], 'avg_emb': Tensor[D], 'meta': ...}
    step: int = 0


class MemoryLearner(nn.Module):
    """Simple MLP regressor that maps averaged embeddings to stored latent codes."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        prev_dim = in_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompressedMemory:
    def __init__(
        self,
        max_items: int = 32000,
        device: str = "cpu",
        enable_learning: bool = True,
        learn_hidden: int = 256,
        learn_depth: int = 2,
        learn_rate: float = 1e-3,
        learn_weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        ema_decay: float = 0.9,
        min_train_items: int = 16,
        # Long-term memory policy knobs (3brain manual)
        shadow_bank_cap: int = 300,
        write_per_turn: int = 1,
        write_per_100: int = 6,
        dup_skip_thr: float = 0.95,
        near_merge_thr: float = 0.84,
        stage_mid_low: float = 0.73,
        entropy_floor: float = 0.3,
        log_dir: str = ".",
        # extras for Luria manual
        warmup_steps: int = 200,
        tail_write_per_100: int = 2,
        topk_default: int = 6,
        softmax_temp: float = 0.2,
        lambda_penalty: float = 0.22,
        # Autonomy/salience controls (Copya.txt prescription)
        autonomy_gain: float = 1.0,
        salience_commit_bias: float = 0.0,
    ):
        self.items: List[MemoryItem] = []
        self.max_items = int(max_items)
        self.device = device

        # Learning-related configuration/state
        self.enable_learning = enable_learning
        self.learn_hidden = learn_hidden
        self.learn_depth = learn_depth
        self.learn_rate = learn_rate
        self.learn_weight_decay = learn_weight_decay
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.min_train_items = min_train_items

        self._learner: Optional[MemoryLearner] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._ema_loss: Optional[float] = None
        self._learn_fields: Optional[Tuple[str, str]] = None  # (key_field, target_field)

        # Real-time state tracking
        self._session_state: Dict[str, Any] = {}
        self._input_history: List[torch.Tensor] = []
        self._state_history: List[Dict[str, Any]] = []
        self._max_history: int = 100

        # --- Long-term memory policy state ---
        self.shadow_bank: List[MemoryItem] = []
        self.shadow_bank_cap = shadow_bank_cap
        self.write_per_turn = write_per_turn
        self.write_per_100 = write_per_100
        self.tail_write_per_100 = tail_write_per_100
        self.dup_skip_thr = dup_skip_thr
        self.near_merge_thr = near_merge_thr
        self.stage_mid_low = stage_mid_low
        self.entropy_floor = entropy_floor
        self.log_dir = log_dir
        self._write_steps: List[int] = []  # accepted write steps for rate limiting
        self._tail_write_steps: List[int] = []  # tail bucket write steps
        self._surprise_hist: List[float] = []  # rolling surprise values
        self._entropy_hist: List[float] = []  # retrieval entropy snapshot
        self._last_step_seen: int = -1
        self._rng = random.Random(17)
        self.warmup_steps = warmup_steps
        self.topk = topk_default
        self.softmax_temp = softmax_temp
        self.lambda_penalty = lambda_penalty
        self.last_meta_entropy: Optional[float] = None
        # Autonomy/salience controls
        self.autonomy_gain: float = float(max(0.0, autonomy_gain))
        # Bias added to salience score before comparing with gate; useful to globally tighten/loosen survival
        self.salience_commit_bias: float = float(salience_commit_bias)
        # Diversity/cooldown bookkeeping
        self._last_used_step_per_idx: Dict[int, int] = {}
        self._usage_tick: int = 0
        self.cooldown_steps: int = 80
        self.intent_write_toggle: Optional[bool] = None

        # Lekiltan enhancements
        self._surprise_ema8: Optional[float] = None  # EMA with alpha=0.25 (??-step window)
        self._model_entropy_window: List[float] = []  # rolling window for model entropy (256 steps)
        self._alert_active_until: int = -1  # surprise alert end step
        self._dampen_active_until: int = -1  # z-score dampen end step
        self._warmup_ramp_steps: int = 1000  # warmup ramp duration

        # Luria: shadow bank decay
        self.shadow_decay_interval: int = 500  # decay every N steps
        self.shadow_size_hard_cap: int = 100  # force decay if size >= this
        self._last_shadow_decay: int = -1  # last decay step

        # L2 regularization
        self.key_norm_mode: str = "scaled_l2"  # "unit" or "scaled_l2"
        self.key_norm_scale: float = 0.90  # Choebang-yak: scaled_l2 target up (0.85->0.90)
        self.l2_reg_weight: float = 5e-5  # Choebang-yak: reduce L2 penalty (1e-4->5e-5)
        self.enforce_unit_norm: bool = True  # enforce L2 norm on keys/z_slow

        # Promotion relax control (Choebang-yak): if two consecutive cycles yield no promotions,
        # relax surprise percentile by 5p on the next cycle.
        self._rebalance_no_promo_runs: int = 0
        self._rebalance_relax_bonus: int = 0  # in percentile points (0 or 5)

        # Dynamic surprise threshold
        self._surprise_threshold_ema: Optional[float] = None  # EMA of surprise threshold
        self._surprise_threshold_alpha: float = 0.1  # slower adaptation for threshold
        self.write_threshold_base: float = 0.05  # base threshold (set to 0.05)
        self.write_threshold_alpha: float = -0.20  # adjustment coefficient (negative = lower when surprise high)
        self.write_threshold_min: float = 0.05  # clamp min (set to 0.05)
        self.write_threshold_max: float = 0.40  # clamp max (softened)
        self.write_threshold_warmup_end: int = 1000  # warmup duration for threshold

        # Sigmoid scaling for smooth transitions
        self.sigmoid_k: float = 12.0  # scale parameter (8~16 range for exploration)

        # --- Curiosity logging (Phase 1: metrics only, no behavior change) ---
        self._curiosity_score: Optional[float] = None
        self._curiosity_reservoir: Optional[float] = None
        self._curiosity_alpha: float = 0.2  # EMA for reservoir
        self._prev_ema_loss_for_log: Optional[float] = None
        self._learner_progress_for_log: Optional[float] = None
        self._model_entropy_last_for_log: Optional[float] = None

    # -------------------------
    # Utility: logging helpers
    # -------------------------
    def _append_csv(self, filename: str, row: Dict[str, Any]):
        import csv, os
        os.makedirs(self.log_dir, exist_ok=True)
        path = f"{self.log_dir}/{filename}"
        file_exists = False
        try:
            file_exists = os.path.exists(path)
        except Exception:
            pass
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    w.writeheader()
                w.writerow(row)
        except Exception:
            # logging should not break compute path
            pass

    def _log_write(
        self,
        step: int,
        action: str,
        reason: Optional[str] = None,
        sim_max: Optional[float] = None,
        surprise: Optional[float] = None,
        surp_norm: Optional[float] = None,
        entropy: Optional[float] = None,
        entropy_raw: Optional[float] = None,
        topk: Optional[int] = None,
        used_key_id: Optional[int] = None,
        l2_penalty: Optional[float] = None,
        surprise_threshold: Optional[float] = None,
        top5_neighbors_dist: Optional[List[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        row = {
            "step": int(step),
            "action": action,
            "reason": reason,
            "sim_max": sim_max if sim_max is None else float(sim_max),
            "surprise": surprise if surprise is None else float(surprise),
            "surp_norm": surp_norm if surp_norm is None else float(surp_norm),
            "entropy": entropy if entropy is None else float(entropy),
            "entropy_raw": entropy_raw if entropy_raw is None else float(entropy_raw),
            "topk": int(topk if topk is not None else self.topk),
            "used_key_id": used_key_id,
            "shadow_size": int(len(self.shadow_bank)),
            "mem_size": int(len(self.items)),
            "l2_penalty": l2_penalty if l2_penalty is None else float(l2_penalty),
            "surprise_threshold": surprise_threshold if surprise_threshold is None else float(surprise_threshold),
        }
        # Curiosity metrics (logging only)
        try:
            if self._curiosity_score is not None:
                row["curiosity_score"] = float(self._curiosity_score)
            if self._curiosity_reservoir is not None:
                row["curiosity_reservoir"] = float(self._curiosity_reservoir)
            if self._ema_loss is not None:
                row["learner_loss_ema"] = float(self._ema_loss)
            if self._learner_progress_for_log is not None:
                row["learner_progress"] = float(self._learner_progress_for_log)
            if self._model_entropy_last_for_log is not None:
                row["model_entropy_last"] = float(self._model_entropy_last_for_log)
        except Exception:
            # logging must not break compute path
            pass
        # Luria: add top5 neighbor distances for novelty analysis
        if top5_neighbors_dist is not None:
            for i, dist in enumerate(top5_neighbors_dist[:5]):
                row[f"top5_dist_{i+1}"] = round(float(dist), 6)
        if extra:
            try:
                row.update(extra)
            except Exception:
                pass
        self._append_csv("writes.csv", row)

    # -------------------------
    # Policy: retrieval entropy
    # -------------------------
    def _retrieval_entropy(self, query: torch.Tensor, topk: int = 8) -> Tuple[float, float]:
        """Return (entropy_normalized, entropy_raw)."""
        if not self.items:
            return 0.0, 0.0
        keys = torch.stack([it.key for it in self.items], dim=0)
        q = query.unsqueeze(0)
        sim = F.cosine_similarity(keys, q.expand_as(keys), dim=-1)  # [N]

        # ?곸쐞 k留??ъ슜??遺꾪룷 援ъ꽦
        k = min(topk, sim.numel())
        vals, idx = torch.topk(sim, k=k, largest=True)
        sel = vals

        # softmax with clamping
        temp = max(float(self.softmax_temp), 1e-4)
        probs = torch.softmax(sel / temp, dim=-1)
        probs = probs.clamp_min(1e-9)  # 0 ?뺣쪧 湲덉?

        ent_raw = float(-(probs * probs.log()).sum().item())

        # Soft clipping with tanh (instead of hard clipping)
        # Maps [0, log(k)] ??[0, 1] smoothly
        import math
        max_ent = math.log(float(k))  # theoretical max for k items
        if max_ent > 1e-6:
            # tanh-based soft normalization
            ent_norm = float(torch.tanh(torch.tensor(ent_raw / max_ent)).item())
        else:
            ent_norm = 0.0

        # track
        self._entropy_hist.append(ent_norm)
        self.last_meta_entropy = ent_norm
        self._append_csv("entropy.csv", {
            "step": self._last_step_seen,
            "entropy": round(ent_norm, 6),
            "entropy_raw": round(ent_raw, 6),
            "entropy_max": round(max_ent, 6)
        })
        return ent_norm, ent_raw

    # -------------------------
    # Policy: salience ("trauma") scoring
    # -------------------------
    def _compute_salience(
        self,
        *,
        surprise: Optional[float],
        ent_norm: Optional[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Composite salience score in [0,1] inspired by Copya.txt.

        - surprise: normalized surprise in [0,1] if available
        - ent_norm: retrieval entropy in [0,1] (higher = broader support)
        - meta: may contain 'composite_score', 'brier', 'shock', 'intent_bias'

        Returns a soft score; higher = more likely to commit even under strict gates.
        """
        try:
            s = float(max(0.0, min(1.0, surprise if surprise is not None else 0.0)))
        except Exception:
            s = 0.0
        try:
            e = float(max(0.0, min(1.0, ent_norm if ent_norm is not None else (self.last_meta_entropy or 0.0))))
        except Exception:
            e = float(self.last_meta_entropy or 0.0)
        # High surprise and moderately high entropy implies "new and impactful".
        novelty = 0.6 * s + 0.4 * e

        comp = 0.0
        brier = 0.0
        shock_meta = 0.0
        intent_bias = 0.0
        if meta:
            # composite_score is an unbounded positive scalar; squash to [0,1]
            try:
                c = float(meta.get("composite_score", 0.0))
                comp = 1.0 - float(torch.exp(torch.tensor(-max(0.0, c) / 5.0)).item())  # ~1-exp(-c/5)
            except Exception:
                comp = 0.0
            try:
                brier = float(meta.get("brier", 0.0))  # [0, +]
                # Convert to [0,1] where higher = more surprising/impactful error
                brier = max(0.0, 1.0 - float(torch.exp(torch.tensor(-brier)).item()))
            except Exception:
                brier = 0.0
            try:
                shock_meta = float(meta.get("shock", 0.0))
            except Exception:
                shock_meta = 0.0
            try:
                intent_bias = float(meta.get("intent_bias", 0.0))
            except Exception:
                intent_bias = 0.0

        # Combine: novelty + supervised miss (brier) + external shock + pro-intent bias
        sal = 0.45 * novelty + 0.30 * brier + 0.15 * comp + 0.10 * max(0.0, shock_meta)
        sal = max(0.0, min(1.0, sal))
        # Intent bias acts as small bonus (subjective prioritization)
        sal = sal + 0.10 * max(0.0, intent_bias)
        # Global bias knob
        sal = min(1.0, max(0.0, sal + float(self.salience_commit_bias)))
        return float(sal)

    # -------------------------
    # Policy: similarity helpers
    # -------------------------
    def _nearest_idx_and_sim(self, key: torch.Tensor) -> Tuple[Optional[int], float]:
        if not self.items:
            return None, -1.0
        keys = torch.stack([it.key for it in self.items], dim=0)
        sim = F.cosine_similarity(keys, key.unsqueeze(0).expand_as(keys), dim=-1)
        # Re-use penalty based on similarity to recent queries
        if self._input_history:
            recent = torch.stack(self._input_history[-min(32, len(self._input_history)):], dim=0).to(keys.device)
            recent = F.normalize(recent, dim=-1)
            sim_recent = torch.matmul(keys, recent.transpose(0, 1)).mean(dim=1)
            sim = sim - float(self.lambda_penalty) * sim_recent
        # Cooldown penalty for recently used items
        if self._last_used_step_per_idx:
            cool_pen = torch.zeros_like(sim)
            for i, last in self._last_used_step_per_idx.items():
                if 0 <= i < sim.numel() and (self._usage_tick - last) <= self.cooldown_steps:
                    cool_pen[i] = 0.2
            sim = sim - cool_pen
        val, idx = torch.max(sim, dim=0)
        return int(idx.item()), float(val.item())

    def _update_curiosity_metrics(self, ent: Optional[float] = None, surp_norm: Optional[float] = None, step: Optional[int] = None) -> None:
        """Update curiosity metrics for logging only.

        Args:
            ent: Retrieval entropy (normalized 0..1) if available.
            surp_norm: Normalized surprise (0..1) if available.
            step: Current step (unused, reserved for future diagnostics).
        """
        try:
            # Inputs
            c_entropy = float(ent) if ent is not None else float(self.last_meta_entropy or 0.0)
            c_surprise = float(surp_norm) if surp_norm is not None else 0.0
            c_entropy = max(0.0, min(1.0, c_entropy))
            c_surprise = max(0.0, min(1.0, c_surprise))

            # CuriosityScore: simple blend (no behavior change)
            score = 0.6 * c_surprise + 0.4 * c_entropy
            score = max(0.0, min(1.0, score))
            self._curiosity_score = score

            # Reservoir EMA for coarse trend observation
            if self._curiosity_reservoir is None:
                self._curiosity_reservoir = score
            else:
                a = float(self._curiosity_alpha)
                self._curiosity_reservoir = (1.0 - a) * float(self._curiosity_reservoir) + a * score

            # Learner progress (EMA loss delta, positive when improving)
            cur_ema = self._ema_loss
            prev_ema = self._prev_ema_loss_for_log
            if cur_ema is not None:
                cur = float(cur_ema)
                if prev_ema is not None:
                    prog = max(0.0, float(prev_ema) - cur)
                    self._learner_progress_for_log = prog
                self._prev_ema_loss_for_log = cur

            # Last model entropy value (if present)
            if self._model_entropy_window:
                self._model_entropy_last_for_log = float(self._model_entropy_window[-1])
        except Exception:
            # Never break main path for logging aids
            pass

    def _top5_neighbors_dist(self, key: torch.Tensor) -> List[float]:
        """Compute top-5 neighbor distances (1 - cosine_sim) for novelty analysis (Luria improvement).

        Returns:
            List of up to 5 distances, sorted from nearest to farthest.
        """
        if not self.items:
            return []

        keys = torch.stack([it.key for it in self.items], dim=0)
        sim = F.cosine_similarity(keys, key.unsqueeze(0).expand_as(keys), dim=-1)

        # No penalties here - we want raw similarity for analysis
        k = min(5, sim.numel())
        vals, _ = torch.topk(sim, k=k, largest=True)
        # Convert similarity to distance: distance = 1 - sim
        distances = [float(1.0 - v.item()) for v in vals]
        return distances

    def _k3_avg_sim(self, key: torch.Tensor) -> Tuple[float, bool]:
        """Compute k=3 average similarity with hard negative check (Luria improvement).

        Returns:
            (avg_sim, hard_neg_pass): Average top-3 similarity and whether hard negative check passed
        """
        if not self.items:
            return -1.0, True

        if len(self.items) < 3:
            # Fallback to single nearest
            _, smax = self._nearest_idx_and_sim(key)
            return smax, True

        keys = torch.stack([it.key for it in self.items], dim=0)
        sim = F.cosine_similarity(keys, key.unsqueeze(0).expand_as(keys), dim=-1)

        # Apply same penalties as _nearest_idx_and_sim
        if self._input_history:
            recent = torch.stack(self._input_history[-min(32, len(self._input_history)):], dim=0).to(keys.device)
            recent = F.normalize(recent, dim=-1)
            sim_recent = torch.matmul(keys, recent.transpose(0, 1)).mean(dim=1)
            sim = sim - float(self.lambda_penalty) * sim_recent

        if self._last_used_step_per_idx:
            cool_pen = torch.zeros_like(sim)
            for i, last in self._last_used_step_per_idx.items():
                if 0 <= i < sim.numel() and (self._usage_tick - last) <= self.cooldown_steps:
                    cool_pen[i] = 0.2
            sim = sim - cool_pen

        # Top-3 average
        k = min(3, sim.numel())
        vals, _ = torch.topk(sim, k=k, largest=True)
        avg_sim = float(vals.mean().item())

        # Hard negative check: sample 1-2 random items and check dissimilarity
        hard_neg_pass = True
        if len(self.items) >= 10:
            try:
                n_samples = min(2, len(self.items) - 3)
                hard_neg_indices = self._rng.sample(range(len(self.items)), n_samples)
                hard_neg_keys = torch.stack([self.items[i].key for i in hard_neg_indices], dim=0)
                hard_neg_sim = F.cosine_similarity(hard_neg_keys, key.unsqueeze(0).expand_as(hard_neg_keys), dim=-1)
                max_hard_neg_sim = float(hard_neg_sim.max().item())
                # If too similar to hard negatives, this might not be a true near-duplicate
                if max_hard_neg_sim > 0.70:
                    hard_neg_pass = False
            except Exception:
                pass

        return avg_sim, hard_neg_pass

    # -------------------------
    # L2 regularization helpers
    # -------------------------
    def _apply_l2_norm(self, tensor: torch.Tensor, target_norm: Optional[float] = None) -> torch.Tensor:
        """Apply L2 normalization based on configured mode.

        Modes:
        - "unit": normalize to unit norm (L2=1.0)
        - "scaled_l2": normalize to scaled norm (L2=key_norm_scale, e.g., 0.85)
        """
        if not self.enforce_unit_norm:
            return tensor
        if tensor.ndim == 0:
            return tensor

        # Determine target norm
        if target_norm is None:
            if self.key_norm_mode == "scaled_l2":
                target_norm = self.key_norm_scale
            else:  # "unit" or default
                target_norm = 1.0

        norm = tensor.norm(dim=-1, keepdim=True)
        # Avoid division by zero
        norm = torch.clamp(norm, min=1e-8)
        return tensor * (target_norm / norm)

    def _l2_penalty(self, tensor: torch.Tensor) -> float:
        """Compute L2 penalty for regularization loss."""
        if tensor.ndim == 0:
            return 0.0
        return float(self.l2_reg_weight * (tensor ** 2).sum().item())

    def _sigmoid_scale(self, x: float, k: Optional[float] = None) -> float:
        """Apply sigmoid scaling for smooth transitions: sigmoid(k*(x-0.5))."""
        if k is None:
            k = self.sigmoid_k
        import math
        return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))

    # -------------------------
    # Lekiltan enhancements
    # -------------------------
    def _warmup_ramp_factor(self, step: int) -> float:
        """Linear ramp from 0.0 to 1.0 over [0, warmup_ramp_steps]."""
        if step >= self._warmup_ramp_steps:
            return 1.0
        return max(0.0, float(step) / float(self._warmup_ramp_steps))

    def _update_surprise_ema(self, surprise: float):
        """Update EMA with alpha=0.25 (??-step window)."""
        alpha = 0.25
        if self._surprise_ema8 is None:
            self._surprise_ema8 = surprise
        else:
            self._surprise_ema8 = alpha * surprise + (1.0 - alpha) * self._surprise_ema8

    def _check_surprise_alert(self, step: int) -> bool:
        """Check if surprise EMA exceeds median+1?; activate 8-step alert."""
        if self._surprise_ema8 is None or len(self._surprise_hist) < 30:
            return False
        import statistics
        median = statistics.median(self._surprise_hist[-100:])
        try:
            stdev = statistics.stdev(self._surprise_hist[-100:])
        except:
            stdev = 0.0
        threshold = median + stdev
        if self._surprise_ema8 > threshold:
            self._alert_active_until = step + 8
            return True
        return step < self._alert_active_until

    def _update_model_entropy_window(self, model_entropy: float):
        """Track model entropy in a rolling window (256 steps)."""
        self._model_entropy_window.append(model_entropy)
        if len(self._model_entropy_window) > 256:
            self._model_entropy_window.pop(0)

    def _check_zscore_dampen(self, step: int) -> bool:
        """Check if model_entropy |z-score| >= 3; activate 16-step dampen."""
        if len(self._model_entropy_window) < 30:
            return False
        import statistics
        try:
            mean = statistics.mean(self._model_entropy_window)
            stdev = statistics.stdev(self._model_entropy_window)
        except:
            return False
        if stdev < 1e-6:
            return False
        z_score = abs((self._model_entropy_window[-1] - mean) / stdev)
        if z_score >= 3.0:
            self._dampen_active_until = step + 16
            return True
        return step < self._dampen_active_until

    def _update_surprise_threshold(self, current_surprise: float):
        """Update dynamic surprise threshold using EMA."""
        if self._surprise_threshold_ema is None:
            # Initialize with current percentile if available
            if len(self._surprise_hist) >= 20:
                self._surprise_threshold_ema = self._percentile(self._surprise_hist, 85)
            else:
                self._surprise_threshold_ema = 0.15
        else:
            # EMA update
            alpha = self._surprise_threshold_alpha
            self._surprise_threshold_ema = alpha * current_surprise + (1.0 - alpha) * self._surprise_threshold_ema

    def _get_surprise_threshold(self, step: int) -> float:
        """Get dynamic surprise threshold based on EMA and surprise signal strength.

        Formula: threshold = base + alpha * surprise_ema * sigmoid_ramp
        - base: write_threshold_base (0.05)
        - alpha: write_threshold_alpha (-0.20, negative = lower threshold when surprise high)
        - sigmoid_ramp: smooth warmup using sigmoid scaling
        - clamped to [write_threshold_min, write_threshold_max] (0.05~0.40)
        """
        # Warmup ramp using sigmoid for smoothness
        if step < self.write_threshold_warmup_end:
            ramp = self._sigmoid_scale(float(step) / float(self.write_threshold_warmup_end))
        else:
            ramp = 1.0

        # Base threshold
        threshold = self.write_threshold_base

        # Adjust based on surprise EMA if available
        if self._surprise_threshold_ema is not None:
            # Normalize surprise_ema to [0, 1] range (assuming it's typically < 1.0)
            surp_norm = min(1.0, max(0.0, self._surprise_threshold_ema))
            adjustment = self.write_threshold_alpha * surp_norm * ramp
            threshold += adjustment

        # Clamp to safe range
        threshold = max(self.write_threshold_min, min(self.write_threshold_max, threshold))
        return threshold

    # -------------------------
    # Write policy entrypoint
    # -------------------------
    def add_with_policy(
        self,
        key: torch.Tensor,
        val: Dict[str, Any],
        step: int,
        meta: Optional[Dict[str, Any]] = None,
        bench_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Apply 3brain long-term memory policies when writing.

        Returns a decision dict: {status: 'accepted'|'merged'|'staged'|'skipped'|'deferred', reason: str}
        """
        self._last_step_seen = int(step)
        meta = dict(meta) if meta else {}
        intent_toggle = meta.pop("luria_intent_force", None)
        if intent_toggle is None:
            intent_toggle = self.intent_write_toggle
        # Autonomy mode: do not let intent toggle bypass policy gates; use only for upstream logic
        force_accept = False
        force_block = False

        # Luria: periodic shadow bank decay
        self._decay_shadow_bank(step)

        # Kill switch via environment
        try:
            if os.environ.get("UZR_MEM_WRITE", "1") == "0":
                self._log_write(step, action="defer", reason="kill_switch")
                return {"status": "staged", "reason": "kill_switch"}
        except Exception:
            pass

        # In autonomy mode, ignore explicit block from intent as well

        # Apply L2 normalization using configured mode (unit or scaled_l2)
        key = self._apply_l2_norm(key)
        if torch.is_tensor(val.get("z_slow")):
            val["z_slow"] = self._apply_l2_norm(val["z_slow"])

        # Compute L2 penalty for logging/monitoring
        l2_penalty = 0.0
        if torch.is_tensor(val.get("z_slow")):
            l2_penalty = self._l2_penalty(val["z_slow"])

        # Verify normalization (with tolerance for scaled_l2 mode)
        try:
            expected_norm = self.key_norm_scale if self.key_norm_mode == "scaled_l2" else 1.0
            if key.ndim == 1:
                assert torch.allclose(key.norm(dim=-1), torch.tensor(expected_norm, device=key.device), atol=1e-3)
            if torch.is_tensor(val.get("z_slow")) and val["z_slow"].ndim == 1:
                assert torch.allclose(val["z_slow"].norm(dim=-1), torch.tensor(expected_norm, device=key.device), atol=1e-3)
        except Exception:
            pass

        # ?쒖뾽 癒쇱?: ?쒖뾽 ?숈븞???ㅽ뀒?댁쭠議곗감 ?섏? ?딆쓬(?ㅼ뿼 諛⑹?)
        if step < self.warmup_steps:
            self._log_write(step, action="defer", reason="warmup")
            return {"status": "deferred", "reason": "warmup"}

        # ?뷀듃濡쒗뵾 寃뚯씠?몃뒗 遺?몄뒪?몃옪 ?댄썑?먮쭔
        entropy_check_start = getattr(self, "entropy_check_start", 32)
        apply_entropy_gate = (len(self.items) >= entropy_check_start)
        ent, ent_raw = self._retrieval_entropy(key) if apply_entropy_gate else (1.0, 1.0)
        # Curiosity metrics pre-update (surprise may be unavailable yet)
        self._update_curiosity_metrics(ent=ent, surp_norm=None, step=step)

        if apply_entropy_gate and (ent < self.entropy_floor) and (len(self.items) > 0):
            self._log_write(step, action="defer", reason="entropy_floor", entropy=ent, entropy_raw=ent_raw)
            self._stage_shadow(key, val, step, score=0.0, surprise=None)
            return {"status": "staged", "reason": f"entropy<{self.entropy_floor:.2f}"}

        # Similarity 怨꾩궛 (merge? skip 紐⑤몢 ?꾩슂)
        idx, smax = self._nearest_idx_and_sim(key)

        # Compute predicted z to estimate surprise
        surprise = None
        surp_norm = None
        # Diagnostics for surprise calculation (why missing?)
        _log_surp_diag = {
            "diag_surp_has_z": 1 if torch.is_tensor(val.get("z_slow")) else 0,
            "diag_surp_knn_tried": 0,
            "diag_surp_knn_with_z": 0,
            "diag_surp_computed": 0,
            "diag_surp_reason": "init",
        }
        try:
            z_new = val.get("z_slow")
            z_pred = None
            if torch.is_tensor(z_new):
                # try simple KNN prediction
                neigh = self.retrieve(key, topk=max(4, self.topk))
                if neigh:
                    _log_surp_diag["diag_surp_knn_tried"] = int(len(neigh))
                    cand = [it.val.get("z_slow") for it in neigh if torch.is_tensor(it.val.get("z_slow"))]
                    _log_surp_diag["diag_surp_knn_with_z"] = int(len(cand))
                    if cand:
                        z_pred = torch.stack(cand, dim=0).mean(dim=0)
            if torch.is_tensor(z_new) and torch.is_tensor(z_pred):
                z_new_n = F.normalize(z_new, dim=-1)
                z_pred_n = F.normalize(z_pred, dim=-1)
                surprise = float(1.0 - torch.clamp(F.cosine_similarity(z_new_n.unsqueeze(0), z_pred_n.unsqueeze(0), dim=-1), -1, 1).item())
                self._surprise_hist.append(surprise)
                _log_surp_diag["diag_surp_computed"] = 1
                _log_surp_diag["diag_surp_reason"] = "ok"
                # Update surprise EMA (Lekiltan)
                self._update_surprise_ema(surprise)
                # Update dynamic surprise threshold
                self._update_surprise_threshold(surprise)
                hist = self._surprise_hist[-2000:]
                if len(hist) >= 50:
                    p50 = self._percentile(hist, 50)
                    pTau = self._percentile(hist, 85)
                    surp_norm = max(0.0, (surprise - p50) / (pTau - p50 + 1e-6))
            else:
                # capture reason for missing surprise
                if not torch.is_tensor(z_new):
                    _log_surp_diag["diag_surp_reason"] = "no_z_slow"
                elif _log_surp_diag["diag_surp_knn_tried"] == 0:
                    _log_surp_diag["diag_surp_reason"] = "no_neighbors"
                elif _log_surp_diag["diag_surp_knn_with_z"] == 0:
                    _log_surp_diag["diag_surp_reason"] = "neighbors_no_z"
                else:
                    _log_surp_diag["diag_surp_reason"] = "unknown"
        except Exception:
            # leave diagnostics as-is
            pass

        # Curiosity metrics update with (possibly None) surp_norm after surprise/diagnostics are set
        self._update_curiosity_metrics(ent=ent, surp_norm=surp_norm, step=step)

        # Track model entropy if provided in meta (Lekiltan)
        if meta and "model_entropy" in meta:
            self._update_model_entropy_window(float(meta["model_entropy"]))

        # Near-duplicate merge (Luria: k=3 average similarity with hard negative check)
        k3_for_log = None
        if idx is not None:
            k3_sim, hard_neg_ok = self._k3_avg_sim(key)
            k3_for_log = float(k3_sim) if k3_sim is not None else None
            # Merge only when we have at least 3 neighbors (cluster) and
            # k=3 avg similarity exceeds threshold, with hard negative check.
            if (len(self.items) >= 3) and (k3_sim >= self.near_merge_thr) and hard_neg_ok:
                BETA_MIN, BETA_MAX = 0.10, 0.30
                if meta and meta.get("bucket") == "tail":
                    BETA_MAX = 0.35
                beta = 0.2
                if surp_norm is not None:
                    beta = float(max(BETA_MIN, min(BETA_MAX, BETA_MIN + 0.25 * surp_norm)))
                elif surprise is not None:
                    beta = float(max(BETA_MIN, min(BETA_MAX, 0.1 + 0.4 * surprise)))
                # Get surprise threshold for logging
                surprise_thr = self._get_surprise_threshold(step) if surprise is not None else None
                self._merge_into(idx, val, beta=beta)
                self._log_write(step, action="merge", reason="near_merge_k3", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, entropy_raw=ent_raw, used_key_id=idx,
                               l2_penalty=l2_penalty, surprise_threshold=surprise_thr, extra={**_log_surp_diag, "k3_sim": round(k3_sim, 4)})
                return {"status": "merged", "reason": f"near_merge_k3>={self.near_merge_thr}"}
            elif (len(self.items) >= 3) and (k3_sim >= self.near_merge_thr) and not hard_neg_ok:
                # Hard negative check failed: log and skip merge
                self._log_write(step, action="skip", reason="hard_neg_fail", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, entropy_raw=ent_raw, used_key_id=idx,
                               l2_penalty=l2_penalty, extra={**_log_surp_diag, "k3_sim": round(k3_sim, 4)})

        # Duplicate skip - 癒몄? ?댄썑??泥댄겕
        if idx is not None and smax >= self.dup_skip_thr:
            surprise_thr = self._get_surprise_threshold(step) if surprise is not None else None
            extra_fields = dict(_log_surp_diag)
            if k3_for_log is not None:
                extra_fields["k3_sim"] = round(float(k3_for_log), 4)
            self._log_write(step, action="skip", reason="dup_skip", sim_max=smax, entropy=ent, entropy_raw=ent_raw, used_key_id=idx,
                           l2_penalty=l2_penalty, surprise_threshold=surprise_thr, extra=extra_fields)
            return {"status": "skipped", "reason": f"dup>={self.dup_skip_thr}"}

        # Stage range
        if idx is not None and smax >= self.stage_mid_low:
            self._stage_shadow(key, val, step, score=smax, surprise=surprise)
            return {"status": "staged", "reason": f"mid_range>={self.stage_mid_low}"}

        # Joy override (Add The Joy To Luria): detect Aha moments from meta
        joy_val = 0.0
        try:
            if meta is not None:
                if str(meta.get("type", "")).lower() == "joy":
                    joy_val = float(meta.get("score", 0.0))
                elif "joy_score" in meta:
                    joy_val = float(meta.get("joy_score", 0.0))
        except Exception:
            joy_val = 0.0
        joy_thr = 0.5
        joy_force_commit = (joy_val > joy_thr)

        # Write-on-surprise gate with dynamic threshold (Luria improvement: sim_max-based relaxation)
        surprise_thr = self._get_surprise_threshold(step) if surprise is not None else None
        if surprise is not None and surprise_thr is not None:
            # Luria improvement: lower sim_max allows more commits
            if smax <= 0.30:
                # Very novel (sim_max ??0.30): skip surprise gate entirely
                pass
            elif smax <= 0.45:
                # Novel (sim_max <=0.45): relax surprise gate to 50% of threshold
                if surprise < surprise_thr * 0.35 and not joy_force_commit:
                    self._stage_shadow(key, val, step, score=surprise, surprise=surprise)
                    return {"status": "staged", "reason": f"surprise<{surprise_thr*0.35:.3f}(relaxed@sim={smax:.2f})"}
            else:
                # Use dynamic threshold based on surprise EMA and warmup ramp
                if surprise < surprise_thr and not joy_force_commit:
                    self._stage_shadow(key, val, step, score=surprise, surprise=surprise)
                    return {"status": "staged", "reason": f"surprise<{surprise_thr:.3f}(dynamic)"}

        # Salience ("trauma") override: if composite salience is strong and autonomy_gain is high,
        # allow commit to bypass some gates (but still respect duplicate-skip already handled above).
        try:
            sal = self._compute_salience(surprise=surp_norm if surp_norm is not None else surprise,
                                         ent_norm=self.last_meta_entropy,
                                         meta=meta)
        except Exception:
            sal = 0.0
        # Autonomy scaling: map autonomy_gain to an effective threshold relaxation
        # gain >= 25 → lower required salience; else stricter.
        gain = max(1.0, float(self.autonomy_gain or 1.0))
        # Required salience to force-commit, shaped so that higher gain reduces the bar
        # base_thr ~ 0.92, with gain=25 → ~0.92 - 0.20 ≈ 0.72
        base_thr = 0.92
        sal_thr = max(0.60, base_thr - 0.20 * min(1.0, (gain / 25.0)))
        salience_force_commit = (sal >= sal_thr)

        # Rate limiting (with tail bucket budget)
        bucket = meta.get("bucket")
        if (not salience_force_commit) and (not joy_force_commit) and (not self._rate_limit_ok(step, bucket=bucket)):
            extra_fields = dict(_log_surp_diag)
            if k3_for_log is not None:
                extra_fields["k3_sim"] = round(float(k3_for_log), 4)
            self._log_write(step, action="defer", reason="rate_limited", surprise=surprise, surp_norm=surp_norm, entropy=ent, entropy_raw=ent_raw,
                           l2_penalty=l2_penalty, surprise_threshold=surprise_thr, extra=extra_fields)
            self._stage_shadow(key, val, step, score=0.0, surprise=surprise)
            return {"status": "staged", "reason": "rate_limited"}

        # Lekiltan: check surprise alert state and log
        alert_active = self._check_surprise_alert(step)

        # Luria: compute top5 neighbors distances for commit logging
        top5_dist = self._top5_neighbors_dist(key)

        # Accept and add w/ optional 2PC bench callback
        commit_ok = True
        if bench_callback is not None:
            try:
                # Stage proposal
                extra_fields = {**_log_surp_diag, "alert_active": alert_active}
                if k3_for_log is not None:
                    extra_fields["k3_sim"] = round(float(k3_for_log), 4)
                self._log_write(step, action="stage", reason="2pc_proposal", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, entropy_raw=ent_raw, used_key_id=idx,
                               l2_penalty=l2_penalty, surprise_threshold=surprise_thr, top5_neighbors_dist=top5_dist,
                               extra=extra_fields)
                commit_ok = bool(bench_callback())
            except Exception:
                commit_ok = False
        if commit_ok or salience_force_commit or joy_force_commit:
            self.add(key, val, step)
            if bucket == "tail":
                self._tail_write_steps.append(int(step))
            self._write_steps.append(int(step))
            extra_fields = {**_log_surp_diag, "alert_active": alert_active}
            if k3_for_log is not None:
                extra_fields["k3_sim"] = round(float(k3_for_log), 4)
            reason_str = "add"
            if salience_force_commit:
                reason_str = "salience_force"
            if joy_force_commit:
                reason_str = "joy_override"
            self._log_write(step, action="commit", reason=reason_str, sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, entropy_raw=ent_raw, used_key_id=idx,
                            l2_penalty=l2_penalty, surprise_threshold=surprise_thr, top5_neighbors_dist=top5_dist,
                            extra=extra_fields)
            return {"status": "accepted", "reason": "add"}
        else:
            self._append_csv("rollbacks.csv", {
                "step": int(step),
                "reason": "bench_fail",
                "surprise": round(float(surprise), 6) if surprise is not None else None,
                "entropy": round(float(ent), 6),
                "entropy_raw": round(float(ent_raw), 6),
                "l2_penalty": round(float(l2_penalty), 6) if l2_penalty is not None else None,
                "surprise_threshold": round(float(surprise_thr), 6) if surprise_thr is not None else None,
                # autonomy mode: do not log intent toggle
            })
            extra_fields = {**_log_surp_diag, "alert_active": alert_active}
            if k3_for_log is not None:
                extra_fields["k3_sim"] = round(float(k3_for_log), 4)
            self._log_write(step, action="rollback", reason="bench_fail", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, entropy_raw=ent_raw, used_key_id=idx,
                           l2_penalty=l2_penalty, surprise_threshold=surprise_thr, top5_neighbors_dist=top5_dist,
                           extra=extra_fields)
            return {"status": "rolled_back", "reason": "bench_fail"}

    def _merge_into(self, idx: int, val: Dict[str, Any], beta: float = 0.2):
        try:
            tgt = self.items[idx]
            if torch.is_tensor(val.get("z_slow")) and torch.is_tensor(tgt.val.get("z_slow")):
                z_old = tgt.val["z_slow"]
                z_new = val["z_slow"]
                z_m = F.normalize((1.0 - beta) * z_old + beta * z_new, dim=-1)
                tgt.val["z_slow"] = z_m.detach().to(self.device)
        except Exception:
            pass

    def _decay_shadow_bank(self, step: int):
        """Luria improvement: periodic shadow bank decay to prevent unbounded growth.

        - Decays every shadow_decay_interval steps (default: 500)
        - Forces decay if size >= shadow_size_hard_cap (default: 100)
        - Keeps top 50% by score
        """
        if not self.shadow_bank:
            return

        # Check if decay is needed
        should_decay = False
        reason = ""
        if len(self.shadow_bank) >= self.shadow_size_hard_cap:
            # Force decay if size >= hard cap
            should_decay = True
            reason = f"size>={self.shadow_size_hard_cap}"
        elif self._last_shadow_decay >= 0 and (step - self._last_shadow_decay) >= self.shadow_decay_interval:
            # Periodic decay
            should_decay = True
            reason = f"interval={self.shadow_decay_interval}"

        if should_decay:
            size_before = len(self.shadow_bank)
            # Keep top 50% by score
            scored = [(getattr(it, "_score", 0.0), it) for it in self.shadow_bank]
            scored.sort(key=lambda x: x[0], reverse=True)
            keep_n = max(1, len(scored) // 2)
            self.shadow_bank = [it for _, it in scored[:keep_n]]
            self._last_shadow_decay = step
            self._log_write(step, action="shadow_decay", reason=reason, extra={
                "size_before": size_before,
                "size_after": len(self.shadow_bank),
                "discarded": size_before - len(self.shadow_bank)
            })

    def _stage_shadow(self, key: torch.Tensor, val: Dict[str, Any], step: int, score: float, surprise: Optional[float] = None):
        try:
            if len(self.shadow_bank) >= self.shadow_bank_cap:
                # drop worst scored
                worst_idx = min(range(len(self.shadow_bank)), key=lambda i: getattr(self.shadow_bank[i], "_score", -1.0))
                self.shadow_bank.pop(worst_idx)
            it = MemoryItem(key=key.detach().to(self.device), val={k: (v.detach().to(self.device) if torch.is_tensor(v) else v) for k, v in val.items()}, step=step)
            setattr(it, "_score", float(score))  # attach ephemeral score
            if surprise is not None:
                setattr(it, "_surprise", float(surprise))  # Luria: track surprise for promote gate
            self.shadow_bank.append(it)
            self._log_write(step, action="stage", reason="staged", extra={"score": round(float(score), 4), "surprise": round(float(surprise), 4) if surprise is not None else None})
        except Exception:
            pass

    def _rate_limit_ok(self, step: int, bucket: Optional[str] = None) -> bool:
        # allow at most write_per_turn per exact step id
        per_turn = sum(1 for s in self._write_steps if s == step)
        if per_turn >= self.write_per_turn:
            return False

        # Lekiltan: apply warmup ramp and dampen adjustments
        ramp = self._warmup_ramp_factor(step)
        dampen_active = self._check_zscore_dampen(step)

        # Effective write budget with warmup ramp
        effective_write_per_100 = int(self.write_per_100 * ramp)
        # Dampen: reduce budget by 2 if active
        if dampen_active:
            effective_write_per_100 = max(0, effective_write_per_100 - 2)

        # allow at most effective_write_per_100 per rolling window of 100 steps
        window = [s for s in self._write_steps if (step - 100) < s <= step]
        if len(window) >= effective_write_per_100:
            return False

        if bucket == "tail":
            effective_tail = int(self.tail_write_per_100 * ramp)
            if dampen_active:
                effective_tail = max(0, effective_tail - 1)
            tail_window = [s for s in self._tail_write_steps if (step - 100) < s <= step]
            if len(tail_window) >= effective_tail:
                return False
        return True

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        data_sorted = sorted(data)
        k = max(0, min(len(data_sorted) - 1, int(round((p / 100.0) * (len(data_sorted) - 1)))))
        return data_sorted[k]

    # -------------------------
    # Core memory operations
    # -------------------------
    def _prune(self):
        if len(self.items) > self.max_items:
            # simple FIFO prune
            self.items = self.items[-self.max_items:]

    def add(self, key: torch.Tensor, val: Dict[str, Any], step: int):
        packed_val = {}
        for k, v in val.items():
            if torch.is_tensor(v):
                packed_val[k] = v.detach().to(self.device)
            else:
                packed_val[k] = v
        self.items.append(
            MemoryItem(
                key=key.detach().to(self.device),
                val=packed_val,
                step=step,
            )
        )
        self._prune()
        if self.enable_learning:
            self._ensure_learner()

    def set_write_intent(self, toggle: Optional[bool]) -> None:
        """Allow external controllers (e.g., Luria identity intent) to override write gating."""
        if toggle is None:
            self.intent_write_toggle = None
        else:
            self.intent_write_toggle = bool(toggle)

    def retrieve(self, query: torch.Tensor, topk: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        self._usage_tick += 1
        keys = torch.stack([it.key for it in self.items], dim=0)  # [N, D]
        q = query.unsqueeze(0)                                    # [1, D]
        sim = F.cosine_similarity(keys, q.expand_as(keys), dim=-1)  # [N]

        # Add step-based weighting for diversity (EMA-based selection)
        import math
        max_step = max(it.step for it in self.items)
        step_weights = torch.tensor(
            [math.exp(-0.001 * (max_step - it.step)) for it in self.items],
            device=keys.device
        )
        # Re-use penalty channel: similarity to recent queries
        if self._input_history:
            recent = torch.stack(self._input_history[-min(32, len(self._input_history)):], dim=0).to(keys.device)
            recent = F.normalize(recent, dim=-1)
            sim_recent = torch.matmul(keys, recent.transpose(0, 1)).mean(dim=1)
            sim = sim - float(self.lambda_penalty) * sim_recent
        # Cooldown penalty
        if self._last_used_step_per_idx:
            cool_pen = torch.zeros_like(sim)
            for i, last in self._last_used_step_per_idx.items():
                if 0 <= i < sim.numel() and (self._usage_tick - last) <= self.cooldown_steps:
                    cool_pen[i] = 0.2
            sim = sim - cool_pen
        sim_weighted = sim * (0.7 + 0.3 * step_weights)

        vals, idx = torch.topk(sim_weighted, k=min(topk, keys.size(0)))
        selected = idx.tolist()
        for i in selected:
            self._last_used_step_per_idx[int(i)] = self._usage_tick
        return [self.items[i] for i in selected]

    def sample(self, batch: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        return random.sample(self.items, k=min(batch, len(self.items)))

    # -------------------------
    # Learning helpers
    # -------------------------
    def _iter_items_with_fields(self, key_field: str, target_field: str):
        for it in self.items:
            src = it.val.get(key_field)
            tgt = it.val.get(target_field)
            if torch.is_tensor(src) and torch.is_tensor(tgt):
                yield it

    def _ensure_learner(self, key_field: str = "avg_emb", target_field: str = "z_slow"):
        if not self.enable_learning or self._learner is not None:
            return
        for it in reversed(self.items):
            src = it.val.get(key_field)
            tgt = it.val.get(target_field)
            if torch.is_tensor(src) and torch.is_tensor(tgt):
                in_dim = src.numel()
                out_dim = tgt.numel()
                self._learn_fields = (key_field, target_field)
                self._learner = MemoryLearner(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=self.learn_hidden,
                    depth=self.learn_depth,
                ).to(self.device)
                self._optimizer = optim.Adam(
                    self._learner.parameters(),
                    lr=self.learn_rate,
                    weight_decay=self.learn_weight_decay,
                )
                break

    def has_learner(self) -> bool:
        return self._learner is not None

    @property
    def learner_loss(self) -> Optional[float]:
        """Exponential moving average of training loss (if learning is enabled)."""
        return self._ema_loss

    def train_model(
        self,
        steps: int = 1,
        batch_size: int = 32,
        key_field: str = "avg_emb",
        target_field: str = "z_slow",
        shuffle: bool = True,
    ) -> Optional[float]:
        """Train the internal learner to map stored embeddings to latent codes.

        Returns the average loss over the executed steps (MSE) or ``None`` if no
        update was applied (e.g. insufficient data or learning disabled).
        """
        if not self.enable_learning:
            raise RuntimeError("Learning is disabled for this memory instance.")

        available = [it for it in self._iter_items_with_fields(key_field, target_field)]
        if not available:
            return None
        if len(available) < min(self.min_train_items, batch_size) and len(available) < 2:
            return None

        if self._learn_fields is None:
            self._learn_fields = (key_field, target_field)
        elif self._learn_fields != (key_field, target_field):
            raise ValueError(
                f"Learner already initialised for fields {self._learn_fields},"
                f" cannot retrain with ({key_field}, {target_field})."
            )

        self._ensure_learner(key_field=key_field, target_field=target_field)
        if self._learner is None or self._optimizer is None:
            return None

        self._learner.train()
        losses = []
        for _ in range(steps):
            if shuffle:
                batch = random.sample(available, k=min(batch_size, len(available)))
            else:
                batch = available[:min(batch_size, len(available))]
            inputs = torch.stack([it.val[key_field] for it in batch], dim=0).to(self.device)
            targets = torch.stack([it.val[target_field] for it in batch], dim=0).to(self.device)

            preds = self._learner(inputs)
            loss = F.mse_loss(preds, targets)

            self._optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._learner.parameters(), self.grad_clip)
            self._optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            if self._ema_loss is None:
                self._ema_loss = loss_val
            else:
                self._ema_loss = self.ema_decay * self._ema_loss + (1 - self.ema_decay) * loss_val
            losses.append(loss_val)

        if not losses:
            return None
        return sum(losses) / len(losses)

    def predict(
        self,
        avg_emb: torch.Tensor,
        topk: int = 4,
        blend: float = 0.5,
        key_field: str = "avg_emb",
        target_field: str = "z_slow",
        return_components: bool = False,
    ):
        """Infer a latent vector from memory using learned mapping + nearest neighbours.

        Args:
            avg_emb: Un-normalised average embedding vector of shape [D].
            topk: Number of neighbours to use for the retrieval component.
            blend: Weight for the learner prediction when both sources are available.
            key_field: Field name that stores the embedding in memory items.
            target_field: Field name for the latent/slow vector to recover.
            return_components: If True, also return the raw components and items.

        Returns:
            prediction (Tensor or None) if ``return_components`` is False.
            Otherwise ``(prediction, neighbours, info_dict)`` where ``info_dict``
            exposes the individual contributions.
        """
        if avg_emb.dim() != 1:
            raise ValueError("avg_emb must be a 1D tensor")

        device = torch.device(self.device)
        avg_emb = avg_emb.to(device)
        normed = F.normalize(avg_emb.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)

        neighbours = self.retrieve(normed, topk=topk) if topk > 0 else []
        knn_vec = None
        if neighbours:
            cand = [it.val.get(target_field) for it in neighbours if torch.is_tensor(it.val.get(target_field))]
            if cand:
                knn_vec = torch.stack(cand, dim=0).mean(dim=0)

        model_vec = None
        if self._learner is not None and self.enable_learning:
            self._learner.eval()
            with torch.no_grad():
                model_vec = self._learner(avg_emb.unsqueeze(0)).squeeze(0)

        prediction = None
        if model_vec is not None and knn_vec is not None:
            prediction = blend * model_vec + (1.0 - blend) * knn_vec
        elif model_vec is not None:
            prediction = model_vec
        else:
            prediction = knn_vec

        info = {
            "model_pred": model_vec.detach() if isinstance(model_vec, torch.Tensor) else None,
            "knn_pred": knn_vec.detach() if isinstance(knn_vec, torch.Tensor) else None,
            "ema_loss": self._ema_loss,
            "topk": topk,
            "blend": blend,
        }

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach()
        if not return_components:
            return prediction
        return prediction, neighbours, info

    # -------------------------
    # Real-time state tracking
    # -------------------------
    def update_state(self, key: str, value: Any):
        """Update session state with a key-value pair."""
        self._session_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        return self._session_state.get(key, default)

    def clear_state(self):
        """Clear all session state."""
        self._session_state.clear()
        self._input_history.clear()
        self._state_history.clear()

    def track_input(self, avg_emb: torch.Tensor, z_result: Optional[torch.Tensor] = None):
        """Track input embedding and optionally its result for history."""
        # Skip duplicate inputs to maintain memory diversity
        if len(self._input_history) > 0:
            last = self._input_history[-1]
            sim = F.cosine_similarity(avg_emb.detach().cpu().unsqueeze(0), last.unsqueeze(0), dim=1)
            if sim.item() > 0.98:
                return  # skip duplicate

        self._input_history.append(avg_emb.detach().cpu())
        if len(self._input_history) > self._max_history:
            self._input_history = self._input_history[-self._max_history:]

        if z_result is not None:
            # Also check for duplicate in state history
            if len(self._state_history) > 0:
                last_state = self._state_history[-1]
                sim = F.cosine_similarity(avg_emb.detach().cpu().unsqueeze(0), last_state["avg_emb"].unsqueeze(0), dim=1)
                if sim.item() > 0.98:
                    return  # skip duplicate

            state = {"avg_emb": avg_emb.detach().cpu(), "z": z_result.detach().cpu()}
            self._state_history.append(state)
            if len(self._state_history) > self._max_history:
                self._state_history = self._state_history[-self._max_history:]

    def get_or_predict_z(
        self,
        avg_emb: torch.Tensor,
        z_init: Optional[torch.Tensor] = None,
        topk: int = 4,
        blend: float = 0.5,
        use_predictor: bool = True,
        key_field: str = "avg_emb",
        target_field: str = "z_slow",
    ) -> torch.Tensor:
        """Get or predict z from memory using learned predictor and retrieval.

        Args:
            avg_emb: Average embedding tensor [D]
            z_init: Initial z to use if memory has no prediction (optional)
            topk: Number of neighbours for retrieval
            blend: Blend factor between learned and retrieval predictions
            use_predictor: Whether to use the learned predictor
            key_field: Field name for embeddings in memory
            target_field: Field name for z vectors in memory

        Returns:
            Predicted or retrieved z tensor
        """
        if avg_emb.dim() != 1:
            raise ValueError("avg_emb must be a 1D tensor")

        # Try to predict from memory if learning is enabled
        if use_predictor and self.enable_learning and self.has_learner():
            prediction = self.predict(
                avg_emb=avg_emb,
                topk=topk,
                blend=blend,
                key_field=key_field,
                target_field=target_field,
                return_components=False,
            )
            if prediction is not None:
                self.track_input(avg_emb, prediction)
                return prediction.to(self.device)

        # Fallback to retrieval-only
        device = torch.device(self.device)
        avg_emb_dev = avg_emb.to(device)
        normed = F.normalize(avg_emb_dev.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)

        neighbours = self.retrieve(normed, topk=topk) if topk > 0 else []
        if neighbours:
            cand = [it.val.get(target_field) for it in neighbours if torch.is_tensor(it.val.get(target_field))]
            if cand:
                z_pred = torch.stack(cand, dim=0).mean(dim=0)
                self.track_input(avg_emb, z_pred)
                return z_pred.to(device)

        # Final fallback to provided init or zero
        if z_init is not None:
            self.track_input(avg_emb, z_init)
            return z_init.to(device)

        # Return zeros if nothing else available
        if neighbours and len(neighbours) > 0:
            dim = neighbours[0].val.get(target_field).numel()
            z_zero = torch.zeros(dim, device=device)
        else:
            z_zero = torch.zeros(128, device=device)  # default dim

        self.track_input(avg_emb, z_zero)
        return z_zero

    def get_recent_context(self, n: int = 5) -> List[Dict[str, torch.Tensor]]:
        """Get the n most recent state history entries."""
        return self._state_history[-n:] if self._state_history else []

    # -------------------------
    # Serialization
    # -------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Export memory state for saving to checkpoint."""
        state = {
            "items": self.items,
            "shadow_bank": self.shadow_bank,
            "config": {
                "max_items": self.max_items,
                "device": self.device,
                "enable_learning": self.enable_learning,
                "learn_hidden": self.learn_hidden,
                "learn_depth": self.learn_depth,
                "learn_rate": self.learn_rate,
                "learn_weight_decay": self.learn_weight_decay,
                "grad_clip": self.grad_clip,
                "ema_decay": self.ema_decay,
                "min_train_items": self.min_train_items,
                "shadow_bank_cap": self.shadow_bank_cap,
                "write_per_turn": self.write_per_turn,
                "write_per_100": self.write_per_100,
                "tail_write_per_100": self.tail_write_per_100,
                "dup_skip_thr": self.dup_skip_thr,
                "near_merge_thr": self.near_merge_thr,
                "stage_mid_low": self.stage_mid_low,
                "entropy_floor": self.entropy_floor,
                "log_dir": self.log_dir,
                "warmup_steps": self.warmup_steps,
                "topk": self.topk,
                "softmax_temp": self.softmax_temp,
                "lambda_penalty": self.lambda_penalty,
            },
            "learner_state": self._learner.state_dict() if self._learner is not None else None,
            "optimizer_state": self._optimizer.state_dict() if self._optimizer is not None else None,
            "ema_loss": self._ema_loss,
            "learn_fields": self._learn_fields,
            # Lekiltan state
            "lekiltan": {
                "surprise_ema8": self._surprise_ema8,
                "model_entropy_window": self._model_entropy_window[-256:] if self._model_entropy_window else [],
                "alert_active_until": self._alert_active_until,
                "dampen_active_until": self._dampen_active_until,
                "warmup_ramp_steps": self._warmup_ramp_steps,
                # L2 regularization
                "key_norm_mode": self.key_norm_mode,
                "key_norm_scale": self.key_norm_scale,
                "l2_reg_weight": self.l2_reg_weight,
                "enforce_unit_norm": self.enforce_unit_norm,
                # Dynamic surprise threshold
                "surprise_threshold_ema": self._surprise_threshold_ema,
                "surprise_threshold_alpha": self._surprise_threshold_alpha,
                "write_threshold_base": self.write_threshold_base,
                "write_threshold_alpha": self.write_threshold_alpha,
                "write_threshold_min": self.write_threshold_min,
                "write_threshold_max": self.write_threshold_max,
                "write_threshold_warmup_end": self.write_threshold_warmup_end,
                "sigmoid_k": self.sigmoid_k,
            },
        }
        # Include lightweight runtime session history (optional context)
        try:
            hist_cap = int(getattr(self, "_max_history", 100))
            input_hist = []
            for t in self._input_history[-hist_cap:]:
                if torch.is_tensor(t):
                    input_hist.append(t.detach().cpu())
            state_hist = []
            for st in self._state_history[-hist_cap:]:
                try:
                    avg = st.get("avg_emb"); zt = st.get("z")
                    rec = {
                        "avg_emb": avg.detach().cpu() if torch.is_tensor(avg) else avg,
                        "z": zt.detach().cpu() if torch.is_tensor(zt) else zt,
                    }
                    state_hist.append(rec)
                except Exception:
                    pass
            state["rt_state"] = {
                "session_state": dict(self._session_state),
                "state_history": state_hist,
                "input_history": input_hist,
            }
        except Exception:
            # Runtime history is optional; never break checkpointing
            pass
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore memory state from checkpoint."""
        self.items = state.get("items", [])
        self.shadow_bank = state.get("shadow_bank", [])

        # Ensure tensors in restored items live on the configured device
        if self.items:
            dev = self.device
            moved_items: List[MemoryItem] = []
            for it in self.items:
                key = it.key
                try:
                    key = key.to(dev)
                except Exception:
                    # Fallback: keep as-is if device move fails
                    pass

                packed_val = {}
                for k, v in it.val.items():
                    if torch.is_tensor(v):
                        try:
                            packed_val[k] = v.to(dev)
                        except Exception:
                            packed_val[k] = v
                    else:
                        packed_val[k] = v
                moved_items.append(MemoryItem(key=key, val=packed_val, step=it.step))
            self.items = moved_items

        # Restore learner if it was saved
        if state.get("learner_state") is not None and state.get("learn_fields") is not None:
            self._learn_fields = state["learn_fields"]
            key_field, target_field = self._learn_fields

            # Infer dimensions from saved items
            for it in reversed(self.items):
                src = it.val.get(key_field)
                tgt = it.val.get(target_field)
                if torch.is_tensor(src) and torch.is_tensor(tgt):
                    in_dim = src.numel()
                    out_dim = tgt.numel()

                    self._learner = MemoryLearner(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        hidden_dim=self.learn_hidden,
                        depth=self.learn_depth,
                    ).to(self.device)
                    self._learner.load_state_dict(state["learner_state"])

                    self._optimizer = optim.Adam(
                        self._learner.parameters(),
                        lr=self.learn_rate,
                        weight_decay=self.learn_weight_decay,
                    )
                    if state.get("optimizer_state") is not None:
                        self._optimizer.load_state_dict(state["optimizer_state"])

                    break

        self._ema_loss = state.get("ema_loss")

        # Restore optional runtime state
        rt = state.get("rt_state", None)
        if rt:
            try:
                self._session_state = dict(rt.get("session_state", {}))
            except Exception:
                self._session_state = {}
            # input history tensors on CPU for consistency
            self._input_history = []
            for t in rt.get("input_history", []) or []:
                if torch.is_tensor(t):
                    try:
                        self._input_history.append(t.detach().cpu())
                    except Exception:
                        pass
            # state history entries with tensors on CPU
            self._state_history = []
            for st in rt.get("state_history", []) or []:
                try:
                    avg = st.get("avg_emb"); zt = st.get("z")
                    rec = {
                        "avg_emb": avg.detach().cpu() if torch.is_tensor(avg) else avg,
                        "z": zt.detach().cpu() if torch.is_tensor(zt) else zt,
                    }
                    self._state_history.append(rec)
                except Exception:
                    pass

        # Restore Lekiltan state
        lek = state.get("lekiltan", {})
        if lek:
            try:
                self._surprise_ema8 = lek.get("surprise_ema8")
                self._model_entropy_window = list(lek.get("model_entropy_window", []))
                self._alert_active_until = lek.get("alert_active_until", -1)
                self._dampen_active_until = lek.get("dampen_active_until", -1)
                self._warmup_ramp_steps = lek.get("warmup_ramp_steps", 1000)
                # L2 regularization
                self.key_norm_mode = lek.get("key_norm_mode", "scaled_l2")
                self.key_norm_scale = lek.get("key_norm_scale", 0.85)
                self.l2_reg_weight = lek.get("l2_reg_weight", 1e-4)
                self.enforce_unit_norm = lek.get("enforce_unit_norm", True)
                # Dynamic surprise threshold
                self._surprise_threshold_ema = lek.get("surprise_threshold_ema")
                self._surprise_threshold_alpha = lek.get("surprise_threshold_alpha", 0.1)
                self.write_threshold_base = lek.get("write_threshold_base", 0.05)
                self.write_threshold_alpha = lek.get("write_threshold_alpha", -0.20)
                self.write_threshold_min = lek.get("write_threshold_min", 0.05)
                self.write_threshold_max = lek.get("write_threshold_max", 0.40)
                self.write_threshold_warmup_end = lek.get("write_threshold_warmup_end", 1000)
                self.sigmoid_k = lek.get("sigmoid_k", 12.0)
            except Exception:
                pass

    # -------------------------
    # Maintenance
    # -------------------------
    def rebalance(self, max_checks: int = 256):
        """Periodic maintenance: prune duplicates, apply TTL/decay, promote staged.

        - Remove exact dups (cos>dup_skip_thr) keeping latest.
        - Promote best shadow items if below write budget.
        """
        step = self._last_step_seen
        if len(self.items) <= 1:
            return
        # Dedup pass (local, not O(N^2) full): sample anchors
        idxs = list(range(len(self.items)))
        self._rng.shuffle(idxs)
        idxs = idxs[: min(max_checks, len(idxs))]
        to_drop = set()
        for i in idxs:
            if i in to_drop:
                continue
            key_i = self.items[i].key
            j, s = self._nearest_idx_and_sim(key_i)
            if j is not None and j != i and s >= self.dup_skip_thr:
                # drop older one
                if self.items[i].step <= self.items[j].step:
                    to_drop.add(i)
                else:
                    to_drop.add(j)
        if to_drop:
            self.items = [it for k,it in enumerate(self.items) if k not in to_drop]

        # Promote from shadow if we have budget (hotfix: ensure attempts happen with relaxed gates)
        shadow_size = len(self.shadow_bank)
        budget = max(0, self.write_per_100 - len([s for s in self._write_steps if (self._last_step_seen - 100) < s <= self._last_step_seen]))
        if not self.shadow_bank:
            return
        if shadow_size <= 20:
            # Too small to rebalance; log skip with reason
            try:
                self._append_csv("rebalance.csv", {
                    "step": int(step), "event": "RB_SKIP", "reason": "shadow_too_small",
                    "promote_fail_reason": "-", "shadow_size": int(shadow_size),
                    "eligible_cnt": 0, "promoted": 0, "cutoff_surprise": "-"
                })
            except Exception:
                pass
            return

        # Proceed with promotion selection
        if budget >= 0:  # allow safety pin even when budget==0
            # Promotion scoring: 0.6*score + 0.1*recency + 0.3*diversity bonus (1 - max_sim)
            if self.items:
                mem_keys = torch.stack([it.key for it in self.items], dim=0)
            else:
                mem_keys = None
            max_step = max((it.step for it in self.shadow_bank), default=0)
            min_step = min((it.step for it in self.shadow_bank), default=0)
            span = max(1, max_step - min_step)

            def promo_score(it: MemoryItem) -> float:
                base = float(getattr(it, "_score", 0.0))
                rec = float((it.step - min_step) / span)
                div = 0.0
                try:
                    if mem_keys is not None and mem_keys.numel() > 0:
                        sims = torch.matmul(mem_keys, it.key.to(mem_keys.device))
                        max_sim = float(torch.max(sims).item())
                        div = 1.0 - max_sim
                except Exception:
                    pass
                return 0.6 * base + 0.1 * rec + 0.3 * div

            ranked = sorted(self.shadow_bank, key=promo_score, reverse=True)

            # Promotion gates per Choebang-yak: primary p55 with fallback p50, plus next-cycle 5p relax on repeated failures
            surprises_all = [getattr(it, "_surprise", None) for it in self.shadow_bank]
            surprises_vals = [float(s) for s in surprises_all if isinstance(s, (int, float))]
            p_used = None
            thr_val = None
            # Dual-cut params (percentile + z-cut)
            z_cut = 0.25
            thr_z = None
            surprise_cut = None
            eligible_before = 0
            eligible_after = 0
            retry_count = 0
            util_ratio = 0.33

            # Start by taking a generous slice bounded by budget; decouple with minimums below
            slice_len = max(budget, 1) if budget > 0 else 1
            # Ensure exploration capacity (decouple from budget)
            promote_attempts_min = 6
            eps_explore = 4  # attempts even when budget==0
            # Pre-expand slice to allow further filtering
            slice_len = max(slice_len, promote_attempts_min * 2)
            candidates = ranked[: min(len(ranked), max(slice_len, 1))]

            promote = []
            eligible_cnt = 0
            if surprises_vals:
                # Primary cutoff p55, optionally relaxed by 5p if prior two cycles had no promotions
                import math
                p_primary = max(5, 55 - int(self._rebalance_relax_bonus))
                p_used = p_primary
                try:
                    thr_val = self._percentile(surprises_vals, p_used)
                except Exception:
                    thr_val = None
                # z-cut (EMA-free, per-round from shadow stats)
                try:
                    mu = sum(surprises_vals) / max(1, len(surprises_vals))
                    var = sum((x - mu) ** 2 for x in surprises_vals) / max(1, len(surprises_vals))
                    sigma = math.sqrt(max(0.0, var))
                    thr_z = mu + z_cut * sigma
                except Exception:
                    thr_z = None
                if thr_val is not None:
                    surprise_cut = max(thr_val, (thr_z if thr_z is not None else thr_val))
                    eligible = [it for it in candidates if (getattr(it, "_surprise", -1.0) is not None and float(getattr(it, "_surprise", -1.0)) >= surprise_cut)]
                    eligible_cnt = len(eligible)
                    eligible_before = eligible_cnt
                    promote = list(eligible)
                # Fallback to p50 within the same round if empty
                if not promote:
                    retry_count += 1
                    p_used = 50
                    try:
                        thr_val = self._percentile(surprises_vals, p_used)
                    except Exception:
                        thr_val = None
                    # relax z_cut on retry
                    z_cut = 0.10
                    try:
                        mu = sum(surprises_vals) / max(1, len(surprises_vals))
                        var = sum((x - mu) ** 2 for x in surprises_vals) / max(1, len(surprises_vals))
                        sigma = math.sqrt(max(0.0, var))
                        thr_z = mu + z_cut * sigma
                    except Exception:
                        thr_z = None
                    if thr_val is not None:
                        surprise_cut = max(thr_val, (thr_z if thr_z is not None else thr_val))
                        eligible = [it for it in candidates if (getattr(it, "_surprise", -1.0) is not None and float(getattr(it, "_surprise", -1.0)) >= surprise_cut)]
                        eligible_cnt = len(eligible)
                        promote = list(eligible)
                # If still empty, expand utility channel ratio 1/3 -> 1/2 and retry selection augment
                if not promote:
                    retry_count += 1
                    util_ratio = 0.5
            # If still empty or no surprises recorded, fall back to top-k by surprise across bank
            if not promote:
                # Build a sorted index by surprise (missing surprise treated as very low)
                scored_indices = list(range(len(self.shadow_bank)))
                scored_indices.sort(key=lambda i: float(getattr(self.shadow_bank[i], "_surprise", -1.0)), reverse=True)
                promote = [self.shadow_bank[i] for i in scored_indices[: max(1, slice_len)]]

            # Auxiliary channel: allow plain-but-useful samples with low k3_avg_sim (<=0.58)
            try:
                util_extra = []
                for it in candidates:
                    if it in promote:
                        continue
                    sim3, _hn = self._k3_avg_sim(it.key)
                    if sim3 >= 0 and sim3 <= 0.58:
                        util_extra.append(it)
                # Add a number of utility items (ratio of candidate slice, default 1/3; on retry up to 1/2)
                if util_extra:
                    limit = max(1, int(max(1, len(candidates)) * float(util_ratio)))
                    promote.extend(util_extra[:limit])
            except Exception:
                pass

            # Minimum candidates and minimum shadow gate
            min_shadow_to_promote = 40
            if shadow_size < min_shadow_to_promote and len(promote) == 0:
                try:
                    self._append_csv("rebalance.csv", {
                        "step": int(step), "event": "RB_SKIP", "reason": "below_min_shadow",
                        "promote_fail_reason": "-", "shadow_size": int(shadow_size),
                        "eligible_cnt": 0, "promoted": 0, "cutoff_surprise": "-"
                    })
                except Exception:
                    pass
                return

            min_candidates = max(16, int(0.05 * shadow_size))
            if len(promote) < min_candidates:
                # Fill from top by surprise across bank
                scored_indices = list(range(len(self.shadow_bank)))
                scored_indices.sort(key=lambda i: float(getattr(self.shadow_bank[i], "_surprise", -1.0)), reverse=True)
                need = min_candidates - len(promote)
                extra = []
                for i in scored_indices:
                    it = self.shadow_bank[i]
                    if it not in promote:
                        extra.append(it)
                        if len(extra) >= need:
                            break
                promote = promote + extra

            # Cap attempts per rebalance window; decouple from budget with minimums and epsilon explore
            max_attempts = 6
            if budget > 0:
                max_attempts = min(max_attempts, budget)
            # Ensure minimum attempts: 6 when budget allows, else epsilon=4 even if budget==0
            if budget > 0:
                min_attempts = min(len(promote), promote_attempts_min)
            else:
                min_attempts = min(len(promote), eps_explore)
            target_attempts = min(max_attempts, len(promote))
            if budget > 0:
                target_attempts = max(target_attempts, min_attempts)
            else:
                target_attempts = max(target_attempts, min_attempts)
            # Expand slice to be at least 2x attempts
            slice_len = max(slice_len, target_attempts * 2)
            promote = promote[: max(1, target_attempts)]

            attempts = 0
            success = 0
            try:
                self._append_csv("rebalance.csv", {
                    "step": int(step), "event": "RB_START", "shadow_size": int(shadow_size),
                    "p": p_used if p_used is not None else "-",
                    "cutoff_pctl": p_used if p_used is not None else "-",
                    "thr": thr_val if thr_val is not None else "-",
                    "cutoff_surprise": thr_val if thr_val is not None else "-",
                    "cutoff_z": thr_z if thr_z is not None else "-",
                    "eligible_cnt": int(eligible_cnt), "k": len(promote),
                    "slice_len": int(slice_len),
                    "util_ratio": float(util_ratio),
                    "relax_bonus": int(self._rebalance_relax_bonus)
                })
            except Exception:
                pass

            # Execute promotions
            picked = []
            for it in promote:
                attempts += 1
                picked.append(it)
                self.add(it.key, it.val, it.step)
                self._log_write(it.step, action="promote", reason="shadow_promote")
                success += 1

            # Safety pin: ensure a small number of constrained promotions if none happened and bank is sizable
            if attempts == 0 and shadow_size > 80:
                scored_indices = list(range(len(self.shadow_bank)))
                scored_indices.sort(key=lambda i: float(getattr(self.shadow_bank[i], "_surprise", -1.0)), reverse=True)
                topK = [self.shadow_bank[i] for i in scored_indices[: min(8, len(scored_indices))]]
                safe = []
                for it in topK:
                    try:
                        k3s, _hn = self._k3_avg_sim(it.key)
                    except Exception:
                        k3s = -1.0
                    # recency: top 50%
                    rec = 0.0
                    try:
                        rec = float((it.step - min_step) / span)
                    except Exception:
                        pass
                    sim_max = 1.0
                    try:
                        if mem_keys is not None and mem_keys.numel() > 0:
                            sims = torch.matmul(mem_keys, it.key.to(mem_keys.device))
                            sim_max = float(torch.max(sims).item())
                    except Exception:
                        pass
                    if (k3s >= 0 and k3s <= 0.62) and (rec >= 0.5) and (sim_max <= 0.92):
                        safe.append(it)
                        if len(safe) >= min(3, max(1, promote_attempts_min // 2)):
                            break
                for it in safe:
                    self.add(it.key, it.val, it.step)
                    self._log_write(it.step, action="promote", reason="shadow_promote_safety")
                if safe:
                    attempts += len(safe)
                    success += len(safe)
                    picked.extend(safe)

            # Remove promoted from shadow bank
            if picked:
                remaining = [it for it in self.shadow_bank if it not in picked]
                self.shadow_bank = remaining

            # Update relax bonus state for next cycle
            if success == 0:
                self._rebalance_no_promo_runs += 1
            else:
                self._rebalance_no_promo_runs = 0
            self._rebalance_relax_bonus = 5 if self._rebalance_no_promo_runs >= 2 else 0

            try:
                eligible_after = len(promote)
                self._append_csv("rebalance.csv", {
                    "step": int(step), "event": "RB_DONE",
                    "promote_attempts": attempts, "promote_success": success, "actual_promoted": success,
                    "promoted": success,
                    "eligible_cnt": int(eligible_cnt),
                    "eligible_before": int(eligible_before),
                    "eligible_after": int(eligible_after),
                    "retry_count": int(retry_count),
                    "util_ratio": float(util_ratio),
                    "cutoff_surprise": thr_val if thr_val is not None else "-",
                    "cutoff_z": thr_z if thr_z is not None else "-",
                    "promote_fail_reason": "-" if success > 0 else "no_eligible_after_retry"
                })
            except Exception:
                pass

    # -------------------------
    # Policy tuning support
    # -------------------------
    def set_policy_thresholds(
        self,
        *,
        dup_skip_thr: Optional[float] = 0.95,
        near_merge_thr: Optional[float] = 0.84,
        stage_mid_low: Optional[float] = 0.73,
        entropy_floor: Optional[float] = 0.15,
        write_per_turn: Optional[int] = 1,
        write_per_100: Optional[int] = 12,
        tail_write_per_100: Optional[int] = 6,
    ):
        if dup_skip_thr is not None:
            self.dup_skip_thr = dup_skip_thr
        if near_merge_thr is not None:
            self.near_merge_thr = near_merge_thr
        if stage_mid_low is not None:
            self.stage_mid_low = stage_mid_low
        if entropy_floor is not None:
            self.entropy_floor = entropy_floor
        if write_per_turn is not None:
            self.write_per_turn = int(write_per_turn)
        if write_per_100 is not None:
            self.write_per_100 = int(write_per_100)
        if tail_write_per_100 is not None:
            self.tail_write_per_100 = int(tail_write_per_100)

    # -------------------------
    # Exploration pulse context manager
    # -------------------------
    @contextlib.contextmanager
    def exploration_pulse(self, topk: int = 8, lambda_pulse: Optional[float] = None, temp_pulse: Optional[float] = None):
        old = (self.topk, self.lambda_penalty, self.softmax_temp)
        try:
            self.topk = int(topk)
            if lambda_pulse is not None:
                self.lambda_penalty = float(lambda_pulse)
            if temp_pulse is not None:
                self.softmax_temp = float(temp_pulse)
            yield
        finally:
            self.topk, self.lambda_penalty, self.softmax_temp = old

    # -------------------------
    # Dashboard statistics
    # -------------------------
    def get_memory_stats(self, step: int) -> Dict[str, Any]:
        """Get comprehensive memory system statistics for dashboard.

        Args:
            step: Current training step

        Returns:
            Dictionary with memory, shadow, write_rate, and threshold statistics
        """
        # Write history analysis
        recent_100 = [s for s in self._write_steps if s > step - 100]

        return {
            "step": step,
            "memory": {
                "size": len(self.items),
                "capacity": self.max_items,
                "used_pct": len(self.items) / self.max_items * 100 if self.max_items > 0 else 0.0,
            },
            "shadow": {
                "size": len(self.shadow_bank),
                "capacity": self.shadow_bank_cap,
                "used_pct": len(self.shadow_bank) / self.shadow_bank_cap * 100 if self.shadow_bank_cap > 0 else 0.0,
            },
            "write_rate": {
                "total_writes": len(self._write_steps),
                "recent_100": len(recent_100),
                "rate_per_100": len(recent_100),
            },
            "thresholds": {
                "near_merge_thr": self.near_merge_thr,
                "dup_skip_thr": self.dup_skip_thr,
                "stage_mid_low": self.stage_mid_low,
                "entropy_floor": self.entropy_floor,
            }
        }


    # -------------------------
    # Memory operations (autonomous, 650+ steps)
    # -------------------------
    def synthesize_memories(self, indices: List[int], weights: Optional[List[float]] = None) -> Optional[MemoryItem]:
        """?ㅼ쨷 硫붾え由??⑹꽦: ?щ윭 硫붾え由щ? 媛以묓룊洹좎쑝濡??⑹꽦

        Args:
            indices: ?⑹꽦??硫붾え由??몃뜳??由ъ뒪??            weights: 媛?硫붾え由ъ쓽 媛以묒튂 (None?대㈃ 洹좊벑)

        Returns:
            ?⑹꽦????MemoryItem ?먮뒗 None
        """
        if not indices or not self.items:
            return None

        # Valid indices留??꾪꽣
        valid_indices = [i for i in indices if 0 <= i < len(self.items)]
        if len(valid_indices) < 2:
            return None

        # Weights ?ㅼ젙
        if weights is None:
            weights = [1.0 / len(valid_indices)] * len(valid_indices)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        # Key ?⑹꽦
        keys = [self.items[i].key for i in valid_indices]
        key_synth = sum(w * k for w, k in zip(weights, keys))
        key_synth = F.normalize(key_synth, dim=-1)

        # z_slow ?⑹꽦
        z_slows = [self.items[i].val.get("z_slow") for i in valid_indices]
        z_slows = [z for z in z_slows if torch.is_tensor(z)]
        if z_slows:
            z_synth = sum(w * z for w, z in zip(weights[:len(z_slows)], z_slows))
            z_synth = F.normalize(z_synth, dim=-1)
        else:
            return None

        # avg_emb ?⑹꽦
        avg_embs = [self.items[i].val.get("avg_emb") for i in valid_indices]
        avg_embs = [e for e in avg_embs if torch.is_tensor(e)]
        if avg_embs:
            avg_emb_synth = sum(w * e for w, e in zip(weights[:len(avg_embs)], avg_embs))
        else:
            avg_emb_synth = key_synth  # fallback

        # ??MemoryItem ?앹꽦
        val = {
            "z_slow": z_synth.detach().to(self.device),
            "avg_emb": avg_emb_synth.detach().to(self.device),
            "meta": {"synthesized": True, "source_indices": valid_indices}
        }

        return MemoryItem(
            key=key_synth.detach().to(self.device),
            val=val,
            step=self._last_step_seen
        )

    def split_memory(self, idx: int, noise_scale: float = 0.1) -> Tuple[Optional[MemoryItem], Optional[MemoryItem]]:
        """硫붾え由?遺꾪븷: ?섎굹??硫붾え由щ? ??媛쒕줈 ?섎늻湲?
        Args:
            idx: 遺꾪븷??硫붾え由??몃뜳??            noise_scale: 遺꾪븷 ??異붽????몄씠利??ш린

        Returns:
            (item1, item2) ?먮뒗 (None, None)
        """
        if not self.items or idx < 0 or idx >= len(self.items):
            return None, None

        original = self.items[idx]
        key_orig = original.key
        z_orig = original.val.get("z_slow")

        if not torch.is_tensor(z_orig):
            return None, None

        # ?몄씠利?異붽?濡???諛⑺뼢 ?앹꽦
        noise1 = torch.randn_like(z_orig) * noise_scale
        noise2 = -noise1  # 諛섎? 諛⑺뼢

        z1 = F.normalize(z_orig + noise1, dim=-1)
        z2 = F.normalize(z_orig + noise2, dim=-1)

        key1 = F.normalize(key_orig + torch.randn_like(key_orig) * noise_scale, dim=-1)
        key2 = F.normalize(key_orig - torch.randn_like(key_orig) * noise_scale, dim=-1)

        # avg_emb??遺꾪븷
        avg_emb = original.val.get("avg_emb")
        if torch.is_tensor(avg_emb):
            avg_emb1 = avg_emb + torch.randn_like(avg_emb) * noise_scale
            avg_emb2 = avg_emb - torch.randn_like(avg_emb) * noise_scale
        else:
            avg_emb1 = key1
            avg_emb2 = key2

        item1 = MemoryItem(
            key=key1.detach().to(self.device),
            val={
                "z_slow": z1.detach().to(self.device),
                "avg_emb": avg_emb1.detach().to(self.device),
                "meta": {"split": True, "split_from": idx, "split_id": 0}
            },
            step=self._last_step_seen
        )

        item2 = MemoryItem(
            key=key2.detach().to(self.device),
            val={
                "z_slow": z2.detach().to(self.device),
                "avg_emb": avg_emb2.detach().to(self.device),
                "meta": {"split": True, "split_from": idx, "split_id": 1}
            },
            step=self._last_step_seen
        )

        return item1, item2

    def interpolate_memories(self, idx1: int, idx2: int, alpha: float = 0.5) -> Optional[MemoryItem]:
        """硫붾え由?蹂닿컙: ??硫붾え由??ъ씠??以묎컙媛??앹꽦

        Args:
            idx1, idx2: 蹂닿컙??硫붾え由??몃뜳??            alpha: 蹂닿컙 鍮꾩쑉 (0=idx1, 1=idx2, 0.5=以묎컙)

        Returns:
            蹂닿컙??MemoryItem ?먮뒗 None
        """
        if not self.items or idx1 < 0 or idx2 < 0 or idx1 >= len(self.items) or idx2 >= len(self.items):
            return None

        if idx1 == idx2:
            return None

        item1 = self.items[idx1]
        item2 = self.items[idx2]

        # Key 蹂닿컙
        key_interp = (1 - alpha) * item1.key + alpha * item2.key
        key_interp = F.normalize(key_interp, dim=-1)

        # z_slow 蹂닿컙
        z1 = item1.val.get("z_slow")
        z2 = item2.val.get("z_slow")
        if torch.is_tensor(z1) and torch.is_tensor(z2):
            z_interp = (1 - alpha) * z1 + alpha * z2
            z_interp = F.normalize(z_interp, dim=-1)
        else:
            return None

        # avg_emb 蹂닿컙
        e1 = item1.val.get("avg_emb")
        e2 = item2.val.get("avg_emb")
        if torch.is_tensor(e1) and torch.is_tensor(e2):
            avg_emb_interp = (1 - alpha) * e1 + alpha * e2
        else:
            avg_emb_interp = key_interp

        return MemoryItem(
            key=key_interp.detach().to(self.device),
            val={
                "z_slow": z_interp.detach().to(self.device),
                "avg_emb": avg_emb_interp.detach().to(self.device),
                "meta": {"interpolated": True, "source": [idx1, idx2], "alpha": alpha}
            },
            step=self._last_step_seen
        )

    def crossover_memories(self, idx1: int, idx2: int, crossover_point: float = 0.5) -> Tuple[Optional[MemoryItem], Optional[MemoryItem]]:
        """硫붾え由?援먯감: ??硫붾え由ъ쓽 ?뱀쭠??援먯감?섏뿬 ?덈줈????媛??앹꽦

        Args:
            idx1, idx2: 援먯감??硫붾え由??몃뜳??            crossover_point: 援먯감 吏??(0-1, 踰≫꽣 李⑥썝??鍮꾩쑉)

        Returns:
            (child1, child2) ?먮뒗 (None, None)
        """
        if not self.items or idx1 < 0 or idx2 < 0 or idx1 >= len(self.items) or idx2 >= len(self.items):
            return None, None

        if idx1 == idx2:
            return None, None

        item1 = self.items[idx1]
        item2 = self.items[idx2]

        z1 = item1.val.get("z_slow")
        z2 = item2.val.get("z_slow")

        if not torch.is_tensor(z1) or not torch.is_tensor(z2):
            return None, None

        # Crossover point 寃곗젙
        dim = z1.shape[-1]
        cut = int(dim * crossover_point)

        # z_slow 援먯감
        z_child1 = torch.cat([z1[:cut], z2[cut:]], dim=0)
        z_child2 = torch.cat([z2[:cut], z1[cut:]], dim=0)
        z_child1 = F.normalize(z_child1, dim=-1)
        z_child2 = F.normalize(z_child2, dim=-1)

        # Key 援먯감
        key_child1 = torch.cat([item1.key[:cut], item2.key[cut:]], dim=0)
        key_child2 = torch.cat([item2.key[:cut], item1.key[cut:]], dim=0)
        key_child1 = F.normalize(key_child1, dim=-1)
        key_child2 = F.normalize(key_child2, dim=-1)

        # avg_emb 援먯감
        e1 = item1.val.get("avg_emb")
        e2 = item2.val.get("avg_emb")
        if torch.is_tensor(e1) and torch.is_tensor(e2):
            e_child1 = torch.cat([e1[:cut], e2[cut:]], dim=0)
            e_child2 = torch.cat([e2[:cut], e1[cut:]], dim=0)
        else:
            e_child1 = key_child1
            e_child2 = key_child2

        child1 = MemoryItem(
            key=key_child1.detach().to(self.device),
            val={
                "z_slow": z_child1.detach().to(self.device),
                "avg_emb": e_child1.detach().to(self.device),
                "meta": {"crossover": True, "parents": [idx1, idx2], "child_id": 0}
            },
            step=self._last_step_seen
        )

        child2 = MemoryItem(
            key=key_child2.detach().to(self.device),
            val={
                "z_slow": z_child2.detach().to(self.device),
                "avg_emb": e_child2.detach().to(self.device),
                "meta": {"crossover": True, "parents": [idx1, idx2], "child_id": 1}
            },
            step=self._last_step_seen
        )

        return child1, child2


def make_sketch(avg_emb: torch.Tensor, z_slow: torch.Tensor, meta: Dict[str, Any] = None, repr_text: str = None):
    key = F.normalize(avg_emb, dim=-1)
    val = {"avg_emb": avg_emb, "z_slow": z_slow}
    if meta:
        val["meta"] = meta
    if repr_text is not None:
        val["repr_text"] = repr_text[:2048]  # 濡쒓렇/硫붾え由???＜ 諛⑹?
    return key, val
