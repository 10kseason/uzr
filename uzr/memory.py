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
        max_items: int = 100000,
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
        dup_skip_thr: float = 0.98,
        near_merge_thr: float = 0.90,
        stage_mid_low: float = 0.70,
        entropy_floor: float = 0.3,
        log_dir: str = ".",
        # extras for Luria manual
        warmup_steps: int = 100,
        tail_write_per_100: int = 2,
        topk_default: int = 6,
        softmax_temp: float = 0.2,
        lambda_penalty: float = 0.22,
    ):
        self.items: List[MemoryItem] = []
        self.max_items = max_items
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
        # Diversity/cooldown bookkeeping
        self._last_used_step_per_idx: Dict[int, int] = {}
        self._usage_tick: int = 0
        self.cooldown_steps: int = 80

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
        topk: Optional[int] = None,
        used_key_id: Optional[int] = None,
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
            "topk": int(topk if topk is not None else self.topk),
            "used_key_id": used_key_id,
            "shadow_size": int(len(self.shadow_bank)),
            "mem_size": int(len(self.items)),
        }
        if extra:
            try:
                row.update(extra)
            except Exception:
                pass
        self._append_csv("writes.csv", row)

    # -------------------------
    # Policy: retrieval entropy
    # -------------------------
    def _retrieval_entropy(self, query: torch.Tensor, topk: int = 8) -> float:
        if not self.items:
            return 0.0
        keys = torch.stack([it.key for it in self.items], dim=0)
        q = query.unsqueeze(0)
        sim = F.cosine_similarity(keys, q.expand_as(keys), dim=-1)  # [N]

        # 상위 k만 사용해 분포 구성
        k = min(topk, sim.numel())
        vals, idx = torch.topk(sim, k=k, largest=True)
        sel = vals

        # softmax with clamping
        temp = max(float(self.softmax_temp), 1e-4)
        probs = torch.softmax(sel / temp, dim=-1)
        probs = probs.clamp_min(1e-9)  # 0 확률 금지

        ent = float(-(probs * probs.log()).sum().item())
        # track
        self._entropy_hist.append(ent)
        self.last_meta_entropy = ent
        self._append_csv("entropy.csv", {"step": self._last_step_seen, "entropy": round(ent, 6)})
        return ent

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

        Returns a decision dict with keys: {status: 'accepted'|'merged'|'staged'|'skipped', reason: str}
        """
        self._last_step_seen = int(step)

        # Kill switch via environment
        try:
            if os.environ.get("UZR_MEM_WRITE", "1") == "0":
                self._log_write(step, action="defer", reason="kill_switch")
                return {"status": "staged", "reason": "kill_switch"}
        except Exception:
            pass

        # L2-normalize inputs and enforce
        key = F.normalize(key, dim=-1)
        if torch.is_tensor(val.get("z_slow")):
            val["z_slow"] = F.normalize(val["z_slow"], dim=-1)
        try:
            if key.ndim == 1:
                assert torch.allclose(key.norm(dim=-1), torch.ones((), device=key.device), atol=1e-3)
            if torch.is_tensor(val.get("z_slow")) and val["z_slow"].ndim == 1:
                assert torch.allclose(val["z_slow"].norm(dim=-1), torch.ones((), device=key.device), atol=1e-3)
        except Exception:
            pass

        # 웜업 먼저: 웜업 동안엔 스테이징조차 하지 않음(오염 방지)
        if step < self.warmup_steps:
            self._log_write(step, action="defer", reason="warmup")
            return {"status": "staged", "reason": "warmup"}

        # 엔트로피 게이트는 부트스트랩 이후에만
        entropy_check_start = getattr(self, "entropy_check_start", 32)
        apply_entropy_gate = (len(self.items) >= entropy_check_start)
        ent = self._retrieval_entropy(key) if apply_entropy_gate else 1.0

        if apply_entropy_gate and (ent < self.entropy_floor) and (len(self.items) > 0):
            self._log_write(step, action="defer", reason="entropy_floor", entropy=ent)
            self._stage_shadow(key, val, step, score=0.0)
            return {"status": "staged", "reason": f"entropy<{self.entropy_floor:.2f}"}

        # Similarity 계산 (merge와 skip 모두 필요)
        idx, smax = self._nearest_idx_and_sim(key)

        # Compute predicted z to estimate surprise
        surprise = None
        surp_norm = None
        try:
            z_new = val.get("z_slow")
            z_pred = None
            if torch.is_tensor(z_new):
                # try simple KNN prediction
                neigh = self.retrieve(key, topk=max(4, self.topk))
                if neigh:
                    cand = [it.val.get("z_slow") for it in neigh if torch.is_tensor(it.val.get("z_slow"))]
                    if cand:
                        z_pred = torch.stack(cand, dim=0).mean(dim=0)
            if torch.is_tensor(z_new) and torch.is_tensor(z_pred):
                z_new_n = F.normalize(z_new, dim=-1)
                z_pred_n = F.normalize(z_pred, dim=-1)
                surprise = float(1.0 - torch.clamp(F.cosine_similarity(z_new_n.unsqueeze(0), z_pred_n.unsqueeze(0), dim=-1), -1, 1).item())
                self._surprise_hist.append(surprise)
                hist = self._surprise_hist[-2000:]
                if len(hist) >= 50:
                    p50 = self._percentile(hist, 50)
                    pTau = self._percentile(hist, 85)
                    surp_norm = max(0.0, (surprise - p50) / (pTau - p50 + 1e-6))
        except Exception:
            pass

        # Near-duplicate merge (on key similarity) - 머지 우선
        if idx is not None and smax >= self.near_merge_thr:
            BETA_MIN, BETA_MAX = 0.10, 0.30
            if meta and meta.get("bucket") == "tail":
                BETA_MAX = 0.35
            beta = 0.2
            if surp_norm is not None:
                beta = float(max(BETA_MIN, min(BETA_MAX, BETA_MIN + 0.25 * surp_norm)))
            elif surprise is not None:
                beta = float(max(BETA_MIN, min(BETA_MAX, 0.1 + 0.4 * surprise)))
            self._merge_into(idx, val, beta=beta)
            self._log_write(step, action="merge", reason="near_merge", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, used_key_id=idx)
            return {"status": "merged", "reason": f"near_merge>={self.near_merge_thr}"}

        # Duplicate skip - 머지 이후에 체크
        if idx is not None and smax >= self.dup_skip_thr:
            self._log_write(step, action="skip", reason="dup_skip", sim_max=smax, entropy=ent, used_key_id=idx)
            return {"status": "skipped", "reason": f"dup>={self.dup_skip_thr}"}

        # Stage range
        if idx is not None and smax >= self.stage_mid_low:
            self._stage_shadow(key, val, step, score=smax)
            return {"status": "staged", "reason": f"mid_range>={self.stage_mid_low}"}

        # Write-on-surprise gate
        if surprise is not None:
            thr = self._percentile(self._surprise_hist, 85) if len(self._surprise_hist) >= 20 else 0.15
            if surprise < thr:
                self._stage_shadow(key, val, step, score=surprise)
                return {"status": "staged", "reason": f"surprise<{thr:.3f}"}

        # Rate limiting (with tail bucket budget)
        bucket = meta.get("bucket") if meta else None
        if not self._rate_limit_ok(step, bucket=bucket):
            self._log_write(step, action="defer", reason="rate_limited", surprise=surprise, surp_norm=surp_norm, entropy=ent)
            self._stage_shadow(key, val, step, score=0.0)
            return {"status": "staged", "reason": "rate_limited"}

        # Accept and add w/ optional 2PC bench callback
        commit_ok = True
        if bench_callback is not None:
            try:
                # Stage proposal
                self._log_write(step, action="stage", reason="2pc_proposal", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, used_key_id=idx)
                commit_ok = bool(bench_callback())
            except Exception:
                commit_ok = False
        if commit_ok:
            self.add(key, val, step)
            if bucket == "tail":
                self._tail_write_steps.append(int(step))
            self._write_steps.append(int(step))
            self._log_write(step, action="commit", reason="add", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, used_key_id=idx)
            return {"status": "accepted", "reason": "add"}
        else:
            self._append_csv("rollbacks.csv", {
                "step": int(step),
                "reason": "bench_fail",
                "surprise": round(float(surprise), 6) if surprise is not None else None,
                "entropy": round(float(ent), 6),
            })
            self._log_write(step, action="rollback", reason="bench_fail", sim_max=smax, surprise=surprise, surp_norm=surp_norm, entropy=ent, used_key_id=idx)
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

    def _stage_shadow(self, key: torch.Tensor, val: Dict[str, Any], step: int, score: float):
        try:
            if len(self.shadow_bank) >= self.shadow_bank_cap:
                # drop worst scored
                worst_idx = min(range(len(self.shadow_bank)), key=lambda i: getattr(self.shadow_bank[i], "_score", -1.0))
                self.shadow_bank.pop(worst_idx)
            it = MemoryItem(key=key.detach().to(self.device), val={k: (v.detach().to(self.device) if torch.is_tensor(v) else v) for k, v in val.items()}, step=step)
            setattr(it, "_score", float(score))  # attach ephemeral score
            self.shadow_bank.append(it)
            self._log_write(step, action="stage", reason="staged", extra={"score": round(float(score), 4)})
        except Exception:
            pass

    def _rate_limit_ok(self, step: int, bucket: Optional[str] = None) -> bool:
        # allow at most write_per_turn per exact step id
        per_turn = sum(1 for s in self._write_steps if s == step)
        if per_turn >= self.write_per_turn:
            return False
        # allow at most write_per_100 per rolling window of 100 steps
        window = [s for s in self._write_steps if (step - 100) < s <= step]
        if len(window) >= self.write_per_100:
            return False
        if bucket == "tail":
            tail_window = [s for s in self._tail_write_steps if (step - 100) < s <= step]
            if len(tail_window) >= self.tail_write_per_100:
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

    # -------------------------
    # Maintenance
    # -------------------------
    def rebalance(self, max_checks: int = 256):
        """Periodic maintenance: prune duplicates, apply TTL/decay, promote staged.

        - Remove exact dups (cos>dup_skip_thr) keeping latest.
        - Promote best shadow items if below write budget.
        """
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

        # Promote from shadow if we have budget
        budget = max(0, self.write_per_100 - len([s for s in self._write_steps if (self._last_step_seen - 100) < s <= self._last_step_seen]))
        if budget > 0 and self.shadow_bank:
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
            promote = ranked[: min(len(ranked), budget)] if budget > 0 else []
            # remove promoted from shadow_bank
            if promote:
                remaining = [it for it in self.shadow_bank if it not in promote]
                self.shadow_bank = remaining
                for it in promote:
                    self.add(it.key, it.val, it.step)
                    self._log_write(it.step, action="promote", reason="shadow_promote")

    # -------------------------
    # Policy tuning support
    # -------------------------
    def set_policy_thresholds(
        self,
        *,
        dup_skip_thr: Optional[float] = 0.92,
        near_merge_thr: Optional[float] = 0.85,
        stage_mid_low: Optional[float] = 0.60,
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


def make_sketch(avg_emb: torch.Tensor, z_slow: torch.Tensor, meta: Dict[str, Any] = None, repr_text: str = None):
    key = F.normalize(avg_emb, dim=-1)
    val = {"avg_emb": avg_emb, "z_slow": z_slow}
    if meta:
        val["meta"] = meta
    if repr_text is not None:
        val["repr_text"] = repr_text[:2048]  # 로그/메모리 폭주 방지
    return key, val
