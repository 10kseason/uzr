
import math, unicodedata
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

# ---- Codebook + Transition helpers ----
class CodebookEncoder(nn.Module):
    """codebook token ids -> pooled embedding.

    Supports mean/max/attention pooling over a sequence of token ids.
    """
    def __init__(self, vocab_size: int, emb_dim: int, pool: str = "mean"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        assert pool in ("mean", "max", "attn")
        self.pool = pool
        if pool == "attn":
            self.q = nn.Linear(emb_dim, emb_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L]
        E = self.emb(token_ids)  # [B, L, E]
        if self.pool == "mean":
            return E.mean(dim=1)
        if self.pool == "max":
            return E.max(dim=1).values
        # attention pooling
        q = self.q(E.mean(dim=1)).unsqueeze(1)  # [B,1,E]
        attn = torch.softmax((q * E).sum(dim=-1), dim=1)  # [B,L]
        return (attn.unsqueeze(-1) * E).sum(dim=1)  # [B,E]


class MMFuse(nn.Module):
    """concat(norm(z), u, cb_enc) -> fused representation."""
    def __init__(self, z_dim: int, u_dim: int, cb_dim: int, fused_dim: int):
        super().__init__()
        in_dim = z_dim + u_dim + cb_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, fused_dim),
            nn.ReLU(),
        )

    def forward(self, z_norm: torch.Tensor, u: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_norm, u, cb], dim=-1)
        return self.net(x)


class TransHeadZ(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, hidden_mult: int = 2, spectral: bool = True):
        super().__init__()
        H = max(z_dim * hidden_mult, 32)
        def Lin(a, b):
            base = nn.Linear(a, b)
            return nn.utils.parametrizations.spectral_norm(base) if spectral else base
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            Lin(in_dim, H), nn.ReLU(),
            Lin(H, z_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)  # Δz_pred


class TransHeadCB(nn.Module):
    """next codebook distribution (BOW) or next-token logits."""
    def __init__(self, in_dim: int, vocab_size: int, mode: str = "bag", hidden_mult: int = 2):
        super().__init__()
        H = max(in_dim * hidden_mult, 64)
        self.mode = mode  # "bag": multi-label (BCE), "lm": next-token
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, H), nn.ReLU(),
            nn.Linear(H, vocab_size),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        logits = self.proj(fused)
        return logits

# Optional meta-cognition utilities (SelfEval, entropy, brier)
try:
    from .meta_core import (
        SelfEvalHead,
        sequence_entropy_from_logits as _seq_ent,
        brier_loss_from_conf as _brier_from_conf,
    )
except Exception:  # fallback if not available
    SelfEvalHead = None  # type: ignore
    def _seq_ent(logits, topk=None):
        probs = torch.softmax(logits, dim=-1)
        ent_tok = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
        return ent_tok.mean(dim=1)
    def _brier_from_conf(logits, targets, conf, ignore_index=0):
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = (targets != ignore_index).float()
            correct = (pred == targets).float() * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            acc = (correct.sum(dim=1) / denom)
        return torch.mean((conf - acc) ** 2)

# ---- Byte-level tokenizer (simple) ----
class ByteTokenizer:
    def __init__(self, max_len: int = 128):
        self.vocab_size = 258  # 256 bytes + BOS + EOS
        self.BOS = 256
        self.EOS = 257
        self.max_len = max_len

    def encode(self, s: str):
        b = s.encode("utf-8", errors="ignore")[: self.max_len - 2]
        ids = [self.BOS] + list(b) + [self.EOS]
        pad = [0] * (self.max_len - len(ids))
        return torch.tensor(ids + pad, dtype=torch.long)

    def decode(self, ids):
        # Stop at first EOS, drop BOS/pad, and ignore invalid UTF-8 tails.
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i in (self.BOS, 0):
                continue
            out.append(int(i))
        try:
            return bytes(out).decode("utf-8", errors="ignore")
        except Exception:
            return ""

# ---- Minimal KO+EN tokenizer (character-level, Unicode-aware) ----
class KoEnTokenizer:
    def __init__(self, max_len: int = 128, add_bos: bool = True, add_eos: bool = True):
        self.max_len = max_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3

        chars = []

        # ASCII basic printable (32..126)
        chars.extend(chr(i) for i in range(32, 127))
        # Include newline for safety
        chars.append("\n")

        # Hangul Syllables
        chars.extend(chr(i) for i in range(0xAC00, 0xD7A4))

        # Hangul Compatibility Jamo
        chars.extend(chr(i) for i in range(0x3130, 0x3190))

        # Hangul Jamo blocks
        for start, end in [(0x1100, 0x115F), (0x1160, 0x11A7), (0x11A8, 0x11FF)]:
            chars.extend(chr(i) for i in range(start, end + 1))

        self.itos = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        if not isinstance(text, str):
            text = str(text)
        s = unicodedata.normalize("NFKC", text)

        ids = []
        if self.add_bos:
            ids.append(self.BOS)

        for ch in s:
            idx = self.stoi.get(ch, self.UNK)
            ids.append(idx)
            if len(ids) >= self.max_len - (1 if self.add_eos else 0):
                break

        if self.add_eos:
            ids.append(self.EOS)

        if len(ids) < self.max_len:
            ids.extend([self.PAD] * (self.max_len - len(ids)))

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        out = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = int(idx.item())
            if idx == self.PAD:
                continue
            if idx == self.BOS:
                continue
            if idx == self.EOS:
                break
            if 0 <= idx < len(self.itos):
                ch = self.itos[idx]
                if ch in {"[PAD]", "[BOS]", "[EOS]"}:
                    continue
                if ch == "[UNK]":
                    out.append(" ")
                else:
                    out.append(ch)
            else:
                out.append(" ")
        return "".join(out)

# ---- FiLM conditioning block ----
class FiLM(nn.Module):
    def __init__(self, z_dim: int, d_model: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, d_model * 2)

    def forward(self, h, z):
        # h: [B, T, D]
        gamma, beta = self.fc(z).chunk(2, dim=-1)  # [B, D], [B, D]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return h * (1 + torch.tanh(gamma)) + beta

# ---- Tiny Transformer Encoder ----
class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_head: int = 4, n_layer: int = 4, max_len: int = 128):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.max_len = max_len
        # Toggle via environment variable set by the training script
        # UZR_GRADIENT_CHECKPOINTING in {"1","true","on"} enables layer-wise checkpointing
        try:
            _env_gc = str(os.environ.get("UZR_GRADIENT_CHECKPOINTING", "0")).strip().lower()
        except Exception:
            _env_gc = "0"
        self.use_checkpoint = _env_gc in {"1", "true", "on"}

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos)
        # Use gradient checkpointing only when explicitly enabled, in training, and when grads are active
        use_ckpt = bool(self.use_checkpoint and self.training and h.requires_grad)
        if not use_ckpt:
            h = self.enc(h)
            return h  # [B, T, D]

        # Layer-wise checkpointing through TransformerEncoder layers
        # Guard for implementations without public .layers (older PyTorch) — fallback to regular path
        layers = getattr(self.enc, "layers", None)
        if layers is None:
            h = self.enc(h)
            return h

        for lyr in layers:
            # Wrap single-arg call for checkpoint
            h = _ckpt(lambda inp: lyr(inp), h)
        norm = getattr(self.enc, "norm", None)
        if norm is not None:
            h = norm(h)
        return h  # [B, T, D]


class MultiTimeScaleAdapter(nn.Module):
    """Self-referential gate with fast (learned) and slow (EMA) components."""

    def __init__(self, dim: int, slow_momentum: float = 0.995):
        super().__init__()
        self.dim = int(dim)
        self.slow_momentum = float(slow_momentum)
        if self.dim > 0:
            self.fast = nn.Parameter(torch.zeros(self.dim))
            self.register_buffer("slow", torch.zeros(self.dim))
        else:
            self.fast = None
            self.register_buffer("slow", torch.zeros(1))

    def forward(self) -> torch.Tensor:
        if self.dim == 0 or self.fast is None:
            return self.slow.new_zeros(1)
        return torch.tanh(self.fast + self.slow)

    @torch.no_grad()
    def update_slow(self):
        if self.dim == 0 or self.fast is None:
            return
        momentum = self.slow_momentum
        self.slow.mul_(momentum).add_((1.0 - momentum) * self.fast.detach())

# ---- Universal Z-Rule Runner f_theta ----
class UZRModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, z_dim: int = 128, n_head: int = 4, n_layer: int = 4, max_len: int = 128,
                 z_think_dim: int = 64, z_lang_dim: int = 32, num_langs: int = 4, identity_self_dim: int = 32,
                 identity_intent_dim: Optional[int] = 16, memory=None,
                 z_slow_lang_dim: int = 96, z_slow_logic_dim: int = 96, z_bridge_dim: int = 64,
                 use_self_eval: bool = True,
                 # Transition (optional; initialized later via init_transition_module if needed)
                 ):
        super().__init__()
        self.encoder = TinyEncoder(vocab_size, d_model, n_head, n_layer, max_len)
        self.film = FiLM(z_dim, d_model)
        self.readout = nn.Linear(d_model, vocab_size)  # token-level logits
        # Tokenizer compatibility (PAD/BOS/EOS)
        self.pad_token_id: int = 0
        self.bos_token_id: int = 1
        self.eos_token_id: int = 2
        self.z_dim = z_dim
        self.z_think_dim = z_think_dim
        self.z_lang_dim = z_lang_dim
        self.num_langs = num_langs
        self.identity_self_dim = identity_self_dim
        if identity_intent_dim is None:
            # Default intent space is half the identity (capped at 16) when unspecified.
            desired_intent = min(16, max(0, identity_self_dim // 2))
        else:
            desired_intent = int(max(0, identity_intent_dim))
        if desired_intent >= identity_self_dim:
            desired_intent = max(0, identity_self_dim - 1)
        self.identity_intent_dim = desired_intent
        self.identity_core_dim = self.identity_self_dim - self.identity_intent_dim

        # 3brains dimensions
        self.z_slow_lang_dim = z_slow_lang_dim
        self.z_slow_logic_dim = z_slow_logic_dim
        self.z_bridge_dim = z_bridge_dim

        self.lang_embed = nn.Embedding(num_langs, z_lang_dim)

        # Learned identity embedding for self-awareness (e.g., "루리아")
        self.identity_self = nn.Parameter(torch.randn(identity_self_dim) * 0.02)
        self.identity_self.requires_grad_(False)
        if self.identity_intent_dim > 0:
            self.intent_norm = nn.LayerNorm(self.identity_intent_dim)
            self.intent_state = nn.Sequential(
                nn.Linear(self.identity_intent_dim, self.identity_intent_dim),
                nn.Tanh(),
            )
            self.intent_rule_proj = nn.Linear(self.identity_intent_dim, z_dim)
            self.intent_think_proj = nn.Linear(self.identity_intent_dim, z_think_dim)
            self.identity_abstain_head = nn.Linear(self.identity_intent_dim, 1)
            self.identity_toggle_head = nn.Linear(self.identity_intent_dim, 1)
            self._last_intent_bias = 0.0
            self._last_intent_toggle = 0.0
        else:
            self.intent_norm = None
            self.intent_state = None
            self.intent_rule_proj = None
            self.intent_think_proj = None
            self.identity_abstain_head = None
            self.identity_toggle_head = None
            self._last_intent_bias = 0.0
            self._last_intent_toggle = 0.0

        # Updated fuse dimension to include identity_self interactions
        # Original: z_dim + z_think_dim + z_lang_dim + z_dim + z_dim + z_think_dim
        # Add: identity_core (remaining dims) + intent_state + rule*identity + think*identity + intent-modulated views
        fuse_dim = (
            z_dim + z_think_dim + z_lang_dim + z_dim + z_dim + z_think_dim
            + max(0, self.identity_core_dim)
            + z_dim + z_think_dim
        )
        if self.identity_intent_dim > 0:
            fuse_dim += self.identity_intent_dim + z_dim + z_think_dim
        self.fuse_proj = nn.Linear(fuse_dim, z_dim)

        # 3brains fuse projection: combines weighted sum of all brains + bridge context
        max_3brains_dim = max(z_slow_lang_dim, z_slow_logic_dim, z_bridge_dim)
        fuse_3brains_dim = max_3brains_dim + z_bridge_dim
        self.fuse_proj_3brains = nn.Linear(fuse_3brains_dim, z_dim)

        # learned z initializer for the rule channel
        self.z0 = nn.Parameter(torch.zeros(z_dim))

        # Self-referential adapters (fast vs slow timescales)
        self.self_ref_rule = MultiTimeScaleAdapter(z_dim)
        self.self_ref_think = MultiTimeScaleAdapter(z_think_dim)
        self.self_ref_fused = MultiTimeScaleAdapter(z_dim)

        # Optional memory reference for real-time state tracking
        self.memory = memory

        # Optional self-evaluation head (confidence estimator)
        # Allow override via environment variable UZR_SELF_EVAL in {"0","1","off","on"}
        env_self = os.environ.get("UZR_SELF_EVAL")
        if env_self is not None:
            try:
                use_self_eval = str(env_self).strip().lower() not in {"0", "false", "off"}
            except Exception:
                pass
        self.self_eval = None
        if use_self_eval and SelfEvalHead is not None:
            try:
                self.self_eval = SelfEvalHead(d_model)
            except Exception:
                self.self_eval = None

        # Register 3brains fusion weights as buffer (avoid recreating every forward pass)
        self.register_buffer("brains_weights", torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32))

        # Optional external inference engine (ORT/QNN). Training 경로에는 영향 없음.
        self._npu_engine = None

        # Luria Brains: Supervisor (Nested Learning meta-controller)
        try:
            self.supervisor = LuriaSupervisor(d_model, max_steps=25)
        except Exception:
            self.supervisor = None

    def init_z(self, batch_size: int = 1):
        return self.z0.unsqueeze(0).expand(batch_size, -1).clone().detach()

    def init_z_thinking(self, batch_size: int = 1):
        return self.z0.new_zeros(batch_size, self.z_think_dim)

    def init_z_slow_lang(self, batch_size: int = 1):
        """Initialize language artisan z (언어 장인)"""
        return self.z0.new_zeros(batch_size, self.z_slow_lang_dim)

    def init_z_slow_logic(self, batch_size: int = 1):
        """Initialize logic artisan z (논리 장인)"""
        return self.z0.new_zeros(batch_size, self.z_slow_logic_dim)

    def init_z_bridge(self, batch_size: int = 1):
        """Initialize bridge z (브릿지/지휘자)"""
        return self.z0.new_zeros(batch_size, self.z_bridge_dim)

    def avg_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Always return [B, D] - squeeze 금지"""
        with torch.no_grad():
            h = self.encoder(x)      # [B, T, D]
            avg = h.mean(dim=1)      # [B, D]
            return avg               # squeeze 금지

    def confidence(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Estimate per-sample confidence in [0,1] using SelfEvalHead.

        Returns None if self-eval head is not available.
        """
        if self.self_eval is None:
            return None
        with torch.no_grad():
            h = self.encoder(x)      # [B, T, D]
            conf = self.self_eval(h.mean(1))  # [B, D] -> [B]
            return conf

    @torch.no_grad()
    def unconscious_loop(self, x: torch.Tensor, rounds: int = 3) -> dict:
        """Run a brief, silent inner loop before answering.

        Copya.txt asks for a non-verbal "무의식 루프" (latency) where the
        model reflects without emitting tokens. Here we implement a light-weight
        version that collects meta-signals and nudges self-referential adapters
        using only their slow EMA path (no gradient / no parameter updates).

        Args:
            x: input tokens [B, T]
            rounds: number of reflection passes (small integer)

        Returns:
            telemetry dict with simple aggregates for potential logging.
        """
        rounds = int(max(1, min(16, rounds)))
        conf_hist = []
        ent_hist = []
        for _ in range(rounds):
            # encode once per round to observe stability
            h = self.encoder(x)
            logits = self.readout(h)
            # entropy per sample
            ent = self.sequence_entropy(logits)
            ent_hist.append(float(ent.mean().item()))
            # confidence if available
            c = self.confidence(x)
            if c is not None:
                conf_hist.append(float(c.mean().item()))
            # very small EMA nudge on self-referential slow buffers
            try:
                if self.self_ref_rule is not None:
                    self.self_ref_rule.update_slow()
                if self.self_ref_think is not None:
                    self.self_ref_think.update_slow()
                if self.self_ref_fused is not None:
                    self.self_ref_fused.update_slow()
            except Exception:
                pass
        tel = {
            "rounds": rounds,
            "conf_mean": (sum(conf_hist) / len(conf_hist)) if conf_hist else None,
            "ent_mean": (sum(ent_hist) / len(ent_hist)) if ent_hist else None,
        }
        return tel

    @staticmethod
    def sequence_entropy(logits: torch.Tensor, topk: Optional[int] = None) -> torch.Tensor:
        """Mean entropy per sample from token logits."""
        return _seq_ent(logits, topk=topk)

    def brier_from_logits_conf(self, logits: torch.Tensor, targets: torch.Tensor, conf: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
        """Brier loss between scalar conf and token-accuracy proxy.

        If ignore_index is not provided, use the model's pad_token_id.
        """
        if ignore_index is None:
            ignore_index = int(getattr(self, 'pad_token_id', 0))
        return _brier_from_conf(logits, targets, conf, ignore_index=ignore_index)

    def set_tokenizer_specials(self, pad_id: Optional[int] = None, bos_id: Optional[int] = None, eos_id: Optional[int] = None) -> None:
        """Configure special token ids for loss masking and metrics."""
        if pad_id is not None:
            self.pad_token_id = int(pad_id)
        if bos_id is not None:
            self.bos_token_id = int(bos_id)
        if eos_id is not None:
            self.eos_token_id = int(eos_id)

    def get_z_from_memory(self, x: torch.Tensor, z_init: Optional[torch.Tensor] = None,
                          topk: Optional[int] = None, blend: float = 0.5) -> Optional[torch.Tensor]:
        """Get z from memory based on input embeddings.

        Args:
            x: Input tokens [B, T] or [1, T]
            z_init: Initial z if memory fails
            topk: Number of neighbors to retrieve
            blend: Blend factor for predictor (will be adaptive if z_init provided)

        Returns:
            Predicted z or None if memory not available
        """
        if self.memory is None:
            return None

        # Get average embedding [B, D] -> [D]
        avg_emb = self.avg_embed(x)  # [B, D]
        if avg_emb.dim() == 2:
            avg_emb = avg_emb.squeeze(0)  # [D]

        # Adaptive blend based on z_init norm
        if z_init is not None:
            z_norm = torch.norm(z_init)
            blend = 0.3 + 0.4 * torch.sigmoid(z_norm - 1.0).item()

        # Respect dynamic memory settings if topk is not provided
        k = self.memory.topk if topk is None else topk
        z = self.memory.get_or_predict_z(
            avg_emb=avg_emb,
            z_init=z_init,
            topk=k,
            blend=blend,
            use_predictor=True,
        )

        # Normalize and ensure z matches model z_dim before returning
        if z is not None:
            z = F.normalize(z, p=2, dim=-1)
            try:
                if z.size(-1) != self.z_dim:
                    z = self._pad_or_trim(z, self.z_dim)
            except Exception:
                pass

        return z

    def update_memory_state(self, x: torch.Tensor, z: torch.Tensor):
        """Update memory state with current input and z.

        Args:
            x: Input tokens [B, T]
            z: Current z vector [D] or [B, D]
        """
        if self.memory is None:
            return

        avg_emb = self.avg_embed(x)
        if avg_emb.dim() > 1:
            avg_emb = avg_emb.mean(dim=0)
        if z.dim() > 1:
            z = z.mean(dim=0)

        # Normalize before tracking
        avg_emb = F.normalize(avg_emb, p=2, dim=-1)
        z = F.normalize(z, p=2, dim=-1)

        self.memory.track_input(avg_emb.detach(), z.detach())

    def build_text_repr(self, x: torch.Tensor, tokenizer=None) -> str:
        """
        x: [B, T] 입력 토큰
        tokenizer: decode가 가능한 토크나이저 (KoEnTokenizer 권장)
        return: 한 줄짜리 설명 문자열 (한글 비율, 영문 비율, 숫자비율, 길이, 대표 n그램 등)
        """
        try:
            # 배치 평균으로 간단화
            ids = x[0].tolist() if x.dim() == 2 else x.tolist()
        except Exception:
            ids = []
        text = ""
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                text = tokenizer.decode(ids)
            except Exception:
                text = ""

        # 통계
        hangul = sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7A3)
        latin  = sum(1 for ch in text if (u"a" <= ch <= u"z") or (u"A" <= ch <= u"Z"))
        digit  = sum(1 for ch in text if ch.isdigit())
        total  = max(1, len(text))
        ratio  = lambda n: f"{(100.0*n/total):.1f}%"

        # n그램 샘플 (2~3그램 일부)
        grams = []
        for n in (2, 3):
            seen = set()
            for i in range(len(text)-n+1):
                g = text[i:i+n]
                if g.strip() and g not in seen:
                    seen.add(g)
                    grams.append(g)
                if len(grams) >= 12:
                    break
            if len(grams) >= 12:
                break
        grams = grams[:12]

        return f"|len={len(text)}|ko={ratio(hangul)}|en={ratio(latin)}|num={ratio(digit)}|ngrams={','.join(grams)}"

    @staticmethod
    def _pad_or_trim(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        if vec.size(-1) == target_dim:
            return vec
        if vec.size(-1) > target_dim:
            return vec[..., :target_dim]
        pad_shape = list(vec.shape)
        pad_shape[-1] = target_dim - vec.size(-1)
        pad = vec.new_zeros(*pad_shape)
        return torch.cat([vec, pad], dim=-1)

    def _split_identity(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        identity = self.identity_self.unsqueeze(0).expand(batch_size, -1)
        if self.identity_intent_dim > 0:
            if self.identity_core_dim > 0:
                identity_core = identity[..., : self.identity_core_dim]
            else:
                identity_core = None
            identity_intent = identity[..., -self.identity_intent_dim :]
        else:
            identity_core = identity
            identity_intent = None
        return identity_core, identity_intent

    def _compute_intent_features(self, identity_intent: torch.Tensor):
        if identity_intent is None or self.identity_intent_dim <= 0:
            return None, None, None
        intent_normed = self.intent_norm(identity_intent)
        intent_state = self.intent_state(intent_normed)
        bias = torch.tanh(self.identity_abstain_head(intent_state))
        toggle = torch.tanh(self.identity_toggle_head(intent_state))
        self._last_intent_bias = float(bias.mean().detach().cpu())
        self._last_intent_toggle = float(toggle.mean().detach().cpu())
        return intent_state, bias, toggle

    @torch.no_grad()
    def identity_intent_control(self) -> Tuple[float, float]:
        """Return (bias, toggle) derived from identity intent slice."""
        if self.identity_intent_dim <= 0:
            return 0.0, 0.0
        core = self.identity_self
        identity_intent = core[-self.identity_intent_dim :].unsqueeze(0)
        _, bias, toggle = self._compute_intent_features(identity_intent)
        if bias is None or toggle is None:
            return 0.0, 0.0
        return (
            float(bias.mean().detach().cpu()),
            float(toggle.mean().detach().cpu()),
        )

    @torch.no_grad()
    def set_identity_self(self, new_identity: torch.Tensor) -> None:
        if new_identity.dim() != 1 or new_identity.size(0) != self.identity_self_dim:
            raise ValueError("new_identity must be 1-D and match identity_self_dim")
        self.identity_self.copy_(new_identity)

    @torch.no_grad()
    def update_self_referential(self):
        for adapter in (self.self_ref_rule, self.self_ref_think, self.self_ref_fused):
            if adapter is not None:
                adapter.update_slow()

    def _fuse_z(self, z_rule: torch.Tensor, z_think: torch.Tensor, lang_id):
        if z_rule.dim() == 1:
            z_rule = z_rule.unsqueeze(0)
        if z_think.dim() == 1:
            z_think = z_think.unsqueeze(0)

        batch_size = z_rule.size(0)

        # Safety: ensure expected channel dimensions
        try:
            z_rule = self._pad_or_trim(z_rule, self.z_dim)
            z_think = self._pad_or_trim(z_think, self.z_think_dim)
        except Exception:
            pass
        if self.self_ref_rule.fast is not None:
            z_rule = z_rule * (1 + self.self_ref_rule().unsqueeze(0))
        if self.self_ref_think.fast is not None:
            z_think = z_think * (1 + self.self_ref_think().unsqueeze(0))

        lang_tensor = torch.as_tensor(lang_id, device=z_rule.device, dtype=torch.long)
        if lang_tensor.dim() == 0:
            lang_tensor = lang_tensor.unsqueeze(0)
        if lang_tensor.numel() == 1 and z_rule.size(0) > 1:
            lang_tensor = lang_tensor.expand(z_rule.size(0))
        if lang_tensor.size(0) != z_rule.size(0):
            raise ValueError(f"lang_id batch ({lang_tensor.size(0)}) must match z batch ({z_rule.size(0)})")

        lang_e = self.lang_embed(lang_tensor)
        lang_for_rule = self._pad_or_trim(lang_e, z_rule.size(-1))
        think_for_rule = self._pad_or_trim(z_think, z_rule.size(-1))
        lang_for_think = self._pad_or_trim(lang_e, z_think.size(-1))

        rule_lang = z_rule * lang_for_rule
        rule_think = z_rule * think_for_rule
        think_lang = z_think * lang_for_think

        # Add identity_self interactions (core + intent split)
        identity_core, identity_intent = self._split_identity(batch_size)
        if identity_core is not None and identity_core.size(-1) > 0:
            identity_for_rule = self._pad_or_trim(identity_core, z_rule.size(-1))
            identity_for_think = self._pad_or_trim(identity_core, z_think.size(-1))
            rule_identity = z_rule * identity_for_rule  # [B, z_dim]
            think_identity = z_think * identity_for_think  # [B, z_think_dim]
        else:
            identity_for_rule = z_rule.new_zeros(z_rule.size())
            identity_for_think = z_think.new_zeros(z_think.size())
            rule_identity = identity_for_rule
            think_identity = identity_for_think

        intent_state = None
        intent_rule_gate = None
        intent_think_gate = None
        if identity_intent is not None:
            intent_state, _, _ = self._compute_intent_features(identity_intent)
            intent_rule_gate = torch.tanh(self.intent_rule_proj(intent_state))
            intent_think_gate = torch.tanh(self.intent_think_proj(intent_state))

        fuse_parts = [z_rule, z_think, lang_e, rule_lang, rule_think, think_lang]
        if identity_core is not None and identity_core.size(-1) > 0:
            fuse_parts.append(identity_core)
        fuse_parts.extend([rule_identity, think_identity])
        if intent_state is not None:
            rule_intent = z_rule * intent_rule_gate
            think_intent = z_think * intent_think_gate
            fuse_parts.extend([intent_state, rule_intent, think_intent])

        fuse = torch.cat(fuse_parts, dim=-1)
        z_fused = self.fuse_proj(fuse)
        if self.self_ref_fused.fast is not None:
            z_fused = z_fused * (1 + self.self_ref_fused().unsqueeze(0))
        return z_fused

    def _fuse_z_3brains(self, z_slow_lang: torch.Tensor, z_slow_logic: torch.Tensor, z_bridge: torch.Tensor):
        """
        Fuse 3brains: language artisan + logic artisan + bridge
        Bridge acts as the conductor, synthesizing both interpretations.
        All three brains are active simultaneously (NOT MoE).
        """
        if z_slow_lang.dim() == 1:
            z_slow_lang = z_slow_lang.unsqueeze(0)
        if z_slow_logic.dim() == 1:
            z_slow_logic = z_slow_logic.unsqueeze(0)
        if z_bridge.dim() == 1:
            z_bridge = z_bridge.unsqueeze(0)

        # Pad all vectors to the same dimension for weighted sum
        max_dim = max(z_slow_lang.size(-1), z_slow_logic.size(-1), z_bridge.size(-1))
        z_lang_padded = self._pad_or_trim(z_slow_lang, max_dim)
        z_logic_padded = self._pad_or_trim(z_slow_logic, max_dim)
        z_bridge_padded = self._pad_or_trim(z_bridge, max_dim)

        # Weighted sum for conductor effect (using pre-registered buffer)
        weights = torch.softmax(self.brains_weights, dim=0)
        fuse = z_lang_padded * weights[0] + z_logic_padded * weights[1] + z_bridge_padded * weights[2]

        identity_core, identity_intent = self._split_identity(z_bridge.size(0))
        if identity_core is not None and identity_core.size(-1) > 0:
            identity_pad = self._pad_or_trim(identity_core, fuse.size(-1))
            fuse = fuse + identity_pad
        if identity_intent is not None:
            intent_state, _, _ = self._compute_intent_features(identity_intent)
            intent_bridge = self._pad_or_trim(intent_state, z_bridge.size(-1))
            z_bridge = z_bridge + intent_bridge

        # Concatenate with original bridge for additional context
        z_fused = self.fuse_proj_3brains(torch.cat([fuse, z_bridge], dim=-1))
        if self.self_ref_fused.fast is not None:
            z_fused = z_fused * (1 + self.self_ref_fused().unsqueeze(0))
        return z_fused

    def forward(self, x, z):
        if isinstance(z, dict):
            # Check if using 3brains mode
            if "slow_lang" in z and "slow_logic" in z and "bridge" in z:
                z_slow_lang = z["slow_lang"]
                z_slow_logic = z["slow_logic"]
                z_bridge = z["bridge"]
                z = self._fuse_z_3brains(z_slow_lang, z_slow_logic, z_bridge)
            # Legacy 2-brain mode (rule + think)
            elif "rule" in z and "think" in z and "lang_id" in z:
                z_rule = z["rule"]
                z_think = z["think"]
                lang_id = z["lang_id"]
                z = self._fuse_z(z_rule, z_think, lang_id)
            else:
                raise ValueError("z dict must contain either (slow_lang, slow_logic, bridge) or (rule, think, lang_id)")
        elif not torch.is_tensor(z):
            raise TypeError("z must be a Tensor or dict")

        # Normalize z after fusion
        z = F.normalize(z, p=2, dim=-1)

        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.size(0) != x.size(0):
            if z.size(0) == 1:
                z = z.expand(x.size(0), -1)
            else:
                raise ValueError(f"z batch ({z.size(0)}) must match input batch ({x.size(0)})")

        h = self.encoder(x)
        h = self.film(h, z)
        logits = self.readout(h)
        return logits

    # ---- External inference (ORT/QNN) optional helpers ----
    def set_npu_engine(self, engine) -> None:
        """외부 추론 엔진(예: ORT/QNN) 등록. 훈련 경로에는 영향을 주지 않습니다."""
        self._npu_engine = engine

    def clear_npu_engine(self) -> None:
        """외부 추론 엔진 등록 해제."""
        self._npu_engine = None

    def npu_run_logits(self, input_ids) -> Optional[torch.Tensor]:
        """등록된 NPU 엔진이 있으면 logits 텐서를 반환, 없거나 실패 시 None.

        엔진은 ONNX Runtime 세션을 가정합니다. 입력은 numpy 배열이어야 하므로 내부에서 변환합니다.
        """
        if self._npu_engine is None:
            return None
        try:
            out = self._npu_engine.run(input_ids=input_ids.detach().cpu().numpy())
            if "logits" in out:
                logits_np = out["logits"]
            else:
                logits_np = list(out.values())[0]
            logits = torch.from_numpy(logits_np).to(input_ids.device)
            return logits
        except Exception:
            return None

    # ---- Transition (multimodal) module support ----
    def init_transition_module(
        self,
        z_bridge_dim: int,
        u_dim: int,
        cb_vocab: int,
        cb_emb: int = 64,
        fused_dim: int = 256,
        lam_trans: float = 8e-4,
        lam_cos: float = 4e-4,
        lam_roll: float = 4e-4,
        lam_jac: float = 1e-5,
        lam_cb: float = 8e-4,
        lam_align: float = 4e-4,
    ) -> None:
        """Initialize multimodal transition heads and statistics buffers.

        This is optional and can be called after model construction to avoid changing
        the public constructor signature across training scripts.
        """
        # EMA stats for z normalization
        if not hasattr(self, "ema_mean"):
            self.register_buffer("ema_mean", torch.zeros(z_bridge_dim))
            self.register_buffer("ema_std", torch.ones(z_bridge_dim))
            self.register_buffer("ema_count", torch.tensor(0.0))
        else:
            # Resize if dimensions changed
            if self.ema_mean.numel() != z_bridge_dim:
                self.ema_mean = torch.zeros(z_bridge_dim, device=self.ema_mean.device)
                self.ema_std = torch.ones(z_bridge_dim, device=self.ema_std.device)
                self.ema_count = torch.tensor(0.0, device=self.ema_count.device)

        # Modules
        self.cb_enc = CodebookEncoder(cb_vocab, cb_emb, pool="mean")
        self.fuse = MMFuse(z_bridge_dim, u_dim, cb_emb, fused_dim)
        self.trans_z = TransHeadZ(fused_dim, z_bridge_dim, spectral=True)
        self.trans_cb = TransHeadCB(fused_dim, cb_vocab, mode="bag")

        # Lambda weights as buffers
        for k, v in dict(
            lam_trans=lam_trans,
            lam_cos=lam_cos,
            lam_roll=lam_roll,
            lam_jac=lam_jac,
            lam_cb=lam_cb,
            lam_align=lam_align,
        ).items():
            if hasattr(self, k):
                # overwrite existing buffer value
                getattr(self, k).data = torch.tensor(v, device=self.readout.weight.device)
            else:
                self.register_buffer(k, torch.tensor(v))

        # For simple guards elsewhere
        self._trans_u_dim = int(u_dim)
        self._trans_cb_vocab = int(cb_vocab)

    @torch.no_grad()
    def update_ema_stats(self, z: torch.Tensor, alpha: float = 0.01) -> None:
        """Update EMA statistics for bridge z normalization."""
        if not hasattr(self, "ema_mean"):
            return
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if self.ema_count.item() == 0:
            self.ema_mean.copy_(z.mean(0))
            # Use population std (unbiased=False) to avoid DoF<=0 warnings when B=1
            s = z.std(0, unbiased=False)
            # Guard against non-finite values
            s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
            self.ema_std.copy_(s + 1e-5)
            self.ema_count.fill_(1)
            return
        self.ema_mean.mul_(1 - alpha).add_(alpha * z.mean(0))
        # Population std and finite guard during EMA updates as well
        s = z.std(0, unbiased=False)
        s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
        self.ema_std.mul_(1 - alpha).add_(alpha * (s + 1e-5))

    def norm_z(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize z with EMA stats if available."""
        if not hasattr(self, "ema_mean"):
            return z
        return (z - self.ema_mean) / (self.ema_std + 1e-5)


class LuriaSupervisor(nn.Module):
    """
    Nested Learning meta-controller: decides inner-loop steps, learning rate, and sparsity
    based on the input’s embedding summary. This enables differentiable control signals
    flowing from the outer loop to the inner thinking loop.
    """

    def __init__(self, d_model: int, max_steps: int = 25):
        super().__init__()
        self.max_steps = int(max(1, max_steps))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 3)  # [steps, eta, lambda]
        )

    def forward(self, h_avg: torch.Tensor):
        # h_avg: [B, D]
        out = self.mlp(h_avg)  # [B,3]
        steps_logits = out[:, 0]
        eta_logits = out[:, 1]
        lam_logits = out[:, 2]
        # Map to usable ranges with smooth squashing
        steps = torch.clamp(torch.sigmoid(steps_logits) * self.max_steps, min=1.0, max=float(self.max_steps))
        eta = torch.sigmoid(eta_logits) * 0.99 + 0.01   # [0.01, 1.0]
        lam = torch.sigmoid(lam_logits) * 1e-2          # [0, 1e-2]
        return steps, eta, lam

# ---- Loss helpers ----
def seq_ce_loss(logits, target, ignore_index=0):
    # logits: [B, T, V], target: [B, T]
    return F.cross_entropy(logits.transpose(1, 2), target, ignore_index=ignore_index)

def soft_threshold(z, thr):
    # proximal operator for L1
    return torch.sign(z) * torch.clamp(torch.abs(z) - thr, min=0.0)

def confidence_from_logits(logits, target, ignore_index=0):
    # Use negative CE as confidence proxy (higher is better)
    with torch.no_grad():
        ce = F.cross_entropy(logits.transpose(1,2), target, ignore_index=ignore_index, reduction="none")
        # average over non-ignored tokens
        mask = (target != ignore_index).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        c = (- (ce * mask).sum(dim=1) / denom)  # [B]
        # squash to [0,1] approximately
        c = torch.sigmoid(c / 5.0)
        return c
