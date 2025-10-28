
import math, unicodedata
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos)
        h = self.enc(h)
        return h  # [B, T, D]

# ---- Universal Z-Rule Runner f_theta ----
class UZRModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, z_dim: int = 128, n_head: int = 4, n_layer: int = 4, max_len: int = 128,
                 z_think_dim: int = 64, z_lang_dim: int = 32, num_langs: int = 4, identity_self_dim: int = 2, memory=None,
                 z_slow_lang_dim: int = 96, z_slow_logic_dim: int = 96, z_bridge_dim: int = 64,
                 use_self_eval: bool = True):
        super().__init__()
        self.encoder = TinyEncoder(vocab_size, d_model, n_head, n_layer, max_len)
        self.film = FiLM(z_dim, d_model)
        self.readout = nn.Linear(d_model, vocab_size)  # token-level logits
        self.z_dim = z_dim
        self.z_think_dim = z_think_dim
        self.z_lang_dim = z_lang_dim
        self.num_langs = num_langs
        self.identity_self_dim = identity_self_dim

        # 3brains dimensions
        self.z_slow_lang_dim = z_slow_lang_dim
        self.z_slow_logic_dim = z_slow_logic_dim
        self.z_bridge_dim = z_bridge_dim

        self.lang_embed = nn.Embedding(num_langs, z_lang_dim)

        # Learned identity embedding for self-awareness (e.g., "루리아")
        self.identity_self = nn.Parameter(torch.randn(identity_self_dim) * 0.02)

        # Updated fuse dimension to include identity_self interactions
        # Original: z_dim + z_think_dim + z_lang_dim + z_dim + z_dim + z_think_dim
        # Add: identity_self_dim + z_dim (rule*identity) + z_think_dim (think*identity)
        fuse_dim = z_dim + z_think_dim + z_lang_dim + z_dim + z_dim + z_think_dim + identity_self_dim + z_dim + z_think_dim
        self.fuse_proj = nn.Linear(fuse_dim, z_dim)

        # 3brains fuse projection: combines weighted sum of all brains + bridge context
        max_3brains_dim = max(z_slow_lang_dim, z_slow_logic_dim, z_bridge_dim)
        fuse_3brains_dim = max_3brains_dim + z_bridge_dim
        self.fuse_proj_3brains = nn.Linear(fuse_3brains_dim, z_dim)

        # learned z initializer for the rule channel
        self.z0 = nn.Parameter(torch.zeros(z_dim))

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

    @staticmethod
    def sequence_entropy(logits: torch.Tensor, topk: Optional[int] = None) -> torch.Tensor:
        """Mean entropy per sample from token logits."""
        return _seq_ent(logits, topk=topk)

    @staticmethod
    def brier_from_logits_conf(logits: torch.Tensor, targets: torch.Tensor, conf: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:
        """Brier loss between scalar conf and token-accuracy proxy."""
        return _brier_from_conf(logits, targets, conf, ignore_index=ignore_index)

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

        # Normalize before returning
        if z is not None:
            z = F.normalize(z, p=2, dim=-1)

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

    def _fuse_z(self, z_rule: torch.Tensor, z_think: torch.Tensor, lang_id):
        if z_rule.dim() == 1:
            z_rule = z_rule.unsqueeze(0)
        if z_think.dim() == 1:
            z_think = z_think.unsqueeze(0)

        batch_size = z_rule.size(0)

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

        # Add identity_self interactions
        identity = self.identity_self.unsqueeze(0).expand(batch_size, -1)  # [B, identity_self_dim]
        identity_for_rule = self._pad_or_trim(identity, z_rule.size(-1))
        identity_for_think = self._pad_or_trim(identity, z_think.size(-1))

        rule_identity = z_rule * identity_for_rule  # [B, z_dim]
        think_identity = z_think * identity_for_think  # [B, z_think_dim]

        fuse = torch.cat([z_rule, z_think, lang_e, rule_lang, rule_think, think_lang,
                         identity, rule_identity, think_identity], dim=-1)
        z_fused = self.fuse_proj(fuse)
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

        # Concatenate with original bridge for additional context
        z_fused = self.fuse_proj_3brains(torch.cat([fuse, z_bridge], dim=-1))
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

