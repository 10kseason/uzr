
"""
model_nonvolatile.py

Upgrades over original:
- Non-volatile Z-state snapshots (save/load) + optional EMA consolidation into z0.
- Confidence-gated consolidation (only update when confidence is high enough).
- Korean-friendly text handling: NFKC normalization, basic Hangul detection, higher default max_len.
- Helper: guess_lang_id(text) -> {0:en, 1:ko, 2:ja, 3:base}.
- Retention loss helper to stabilize Z across sessions/chunks.
"""

import os
import json
import math
import unicodedata
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Language helpers ----------

LANG_EN = 0
LANG_KO = 1
LANG_JA = 2
LANG_BASE = 3

def guess_lang_id(text: str) -> int:
    """Very simple heuristic. Prefer KO if any Hangul syllables; else JA if any Hiragana/Katakana;
    else EN if any ASCII letters; else BASE."""
    has_ko = any('\uAC00' <= ch <= '\uD7A3' for ch in text)  # Hangul syllables
    if has_ko:
        return LANG_KO
    has_ja = any(('\u3040' <= ch <= '\u309F') or ('\u30A0' <= ch <= '\u30FF') for ch in text)  # Hiragana/Katakana
    if has_ja:
        return LANG_JA
    has_en = any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in text)
    if has_en:
        return LANG_EN
    return LANG_BASE


# ---------- Byte-level tokenizer (Korean-friendly normalization) ----------
class ByteTokenizer:
    def __init__(self, max_len: int = 256):
        # 256 bytes + BOS + EOS
        self.vocab_size = 258
        self.BOS = 256
        self.EOS = 257
        self.max_len = max_len

    @staticmethod
    def _normalize(s: str) -> str:
        # NFKC normalization improves Korean/JP width variants; strip odd control chars.
        s = unicodedata.normalize("NFKC", s)
        # Replace common zero-width chars
        s = s.replace("\u200b", "").replace("\uFEFF", "")
        return s

    def encode(self, s: str):
        s = self._normalize(s)
        b = s.encode("utf-8", errors="ignore")[: self.max_len - 2]
        ids = [self.BOS] + list(b) + [self.EOS]
        pad = [0] * (self.max_len - len(ids))
        return torch.tensor(ids + pad, dtype=torch.long)

    def decode(self, ids):
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


# ---------- FiLM conditioning block ----------
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


# ---------- Tiny Transformer Encoder ----------
class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_head: int = 4, n_layer: int = 4, max_len: int = 256):
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


# ---------- Universal Z-Rule Runner f_theta ----------
class UZRModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, z_dim: int = 128, n_head: int = 4, n_layer: int = 4, max_len: int = 256,
                 z_think_dim: int = 64, z_lang_dim: int = 32, num_langs: int = 4):
        super().__init__()
        self.encoder = TinyEncoder(vocab_size, d_model, n_head, n_layer, max_len)
        self.film = FiLM(z_dim, d_model)
        self.readout = nn.Linear(d_model, vocab_size)  # token-level logits
        self.z_dim = z_dim
        self.z_think_dim = z_think_dim
        self.z_lang_dim = z_lang_dim
        self.num_langs = num_langs

        self.lang_embed = nn.Embedding(num_langs, z_lang_dim)

        fuse_dim = z_dim + z_think_dim + z_lang_dim + z_dim + z_dim + z_think_dim
        self.fuse_proj = nn.Linear(fuse_dim, z_dim)

        # learned z initializer for the rule channel (acts as "long-term" seed)
        self.z0 = nn.Parameter(torch.zeros(z_dim))

        # --- Non-volatile controls ---
        self.register_buffer("nv_updates", torch.zeros(1, dtype=torch.long))  # counts consolidations

    def init_z(self, batch_size: int = 1):
        return self.z0.unsqueeze(0).expand(batch_size, -1).clone().detach()

    def init_z_thinking(self, batch_size: int = 1):
        return self.z0.new_zeros(batch_size, self.z_think_dim)

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

        fuse = torch.cat([z_rule, z_think, lang_e, rule_lang, rule_think, think_lang], dim=-1)
        z_fused = self.fuse_proj(fuse)
        return z_fused

    def forward(self, x, z):
        if isinstance(z, dict):
            z_rule = z["rule"]
            z_think = z["think"]
            lang_id = z["lang_id"]
            z = self._fuse_z(z_rule, z_think, lang_id)
        elif not torch.is_tensor(z):
            raise TypeError("z must be a Tensor or dict with rule/think/lang_id")

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

    # ---------- Non-volatile features ----------
    @torch.no_grad()
    def consolidate_z0(self, z_rule_batch: torch.Tensor, conf: Optional[torch.Tensor] = None,
                       alpha: float = 0.1, conf_thr: float = 0.6):
        """EMA-update z0 from a batch of 'good' session z_rule (confidence-gated).
        z_rule_batch: [B, z_dim], conf: [B] in [0, 1] (higher=better), alpha: EMA step size.
        """
        if z_rule_batch.dim() == 1:
            z_rule_batch = z_rule_batch.unsqueeze(0)
        if conf is None:
            conf = torch.ones(z_rule_batch.size(0), device=z_rule_batch.device)
        mask = (conf >= conf_thr).float().unsqueeze(-1)  # [B,1]
        if mask.sum() == 0:
            return  # nothing to update
        z_mean = (z_rule_batch * mask).sum(dim=0) / mask.sum()
        self.z0.data = (1 - alpha) * self.z0.data + alpha * z_mean
        self.nv_updates += 1

    def save_nonvolatile(self, path: str, z_snapshot: Optional[Dict[str, torch.Tensor]] = None, save_model: bool = True):
        """Save model weights (including updated z0) and an optional z snapshot dict {'rule','think','lang_id'}."""
        pkg: Dict[str, Any] = {"state_dict": self.state_dict(), "nv_updates": int(self.nv_updates.item())}
        if z_snapshot is not None:
            # store on CPU
            pkg["z_snapshot"] = {
                "rule": z_snapshot["rule"].detach().cpu(),
                "think": z_snapshot["think"].detach().cpu(),
                "lang_id": int(z_snapshot["lang_id"]) if not torch.is_tensor(z_snapshot["lang_id"]) else int(z_snapshot["lang_id"].item()),
            }
        torch.save(pkg, path)

    @staticmethod
    def load_nonvolatile(path: str, map_location: str = "cpu") -> "UZRModel":
        """Load a model package created by save_nonvolatile. Returns the model; z_snapshot is returned separately via load_z_snapshot()."""
        pkg = torch.load(path, map_location=map_location)
        # We cannot reconstruct constructor args from state_dict; caller must create same-arch model first.
        raise RuntimeError("Call `load_weights_into(model, path)` on an already-constructed model with the same architecture.")

    def load_weights_into(self, path: str, map_location: str = "cpu") -> Optional[Dict[str, torch.Tensor]]:
        """Load into an existing instance. Returns z_snapshot dict if present."""
        pkg = torch.load(path, map_location=map_location)
        self.load_state_dict(pkg["state_dict"])
        if "nv_updates" in pkg and isinstance(pkg["nv_updates"], int):
            self.nv_updates = torch.tensor([pkg["nv_updates"]], dtype=torch.long, device=self.z0.device)
        return pkg.get("z_snapshot", None)


# ---------- Loss helpers ----------
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


def retention_loss(z_prev: torch.Tensor, z_curr: torch.Tensor, weight: float = 1e-2, p: int = 2) -> torch.Tensor:
    """Penalize drift between consecutive z states to encourage cross-chunk/session stability."""
    if z_prev.dim() == 1:
        z_prev = z_prev.unsqueeze(0)
    if z_curr.dim() == 1:
        z_curr = z_curr.unsqueeze(0)
    return weight * (z_prev - z_curr).norm(p=p, dim=-1).mean()


# ---------- Simple non-volatile store for z snapshots ----------
class PersistentZStore:
    """Stores/loads z snapshots on disk. You can keep one rolling file or step-indexed files."""
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.last_file = os.path.join(root, "z_last.pt")

    def save(self, z_rule: torch.Tensor, z_think: torch.Tensor, lang_id: int, step: Optional[int] = None):
        pkg = {
            "rule": z_rule.detach().cpu(),
            "think": z_think.detach().cpu(),
            "lang_id": int(lang_id),
            "step": int(step) if step is not None else None,
        }
        torch.save(pkg, self.last_file)

    def load_last(self) -> Optional[Dict[str, torch.Tensor]]:
        if not os.path.exists(self.last_file):
            return None
        pkg = torch.load(self.last_file, map_location="cpu")
        return pkg


# ---------- Minimal usage example ----------
EXAMPLE_USAGE = r"""
# --- Build ---
tok = ByteTokenizer(max_len=256)
model = UZRModel(vocab_size=tok.vocab_size, d_model=256, z_dim=128, z_think_dim=64, z_lang_dim=32, num_langs=4)

# --- Start of a new run (try to restore previous z) ---
z_store = PersistentZStore("./nv_store")
z_last = z_store.load_last()

B = 1
if z_last is None:
    z_state = {"rule": model.init_z(B), "think": model.init_z_thinking(B), "lang_id": LANG_KO}  # default KO if you want
else:
    z_state = {"rule": z_last["rule"], "think": z_last["think"], "lang_id": int(z_last["lang_id"])}

# --- Inference step ---
x = tok.encode("안녕, 세상아!").unsqueeze(0)
logits = model(x, z_state)

# --- Confidence & consolidation ---
conf = confidence_from_logits(logits, target=x)  # or any proxy you prefer
model.consolidate_z0(z_state["rule"], conf=conf, alpha=0.1, conf_thr=0.6)

# --- Persist z for next session ---
z_store.save(z_state["rule"], z_state["think"], z_state["lang_id"], step=123)

# --- Optional: save weights (z0 included) ---
model.save_nonvolatile("./nv_model.pt", z_snapshot=z_state)
"""
