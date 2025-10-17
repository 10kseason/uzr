
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        ids = [i for i in ids if i not in (self.BOS, self.EOS, 0)]
        try:
            return bytes(ids).decode("utf-8", errors="ignore")
        except Exception:
            return ""

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

        # learned z initializer for the rule channel
        self.z0 = nn.Parameter(torch.zeros(z_dim))

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

