from __future__ import annotations

"""Meta-cognition utilities for UZR (Self-eval, Abstain, Error Tagger, Reflection).

This module is designed to be strictly additive and does not modify existing
model/training code. Callers can import and use the utilities on demand.

Key features (from Hyper-meta-update.txt v1.0):
- SelfEvalHead: confidence estimation with Brier regularization
- Abstain policy: suppress updates when confidence is low or entropy is high
- Error tagger: categorize common error modes for balanced replay
- Reflection memo: periodic JSON summary of failures and next actions

No external dependencies are introduced.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Self-evaluation head
# -----------------------------
class SelfEvalHead(nn.Module):
    """Predicts per-sample confidence in [0,1] from hidden states.

    Expected input is a per-sample hidden representation [B, D], typically
    an average-pooled encoder output. If a sequence [B, T, D] is supplied,
    it will be mean-pooled over the time dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        conf = torch.sigmoid(self.fc(hidden)).squeeze(-1)  # [B]
        return conf


def sequence_entropy_from_logits(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    ignore_index: int = 0,
    topk: Optional[int] = None
) -> torch.Tensor:
    """
    logits: [B, T, V], targets: [B, T] (optional)
    return: [B] per-seq mean entropy over non-ignored tokens
    """
    B, T, V = logits.shape
    if topk is None or topk >= V:
        probs = torch.softmax(logits, dim=-1)
    else:
        topv, _ = torch.topk(logits, k=topk, dim=-1)
        probs = torch.softmax(topv, dim=-1)

    # token entropies
    ent_tok = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)  # [B, T]

    if targets is not None:
        mask = (targets != ignore_index).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        ent = (ent_tok * mask).sum(dim=1) / denom
    else:
        ent = ent_tok.mean(dim=1)
    return ent  # [B]



def brier_loss_from_conf(
    logits: torch.Tensor,
    targets: torch.Tensor,
    conf: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Compute Brier loss aligning scalar confidence with token-level accuracy.

    We approximate per-sample correctness by token-level mean accuracy over
    non-ignored targets.
    Args:
        logits: [B, T, V]
        targets: [B, T]
        conf: [B] in [0,1]
    Returns:
        mean Brier loss (scalar)
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)  # [B, T]
        mask = (targets != ignore_index).float()
        correct = (pred == targets).float() * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        acc = (correct.sum(dim=1) / denom)  # [B] in [0,1]
    brier = torch.mean((conf - acc) ** 2)
    return brier


# -----------------------------
# Abstain policy
# -----------------------------
@dataclass
class AbstainThresholds:
    conf_min: float = 0.65
    ent_max: float = 2.2


def maybe_abstain(conf: torch.Tensor, ent: torch.Tensor, thr: AbstainThresholds) -> torch.Tensor:
    """Return a boolean mask [B] to abstain for samples below thresholds.

    Abstain if conf < conf_min or ent > ent_max.
    """
    if conf.dim() != 1:
        conf = conf.view(-1)
    if ent.dim() != 1:
        ent = ent.view(-1)
    return (conf < thr.conf_min) | (ent > thr.ent_max)


# -----------------------------
# Error tagger
# -----------------------------
RULE_CATS = ["reverse", "case", "bracket", "dedupe", "lang_mix", "misc"]


def _is_hangul(ch: str) -> bool:
    code = ord(ch)
    return (
        0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0x1100 <= code <= 0x11FF  # Jamo
        or 0x3130 <= code <= 0x318F  # Compatibility Jamo
    )


def _lang_mix_ratio(s: str) -> float:
    if not s:
        return 0.0
    hangul = sum(1 for ch in s if _is_hangul(ch))
    ascii_letters = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    total = hangul + ascii_letters
    if total == 0:
        return 0.0
    p = hangul / total
    return min(p, 1 - p)  # high when mixed


def tag_error(pred_text: str, target_text: str) -> str:
    """Heuristically categorize error between predicted and target strings.

    Categories: reverse, case, bracket, dedupe, lang_mix, misc
    """
    p = (pred_text or "").strip()
    t = (target_text or "").strip()
    if not p or not t:
        return "misc"

    # reverse: close to reverse order
    if p[::-1] == t or p.split()[::-1] == t.split():
        return "reverse"

    # case: identical when lowercased
    if p.lower() == t.lower() and p != t:
        return "case"

    # bracket: bracket mismatch
    brackets = set("()[]{}<>")
    pb = [ch for ch in p if ch in brackets]
    tb = [ch for ch in t if ch in brackets]
    if pb != tb and sorted(pb) == sorted(tb):
        return "bracket"

    # dedupe: repeated tokens collapsing to target
    def _dedup_tokens(s: str) -> List[str]:
        toks: List[str] = s.split()
        out: List[str] = []
        prev: Optional[str] = None
        for tok in toks:
            if tok != prev:
                out.append(tok)
            prev = tok
        return out

    if " ".join(_dedup_tokens(p)) == t and p != t:
        return "dedupe"

    # lang_mix: strong mixed-language signal vs target
    if _lang_mix_ratio(p) > 0.35 and _lang_mix_ratio(t) < 0.15:
        return "lang_mix"

    return "misc"


# -----------------------------
# Replay buffer (lightweight)
# -----------------------------
@dataclass
class ReplayItem:
    x: str
    y: str
    tag: str


@dataclass
class ReplayBuffer:
    """Category-balanced replay sampler with simple decay on failure counts."""

    maxlen: int = 10000
    fail_decay: float = 0.95
    _items: List[ReplayItem] = field(default_factory=list)
    _fail_counts: Counter = field(default_factory=Counter)

    def add(self, x: str, y: str, tag: str) -> None:
        if len(self._items) >= self.maxlen:
            self._items.pop(0)
        self._items.append(ReplayItem(x=x, y=y, tag=tag))
        self._fail_counts[tag] += 1

    def most_common(self, n: int = 3) -> List[Tuple[str, int]]:
        return Counter(self._fail_counts).most_common(n)

    def decay(self) -> None:
        for k in list(self._fail_counts.keys()):
            self._fail_counts[k] *= self.fail_decay

    def sample(self, k: int) -> List[ReplayItem]:
        if not self._items:
            return []
        # Weighted by inverse category frequency to balance
        cat_counts = Counter(it.tag for it in self._items)
        weights = [1.0 / max(1, cat_counts[it.tag]) for it in self._items]
        total = sum(weights)
        if total <= 0:
            idxs = torch.randint(0, len(self._items), (k,)).tolist()
        else:
            probs = torch.tensor([w / total for w in weights], dtype=torch.float)
            num = min(k, len(self._items))
            idxs = torch.multinomial(probs, num_samples=num, replacement=False).tolist()
        return [self._items[i] for i in idxs]


# -----------------------------
# Reflection memo
# -----------------------------
@dataclass
class ReflectionLogger:
    out_dir: str = "reflection"

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def write_epoch(self, epoch: int, fail_top: Iterable[Tuple[str, int]], metrics: Dict[str, float]) -> str:
        memo = {
            "epoch": int(epoch),
            "top_failures": [k for k, _ in list(fail_top)[:3]],
            "next_action": self._suggest_tweak(fail_top, metrics),
            "summary": f"EMA={metrics.get('ema', float('nan')):.3f}, rule_acc={metrics.get('acc', float('nan')):.2f}",
        }
        path = os.path.join(self.out_dir, f"epoch_{epoch}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(memo, f, ensure_ascii=False, indent=2)
            # also update a rolling summary
            with open(os.path.join(self.out_dir, "reflect_summary.json"), "w", encoding="utf-8") as f:
                json.dump(memo, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return path

    @staticmethod
    def _suggest_tweak(fail_top: Iterable[Tuple[str, int]], metrics: Dict[str, float]) -> str:
        cats = [k for k, _ in list(fail_top)[:3]]
        acc = metrics.get("acc", 0.0)
        tips: List[str] = []
        if any(c in ("case", "lang_mix") for c in cats):
            tips.append("upweight replay for case/lang_mix")
        if "reverse" in cats:
            tips.append("add order-invariant hints or penalties")
        if acc < 0.9:
            tips.append("reduce lr 0.5x")
        return "; ".join(tips) if tips else "monitor"


# -----------------------------
# Config helpers
# -----------------------------
DEFAULT_META_CONFIG: Dict[str, float] = {
    "lambda_brier": 0.3,
    "conf_min": 0.63,
    "brier_max": 0.1,
    "ent_max": 2.2,
    "reflect_interval": 1.0,  # epochs
    "replay_ratio": 0.25,
    "fail_decay": 0.95,
}


def load_meta_config(path: str = "config/meta.json") -> Dict[str, float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {**DEFAULT_META_CONFIG, **data}
    except Exception:
        return dict(DEFAULT_META_CONFIG)


# -----------------------------
# Convenience: one-step meta loss & gates
# -----------------------------
@dataclass
class MetaStepOutput:
    task_loss: torch.Tensor
    brier_loss: torch.Tensor
    total_loss: torch.Tensor
    conf: torch.Tensor  # [B]
    abstain_mask: torch.Tensor  # [B] bool


def meta_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden_for_conf: Optional[torch.Tensor] = None,
    selfeval: Optional[SelfEvalHead] = None,
    lambda_brier: float = DEFAULT_META_CONFIG["lambda_brier"],
    thresholds: Optional[AbstainThresholds] = None,
    ignore_index: int = 0,
) -> MetaStepOutput:
    """Compute task loss + Brier regularizer and abstain decisions."""
    # forward
    logits = model(x, y.new_zeros(y.size(0), getattr(model, "z_dim", 128))) if hasattr(model, "z_dim") else model(x)

    # 2-1) hidden_for_conf 확보 (주어진 게 있으면 절대 덮지 말 것)
    if hidden_for_conf is None:
        if hasattr(model, "avg_embed"):
            try:
                hidden_for_conf = model.avg_embed(x)  # [B, D]
            except Exception:
                hidden_for_conf = logits.mean(dim=1)  # fallback
        else:
            hidden_for_conf = logits.mean(dim=1)
    # 이제서야 detach
    conf_hidden = hidden_for_conf.detach()

    # 2-2) selfeval 준비
    if selfeval is None:
        hidden_dim = int(conf_hidden.size(-1))
        selfeval = SelfEvalHead(hidden_dim)  # 주의: 옵티마이저에 등록은 호출자 책임

    conf = selfeval(conf_hidden)  # [B]

    # 2-3) 손실
    task_loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=ignore_index)
    brier = brier_loss_from_conf(logits, y, conf, ignore_index=ignore_index)
    total = task_loss + float(lambda_brier) * brier

    # 2-4) 엔트로피(마스크 적용)
    ent = sequence_entropy_from_logits(logits, targets=y, ignore_index=ignore_index)

    # 2-5) abstain: 학습 중엔 업데이트 차단 금지, 추론에서만 사용
    if model.training:
        abstain_mask = torch.zeros_like(conf, dtype=torch.bool)
    else:
        thr = thresholds or AbstainThresholds(
            conf_min=DEFAULT_META_CONFIG["conf_min"],
            ent_max=DEFAULT_META_CONFIG["ent_max"]
        )
        abstain_mask = maybe_abstain(conf.detach(), ent.detach(), thr)

    return MetaStepOutput(
        task_loss=task_loss, brier_loss=brier, total_loss=total,
        conf=conf, abstain_mask=abstain_mask
    )


__all__ = [
    "SelfEvalHead",
    "sequence_entropy_from_logits",
    "brier_loss_from_conf",
    "AbstainThresholds",
    "maybe_abstain",
    "inner_steps_from_conf",
    "RULE_CATS",
    "tag_error",
    "ReplayItem",
    "ReplayBuffer",
    "ReflectionLogger",
    "DEFAULT_META_CONFIG",
    "load_meta_config",
    "MetaStepOutput",
    "meta_step",
]

# -----------------------------
# Inner-steps scheduler
# -----------------------------
def inner_steps_from_conf(conf: float, s_max: int = 4, s_min: int = 0, k: float = 10.0, mid: float = 0.7) -> int:
    """Map confidence in [0,1] to an integer inner-step budget.

    Higher confidence yields more steps (exploit), lower yields fewer (explore cheaply).

    Args:
        conf: float confidence in [0,1]
        s_max: maximum steps
        s_min: minimum steps
        k: logistic slope
        mid: midpoint of logistic
    """
    import math
    try:
        c = float(conf)
    except Exception:
        c = 0.0
    frac = 1.0 - 1.0 / (1.0 + math.exp(-k * (c - mid)))
    steps = round(s_min + (s_max - s_min) * frac)
    return max(int(s_min), min(int(s_max), int(steps)))
