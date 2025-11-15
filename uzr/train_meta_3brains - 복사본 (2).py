import argparse, random, os, math, time, unicodedata, sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import namedtuple, deque

# NOTE: Keep imports for backwards compatibility, but we won't use ByteTokenizer now.
from .model import UZRModel, ByteTokenizer, seq_ce_loss, soft_threshold
from .memory import CompressedMemory, make_sketch
from .codebook import CodebookManager
from uzr.tasks import sample_task, DatasetMiTSampler
from uzr.meta_core import load_meta_config, AbstainThresholds, maybe_abstain, ReplayBuffer, ReflectionLogger
from .hooks import should_force_write, apply_force_write, update_force_write_config
from .hooks.stage import log_stage_event
from .luria_logging.shadow_bank import log_sb_event
from .luria_logging.metrics import log_sb_snapshot
from .luria_logging.dedup import log_dedup
from utils.kobert_tokenizer_lite import load_kobert_tokenizer

# -------------------------------
# Identity QA (KO/EN only; JA commented out for future support)
# -------------------------------
IDENTITY_QA_KO = [
    ("너는 누구야?", "나는 {id}입니다."),
    ("네 이름은 뭐야?", "나는 {id}입니다."),
    ("자기소개해봐", "{id}"),
]
IDENTITY_QA_EN = [
    ("Who are you?", "I am {id}."),
    ("What is your name?", "{id}"),
    ("Introduce yourself.", "I am {id}."),
]
# IDENTITY_QA_JA = [
#     ("?귙겒?잆겘?졼굦竊?, "?뤵걼?쀣겘{id}?㎯걲??),
#     ("?듽겒?얇걟??폕", "{id}"),
#     ("?섅걪?쀣굟?녴걢?꾠걮?╉?, "?뤵걼?쀣겘{id}?㎯걲??),
# ]

# Language mapping: KO/EN + base only (JA deferred)
LANG2ID = {"base": 0, "en": 1, "ko": 2}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def detect_lang_from_text(text: str) -> str:
    """Very light detector: KO > EN > BASE (JA deferred)."""
    has_hangul = False
    has_latin = False
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            has_hangul = True
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            has_latin = True
    if has_hangul:
        return "ko"
    if has_latin:
        return "en"
    return "base"


def detect_task_type(text: str, desc: str) -> str:
    """
    Detect if task is logic-oriented or language-oriented.
    Logic: BASE language, math/calculation tasks
    Language: Everything else (ko/en language tasks)
    """
    # Check language
    lang = detect_lang_from_text(text)
    if lang == "base":
        return "logic"

    # Check task description for logic keywords
    logic_keywords = ["calc", "math", "count", "number", "digit", "arithmetic", "compute"]
    desc_lower = desc.lower()
    for keyword in logic_keywords:
        if keyword in desc_lower:
            return "logic"

    return "language"


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# -------------------------------
# Minimal KO+EN tokenizer (character-level, Unicode-aware)
# - Supports Hangul syllables (U+AC00..U+D7A3)
# - Supports Hangul Jamo ranges (U+1100..U+11FF, U+3130..U+318F) for jamo-focused rules
# - Supports common ASCII set
# -------------------------------
class KoEnTokenizer:
    def __init__(self, max_len=512, add_bos=True, add_eos=True):
        self.max_len = max_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.PAD = 0; self.BOS = 1; self.EOS = 2; self.UNK = 3

        chars = []

        # ASCII basic printable (32..126)
        chars.extend(chr(i) for i in range(32, 127))
        # Include newline for safety
        chars.append("\n")

        # Hangul Syllables
        chars.extend(chr(i) for i in range(0xAC00, 0xD7A4))

        # Hangul Compatibility Jamo
        chars.extend(chr(i) for i in range(0x3130, 0x3190))

        # Hangul Jamo (Choseong/Jungseong/Jongseong blocks)
        for start, end in [(0x1100, 0x115F), (0x1160, 0x11A7), (0x11A8, 0x11FF)]:
            chars.extend(chr(i) for i in range(start, end + 1))

        # Latin letters and digits already included by ASCII range

        self.itos = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        # Normalize lightly (NFKC) to reduce weird width differences
        if not isinstance(text, str):
            text = str(text)
        s = unicodedata.normalize("NFKC", text)

        ids = []
        if self.add_bos:
            ids.append(self.BOS)

        for ch in s:
            idx = self.stoi.get(ch, self.UNK)
            ids.append(idx)
            # leave room for EOS
            if len(ids) >= self.max_len - (1 if self.add_eos else 0):
                break

        if self.add_eos:
            ids.append(self.EOS)

        # Pad
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


def batchify(pairs, tok):
    xs, ys = [], []
    for x, y in pairs:
        xs.append(tok.encode(x))
        ys.append(tok.encode(y))
    X = torch.stack(xs, dim=0)
    Y = torch.stack(ys, dim=0)
    return X, Y


def inner_adapt_z(model, Xc, Yc, z, lam=1e-3, eta=0.5, steps=3, ignore_index: int = 0):
    # z-only updates (legacy single-z version)
    z = z.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(Xc, z)
        loss = seq_ce_loss(logits, Yc, ignore_index=ignore_index) + lam * torch.mean(torch.abs(z))
        g = torch.autograd.grad(loss, z, retain_graph=False)[0]
        z = (z - eta * g).detach().requires_grad_(True)
    # proximal soft-threshold (L1)
    with torch.no_grad():
        z = soft_threshold(z, lam * 0.5)
    return z.detach()


def inner_adapt_z_multi(model, Xc, Yc, z_rule, z_think, lang_id, lam_rule=1e-3, lam_think=1e-3, eta=0.5, steps=3, ignore_index: int = 0):
    """Legacy 2-brain adaptation (rule + think)"""
    z_rule = z_rule.clone().detach().requires_grad_(True)
    z_think = z_think.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(Xc, {"rule": z_rule, "think": z_think, "lang_id": lang_id})
        loss = seq_ce_loss(logits, Yc, ignore_index=ignore_index)
        loss = loss + lam_rule * torch.mean(torch.abs(z_rule)) + lam_think * torch.mean(torch.abs(z_think))
        grad_rule, grad_think = torch.autograd.grad(loss, (z_rule, z_think), retain_graph=False)
        z_rule = (z_rule - eta * grad_rule).detach().requires_grad_(True)
        z_think = (z_think - eta * grad_think).detach().requires_grad_(True)
    with torch.no_grad():
        z_rule = soft_threshold(z_rule, lam_rule * 0.5)
        z_think = soft_threshold(z_think, lam_think * 0.5)
    return z_rule.detach(), z_think.detach()


def choose_adaptive_steps(conf, ent, verifier, seq_len, conf_ema, s_max_cap, task_key, cooldown_tracker):
    """
    Choose inner adaptation steps using bucket mapping strategy.
    From 적응형 이너스텝.txt specification.

    Args:
        conf: Current confidence
        ent: Current entropy
        verifier: Verification score (similar to confidence)
        seq_len: Sequence length
        conf_ema: Exponential moving average of confidence
        s_max_cap: Maximum allowed steps (affected by inflation governor)
        task_key: Unique key for task (for cooldown tracking)
        cooldown_tracker: Dict tracking cooldown periods

    Returns:
        chosen_steps: Number of adaptation steps
    """
    # Base settings
    s = 6
    cap = min(25, s_max_cap)

    # Apply cooldown cap if this task recently used high steps
    if task_key in cooldown_tracker and cooldown_tracker[task_key] > 0:
        cap = min(cap, 8)

    # Apply length cap
    if seq_len > 64:
        cap = min(cap, 10)

    # Bucket mapping (check from top to bottom)
    if conf >= 0.78 or verifier >= 0.97:
        s = 6  # G0: High confidence or verifier
    elif conf >= 0.70:
        s = 6  # G1: Good confidence
    elif 0.62 <= conf < 0.70:
        s = 7  # G2: Medium confidence
    elif conf < 0.62 and 0.92 <= verifier < 0.95:
        s = 8  # G3: Gray zone
    elif conf < 0.58 and 0.88 <= verifier < 0.92:
        s = 10  # G4: Low confidence, medium verifier
    elif conf < 0.55 and 0.85 <= verifier < 0.88:
        s = 12  # G5: Very low confidence
    elif conf < 0.50 and verifier < 0.85 and seq_len <= 64:
        s = 14  # G6: Critical zone
    elif conf < 0.45 and verifier < 0.80 and seq_len <= 64:
        s = 15  # G7: Emergency zone
    else:
        s = 7  # G8: Default safe

    # Apply caps
    s = int(max(4, min(s, cap)))

    return s


def inner_adapt_z_3brains(model, Xc, Yc, z_slow_lang, z_slow_logic, z_bridge,
                          lam_lang=1e-3, lam_logic=1e-3, lam_bridge=1e-3, eta=0.5, steps=3, ignore_index: int = 0):
    """
    3brains adaptation: language artisan + logic artisan + bridge
    All three brains adapt simultaneously (NOT MoE)
    """
    z_slow_lang = z_slow_lang.clone().detach().requires_grad_(True)
    z_slow_logic = z_slow_logic.clone().detach().requires_grad_(True)
    z_bridge = z_bridge.clone().detach().requires_grad_(True)

    for _ in range(steps):
        logits = model(Xc, {"slow_lang": z_slow_lang, "slow_logic": z_slow_logic, "bridge": z_bridge})
        loss = seq_ce_loss(logits, Yc, ignore_index=ignore_index)
        loss = loss + lam_lang * torch.mean(torch.abs(z_slow_lang))
        loss = loss + lam_logic * torch.mean(torch.abs(z_slow_logic))
        loss = loss + lam_bridge * torch.mean(torch.abs(z_bridge))

        grad_lang, grad_logic, grad_bridge = torch.autograd.grad(
            loss, (z_slow_lang, z_slow_logic, z_bridge), retain_graph=False
        )

        z_slow_lang = (z_slow_lang - eta * grad_lang).detach().requires_grad_(True)
        z_slow_logic = (z_slow_logic - eta * grad_logic).detach().requires_grad_(True)
        z_bridge = (z_bridge - eta * grad_bridge).detach().requires_grad_(True)

    with torch.no_grad():
        z_slow_lang = soft_threshold(z_slow_lang, lam_lang * 0.5)
        z_slow_logic = soft_threshold(z_slow_logic, lam_logic * 0.5)
        z_bridge = soft_threshold(z_bridge, lam_bridge * 0.5)

    return z_slow_lang.detach(), z_slow_logic.detach(), z_bridge.detach()


def make_identity_batch(tok, identity, batch_lang="mix", n=6):
    pairs = []
    for _ in range(n):
        lang = random.choice(["ko", "en"]) if batch_lang == "mix" else batch_lang
        if lang == "ko":
            q, a = random.choice(IDENTITY_QA_KO)
        else:
            q, a = random.choice(IDENTITY_QA_EN)
        pairs.append((q, a.format(id=identity)))
    return batchify(pairs, tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--z_think_dim", type=int, default=64)
    ap.add_argument("--z_lang_dim", type=int, default=32)
    ap.add_argument("--scale", type=int, default=1, choices=[1, 2, 3, 4], help="multiply core dims by this factor (1..4)")
    ap.add_argument("--n_head", type=int, default=12, help="number of attention heads in the encoder")
    ap.add_argument("--n_layer", type=int, default=12, help="number of transformer encoder layers")
    ap.add_argument("--identity_self_dim", type=int, default=32, help="dimension for self-identity embedding (e.g., '루리아')")
    ap.add_argument("--identity_intent_dim", type=int, default=16, help="intent-specific subspace inside identity_self (set 0 to disable)")
    ap.add_argument("--num_langs", type=int, default=len(LANG2ID))
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--tokenizer", choices=["auto", "koen", "kobert"], default="auto", help="tokenizer choice: auto detects local kobert, else koen")
    ap.add_argument("--dataset_mix_prob", type=float, default=0.0, help="probability of sampling dataset-mit QA tasks per step")
    ap.add_argument("--dataset_mit_path", default="", help="override path for dataset-mit CSV (defaults to dataset-mit/mmlu_KO-KR.csv)")
    ap.add_argument("--kobert_hint", action="store_true", help="attach KoBERT masked-LM hints to dataset-mit prompts")
    ap.add_argument("--kobert_dir", default="kobert", help="local directory containing KoBERT weights")
    ap.add_argument("--kobert_device", default="auto", help="device for KoBERT hints (cpu/cuda/auto)")
    ap.add_argument("--kobert_max_seq_len", type=int, default=384, help="max sequence length for KoBERT hint prompts")
    ap.add_argument("--inner_steps", type=int, default=6)
    ap.add_argument("--inner_eta", type=float, default=0.425)
    ap.add_argument("--save", default="uzr_3brains_ckpt.pt")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--amp", action="store_true")
    # Meta-core integration flags
    ap.add_argument("--self_eval", choices=["on", "off"], help="enable/disable SelfEval head inside model")
    ap.add_argument("--abstain", action="store_true", help="enable abstain gating during training updates")
    ap.add_argument("--summary_csv", default="train3_summary.csv", help="summary CSV path (auto-named in logu/ if default)")
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--autonomous_step", type=int, default=3500, help="step to enable autonomous gating/threshold adaptation (recommended 3000-5000)")
    ap.add_argument("--cosine", action="store_true", help="Use cosine LR decay after warmup")
    ap.add_argument("--single_z", action="store_true", help="Share single z across context batch")
    ap.add_argument("--identity", default="루리아", help="identity string used for identity auxiliary loss")
    ap.add_argument("--id_weight", type=float, default=0.2, help="weight of identity auxiliary loss")
    ap.add_argument("--cb_recent_len", type=int, default=24, help="recent codebook token window for transition model")

    # Legacy regularization params (kept for backward compatibility)
    ap.add_argument("--lam", type=float, default=1e-8, help="L1 regularization on z (legacy)")
    ap.add_argument("--lam_rule", type=float, default=6e-4, help="L1 regularization strength for z_rule (legacy)")
    ap.add_argument("--lam_think", type=float, default=4e-4, help="L1 regularization strength for z_thinking (legacy)")

    # 3brains regularization params (adjusted for balance)
    ap.add_argument("--lam_lang", type=float, default=4e-4, help="L1 regularization for language artisan")
    ap.add_argument("--lam_logic", type=float, default=6e-4, help="L1 regularization for logic artisan")
    ap.add_argument("--lam_bridge", type=float, default=4e-4, help="L1 regularization for bridge")

    # 3brains dimensions
    ap.add_argument("--z_slow_lang_dim", type=int, default=96, help="dimension for language artisan z")
    ap.add_argument("--z_slow_logic_dim", type=int, default=96, help="dimension for logic artisan z")
    ap.add_argument("--z_bridge_dim", type=int, default=64, help="dimension for bridge z")

    # Semantic alignment params (kept from patched_v2)
    ap.add_argument("--lam_sem", type=float, default=6e-4, help="weight for language-agnostic semantic InfoNCE loss")
    ap.add_argument("--lam_transfer", type=float, default=7e-4, help="weight for cross-language cosine alignment loss")
    ap.add_argument("--aux_baux", type=int, default=8, help="aux mini-batch size per language for semantic alignment")
    ap.add_argument("--sem_dim", type=int, default=32, help="shared semantic projection dim")

    args = ap.parse_args()
    # Apply dimension scaling (multiplies defaults or user-provided values)
    _scale = max(1, min(4, int(getattr(args, "scale", 1))))
    if _scale != 1:
        def _mul_arg(name: str):
            setattr(args, name, int(getattr(args, name)) * _scale)
        for _dim_name in [
            "d_model",
            "z_dim",
            "z_think_dim",
            "z_lang_dim",
            "identity_self_dim",
            "identity_intent_dim",
            "z_slow_lang_dim",
            "z_slow_logic_dim",
            "z_bridge_dim",
            "sem_dim",
        ]:
            _mul_arg(_dim_name)
    # Auto-recommend n_head/n_layer unless explicitly provided
    try:
        _argv = sys.argv
    except Exception:
        _argv = []
    _user_set_head = any(tok == "--n_head" for tok in _argv)
    _user_set_layer = any(tok == "--n_layer" for tok in _argv)
    # Recommend heads close to d_model/64, ensure divisibility and reasonable caps
    _target_heads = max(4, min(32, int(round(args.d_model / 64))))
    _candidates = [h for h in range(4, 33) if args.d_model % h == 0]
    if _candidates:
        _rec_head = min(_candidates, key=lambda h: (abs(h - _target_heads), -h))
    else:
        _rec_head = 4
    # Recommend layers by scale (4,6,8,10 for scale=1..4)
    _rec_layer = int(max(4, min(12, 4 + 2 * (_scale - 1))))
    if not _user_set_head:
        args.n_head = _rec_head
    if not _user_set_layer:
        args.n_layer = _rec_layer
    # Basic shape guard for attention head divisibility (post-auto)
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by n_head ({args.n_head})")
    if args.identity_self_dim <= 1:
        args.identity_intent_dim = 0
    else:
        args.identity_intent_dim = max(0, min(args.identity_intent_dim, args.identity_self_dim - 1))
    # Propagate self-eval toggle into model construction via environment
    if args.self_eval is not None:
        os.environ["UZR_SELF_EVAL"] = "1" if args.self_eval == "on" else "0"

    if args.lam_rule is None:
        args.lam_rule = args.lam
    if args.lam_think is None:
        args.lam_think = args.lam

    set_seed(args.seed)
    device = torch.device(args.device)
    # Select tokenizer (auto -> prefer local KoBERT if available)
    _user_set_tok = False
    try:
        _user_set_tok = any(tok == "--tokenizer" for tok in sys.argv)
    except Exception:
        pass
    def _resolve_kobert_dir(pref: str) -> Path:
        cands = []
        try:
            if pref:
                cands.append(Path(pref).expanduser())
        except Exception:
            pass
        try:
            # package-local: uzr/kobert
            cands.append(Path(__file__).resolve().parent / "kobert")
        except Exception:
            pass
        # cwd-based
        cands.append(Path.cwd() / "kobert")
        cands.append(Path.cwd() / "uzr" / "kobert")
        for p in cands:
            try:
                if p.exists() and (p / "vocab.txt").exists():
                    return p
            except Exception:
                continue
        return Path(pref) if pref else Path("kobert")

    tok_choice = getattr(args, "tokenizer", "auto")
    if tok_choice == "auto":
        kobert_root = _resolve_kobert_dir(getattr(args, "kobert_dir", "kobert"))
        if kobert_root.exists() and (kobert_root / "vocab.txt").exists():
            tok_choice = "kobert"
        else:
            tok_choice = "koen"
    if tok_choice == "kobert":
        try:
            _kobert_dir_res = _resolve_kobert_dir(getattr(args, "kobert_dir", "kobert"))
            tok = load_kobert_tokenizer(_kobert_dir_res, max_len=args.max_len)
            print(f"[Tokenizer] KoBERT tokenizer enabled (vocab={tok.vocab_size}, dir={_kobert_dir_res})")
        except Exception as e:
            print(f"[Tokenizer] KoBERT tokenizer load failed ({e}); falling back to KoEnTokenizer")
            tok = KoEnTokenizer(max_len=args.max_len)
    else:
        tok = KoEnTokenizer(max_len=args.max_len)
    dataset_sampler = None
    # Auto-enable dataset-mit mixing if local CSV exists and user didn't override mix prob
    try:
        _user_set_mix = any(tok == "--dataset_mix_prob" for tok in sys.argv)
    except Exception:
        _user_set_mix = False
    try:
        from uzr.tasks import DATASET_MIT_DEFAULT as _DS_DEF
    except Exception:
        _DS_DEF = Path("dataset-mit") / "mmlu_KO-KR.csv"
    ds_path = Path(args.dataset_mit_path).expanduser() if args.dataset_mit_path else _DS_DEF
    if (not _user_set_mix) and args.dataset_mix_prob == 0.0 and ds_path.exists():
        args.dataset_mix_prob = 0.35
        print(f"[Dataset] Auto-enabled dataset-mit mixing: prob={args.dataset_mix_prob} (found {ds_path})")
    if getattr(args, "dataset_mix_prob", 0.0) > 0.0:
        kobert_dir = Path(args.kobert_dir).expanduser() if args.kobert_dir else None
        try:
            dataset_sampler = DatasetMiTSampler(
                csv_path=ds_path,
                mix_prob=args.dataset_mix_prob,
                use_kobert_hint=args.kobert_hint,
                kobert_dir=kobert_dir,
                kobert_device=args.kobert_device,
                kobert_max_seq_len=args.kobert_max_seq_len,
            )
        except Exception as exc:
            raise RuntimeError(f"dataset-mit 샘플러 초기화 실패: {exc}") from exc

    # Meta configuration for thresholds/regularizers
    _meta_cfg = load_meta_config()
    _thr = AbstainThresholds(conf_min=_meta_cfg.get("conf_min", 0.65), ent_max=_meta_cfg.get("ent_max", 2.2))

    # Summary CSV (auto-name under logu/ when default)
    summary_path = args.summary_csv
    try:
        if summary_path == "train3_summary.csv":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("logu"); out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = str(out_dir / f"{ts}_train3_s{args.inner_steps}_t{args.steps}.csv")
    except Exception:
        pass
    _sum_writer = None
    try:
        import csv as _csv
        _sum_file = open(summary_path, "w", newline="", encoding="utf-8")
        _sum_writer = _csv.DictWriter(_sum_file, fieldnames=[
            "step","loss","ema","ema_raw","lambda_abs","ema_raw99","ema_all99","mean_raw_200",
            "perplexity","id_loss","lr","conf_mean","ent_mean","brier","abstain_ratio",
            "chosen_steps","avg_steps","s_max","inflation_rate","high_step_prob","rule_acc","top5_acc","avg_top1_prob",
            # Transition metrics (z/codebook)
            "z_cos","dz_mse","cb_f1","align_cos","jac_surr","dz_norm","dz_norm_std","cb_pos","cb_pred","trans_loss",
            "cloze_transition","gate_pass_60_95","z_lang_norm","z_logic_norm","z_bridge_norm","mem_size",
            "rejector_score","tau_r","tau_r_pi","intent_bias","intent_toggle","coverage","use_autonomous","difficulty_level","composite_score",
            # PI controller diagnostics
            "acc_r_200","acc_t",
            # mode/amp/eval
            "mode","amp","eval_set_id",
            "force_write_enabled","fw_ce","fw_len_pen",
            # diagnostics/modes
            "conserve_on","conserve_reason","lr_scale","fw_prob","surprise_gate_delta",
            # EMA integrity
            "ema_min_csv","ema_min_ckpt","ema_min_delta"
        ])
        _sum_writer.writeheader()
    except Exception:
        _sum_writer = None

    # Create memory first with learning enabled
    mem = CompressedMemory(
        max_items=32000,
        device=device,
        enable_learning=True,
        learn_hidden=int(512 * _scale),
        learn_depth=3,
        learn_rate=1e-3,
        min_train_items=32,
        warmup_steps=args.warmup_steps,
        entropy_floor=0.0,
        softmax_temp=0.8,
    )
    # Memory policy upgrade (lacomi.txt prescription + dynamic learning)
    # - Lower similarity threshold by 0.02 for better diversity
    # - near_merge_thr: 0.285 (initial, will be dynamic after step 350)
    # - Increase write budgets for more diverse memory
    mem.set_policy_thresholds(
        write_per_turn=2,
        write_per_100=14,           # Increased from 12 (more frequent writes)
        tail_write_per_100=8,       # Increased from 6
        entropy_floor=0.0,
        stage_mid_low=0.55,         # Lowered similarity threshold
        near_merge_thr=0.285,       # Initial value (dynamic after step 350)
        dup_skip_thr=0.90,          # Lowered by 0.02 from typical 0.92
    )
    # Enable entropy check only after bootstrapping
    setattr(mem, "entropy_check_start", 32)

    # Create model with memory reference and 3brains dimensions
    model = UZRModel(
        tok.vocab_size,
        d_model=args.d_model,
        z_dim=args.z_dim,
        n_head=args.n_head,
        n_layer=args.n_layer,
        max_len=args.max_len,
        z_think_dim=args.z_think_dim,
        z_lang_dim=args.z_lang_dim,
        num_langs=args.num_langs,
        identity_self_dim=args.identity_self_dim,
        identity_intent_dim=args.identity_intent_dim,
        memory=mem,
        z_slow_lang_dim=args.z_slow_lang_dim,
        z_slow_logic_dim=args.z_slow_logic_dim,
        z_bridge_dim=args.z_bridge_dim,
    ).to(device)
    try:
        model.set_tokenizer_specials(pad_id=int(getattr(tok, 'PAD', 0)), bos_id=int(getattr(tok, 'BOS', 1)), eos_id=int(getattr(tok, 'EOS', 2)))
    except Exception:
        pass

    # Initialize CodebookManager for text+vector representation
    codebook = CodebookManager.init(
        d=args.d_model,
        t_cfg=dict(Gt=6, Kt=128, dt=int(384 * _scale), m=131072, ema_decay=0.995),
        v_cfg=dict(G=6, K=128, beta=0.25, ema_decay=0.995),
        seed=args.seed,
        device=device,
        commit_steps=max(1, int(args.save_every)),
    )

    # Initialize multimodal transition module (z, u, codebook)
    try:
        _cb_vocab = int(codebook.t.Gt * codebook.t.Kt + codebook.v.G * codebook.v.K)
    except Exception:
        _cb_vocab = 1536  # fallback: 2*(6*128)
    _u_dim = 3  # [topk_norm, task_type(binary), lang_idx_norm]
    # Fused dim: modest width, proportional to z_bridge
    _fused_dim = max(128, 2 * int(args.z_bridge_dim))
    # Increase transition strength per request
    model.init_transition_module(
        z_bridge_dim=int(args.z_bridge_dim),
        u_dim=_u_dim,
        cb_vocab=_cb_vocab,
        cb_emb=64,
        fused_dim=_fused_dim,
        lam_trans=1.6e-3,
        lam_cos=8e-4,
        lam_roll=8e-4,
        lam_jac=1e-5,
        lam_cb=1.6e-3,
        lam_align=8e-4,
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- Transition buffer for multimodal dynamics ----
    TransSample = namedtuple("TransSample", ["z_t", "u_t", "cb_t", "z_t1", "cb_t1"])  # [D], [H], [L], [D], [L]
    transition_buffer = deque(maxlen=8192)
    last_pack = None  # (z_batch, u_batch, cb_batch)

    # ---- Generalization helpers (language-agnostic semantic space) ----
    _CONCEPT_VOCAB = 65536
    _concept_table = torch.nn.Embedding(_CONCEPT_VOCAB, args.sem_dim, device=device)
    torch.nn.init.normal_(_concept_table.weight, mean=0.0, std=0.02)
    _W_enc2sem = torch.nn.Parameter(torch.empty(args.d_model, args.sem_dim, device=device))
    _W_con2sem = torch.nn.Parameter(torch.empty(args.sem_dim, args.sem_dim, device=device))
    torch.nn.init.xavier_uniform_(_W_enc2sem); torch.nn.init.xavier_uniform_(_W_con2sem)
    opt.add_param_group({'params': [_W_enc2sem, _W_con2sem, _concept_table.weight], 'lr': args.lr})

    def _hash_ngrams(x_bytes: torch.Tensor, vocab_size=_CONCEPT_VOCAB, nmin: int = 2, nmax: int = 5):
        base = 257; MOD = 2**61 - 1
        B, T = x_bytes.shape
        idx_lists = []
        for b in range(B):
            ints = x_bytes[b].tolist()
            idxs = []
            for n in range(nmin, nmax+1):
                h = 0; p = pow(base, n-1, MOD)
                for i, ch in enumerate(ints):
                    if i < n:
                        h = (h * base + int(ch) + 1) % MOD
                        if i == n-1:
                            idxs.append(h % vocab_size)
                    else:
                        h = (h - (int(ints[i-n]) + 1) * p) % MOD
                        h = (h * base + int(ch) + 1) % MOD
                        idxs.append(h % vocab_size)
            if not idxs: idxs = [0]
            idx_lists.append(torch.tensor(idxs, device=x_bytes.device, dtype=torch.long))
        return idx_lists

    def concept_embed(x_bytes: torch.Tensor):
        idx_lists = _hash_ngrams(x_bytes)
        embs = [ _concept_table(idx).mean(dim=0) for idx in idx_lists ]
        return torch.stack(embs, dim=0)  # (B, args.sem_dim)

    def proj_sem(x, W):
        return torch.nn.functional.normalize(x @ W, dim=-1)

    def info_nce(q, k, tau: float = 0.07):
        logits = (q @ k.t()) / tau
        labels = torch.arange(q.size(0), device=q.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    start_step = 0
    best_loss = float("inf")
    last_path = "uzr_3brains_ckpt_last.pt"
    best_path = "uzr_3brains_ckpt_best.pt"

    # feels-goat: compact run meta header
    try:
        lr_mode = "cosine" if args.cosine else "linear"
        grad_clip_plan = "clip=2.0<2k,1.0>=2k"
        print(f"Meta: seed={args.seed} device={device} lr={lr_mode} save_every={args.save_every} {grad_clip_plan}")
    except Exception:
        pass

    # Rejector-Head parameters (350+ autonomous abstain learning)
    state_dim = 8  # [conf, ent, verifier, seq_len, gray_flag, infl_512, loss_delta, conf_delta]
    rejector_w = torch.nn.Parameter(torch.randn(state_dim, device=device) * 0.01)
    rejector_b = torch.nn.Parameter(torch.zeros(1, device=device))
    tau_r = torch.nn.Parameter(torch.tensor(0.5, device=device))  # learnable threshold
    opt.add_param_group({'params': [rejector_w, rejector_b, tau_r], 'lr': args.lr * 0.5})

    # Memory near-merge threshold parameter (autonomous learning)
    # Initialize around 0.88 as target center, will be clamped during adaptation
    near_merge_thr_param = torch.nn.Parameter(torch.tensor(0.88, device=device))
    opt.add_param_group({'params': [near_merge_thr_param], 'lr': args.lr * 0.3})

    # Memory top-k parameter (350+ autonomous learning)
    top_k_param = torch.nn.Parameter(torch.tensor(16.0, device=device))  # learnable, init=16 (center of 8-25)
    opt.add_param_group({'params': [top_k_param], 'lr': args.lr * 0.2})

    # Autonomous abstain tracking
    abstain_window_100 = []  # 100-step window for runaway detection
    target_coverage = 0.8  # target coverage τ (accept rate)
    prev_loss_val = None
    prev_conf_mean = None

    # feels-goat: EMA integrity tracking (csv vs ckpt)
    ema_min_csv = float("inf")
    ema_min_step_csv = -1

    # Reflection logger for periodic failure summaries (step-based)
    reflect = ReflectionLogger(out_dir="reflection")

    # feels-first: surprise gate temporary shift counter (p65->p60 analogy)
    surprise_shift_counter = 0
    surprise_gate_delta = 0.0
    _orig_write_thr_base = None

    # feels-goat: conserve mode (3k~6k)
    conserve_on = False
    conserve_reason = ""
    _orig_fw_prob = None

    # Memory growth tracking (for near_merge_thr adaptation)
    mem_size_window = []  # Track memory growth
    target_mem_growth_per_100 = 8  # Target: 8 items per 100 steps

    # Memory retrieval tracking (for top-k adaptation)
    retrieval_success_window = []  # Track retrieval success (100-step window)
    target_retrieval_success = 0.75  # Target: 75% successful retrieval

    # Memory operation policy (650+ autonomous learning)
    # 4 operation probabilities: [synthesis, split, interpolate, crossover]
    memop_logits = torch.nn.Parameter(torch.zeros(4, device=device))  # init to 0 (all equal after softmax)
    opt.add_param_group({'params': [memop_logits], 'lr': args.lr * 0.15})

    # Memory operation tracking
    memop_history = []  # Track operation success/failure
    target_accuracy = 0.85  # Target: 85% accuracy
    accuracy_window = []  # Track accuracy (100-step window)

    if args.resume and os.path.exists(args.resume):
        data = torch.load(args.resume, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(data["model"], strict=False)
        except TypeError:
            model.load_state_dict(data["model"])  # older torch compatible
        if "opt" in data:
            opt.load_state_dict(data["opt"])
        if "memory" in data:
            mem.load_state_dict(data["memory"])
            print(f"[resume] restored {len(mem.items)} memory items, learner: {mem.has_learner()}")
        # Load rejector parameters if available
        if "rejector_w" in data:
            rejector_w.data.copy_(data["rejector_w"])
            rejector_b.data.copy_(data["rejector_b"])
            tau_r.data.copy_(data["tau_r"])
            print(f"[resume] restored rejector params (tau_r={tau_r.item():.3f})")
            # Clamp restored tau_r to sane bounds to avoid pathological values
            with torch.no_grad():
                tau_r.clamp_(0.50, 0.90)
                print(f"[resume] clamped tau_r to {tau_r.item():.3f} in [0.50,0.90]")
        # Load memory merge threshold if available
        if "near_merge_thr_param" in data:
            near_merge_thr_param.data.copy_(data["near_merge_thr_param"])
            print(f"[resume] restored near_merge_thr={near_merge_thr_param.item():.3f}")
        # Load memory top-k if available
        if "top_k_param" in data:
            top_k_param.data.copy_(data["top_k_param"])
            print(f"[resume] restored top_k={top_k_param.item():.1f}")
        # Load memory operation policy if available
        if "memop_logits" in data:
            memop_logits.data.copy_(data["memop_logits"])
            probs = torch.softmax(memop_logits, dim=0)
            print(f"[resume] restored memop_logits, probs={probs.tolist()}")
        start_step = data.get("step", 0)
        best_loss = data.get("best_loss", best_loss)
        # Restore EMA integrity (csv minima) if present
        ema_min_csv = data.get("ema_min_csv", ema_min_csv)
        ema_min_step_csv = data.get("ema_min_step_csv", ema_min_step_csv)
        print(f"[resume] loaded {args.resume} from step {start_step} (best_loss={best_loss:.3f})")

    def lr_schedule(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps

        # After 2k steps: apply cosine annealing with warm restarts
        if step >= 2000:
            # Warm restart configuration
            T_0 = 500  # Initial restart period
            T_mult = 2  # Period multiplier after each restart
            eta_min = 0.1  # Minimum LR multiplier (10% of base LR)

            # Calculate which restart cycle we're in
            step_in_schedule = step - 2000
            T_cur = T_0
            accumulated_steps = 0
            cycle = 0

            while accumulated_steps + T_cur <= step_in_schedule:
                accumulated_steps += T_cur
                T_cur = int(T_cur * T_mult)
                cycle += 1

            # Position within current cycle
            step_in_cycle = step_in_schedule - accumulated_steps

            # Cosine annealing formula
            cos_factor = 0.5 * (1 + math.cos(math.pi * step_in_cycle / T_cur))
            lr_mult = eta_min + (1.0 - eta_min) * cos_factor

            return lr_mult

        # Before 2k steps: use original cosine or constant schedule
        if args.cosine:
            t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, t))))
        return 1.0

    ema = None  # EMA_all (after abstain penalty)
    ema_raw_tracker = None  # Track EMA_raw (loss only, beta=0.95)
    ema_raw99_tracker = None  # Track EMA_raw with beta=0.99 for diagnostics
    z_bridge_history = []  # Track z_bridge for decay detection

    # Coverage EMA tracker (rikomi.txt: 100-step smoothing to prevent LR jitter)
    coverage_ema = 0.8  # Start optimistic (high coverage assumption)
    # Acceptance PI controller state (200-step window)
    accept_window_200 = []
    acc_error_int = 0.0
    tau_r_pi = 0.0  # additive bias applied to tau_r for current step
    kp_pi = 0.5
    ki_pi = 0.05
    # Rolling window for mean_last_200(raw)
    loss_window_200 = []

    # Adaptive inner-step parameters (from 적응형 이너스텝.txt + inner-step-tune.txt)
    s_base = 6
    s_max_adaptive = 10  # Start at 10 (from tune.txt), can increase to 15 later
    s_min_adaptive = 4
    conf_ema_tracker = 0.5  # Exponential moving average of confidence

    # Inflation tracker (512-sample window)
    inflation_window = []
    inflation_max_len = 512
    inflation_threshold = 1.30
    inflation_penalty_steps = 0  # Counter for penalty period

    # Cooldown tracker: {task_key: cooldown_remaining}
    cooldown_tracker = {}
    cooldown_duration = 200

    # Abstain ramp-down parameters (1000→3000 steps)
    abstain_hysteresis_state = False  # Track if currently abstaining

    # Quality gate parameters (1k-2k relaxed zone)
    prev_ema = None
    prev_rule_acc = None
    prev_perplexity = None

    # Replay buffer for failed samples with length-balanced sampling
    replay_buffer = ReplayBuffer(maxlen=10000, fail_decay=0.95)

    # Length bin counters for balanced replay (≤16, 17-32, 33-64)
    length_bin_counts = {"short": 0, "medium": 0, "long": 0}

    # Curriculum difficulty parameters (adaptive based on avg_steps/chosen_steps)
    difficulty_level = 0  # 0=easy, 1=medium, 2=hard
    difficulty_update_window = []  # Track cost efficiency
    difficulty_window_size = 100

    # Initialize memory event counters for Luria logging system
    memory_counters = {
        # per-step tick counters (reset each step)
        "promote_tick": 0, "decay_tick": 0, "skip_tick": 0,
        "commit_tick": 0, "merge_tick": 0, "near_merge_tick": 0,
        # totals (never reset)
        "promote_total": 0, "decay_total": 0, "skip_total": 0,
        "commit_total": 0, "merge_total": 0, "near_merge_total": 0,
        # snapshot values
        "mem_size_curr": 0
    }

    # Current stage tracking
    current_stage = "warmup"
    log_stage_event(start_step, current_stage, "training_start", {"steps": args.steps})

    pbar = tqdm(range(start_step, args.steps), desc="3brains-meta-train")
    for step in pbar:
        # rikomi.txt fix: Initialize gate_passed early to prevent UnboundLocalError
        gate_passed = True  # Safe default: allow update unless gate check fails later

        # Determine if autonomous abstain is enabled
        use_autonomous_abstain = (step >= args.autonomous_step)

        # Dynamic near_merge_thr (adapts after autonomous_step)
        if step < args.autonomous_step:
            # Before autonomous: keep fixed threshold at 0.90 to preserve stage path
            near_merge_thr_current = 0.90
        else:
            # Autonomous: learnable parameter, clamped to [0.82, 0.95]
            with torch.no_grad():
                near_merge_thr_param.data.clamp_(0.82, 0.95)
            near_merge_thr_current = near_merge_thr_param.item()

        # Update memory policy with current threshold
        mem.set_policy_thresholds(
            write_per_turn=2,
            write_per_100=14,
            tail_write_per_100=8,
            entropy_floor=0.0,
            stage_mid_low=0.60,
            near_merge_thr=near_merge_thr_current,
            dup_skip_thr=0.97,
        )

        # Determine if force write is enabled for this batch
        force_write_enabled = should_force_write(step)
        fw_ce, fw_len_pen = 0.0, 0.0  # Initialize force write metrics

        # Initialize transition metric placeholders (per-step)
        z_cos_metric = float('nan')
        dz_mse_metric = float('nan')
        cb_f1_metric = float('nan')
        align_cos_metric = float('nan')
        jac_surr_metric = float('nan')
        dz_norm_mean_metric = float('nan')
        dz_norm_std_metric = float('nan')
        cb_pos_rate_metric = float('nan')
        cb_pred_rate_metric = float('nan')
        trans_loss_metric = float('nan')

        # feels-goat: conserve mode (3k~6k) — tighten writes and LR; adjust FW prob
        if 3000 <= step < 6000:
            if not conserve_on:
                conserve_on = True
                conserve_reason = "conserve_window_3k_6k"
                try:
                    if _orig_write_thr_base is None:
                        _orig_write_thr_base = getattr(mem, "write_threshold_base", 0.40)
                    mem.write_threshold_base = float(_orig_write_thr_base + 0.05)
                    surprise_gate_delta = 0.05
                except Exception:
                    pass
                try:
                    from .hooks import get_force_write_stats
                    if _orig_fw_prob is None:
                        _orig_fw_prob = get_force_write_stats().get("prob", None)
                    update_force_write_config(prob=0.05)
                except Exception:
                    pass
        else:
            if conserve_on:
                conserve_on = False
                conserve_reason = ""
                try:
                    if _orig_write_thr_base is not None:
                        mem.write_threshold_base = float(_orig_write_thr_base)
                    surprise_gate_delta = 0.0
                except Exception:
                    pass
                try:
                    if _orig_fw_prob is not None:
                        update_force_write_config(prob=float(_orig_fw_prob))
                except Exception:
                    pass

        # Adaptive difficulty based on cost efficiency (avg_steps / chosen_steps)
        # Difficulty increases when model is efficient (low avg_steps relative to base)
        if step > 0 and step % 100 == 0 and len(difficulty_update_window) >= difficulty_window_size:
            # Calculate cost efficiency: avg_steps / s_base
            avg_cost = sum(difficulty_update_window) / len(difficulty_update_window)
            cost_ratio = avg_cost / s_base

            # Adjust difficulty based on cost ratio
            # If cost_ratio < 1.1: model is efficient -> increase difficulty
            # If cost_ratio > 1.4: model is struggling -> decrease difficulty
            if cost_ratio < 1.1 and difficulty_level < 2:
                difficulty_level += 1  # Increase difficulty
            elif cost_ratio > 1.4 and difficulty_level > 0:
                difficulty_level -= 1  # Decrease difficulty

            # Clear window for next period
            difficulty_update_window = []

        # Map difficulty level to task parameters
        if difficulty_level == 0:
            # Easy: baseline
            n_context, n_query, n_tokens = 6, 8, 5
        elif difficulty_level == 1:
            # Medium: slightly harder
            n_context, n_query, n_tokens = 7, 9, 6
        else:
            # Hard: maximum difficulty
            n_context, n_query, n_tokens = 8, 10, 7

        C_pairs, Q_pairs, desc = sample_task(
            n_context=n_context,
            n_query=n_query,
            n_tokens=n_tokens,
            dataset_sampler=dataset_sampler,
        )

        # Replay injection (manual: mix failed samples with a small ratio)
        try:
            replay_ratio = float(_meta_cfg.get("replay_ratio", 0.25))
        except Exception:
            replay_ratio = 0.25
        try:
            import random as _rnd
            if replay_buffer and _rnd.random() < replay_ratio:
                for it in replay_buffer.sample(k=4):
                    if isinstance(it.x, str) and isinstance(it.y, str) and it.x and it.y:
                        Q_pairs.append((it.x, it.y))
        except Exception:
            pass

        example_text = Q_pairs[0][0] + " " + Q_pairs[0][1]
        lang_key = detect_lang_from_text(example_text)
        lang_idx = LANG2ID.get(lang_key, LANG2ID["base"])

        # Detect task type (logic vs language)
        task_type = detect_task_type(example_text, desc)

        Xc, Yc = batchify(C_pairs, tok)
        Xq, Yq = batchify(Q_pairs, tok)
        Xc, Yc, Xq, Yq = Xc.to(device), Yc.to(device), Xq.to(device), Yq.to(device)

        # Compute initial metrics for adaptive inner steps
        with torch.no_grad():
            conf_vec_init = model.confidence(Xc)
            if conf_vec_init is None:
                conf0 = 0.5  # fallback
                verifier0 = 0.5
            else:
                conf0 = float(conf_vec_init.mean().item())
                # Use confidence as verifier (can be refined with separate verification head)
                verifier0 = conf0

            # Compute initial entropy
            h_init = model.encoder(Xc)
            logits_init = model.readout(h_init)
            ent0 = float(model.sequence_entropy(logits_init).mean().item())

            # Compute sequence length
            seq_len = int((Xc != tok.PAD).sum(dim=1).float().mean().item())

        # Update conf_ema
        conf_ema_tracker = 0.7 * conf_ema_tracker + 0.3 * conf0

        # Update cooldown trackers
        for key in list(cooldown_tracker.keys()):
            cooldown_tracker[key] -= 1
            if cooldown_tracker[key] <= 0:
                del cooldown_tracker[key]

        # Budget governor: check inflation
        if inflation_penalty_steps > 0:
            s_max_adaptive = max(8, s_max_adaptive - 3)
            inflation_penalty_steps -= 1
        elif len(inflation_window) >= inflation_max_len:
            avg_steps = sum(inflation_window) / len(inflation_window)
            inflation = avg_steps / s_base
            if inflation > inflation_threshold:
                # Trigger penalty for next 256 steps
                inflation_penalty_steps = 256
                s_max_adaptive = max(8, s_max_adaptive - 3)

        # Intent-driven budget (autonomy) — do not bypass gates; only bias budgets within guardrails
        # Use identity intent to modulate step cap and final steps slightly
        try:
            intent_bias, intent_toggle = model.identity_intent_control()
        except Exception:
            intent_bias, intent_toggle = 0.0, 0.0
        # Map bias in [-1,1] to cap bonus [-3,+3]
        try:
            _ib = max(-1.0, min(1.0, float(intent_bias)))
        except Exception:
            _ib = 0.0
        cap_bonus = int(round(3.0 * _ib))
        # Apply to s_max with safety clamps (4..25; length and cooldown caps handled in chooser)
        s_max_cap_intent = max(4, min(25, s_max_adaptive + cap_bonus))

        # Create task key for cooldown
        task_key = f"{lang_key}_{task_type}_{desc[:10]}"

        # Choose adaptive steps using bucket mapping
        chosen_steps = choose_adaptive_steps(
            conf=conf0,
            ent=ent0,
            verifier=verifier0,
            seq_len=seq_len,
            conf_ema=conf_ema_tracker,
            s_max_cap=s_max_cap_intent,
            task_key=task_key,
            cooldown_tracker=cooldown_tracker
        )

        # Override inner steps by Luria's will: map intent bias [-1,1] → [4,25]
        try:
            ib01 = 0.5 * (_ib + 1.0)  # [0,1]
            will_steps = int(round(4 + ib01 * (25 - 4)))
            chosen_steps = int(max(4, min(s_max_cap_intent, will_steps)))
        except Exception:
            pass

        # Track chosen steps for inflation calculation
        inflation_window.append(chosen_steps)
        if len(inflation_window) > inflation_max_len:
            inflation_window.pop(0)

        # Track chosen steps for difficulty adaptation
        difficulty_update_window.append(chosen_steps)
        if len(difficulty_update_window) > difficulty_window_size:
            difficulty_update_window.pop(0)

        # Set cooldown if high steps were used
        if chosen_steps >= 12:
            cooldown_tracker[task_key] = cooldown_duration

        # Dynamic top-k (6-18 range, Luria-controlled after 350)
        if step < 350:
            # Before 350: linked to inner steps
            # Linear mapping: steps [4, 25] → top_k [6, 18]
            top_k_dynamic = int(6 + (chosen_steps - 4) * (18 - 6) / (25 - 4))
            top_k_dynamic = max(6, min(18, top_k_dynamic))
        else:
            # After 350: Luria's will directly controls top-k via intent bias
            try:
                ib01 = 0.5 * (_ib + 1.0)  # [0,1]
                top_k_dynamic = int(round(6 + ib01 * (18 - 6)))
            except Exception:
                with torch.no_grad():
                    top_k_param.data.clamp_(6.0, 18.0)
                top_k_dynamic = int(max(6, min(18, round(float(top_k_param.item())))))

        # 3brains adaptation
        if args.single_z:
            enc_avg = model.avg_embed(Xc).mean(dim=0)
            # Initialize z for 3brains
            z_lang0 = model.init_z_slow_lang(batch_size=1)[0].to(device)
            z_logic0 = model.init_z_slow_logic(batch_size=1)[0].to(device)
            z_bridge0 = model.init_z_bridge(batch_size=1)[0].to(device)

            # Try to get better bridge initialization from memory (dynamic top-k)
            z_from_mem = model.get_z_from_memory(Xc, z_init=z_bridge0, topk=top_k_dynamic, blend=0.6)
            retrieval_success = (z_from_mem is not None)
            if retrieval_success:
                # Adapt memory z to bridge dimension
                if z_from_mem.size(-1) >= z_bridge0.size(-1):
                    z_bridge0 = z_from_mem[..., :z_bridge0.size(-1)]
                else:
                    z_bridge0[..., :z_from_mem.size(-1)] = z_from_mem

            z_lang_star, z_logic_star, z_bridge_star = inner_adapt_z_3brains(
                model,
                Xc,
                Yc,
                z_lang0,
                z_logic0,
                z_bridge0,
                lam_lang=args.lam_lang,
                lam_logic=args.lam_logic,
                lam_bridge=args.lam_bridge,
                eta=args.inner_eta,
                steps=chosen_steps,
                ignore_index=int(getattr(tok, 'PAD', 0)),
            )
            z_for_q = {"slow_lang": z_lang_star, "slow_logic": z_logic_star, "bridge": z_bridge_star}
        else:
            # Multi-sample adaptation (ensure z_for_q is always defined before transition collection)
            z_lang0 = model.init_z_slow_lang(batch_size=Xc.size(0)).to(device)
            z_logic0 = model.init_z_slow_logic(batch_size=Xc.size(0)).to(device)
            z_bridge0 = model.init_z_bridge(batch_size=Xc.size(0)).to(device)

            # Try to get better bridge initialization from memory for each sample (dynamic top-k)
            retrieval_success_count = 0
            for b in range(Xc.size(0)):
                z_from_mem = model.get_z_from_memory(Xc[b:b+1], z_init=z_bridge0[b], topk=top_k_dynamic, blend=0.6)
                if z_from_mem is not None:
                    retrieval_success_count += 1
                    # Adapt memory z to bridge dimension
                    if z_from_mem.size(-1) >= z_bridge0[b].size(-1):
                        z_bridge0[b] = z_from_mem[..., :z_bridge0[b].size(-1)].squeeze(0)
                    else:
                        z_bridge0[b, :z_from_mem.size(-1)] = z_from_mem.squeeze(0)
            retrieval_success = (retrieval_success_count > 0)  # At least one success

            zl_list, zg_list, zb_list = [], [], []
            for b in range(Xc.size(0)):
                zl, zg, zb = inner_adapt_z_3brains(
                    model,
                    Xc[b:b + 1],
                    Yc[b:b + 1],
                    z_lang0[b],
                    z_logic0[b],
                    z_bridge0[b],
                    lam_lang=args.lam_lang,
                    lam_logic=args.lam_logic,
                    lam_bridge=args.lam_bridge,
                    eta=args.inner_eta,
                    steps=chosen_steps,
                    ignore_index=int(getattr(tok, 'PAD', 0)),
                )
                zl_list.append(zl)
                zg_list.append(zg)
                zb_list.append(zb)
            z_lang_star = torch.stack(zl_list, dim=0).mean(dim=0)
            z_logic_star = torch.stack(zg_list, dim=0).mean(dim=0)
            z_bridge_star = torch.stack(zb_list, dim=0).mean(dim=0)
            z_for_q = {"slow_lang": z_lang_star, "slow_logic": z_logic_star, "bridge": z_bridge_star}

        # ---- Collect transition samples (build (z,u,cb) pairs across steps) ----
        try:
            # Update EMA for z normalization
            with torch.no_grad():
                model.update_ema_stats(z_bridge_star.detach())

            # Build u_t features (shared across batch)
            top_k_norm = float(max(8, min(25, top_k_dynamic)) - 8) / 17.0
            task_bin = 1.0 if task_type == "language" else 0.0
            lang_norm = float(lang_idx) / max(1.0, float(len(LANG2ID) - 1))
            Bq = int(Xq.size(0))
            u_vec = torch.tensor([top_k_norm, task_bin, lang_norm], device=device, dtype=torch.float32)
            u_batch = u_vec.unsqueeze(0).expand(Bq, -1).contiguous()  # [B, 3]

            # Build codebook token id window per sample
            L_recent = int(args.cb_recent_len)
            try:
                enc_avg_q_batch = model.avg_embed(Xq)  # [B, d]
            except Exception:
                enc_avg_q_batch = torch.zeros(Bq, args.d_model, device=device)

            def _token_to_id(token: str) -> int:
                # Map T tokens (e.g., "KA07") then V tokens (e.g., "A03") into single vocab.
                try:
                    # T tokens start with 'K'
                    if isinstance(token, str) and len(token) >= 3 and token[0] == 'K':
                        g = max(0, ord(token[1]) - ord('A'))
                        idx = int(token[-2:]) if token[-2:].isdigit() else 0
                        return g * getattr(codebook.t, 'Kt', 128) + idx
                    # V tokens: <A..Z><00..>
                    if isinstance(token, str) and len(token) >= 2:
                        g = max(0, ord(token[0]) - ord('A'))
                        idx = int(token[1:]) if token[1:].isdigit() else 0
                        base = getattr(codebook.t, 'Gt', 6) * getattr(codebook.t, 'Kt', 128)
                        return base + g * getattr(codebook.v, 'K', 128) + idx
                except Exception:
                    pass
                return 0

            cb_ids_list = []
            for b in range(Bq):
                try:
                    text_b = tok.decode(Xq[b].tolist())
                except Exception:
                    text_b = ""
                try:
                    zc = codebook.encode(text=text_b, h=enc_avg_q_batch[b], return_parts=True)
                except Exception:
                    zc = {"T": [], "V": []}
                ids = []
                for t in list(zc.get("T", [])) + list(zc.get("V", [])):
                    ids.append(_token_to_id(t))
                # pad/trim to window
                ids = (ids + [0] * L_recent)[:L_recent]
                cb_ids_list.append(torch.tensor(ids, device=device, dtype=torch.long))
            cb_t_ids = torch.stack(cb_ids_list, dim=0)  # [B, L]

            # Pair with last step
            z_now = z_bridge_star.detach().unsqueeze(0).expand(Bq, -1).contiguous().cpu()
            u_now = u_batch.detach().cpu()
            cb_now = cb_t_ids.detach().cpu()
            current_pack = (z_now, u_now, cb_now)
            if last_pack is not None:
                z_prev, u_prev, cb_prev = last_pack
                Bp = min(len(z_prev), len(z_now))
                for i in range(Bp):
                    transition_buffer.append(TransSample(z_prev[i], u_prev[i], cb_prev[i], z_now[i], cb_now[i]))
            last_pack = current_pack
        except Exception:
            # Non-fatal; skip transition collection this step
            pass

        # Z-bridge decay mechanism for stagnation detection
        if step > 7000:  # After ~15 minutes (assuming ~8 steps/sec)
            # Track z_bridge statistics
            z_bridge_history.append(z_bridge_star.detach().cpu())
            if len(z_bridge_history) > 100:
                z_bridge_history = z_bridge_history[-100:]

            # Check variability and apply decay every 50 steps
            if step % 50 == 0 and len(z_bridge_history) >= 50:
                recent = torch.stack(z_bridge_history[-50:])
                mean_vals = recent.mean(dim=0)
                median_vals = recent.median(dim=0).values
                mean_var = mean_vals.std().item()
                median_var = median_vals.std().item()

                # If variation is too small (<0.1%), apply decay
                if mean_var < 0.001 and median_var < 0.001:
                    z_bridge_star = z_bridge_star * 0.95
                    z_for_q["bridge"] = z_bridge_star

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            logits_q = model(Xq, z_for_q)
            loss = seq_ce_loss(logits_q, Yq, ignore_index=int(getattr(tok, 'PAD', 0)))
            # Self-eval metrics on query batch
            conf_vec = model.confidence(Xq)
            if conf_vec is None:
                # fallback proxy from logits/targets
                from .model import confidence_from_logits as _proxy_conf
                conf_vec = _proxy_conf(logits_q, Yq, ignore_index=int(getattr(tok, 'PAD', 0)))
            ent_vec = model.sequence_entropy(logits_q)
            # Accuracy vector (per-sample) for accuracy-aware gating/weighting
            try:
                with torch.no_grad():
                    pred_tokens = torch.argmax(logits_q, dim=-1)
                    mask_acc = (Yq != int(getattr(tok, 'PAD', 0)))
                    correct_tok = (pred_tokens == Yq) & mask_acc
                    acc_vec = correct_tok.float().sum(dim=1) / mask_acc.float().sum(dim=1).clamp(min=1.0)
            except Exception:
                acc_vec = torch.zeros(Xq.size(0), device=logits_q.device)
            # Brier regularizer (align scalar conf with token-correctness)
            brier = model.brier_from_logits_conf(logits_q, Yq, conf_vec, ignore_index=int(getattr(tok, 'PAD', 0)))

            # Identity aux: use 3brains for identity QA
            Xi, Yi = make_identity_batch(tok, args.identity, batch_lang="mix", n=4)
            Xi, Yi = Xi.to(device), Yi.to(device)
            logits_i = model(Xi, {"slow_lang": z_lang_star, "slow_logic": z_logic_star, "bridge": z_bridge_star})
            id_loss = seq_ce_loss(logits_i, Yi, ignore_index=int(getattr(tok, 'PAD', 0)))
            # Combine base loss (task + Brier) with abstain-weighting per manual
            try:
                lambda_brier = float(_meta_cfg.get("lambda_brier", 0.2))
            except Exception:
                lambda_brier = 0.2
            base_total = loss + lambda_brier * brier
            # maybe_abstain: use conf/ent thresholds; apply as soft weight during training
            # Accuracy override: if per-sample accuracy is high, do NOT downweight that sample
            try:
                abstain_mask = maybe_abstain(conf_vec.detach(), ent_vec.detach(), _thr)  # [B] bool
                # thresholds from meta config (fallback defaults)
                try:
                    acc_override_min = float(_meta_cfg.get("acc_override_min", 0.75))
                    min_w = float(_meta_cfg.get("abstain_min_weight", 0.2))
                except Exception:
                    acc_override_min = 0.75
                    min_w = 0.2
                w0 = (1.0 - abstain_mask.float()).detach()  # 1=learn, 0=abstain
                acc_gate = (acc_vec.detach() >= acc_override_min).float()
                w_i = torch.maximum(w0, acc_gate)  # accuracy can only increase learning weight
                w = w_i + min_w * (1.0 - w_i)      # keep a minimum weight
                weighted_base = base_total * w.mean()
            except Exception:
                weighted_base = base_total
            total = weighted_base + args.id_weight * id_loss

            # ---- Multimodal transition loss (if buffer is warmed up) ----
            try:
                if len(transition_buffer) >= 1024:
                    import random as _rnd
                    B_trans = min(256, len(transition_buffer))
                    batch = _rnd.sample(list(transition_buffer), B_trans)
                    z_t = torch.stack([b.z_t for b in batch]).to(device).float()
                    u_t = torch.stack([b.u_t for b in batch]).to(device).float()
                    cb_t = torch.stack([b.cb_t for b in batch]).to(device).long()
                    z_t1 = torch.stack([b.z_t1 for b in batch]).to(device).float()
                    cb_t1 = torch.stack([b.cb_t1 for b in batch]).to(device).long()

                    z_t_norm = model.norm_z(z_t)
                    cb_vec_t = model.cb_enc(cb_t)
                    fused = model.fuse(z_t_norm, u_t, cb_vec_t)

                    # z-head
                    dz_pred = model.trans_z(fused)
                    z_t1_pred = z_t + dz_pred

                    # cb-head (bag-of-words logits)
                    logits_cb = model.trans_cb(fused)
                    # targets: presence-only BOW for next step
                    _cb_vocab = logits_cb.size(-1)
                    bow_t1 = torch.zeros((B_trans, _cb_vocab), device=device)
                    # unique per-sample
                    for i in range(B_trans):
                        uniq = torch.unique(cb_t1[i])
                        bow_t1[i].scatter_(0, uniq, 1.0)

                    # losses
                    dz_tgt = F.normalize(z_t1 - z_t, dim=-1)
                    dz_prd = F.normalize(dz_pred, dim=-1)
                    L_delta = F.mse_loss(dz_prd, dz_tgt)
                    L_cos = 1.0 - F.cosine_similarity(
                        F.normalize(z_t1_pred, dim=-1),
                        F.normalize(z_t1, dim=-1),
                        dim=-1,
                    ).mean()
                    L_cb = F.binary_cross_entropy_with_logits(logits_cb, bow_t1)
                    cb_vec_t1 = model.cb_enc(cb_t1)
                    L_align = 1.0 - F.cosine_similarity(
                        F.normalize(z_t1_pred, dim=-1),
                        F.normalize(cb_vec_t1, dim=-1),
                        dim=-1,
                    ).mean()
                    L_jac = (dz_pred.norm(p=2, dim=-1).mean() / (z_t.norm(p=2, dim=-1).mean() + 1e-6))
                    L_roll = torch.tensor(0.0, device=device)

                    L_trans = (
                        model.lam_trans * L_delta
                        + model.lam_cos * L_cos
                        + model.lam_cb * L_cb
                        + model.lam_align * L_align
                        + model.lam_jac * L_jac
                        + model.lam_roll * L_roll
                    )
                    total = total + L_trans

                    # Transition metrics (detach to scalar floats)
                    with torch.no_grad():
                        # z 1-step cosine (accuracy proxy)
                        z_cos_metric = float(1.0 - L_cos.detach().item())
                        # Δz MSE (directional)
                        dz_mse_metric = float(L_delta.detach().item())
                        # z↔cb align cosine
                        align_cos_metric = float(1.0 - L_align.detach().item())
                        # Jacobian surrogate
                        jac_surr_metric = float(L_jac.detach().item())
                        # Δz norm stats
                        dz_norms = dz_pred.detach().norm(p=2, dim=-1)
                        dz_norm_mean_metric = float(dz_norms.mean().item())
                        dz_norm_std_metric = float(dz_norms.std().item())
                        # codebook F1 (micro, threshold=0.5)
                        preds = (torch.sigmoid(logits_cb.detach()) > 0.5).float()
                        true = bow_t1.detach()
                        tp = (preds * true).sum()
                        pp = preds.sum()
                        tp2 = true.sum()
                        precision = (tp / (pp + 1e-8)).item() if pp.item() > 0 else 0.0
                        recall = (tp / (tp2 + 1e-8)).item() if tp2.item() > 0 else 0.0
                        denom = (precision + recall) if (precision + recall) > 0 else 1e-8
                        cb_f1_metric = float(2.0 * precision * recall / denom)
                        cb_pos_rate_metric = float(true.mean().item())
                        cb_pred_rate_metric = float(preds.mean().item())
                        # total transition loss
                        trans_loss_metric = float(L_trans.detach().item())
            except Exception:
                # If anything goes wrong, proceed without transition loss this step
                pass

            # Force write mini-task: apply extra loss to break uniform distribution fixation
            if force_write_enabled:
                # Create a minimal batch structure for force write
                batch_inputs = {"input_ids": Xq, "labels": Yq}
                fw_loss, fw_info = apply_force_write(batch_inputs, None, logits_q, Yq, step)
                total = total + fw_loss
                fw_ce = fw_info["fw_ce"]
                fw_len_pen = fw_info["fw_len_pen"]

            # ---- MOVED UP: Compute state variables BEFORE using them in autonomous abstain ----
            loss_val = float(loss.item())

            # Compute state vector components
            try:
                conf_mean_current = float(conf_vec.mean().item())
                ent_mean_current = float(ent_vec.mean().item())
                acc_mean_current = float(acc_vec.mean().item())
            except Exception:
                conf_mean_current, ent_mean_current = 0.5, 2.0
                acc_mean_current = 0.0

            # Calculate current inflation rate (needed for state_vec)
            if len(inflation_window) > 0:
                avg_steps_current = sum(inflation_window) / len(inflation_window)
                inflation_rate = avg_steps_current / s_base
            else:
                avg_steps_current, inflation_rate = s_base, 1.0

            # State vector: [conf, ent, verifier, seq_len, gray_flag, infl_512, loss_delta, conf_delta]
            loss_delta = (loss_val - prev_loss_val) if prev_loss_val is not None else 0.0
            conf_delta = (conf_mean_current - prev_conf_mean) if prev_conf_mean is not None else 0.0
            in_gray_zone = (0.85 <= verifier0 < 1.0)  # Expanded to 15% (was 0.90-0.95, 5%)

            state_vec = torch.tensor([
                conf_mean_current,
                ent_mean_current,
                verifier0,
                min(seq_len / 128.0, 1.0),  # normalized
                1.0 if in_gray_zone else 0.0,
                inflation_rate,
                loss_delta,
                conf_delta,
            ], device=device, dtype=torch.float32)
            # ---- END MOVED BLOCK ----

            # Autonomous abstain learning (5k+ steps)
            # Add selective risk + coverage loss
            if use_autonomous_abstain:
                # Rejector score (trainable)
                rejector_score = torch.sigmoid(rejector_w @ state_vec + rejector_b)

                # Coverage (accept rate): should match target_coverage
                coverage = 1.0 - rejector_score
                loss_coverage = ((coverage - target_coverage) ** 2)

                # Selective risk: penalize rejecting good samples or accepting bad ones
                # Proxy for "correctness": high conf + low ent
                quality_proxy = conf_mean_current * (1.0 / (1.0 + ent_mean_current))
                loss_selective = rejector_score * quality_proxy  # penalty for rejecting good samples

                # Rejection cost: penalize abstaining (to prevent "always abstain")
                lambda_rej = 0.1 if step < 1350 else 0.05
                loss_rejection = rejector_score * lambda_rej

                # Cost constraint: inflation penalty
                lambda_infl = 0.02
                loss_cost = lambda_infl * max(0.0, inflation_rate - 1.30) if len(inflation_window) > 0 else 0.0

                # Scheduler for weights (350-1350: supervised only, 1350+: aggressive)
                if step < 1350:
                    # 350-1350: supervised only
                    lambda_cov = 2.0
                    lambda_sel = 1.0
                else:
                    # 1350+: reduce coverage constraint, increase selective risk
                    lambda_cov = 1.0
                    lambda_sel = 2.0

                # Add to total loss
                total = total + lambda_cov * loss_coverage + lambda_sel * loss_selective + loss_rejection + loss_cost

            # ---- Adaptive near_merge_thr learning (350+ steps) ----
            if step >= 350:
                # Track memory growth
                current_mem_size = len(mem.items)
                mem_size_window.append(current_mem_size)
                if len(mem_size_window) > 100:
                    mem_size_window.pop(0)

                # Calculate memory growth rate (items per 100 steps)
                if len(mem_size_window) >= 100 and step % 10 == 0:
                    mem_growth_rate = mem_size_window[-1] - mem_size_window[0]

                    # Memory growth loss: penalize deviation from target
                    # Target: 8 items per 100 steps (balanced growth)
                    growth_error = abs(mem_growth_rate - target_mem_growth_per_100)
                    loss_mem_growth = torch.tensor(growth_error / target_mem_growth_per_100, device=device, dtype=torch.float32)

                    # Diversity loss: penalize extreme thresholds
                    # Prefer staying near center (0.30) of range [0.285, 0.315]
                    diversity_error = (near_merge_thr_param - 0.30) ** 2
                    loss_diversity = 0.5 * diversity_error

                    # Efficiency loss: too high threshold wastes memory, too low reduces diversity
                    # Use memory fill ratio as efficiency metric
                    mem_fill_ratio = current_mem_size / max(1, mem.max_items)
                    if mem_fill_ratio < 0.10:  # Too slow growth
                        loss_efficiency = (0.10 - mem_fill_ratio) * 2.0
                    elif mem_fill_ratio > 0.40:  # Too fast growth (before step 15k)
                        loss_efficiency = (mem_fill_ratio - 0.40) * 1.0
                    else:
                        loss_efficiency = 0.0
                    loss_efficiency = torch.tensor(loss_efficiency, device=device, dtype=torch.float32)

                    # Combine memory-related losses
                    lambda_mem_growth = 0.1
                    lambda_diversity = 0.05
                    lambda_efficiency = 0.08

                    total = total + lambda_mem_growth * loss_mem_growth + lambda_diversity * loss_diversity + lambda_efficiency * loss_efficiency

            # ---- Adaptive top-k learning (350+ steps) ----
            if step >= 350:
                # Track retrieval success
                retrieval_success_window.append(1.0 if retrieval_success else 0.0)
                if len(retrieval_success_window) > 100:
                    retrieval_success_window.pop(0)

                # Calculate retrieval success rate (every 10 steps)
                if len(retrieval_success_window) >= 50 and step % 10 == 0:
                    success_rate = sum(retrieval_success_window) / len(retrieval_success_window)

                    # Retrieval quality loss: penalize low success rate
                    # Target: 75% successful retrieval
                    quality_error = abs(success_rate - target_retrieval_success)
                    loss_retrieval_quality = torch.tensor(quality_error, device=device, dtype=torch.float32)

                    # Efficiency loss: penalize high top-k (computation cost)
                    # Normalize: k ∈ [6, 18] → [0, 1]
                    k_normalized = (top_k_param - 6.0) / 12.0
                    loss_topk_efficiency = 0.5 * k_normalized

                    # Diversity loss: prefer center (12) of range [6, 18]
                    topk_diversity_error = (top_k_param - 12.0) ** 2 / 36.0  # (18-6)^2/4 = 36
                    loss_topk_diversity = 0.3 * topk_diversity_error

                    # Combine top-k related losses
                    lambda_retrieval_quality = 0.15
                    lambda_topk_efficiency = 0.08
                    lambda_topk_diversity = 0.05

                    total = total + lambda_retrieval_quality * loss_retrieval_quality + lambda_topk_efficiency * loss_topk_efficiency + lambda_topk_diversity * loss_topk_diversity

            # ---- Adaptive memory operations (650+ steps) ----
            if step >= 650 and len(mem.items) >= 10:
                # Compute operation probabilities from learned logits
                memop_probs = torch.softmax(memop_logits, dim=0)  # [synthesis, split, interpolate, crossover]

                # Decide which operation to perform (sample from distribution)
                # Only perform operation every 20 steps to avoid overhead
                if step % 20 == 0:
                    # If accuracy improved, let Luria's will choose the operation
                    if 'acc_improved_flag' in locals() and acc_improved_flag:
                        try:
                            if intent_toggle or _ib >= 0.66:
                                op_choice = 0  # synthesis (augment/add)
                            elif _ib >= 0.33:
                                op_choice = 2  # interpolate
                            elif _ib >= -0.33:
                                op_choice = 3  # crossover (augment)
                            else:
                                op_choice = 1  # split (modify)
                        except Exception:
                            op_choice = torch.multinomial(memop_probs, num_samples=1).item()
                    else:
                        op_choice = torch.multinomial(memop_probs, num_samples=1).item()

                    # Select random memory indices
                    n_items = len(mem.items)
                    new_item = None

                    if op_choice == 0:  # Synthesis
                        # Synthesize 2-3 random memories
                        n_synth = min(3, n_items)
                        indices = random.sample(range(n_items), n_synth)
                        new_item = mem.synthesize_memories(indices)

                    elif op_choice == 1:  # Split
                        # Split a random memory
                        idx = random.randint(0, n_items - 1)
                        item1, item2 = mem.split_memory(idx, noise_scale=0.1)
                        if item1 is not None and item2 is not None:
                            # Add both split items
                            mem.add(item1.key, item1.val, step)
                            mem.add(item2.key, item2.val, step)

                    elif op_choice == 2:  # Interpolate
                        # Interpolate between two random memories
                        if n_items >= 2:
                            idx1, idx2 = random.sample(range(n_items), 2)
                            alpha = random.uniform(0.3, 0.7)  # Avoid extremes
                            new_item = mem.interpolate_memories(idx1, idx2, alpha=alpha)

                    elif op_choice == 3:  # Crossover
                        # Crossover two random memories
                        if n_items >= 2:
                            idx1, idx2 = random.sample(range(n_items), 2)
                            child1, child2 = mem.crossover_memories(idx1, idx2, crossover_point=0.5)
                            if child1 is not None and child2 is not None:
                                # Add both children
                                mem.add(child1.key, child1.val, step)
                                mem.add(child2.key, child2.val, step)

                    # Add single new item if applicable
                    if new_item is not None and op_choice != 1 and op_choice != 3:
                        mem.add(new_item.key, new_item.val, step)

                    # Track operation (will be used to compute accuracy-based loss)
                    memop_history.append({
                        "step": step,
                        "op": op_choice,
                        "accuracy": rule_acc  # Use current rule accuracy as feedback
                    })
                    if len(memop_history) > 100:
                        memop_history.pop(0)

                # Calculate accuracy-based loss (every 10 steps)
                if len(memop_history) >= 20 and step % 10 == 0:
                    # Track recent accuracy
                    recent_acc = [item["accuracy"] for item in memop_history[-50:]]
                    avg_accuracy = sum(recent_acc) / len(recent_acc)
                    accuracy_window.append(avg_accuracy)
                    if len(accuracy_window) > 100:
                        accuracy_window.pop(0)

                    # Loss: penalize deviation from target accuracy (0.85)
                    accuracy_error = abs(avg_accuracy - target_accuracy)
                    loss_accuracy = torch.tensor(accuracy_error, device=device, dtype=torch.float32)

                    # Operation diversity loss: penalize extreme probability distributions
                    # Encourage balanced exploration of all 4 operations
                    prob_entropy = -(memop_probs * torch.log(memop_probs + 1e-9)).sum()
                    max_entropy = torch.log(torch.tensor(4.0, device=device))  # log(4) for 4 operations
                    diversity_bonus = prob_entropy / max_entropy  # [0, 1], higher is more diverse

                    # Encourage diversity by penalizing low entropy
                    loss_memop_diversity = (1.0 - diversity_bonus) * 0.5

                    # Operation efficiency loss: penalize if accuracy is decreasing
                    if len(accuracy_window) >= 10:
                        recent_trend = accuracy_window[-1] - accuracy_window[-10]
                        if recent_trend < 0:
                            # Accuracy is decreasing, penalize current distribution
                            loss_efficiency_memop = torch.tensor(abs(recent_trend), device=device, dtype=torch.float32)
                        else:
                            loss_efficiency_memop = torch.tensor(0.0, device=device, dtype=torch.float32)
                    else:
                        loss_efficiency_memop = torch.tensor(0.0, device=device, dtype=torch.float32)

                    # Combine memory operation losses
                    lambda_accuracy = 0.20  # Strong signal for target accuracy
                    lambda_memop_diversity = 0.10
                    lambda_efficiency_memop = 0.08

                    total = total + lambda_accuracy * loss_accuracy + lambda_memop_diversity * loss_memop_diversity + lambda_efficiency_memop * loss_efficiency_memop

            # ---- Language-agnostic semantic & cross-language alignment (KO<->EN) ----
            Baux = args.aux_baux
            if Baux > 0:
                Xko, _ = make_identity_batch(tok, args.identity, batch_lang="ko", n=Baux)
                Xen, _ = make_identity_batch(tok, args.identity, batch_lang="en", n=Baux)
                Xko, Xen = Xko.to(device), Xen.to(device)

                E_ko = model.avg_embed(Xko); E_en = model.avg_embed(Xen)
                C_ko = concept_embed(Xko); C_en = concept_embed(Xen)

                S_ko = proj_sem(E_ko, _W_enc2sem); S_en = proj_sem(E_en, _W_enc2sem)
                K_ko = proj_sem(C_ko, _W_con2sem); K_en = proj_sem(C_en, _W_con2sem)

                loss_sem  = (info_nce(S_ko, K_ko) + info_nce(S_en, K_en)) / 2.0
                loss_xfer = (1 - torch.nn.functional.cosine_similarity(S_ko, S_en, dim=-1)).mean()

                total = total + args.lam_sem * loss_sem + args.lam_transfer * loss_xfer

        # Abstain gating: rule-based (0-5k) → autonomous learning (5k+)
        abstain_ratio = 0.0

        # (state_vec and related variables now computed earlier, before autonomous abstain block)

        intent_bias, intent_toggle = model.identity_intent_control()
        intent_force = None
        if intent_toggle <= -0.5:
            intent_force = False
        elif intent_toggle >= 0.5:
            intent_force = True

        if not use_autonomous_abstain:
            # Rule-based abstain (0-5k steps)
            # TUNED (from inner-step-tune.txt): conf_min -0.02, ent_max +0.2
            if step < 1000:
                # Warmup: disable abstain
                conf_min_adaptive = 0.58
                ent_max_adaptive = 2.8
                abstain_enabled = False
            elif step < 3000:
                # Ramp down: 1000→3000 steps
                progress = (step - 1000) / 2000.0
                conf_min_adaptive = 0.58 + progress * (0.61 - 0.58)  # Target: 0.61 (was 0.63)
                ent_max_adaptive = 2.8 - progress * (2.8 - 2.5)      # Target: 2.5 (was 2.3)
                abstain_enabled = True
            else:
                # Post-ramp: use relaxed target values
                conf_min_adaptive = 0.61  # Relaxed from 0.63
                ent_max_adaptive = 2.5    # Relaxed from 2.3
                abstain_enabled = True

            # Gray-zone buffer (FORCED ON from tune.txt):
            # 0.90 ≤ verifier < 0.95 → abstain FORBIDDEN
            if in_gray_zone:
                # Gray zone: abstain is FORBIDDEN
                should_abstain = False
                abstain_hysteresis_state = False  # Reset hysteresis
            elif abstain_hysteresis_state:
                # Currently abstaining: check exit conditions
                if conf_mean_current > 0.68 and ent_mean_current < 2.2:
                    abstain_hysteresis_state = False
                    should_abstain = False
                else:
                    should_abstain = True
            else:
                # Not abstaining: check entry conditions (RELAXED)
                if conf_mean_current < 0.60 or ent_mean_current > 2.6:
                    abstain_hysteresis_state = True
                    should_abstain = True
                else:
                    should_abstain = False

            # Accuracy override: if batch accuracy is high, do not abstain
            try:
                acc_override_min = float(_meta_cfg.get("acc_override_min", 0.75))
            except Exception:
                acc_override_min = 0.75
            if acc_mean_current >= acc_override_min:
                should_abstain = False
                abstain_hysteresis_state = False

            if intent_force is not None:
                should_abstain = intent_force
                abstain_hysteresis_state = intent_force

            # Respect warmup disable for metric as well
            if not abstain_enabled:
                should_abstain = False
            abstain_ratio = 1.0 if should_abstain else 0.0

        else:
            # Autonomous abstain learning (5k+ steps)
            # Rejector score: r = sigmoid(w @ state + b)
            with torch.no_grad():
                rejector_score = torch.sigmoid(rejector_w @ state_vec + rejector_b).item()

            # Safety guards (from inner-step-tune.txt)
            # 1. Gray-zone protection: abstain FORBIDDEN in gray zone
            if in_gray_zone:
                should_abstain = False
                rejector_score = 0.0  # Force accept
            else:
                # 2. Length bonus: seq_len > 64 → raise threshold (harder to abstain)
                tau_r_adjusted = float(tau_r.item() + tau_r_pi)
                if seq_len > 64:
                    tau_r_adjusted += 0.02
                tau_r_adjusted += intent_bias
                # 3. critic-lite adjustment based on self-eval proxy
                try:
                    selfeval_pass = (conf_mean_current >= _thr.conf_min and ent_mean_current <= _thr.ent_max)
                    # consistency proxy: confidence
                    consistency = max(0.0, min(1.0, conf_mean_current))
                    # spec_coverage proxy: inverse entropy (scaled)
                    spec_cov = max(0.0, min(1.0, 1.0 - ent_mean_current / 3.0))
                    # anti-copy proxy: 1 - trigram overlap between input and target
                    def _trigrams(s: str):
                        return set([s[i:i+3] for i in range(max(0, len(s)-2))]) if s else set()
                    try:
                        x_text = tok.decode(Xq[0].tolist()) if Xq.size(0) > 0 else ""
                        y_text = tok.decode(Yq[0].tolist()) if Yq.size(0) > 0 else ""
                    except Exception:
                        x_text, y_text = "", ""
                    g_in = _trigrams(x_text)
                    g_out = _trigrams(y_text)
                    overlap = len(g_in & g_out) / max(1, len(g_out)) if g_out else 0.0
                    anti_copy = max(0.0, 1.0 - overlap)
                    critic_R = 1.0 * consistency + 0.5 * spec_cov + 0.25 * anti_copy
                    # Apply small bias to threshold
                    eta = 0.05
                    tau_r_adjusted = tau_r_adjusted + (eta * critic_R if selfeval_pass else -eta * critic_R)
                except Exception:
                    critic_R = 0.0

                # 3. Hysteresis (separate entry/exit) with autonomous control
                if abstain_hysteresis_state:
                    # Exit: conf > 0.66 and ent < 2.2
                    if conf_mean_current > 0.66 and ent_mean_current < 2.2:
                        abstain_hysteresis_state = False
                        should_abstain = False
                    else:
                        should_abstain = True
                else:
                    # Entry: rely on learned rejector vs threshold (no forced rule-based entry)
                    should_abstain = (rejector_score > tau_r_adjusted)

            # Accuracy override: if batch accuracy is high, do not abstain
            try:
                acc_override_min = float(_meta_cfg.get("acc_override_min", 0.75))
            except Exception:
                acc_override_min = 0.75
            if acc_mean_current >= acc_override_min:
                should_abstain = False
                abstain_hysteresis_state = False

            if intent_force is not None:
                should_abstain = intent_force
                abstain_hysteresis_state = intent_force

            abstain_ratio = 1.0 if should_abstain else 0.0
            abstain_window_100.append(abstain_ratio)
            if len(abstain_window_100) > 100:
                abstain_window_100.pop(0)

            # 4. Runaway protection: if abstain rate > 0.8 in last 100 steps
            if len(abstain_window_100) >= 100:
                recent_abstain_rate = sum(abstain_window_100) / len(abstain_window_100)
                if recent_abstain_rate > 0.8:
                    # Emergency: increase threshold slightly and clamp to sane bounds
                    with torch.no_grad():
                        tau_r.add_(0.02)
                        tau_r.clamp_(0.50, 0.90)

            abstain_enabled = True

        # Coverage-aware LR adjustment (lacomi.txt + rikomi.txt prescription)
        # coverage < 0.4: maintain LR (don't decay too fast during low coverage)
        # coverage > 0.6: apply additional cooling
        # NOTE: This must be done AFTER abstain_ratio is calculated
        # rikomi.txt fix: Use 100-step EMA to prevent LR jitter from single-batch noise
        coverage_raw = 1.0 - abstain_ratio
        alpha_coverage = 2.0 / (100.0 + 1.0)  # ~0.02 for 100-step EMA
        coverage_ema = alpha_coverage * coverage_raw + (1.0 - alpha_coverage) * coverage_ema

        # UZR-Gaeseon-1: Acceptance PI controller (200-step window)
        # r = recent accept rate, t = scheduled target; update tau_r bias via PI
        accept_window_200.append(coverage_raw)
        if len(accept_window_200) > 200:
            accept_window_200.pop(0)
        r_accept_200 = sum(accept_window_200) / max(1, len(accept_window_200))

        def target_accept_rate(step_: int) -> float:
            # 3k:0.15 → 10k:0.35 → 15k:0.45 (piecewise linear), clamp afterwards
            if step_ <= 3000:
                return 0.15
            elif step_ <= 10000:
                # slope (0.35-0.15) over 7000 steps
                return 0.15 + (0.20 * (step_ - 3000) / 7000.0)
            elif step_ <= 15000:
                # slope (0.45-0.35) over 5000 steps
                return 0.35 + (0.10 * (step_ - 10000) / 5000.0)
            else:
                return 0.45

        acc_t = target_accept_rate(step)
        e_pi = r_accept_200 - acc_t
        acc_error_int += e_pi
        # Update tau_r bias (do not modify the learnable param directly)
        tau_r_pi += (-kp_pi * e_pi - ki_pi * acc_error_int)
        # Clamp bias to reasonable range to avoid extreme effects
        tau_r_pi = max(-0.20, min(0.20, tau_r_pi))

        lr_mult = lr_schedule(step)
        # feels-goat: apply LR scaling within conserve window
        if conserve_on:
            lr_mult = lr_mult * 0.7

        # Apply coverage-based adjustment using smoothed coverage
        if coverage_ema < 0.4:
            # Low coverage: prevent excessive decay ("겁 많은 학습자" 방지)
            # Maintain at least 80% of scheduled LR
            lr_mult = max(lr_mult, 0.8)
        elif coverage_ema > 0.6:
            # High coverage: apply additional cooling (98% per check)
            # Only apply after warmup period
            if step > args.warmup_steps:
                lr_mult = lr_mult * 0.98

        # Update learning rate with coverage-aware adjustment
        cur_lr = args.lr * lr_mult
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # loss_val already computed earlier (before autonomous abstain block)
        # Track EMA_raw (loss only)
        ema_raw_tracker = loss_val if ema_raw_tracker is None else (ema_raw_tracker * 0.95 + loss_val * 0.05)
        # Track EMA_raw99 (beta=0.99) for diagnostics
        ema_raw99_tracker = loss_val if ema_raw99_tracker is None else (ema_raw99_tracker * 0.99 + loss_val * 0.01)
        # Update mean of last 200 raw losses
        loss_window_200.append(loss_val)
        if len(loss_window_200) > 200:
            loss_window_200.pop(0)
        mean_raw_200 = (sum(loss_window_200) / len(loss_window_200)) if loss_window_200 else loss_val
        # Abstain penalty lambda: 1.0 → 2.0 linearly from 3k to 10k (then hold)
        if step <= 3000:
            lambda_abs = 1.0
        elif step <= 10000:
            lambda_abs = 1.0 + (step - 3000) / 7000.0
        else:
            lambda_abs = 2.0
        # EMA_all = EMA_raw + λ_abs * abstain_ratio
        ema = ema_raw_tracker + float(lambda_abs) * float(abstain_ratio)

        # Track csv min EMA integrity
        try:
            if ema is not None and ema < ema_min_csv:
                ema_min_csv = float(ema)
                ema_min_step_csv = int(step + 1)
        except Exception:
            pass

        # Perplexity from loss
        try:
            perplexity = float(math.exp(min(loss_val, 20.0)))  # cap for numerical stability
        except Exception:
            perplexity = float('nan')

        try:
            conf_mean = float(conf_vec.mean().item())
            ent_mean = float(ent_vec.mean().item())
            brier_val = float(brier.item())
        except Exception:
            conf_mean = float('nan'); ent_mean = float('nan'); brier_val = float('nan')

        # Calculate rule_acc (accuracy on query set)
        try:
            with torch.no_grad():
                pred_tokens = torch.argmax(logits_q, dim=-1)
                # Ignore padding tokens (assume PAD=0)
                mask = (Yq != tok.PAD)
                correct = (pred_tokens == Yq) & mask
                rule_acc = float(correct.sum().item() / max(1, mask.sum().item()))
        except Exception:
            rule_acc = float('nan')

        # Calculate top-5 accuracy
        try:
            with torch.no_grad():
                top5_preds = torch.topk(logits_q, k=min(5, logits_q.size(-1)), dim=-1).indices
                mask = (Yq != tok.PAD)
                # Check if true token is in top-5
                correct_top5 = (top5_preds == Yq.unsqueeze(-1)).any(dim=-1) & mask
                top5_acc = float(correct_top5.sum().item() / max(1, mask.sum().item()))
        except Exception:
            top5_acc = float('nan')

        # Calculate average top-1 probability
        try:
            with torch.no_grad():
                probs = torch.softmax(logits_q, dim=-1)
                top1_probs = probs.max(dim=-1).values
                mask = (Yq != tok.PAD)
                avg_top1_prob = float((top1_probs * mask).sum().item() / max(1, mask.sum().item()))
        except Exception:
            avg_top1_prob = float('nan')

        # --- Auto-tuning per manual: adjust abstain thresholds and write budget ---
        try:
            if ((step + 1) % 200) == 0:
                # If accuracy is low, make abstain stricter; else relax
                if not math.isnan(rule_acc) and rule_acc < 0.85:
                    _thr.conf_min = min(0.80, float(_thr.conf_min) + 0.02)
                    _thr.ent_max  = max(1.8,  float(_thr.ent_max)  - 0.05)
                elif not math.isnan(rule_acc):
                    _thr.conf_min = max(0.55, float(_thr.conf_min) - 0.01)
                    _thr.ent_max  = min(2.6,  float(_thr.ent_max)  + 0.05)
                # Memory write budget nudging based on recent write rate
                try:
                    stats = mem.get_memory_stats(step)
                    rate = int(stats.get("write_rate", {}).get("rate_per_100", 0))
                    if rate < 6:
                        mem.set_policy_thresholds(write_per_100=rate + 2)
                except Exception:
                    pass
        except Exception:
            pass

        # Calculate cloze_transition (token-level accuracy for transfer learning)
        try:
            with torch.no_grad():
                # Use identity batch for cloze transition metric
                pred_i = torch.argmax(logits_i, dim=-1)
                mask_i = (Yi != tok.PAD)
                correct_i = (pred_i == Yi) & mask_i
                cloze_transition = float(correct_i.sum().item() / max(1, mask_i.sum().item()))
        except Exception:
            cloze_transition = float('nan')

        # Reflection memo (manual): summarize top failures + metrics periodically
        try:
            if ((step + 1) % 200) == 0:
                _acc = float(rule_acc) if not isinstance(rule_acc, float) or not math.isnan(rule_acc) else float('nan')
                _ema = float(ema) if (ema is not None) else float('nan')
                reflect.write_epoch(int(step + 1), replay_buffer.most_common(3), {"acc": _acc, "ema": _ema})
        except Exception:
            pass

        # Calculate gate_pass_60_95 (percentage of samples with conf in 60-95% range)
        try:
            conf_percent = conf_vec * 100.0
            gate_pass_60_95 = float(((conf_percent >= 60.0) & (conf_percent <= 95.0)).float().mean().item())
        except Exception:
            gate_pass_60_95 = float('nan')

        # Quality gate check (rikomi.txt fix: moved AFTER metric calculation)
        # Check if current update passes quality standards using THIS step's metrics
        gate_passed = True
        if step >= 1000 and prev_ema is not None and prev_rule_acc is not None and prev_perplexity is not None:
            # Compute THIS step's EMA (before updating prev_ema)
            ema_next = ema * 0.95 + loss_val * 0.05 if ema is not None else loss_val

            # Compute deltas using current step values
            d_ema = ema_next - prev_ema
            d_acc = rule_acc - prev_rule_acc
            ppl_rel = perplexity / max(1e-6, prev_perplexity)

            # Set thresholds based on step range
            if step < 2000:
                # 1k-2k: relaxed thresholds
                threshold_d_ema = -0.03
                threshold_d_acc = -0.02
                threshold_ppl_rel = 1.05
            else:
                # 2k+: stricter thresholds
                threshold_d_ema = -0.02
                threshold_d_acc = -0.01
                threshold_ppl_rel = 1.03

            # Check gate conditions
            if d_ema < threshold_d_ema or d_acc < threshold_d_acc or ppl_rel > threshold_ppl_rel:
                gate_passed = False

        # rikomi.txt fix: Update skip/step block moved here (AFTER gate_passed is computed)
        # Now we can safely use gate_passed since it's been calculated above
        did_update = True
        train_allowed = True
        train_block_reason = None
        if step >= args.warmup_steps:
            if intent_force is True:
                train_allowed = False
                train_block_reason = "luria_intent_block"
        else:
            train_allowed = True  # warmup: always allow

        if not train_allowed:
            did_update = False
        else:
            opt.zero_grad()
            if scaler.is_enabled():
                scaler.scale(total).backward()
                # Gradient clipping (after 2k steps, more aggressive)
                if step >= 2000:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                else:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(opt)
                scaler.update()
                try:
                    model.update_self_referential()
                except Exception:
                    pass
            else:
                total.backward()
                # Gradient clipping (after 2k steps, more aggressive)
                if step >= 2000:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()
                try:
                    model.update_self_referential()
                except Exception:
                    pass

        # Calculate z norms for each brain
        try:
            z_lang_norm = float(torch.norm(z_lang_star).item())
            z_logic_norm = float(torch.norm(z_logic_star).item())
            z_bridge_norm = float(torch.norm(z_bridge_star).item())
        except Exception:
            z_lang_norm = float('nan')
            z_logic_norm = float('nan')
            z_bridge_norm = float('nan')

        # Calculate high-step probability
        high_step_count = sum(1 for s in inflation_window if s >= 12)
        high_step_prob = high_step_count / max(1, len(inflation_window))

        # Dynamic s_max recovery: recover to 18, then up to 25
        if len(inflation_window) >= 256:  # Need enough samples
            if high_step_prob <= 0.01 and s_max_adaptive < 18:
                s_max_adaptive = 18
            elif high_step_prob <= 0.005 and s_max_adaptive < 25:
                s_max_adaptive = 25

        # Determine accuracy improvement (before updating prev_* trackers)
        try:
            acc_improved_flag = (prev_rule_acc is not None) and (not math.isnan(rule_acc)) and (rule_acc > prev_rule_acc)
        except Exception:
            acc_improved_flag = False

        # Update previous values for quality gate and state tracking
        prev_ema = ema
        prev_rule_acc = rule_acc
        prev_perplexity = perplexity
        prev_loss_val = loss_val
        prev_conf_mean = conf_mean_current

        # Add failed samples to replay buffer with length-balanced sampling
        # Consider failure if rule_acc < 0.5
        if rule_acc < 0.5:
            for qx, qy in Q_pairs:
                # Determine length bin
                seq_len = len(qx) + len(qy)
                if seq_len <= 16:
                    length_bin = "short"
                elif seq_len <= 32:
                    length_bin = "medium"
                else:
                    length_bin = "long"

                # Add to replay buffer with length tag
                replay_buffer.add(qx, qy, tag=length_bin)
                length_bin_counts[length_bin] += 1

        # Periodically decay replay counters to prioritize recent failures
        if (step + 1) % 50 == 0:
            replay_buffer.decay()

        # Compute rejector metrics for display
        try:
            if use_autonomous_abstain:
                with torch.no_grad():
                    rejector_display = torch.sigmoid(rejector_w @ state_vec + rejector_b).item()
                    tau_r_display = tau_r.item()
            else:
                rejector_display = 0.0
                tau_r_display = 0.5
        except Exception:
            rejector_display = 0.0
            tau_r_display = 0.5

        # 코드북 통계 (주기적으로만 계산)
        if (step + 1) % 100 == 0:
            cb_stats = codebook.stats()
            cb_t_ent = cb_stats.t_entropy
            cb_v_ent = cb_stats.v_entropy
            # feels-first: entropy/dead-ratio driven adjustments
            try:
                import math as _math
                t_uniform = float(_math.log(getattr(codebook.t, 'Kt', 128)))
                v_uniform = float(_math.log(getattr(codebook.v, 'K', 128)))
                # If either codebook entropy falls below 75% of uniform, raise tau_r slightly
                if min(cb_t_ent, cb_v_ent) < 0.75 * min(t_uniform, v_uniform):
                    with torch.no_grad():
                        tau_r.add_(0.01)
                        tau_r.clamp_(0.50, 0.90)
                # If dead ratio too high, temporarily make surprise gate more permissive (p65->p60 analog)
                t_dead = getattr(cb_stats, 't_dead_ratio', 0.0)
                v_dead = getattr(cb_stats, 'v_dead_ratio', 0.0)
                if max(t_dead, v_dead) > 0.35:
                    surprise_shift_counter = 400
            except Exception:
                pass
        else:
            cb_t_ent = getattr(pbar, '_cb_t_ent', 0.0)
            cb_v_ent = getattr(pbar, '_cb_v_ent', 0.0)
        pbar._cb_t_ent = cb_t_ent
        pbar._cb_v_ent = cb_v_ent

        # Apply temporary surprise gate relaxation if requested and not in conserve mode
        try:
            if surprise_shift_counter > 0 and not conserve_on:
                if _orig_write_thr_base is None:
                    _orig_write_thr_base = getattr(mem, "write_threshold_base", 0.40)
                mem.write_threshold_base = float(_orig_write_thr_base - 0.05)
                surprise_gate_delta = -0.05
                surprise_shift_counter -= 1
            elif not conserve_on:
                if _orig_write_thr_base is None:
                    _orig_write_thr_base = getattr(mem, "write_threshold_base", 0.40)
                mem.write_threshold_base = float(_orig_write_thr_base)
                if surprise_shift_counter <= 0:
                    surprise_gate_delta = 0.0
        except Exception:
            pass

        pbar.set_postfix({
            "loss": f"{loss_val:.3f}",
            "ema": f"{ema:.3f}",
            "ema_r": f"{(ema_raw_tracker if ema_raw_tracker is not None else 0.0):.3f}",
            "ppl": f"{perplexity:.1f}",
            "conf": f"{conf_mean:.2f}",
            "s": chosen_steps,
            "k": top_k_dynamic,
            "s_avg": f"{avg_steps_current:.1f}",
            "s_max": s_max_adaptive,
            "infl": f"{inflation_rate:.2f}",
            "p12+": f"{high_step_prob:.2%}",
            "acc": f"{rule_acc:.3f}",
            "gate": f"{gate_pass_60_95:.2f}",
            "abstain": f"{abstain_ratio:.1f}",
            "fw": "Y" if force_write_enabled else "N",
            "rej_r": f"{rejector_display:.2f}" if use_autonomous_abstain else "-",
            "τ_r": f"{tau_r_display:.2f}" if use_autonomous_abstain else "-",
            "ib": f"{float(intent_bias):.2f}",
            "it": f"{float(intent_toggle):.2f}",
            "τ_pi": f"{tau_r_pi:.2f}",
            "nm_thr": f"{near_merge_thr_current:.3f}" if step >= args.autonomous_step else "-",
            "diff": difficulty_level,
            "cb_T": f"{cb_t_ent:.1f}" if (step + 1) % 100 == 0 else "-",
            "cb_V": f"{cb_v_ent:.1f}" if (step + 1) % 100 == 0 else "-",
            "task": desc[:10],
        })

        # Memory tracking (using bridge as primary z) with continuous gate and composite scoring
        with torch.no_grad():
            # Update memory state with query set and adapted bridge z
            model.update_memory_state(Xq, z_bridge_star)

            enc_avg_q = model.avg_embed(Xq).mean(dim=0)

            # Continuous gate: use surprise_eff (entropy + codebook rarity) * z_norm
            try:
                query_text = tok.decode(Xq[0].tolist()) if Xq.size(0) > 0 else ""
            except Exception:
                query_text = ""
            try:
                zc_for_gate = codebook.encode(text=query_text, h=enc_avg_q, return_parts=True)
            except Exception:
                zc_for_gate = {"T": [], "V": []}

            surprise = ent_mean  # base surprise from entropy
            def _unseen_ratio_T(tokens):
                try:
                    if not tokens:
                        return 0.0
                    cnt = 0; total = 0
                    for t in tokens:
                        if not (isinstance(t, str) and len(t) >= 4 and t[0] == 'K'):
                            continue
                        g = max(0, ord(t[1]) - ord('A'))
                        idx = int(t[-2:]) if t[-2:].isdigit() else 0
                        uc = codebook.t.usage_counts[g][idx].item() if (0 <= g < len(codebook.t.usage_counts)) else 0
                        total += 1
                        if uc <= 0:
                            cnt += 1
                    return (cnt / total) if total > 0 else 0.0
                except Exception:
                    return 0.0
            def _unseen_ratio_V(tokens):
                try:
                    if not tokens:
                        return 0.0
                    cnt = 0; total = 0
                    for t in tokens:
                        if not (isinstance(t, str) and len(t) >= 3):
                            continue
                        g = max(0, ord(t[0]) - ord('A'))
                        idx = int(t[1:]) if t[1:].isdigit() else 0
                        uc = codebook.v.usage_counts[g, idx].item() if (0 <= g < codebook.v.usage_counts.size(0) and 0 <= idx < codebook.v.usage_counts.size(1)) else 0
                        total += 1
                        if uc <= 0:
                            cnt += 1
                    return (cnt / total) if total > 0 else 0.0
                except Exception:
                    return 0.0
            try:
                unseen_t = _unseen_ratio_T(zc_for_gate.get("T", []))
                unseen_v = _unseen_ratio_V(zc_for_gate.get("V", []))
                surprise_cb = 0.5 * (unseen_t + unseen_v)
            except Exception:
                surprise_cb = 0.0
            surprise_eff = 0.7 * surprise + 0.3 * surprise_cb
            z_quality = z_bridge_norm
            composite_score = surprise_eff * z_quality

            # Sigmoid-based soft gate (converts score to [0,1] probability)
            # Scale composite_score to reasonable range for sigmoid
            gate_logit = (composite_score - 5.0) / 2.0  # center around 5.0, scale by 2.0
            gate_prob = torch.sigmoid(torch.tensor(gate_logit, device=device))

            # Top-p% threshold (adaptive based on memory size)
            # Start with top 50%, gradually increase selectivity as memory fills
            mem_fill_ratio = len(mem.items) / max(1, mem.max_items)
            target_percentile = 0.50 + 0.30 * mem_fill_ratio  # 50% -> 80% as memory fills

            # Sample from Bernoulli distribution with gate_prob
            should_commit = (gate_prob.item() > (1.0 - target_percentile))

            # If accuracy improved and Luria's will is positive, force commit
            try:
                if acc_improved_flag and (intent_toggle or _ib >= 0.0):
                    should_commit = True
            except Exception:
                pass

            if should_commit:
                # 문자 표상 만들기
                try:
                    repr_text = model.build_text_repr(Xq, tokenizer=tok)
                except Exception:
                    repr_text = None

                # 코드북 인코딩: 텍스트 + 임베딩 → ZC 토큰
                try:
                    # 텍스트 복원 (첫 번째 샘플)
                    query_text = tok.decode(Xq[0].tolist()) if Xq.size(0) > 0 else ""
                    # 코드북 인코딩
                    zc_tokens = zc_for_gate
                    zc_str = codebook.format_zc(zc_tokens)

                    # 온라인 업데이트 (shadow에 누적)
                    if step >= 500 and (step % 4 == 0):
                        codebook.accumulate_update(text=query_text, h=enc_avg_q)
                except Exception:
                    zc_tokens = {"T": [], "V": []}
                    zc_str = ""

                # For memory, we use bridge z as the primary representation
                key, val = make_sketch(enc_avg_q, z_bridge_star, meta={
                    "lang": int(lang_idx), "desc": desc, "type": task_type,
                    "conf": conf_mean, "ent": ent_mean, "brier": brier_val,
                    "composite_score": float(composite_score),
                    "gate_prob": float(gate_prob.item()),
                }, repr_text=repr_text)

                # ZC 토큰 추가
                if zc_str:
                    val["zc"] = zc_str
                    val["zc_tokens"] = zc_tokens
                if hasattr(mem, "add_with_policy"):
                    mem_meta = {
                        "lang": int(lang_idx),
                        "desc": desc,
                        "composite_score": float(composite_score),
                        "luria_intent_force": intent_force,
                    }
                    mem.add_with_policy(key, val, step, meta=mem_meta)
                else:
                    mem.add(key, val, step)

            # Periodic memory maintenance with LRU + quality-based cleanup
            # lacomi.txt: N-step mini rebalance (not weekly, but per-N-steps)
            # Running every 50 steps for more frequent memory refresh
            if (step + 1) % 50 == 0 and hasattr(mem, "rebalance"):
                mem.rebalance()

                # 주기적으로 codebook shadow → active 스왑 시도
                if codebook.maybe_commit():
                    cb_stats = codebook.stats()
                    print(f"\n[CB-SWAP] step={step+1} id={codebook.active_id} "
                          f"T_ent={cb_stats.t_entropy:.2f} V_ent={cb_stats.v_entropy:.2f} "
                          f"T_dead={cb_stats.t_dead_ratio:.2%} V_dead={cb_stats.v_dead_ratio:.2%}")

                # LRU + quality-based cleanup: remove bottom p% by quality score
                if len(mem.items) > mem.max_items * 0.9:  # Trigger when 90% full
                    cleanup_percentile = 0.10  # Remove bottom 10%

                    # Score each item: recency (LRU) + quality (composite_score)
                    item_scores = []
                    for idx, item in enumerate(mem.items):
                        # LRU score: normalize by step range
                        recency_score = (step - item.get('step', 0)) / max(1, step)

                        # Quality score from metadata
                        meta = item.get('meta', {})
                        quality = meta.get('composite_score', 0.0)

                        # Combined score: lower is better (for removal)
                        # Weight: 0.3 * recency (older = higher) + 0.7 * (1/quality)
                        combined = 0.3 * recency_score + 0.7 * (1.0 / (quality + 1e-6))
                        item_scores.append((idx, combined))

                    # Sort by score (highest = worst)
                    item_scores.sort(key=lambda x: x[1], reverse=True)

                    # Remove bottom p%
                    num_to_remove = int(len(mem.items) * cleanup_percentile)
                    indices_to_remove = [idx for idx, _ in item_scores[:num_to_remove]]

                    # Remove in reverse order to maintain indices
                    for idx in sorted(indices_to_remove, reverse=True):
                        if idx < len(mem.items):
                            mem.items.pop(idx)

        # Train memory predictor periodically
        if (step + 1) % 10 == 0 and len(mem.items) >= mem.min_train_items:
            mem_loss = mem.train_model(steps=5, batch_size=64, shuffle=True)
            if mem_loss is not None and mem.learner_loss is not None:
                pbar.set_postfix({
                    "loss": f"{loss_val:.3f}",
                    "ema": f"{ema:.3f}",
                    "ppl": f"{perplexity:.1f}",
                    "mem": f"{mem.learner_loss:.4f}",
                    "s": chosen_steps,
                    "s_max": s_max_adaptive,
                    "infl": f"{inflation_rate:.2f}",
                    "acc": f"{rule_acc:.3f}",
                    "abstain": f"{abstain_ratio:.1f}",
                    "task": desc[:10],
                })

        # Summary CSV logging
        if _sum_writer is not None:
            try:
                # Calculate composite score for logging
                composite_score_log = ent_mean * z_bridge_norm

                # Mode and AMP flags
                _mode = "train"
                _amp = "on" if (args.amp and device.type == "cuda") else "off"
                _eval_id = "-"

                _sum_writer.writerow({
                    "step": step + 1,
                    "loss": round(loss_val, 4),
                    "ema": round(ema, 4),
                    "ema_raw": round(ema_raw_tracker if ema_raw_tracker is not None else loss_val, 4),
                    "lambda_abs": round(lambda_abs, 4),
                    "ema_raw99": round(ema_raw99_tracker if ema_raw99_tracker is not None else loss_val, 4),
                    "ema_all99": round((ema_raw99_tracker if ema_raw99_tracker is not None else loss_val) + float(lambda_abs) * float(abstain_ratio), 4),
                    "mean_raw_200": round(mean_raw_200, 4),
                    "perplexity": round(perplexity, 4),
                    "id_loss": round(float(id_loss.item()), 4),
                    "lr": cur_lr,
                    "conf_mean": round(conf_mean, 4),
                    "ent_mean": round(ent_mean, 4),
                    "brier": round(brier_val, 4),
                    "abstain_ratio": round(abstain_ratio, 4),
                    "chosen_steps": chosen_steps,
                    "avg_steps": round(avg_steps_current, 4),
                    "s_max": s_max_adaptive,
                    "inflation_rate": round(inflation_rate, 4),
                    "high_step_prob": round(high_step_prob, 4),
                    "rule_acc": round(rule_acc, 4),
                    "top5_acc": round(top5_acc, 4),
                    "avg_top1_prob": round(avg_top1_prob, 4),
                    # Transition metrics
                    "z_cos": round(z_cos_metric, 4) if z_cos_metric == z_cos_metric else "-",
                    "dz_mse": round(dz_mse_metric, 6) if dz_mse_metric == dz_mse_metric else "-",
                    "cb_f1": round(cb_f1_metric, 4) if cb_f1_metric == cb_f1_metric else "-",
                    "align_cos": round(align_cos_metric, 4) if align_cos_metric == align_cos_metric else "-",
                    "jac_surr": round(jac_surr_metric, 6) if jac_surr_metric == jac_surr_metric else "-",
                    "dz_norm": round(dz_norm_mean_metric, 4) if dz_norm_mean_metric == dz_norm_mean_metric else "-",
                    "dz_norm_std": round(dz_norm_std_metric, 4) if dz_norm_std_metric == dz_norm_std_metric else "-",
                    "cb_pos": round(cb_pos_rate_metric, 4) if cb_pos_rate_metric == cb_pos_rate_metric else "-",
                    "cb_pred": round(cb_pred_rate_metric, 4) if cb_pred_rate_metric == cb_pred_rate_metric else "-",
                    "trans_loss": round(trans_loss_metric, 6) if trans_loss_metric == trans_loss_metric else "-",
                    "cloze_transition": round(cloze_transition, 4),
                    "gate_pass_60_95": round(gate_pass_60_95, 4),
                    "z_lang_norm": round(z_lang_norm, 4),
                    "z_logic_norm": round(z_logic_norm, 4),
                    "z_bridge_norm": round(z_bridge_norm, 4),
                    "mem_size": len(mem.items),
                    "rejector_score": round(rejector_display, 4),
                    "tau_r": round(tau_r_display, 4),
                    "tau_r_pi": round(tau_r_pi, 4),
                    "intent_bias": round(float(intent_bias), 4),
                    "intent_toggle": round(float(intent_toggle), 4),
                    "acc_r_200": round(r_accept_200, 4),
                    "acc_t": round(acc_t, 4),
                    "mode": _mode,
                    "amp": _amp,
                    "eval_set_id": _eval_id,
                    "coverage": round(1.0 - abstain_ratio, 4),
                    "use_autonomous": 1 if use_autonomous_abstain else 0,
                    "difficulty_level": difficulty_level,
                    "composite_score": round(composite_score_log, 4),
                    "force_write_enabled": 1 if force_write_enabled else 0,
                    "fw_ce": round(fw_ce, 4),
                    "fw_len_pen": round(fw_len_pen, 4),
                    # diagnostics / modes
                    "conserve_on": 1 if conserve_on else 0,
                    "conserve_reason": conserve_reason,
                    "lr_scale": round(lr_mult, 4),
                    "fw_prob": float(_orig_fw_prob if conserve_on and _orig_fw_prob is not None else 0.10),
                    "surprise_gate_delta": round(surprise_gate_delta, 4),
                    # EMA integrity
                    "ema_min_csv": round(ema_min_csv if ema_min_csv != float("inf") else ema, 4),
                    "ema_min_ckpt": round(best_loss, 4),
                    "ema_min_delta": round((best_loss - (ema_min_csv if ema_min_csv != float("inf") else ema)), 4),
                })
                # Flush to disk periodically for real-time monitoring
                if (step + 1) % 10 == 0:
                    _sum_file.flush()
            except Exception:
                pass

        # Luria logging system: Log shadow bank snapshot every 100 steps
        if (step + 1) % 100 == 0:
            try:
                # Update memory size in counters
                memory_counters["mem_size_curr"] = len(mem.items)

                # Log snapshot
                log_sb_snapshot(
                    step=step + 1,
                    stage=current_stage,
                    bank=mem.shadow_bank,
                    counters=memory_counters,
                    wm_curr=len(mem.items),  # Using memory size as working memory indicator
                    recall_topk=mem.topk
                )

                # Log memory stats every 100 steps
                stats = mem.get_memory_stats(step + 1)
                # Optionally log to console or file
                # print(f"[Memory Stats @ {step+1}] {stats}")

                # Reset tick counters after logging
                memory_counters["promote_tick"] = 0
                memory_counters["decay_tick"] = 0
                memory_counters["skip_tick"] = 0
                memory_counters["commit_tick"] = 0
                memory_counters["merge_tick"] = 0
                memory_counters["near_merge_tick"] = 0
            except Exception as e:
                # Don't break training if logging fails
                pass

        # Stage transition detection and logging
        if step == args.warmup_steps and current_stage == "warmup":
            current_stage = "training"
            log_stage_event(step, current_stage, "warmup_complete", {"warmup_steps": args.warmup_steps})
        elif step == 350 and current_stage == "training":
            current_stage = "autonomous"
            log_stage_event(step, current_stage, "autonomous_abstain_enabled", {"step": step})

        if (step + 1) % max(1, args.save_every) == 0 or step + 1 == args.steps:
            payload = {
                "model": model.state_dict(),
                "args": vars(args),
                "opt": opt.state_dict(),
                "step": step + 1,
                "best_loss": best_loss,
                "memory": mem.state_dict(),  # Save memory state
                "rejector_w": rejector_w.data,
                "rejector_b": rejector_b.data,
                "tau_r": tau_r.data,
                "near_merge_thr_param": near_merge_thr_param.data,
                "top_k_param": top_k_param.data,
                "memop_logits": memop_logits.data,
                # EMA integrity (csv minima) for later cross-check
                "ema_min_csv": float(ema_min_csv),
                "ema_min_step_csv": int(ema_min_step_csv),
            }
            torch.save(payload, last_path)
            if ema < best_loss:
                best_loss = ema
                payload["best_loss"] = best_loss
                torch.save(payload, best_path)

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "memory": mem.state_dict(),  # Save memory state in final checkpoint
        "rejector_w": rejector_w.data,
        "rejector_b": rejector_b.data,
        "tau_r": tau_r.data,
        "near_merge_thr_param": near_merge_thr_param.data,
        "top_k_param": top_k_param.data,
        "memop_logits": memop_logits.data,
        "ema_min_csv": float(ema_min_csv),
        "ema_min_step_csv": int(ema_min_step_csv),
    }, args.save)
    print(f"Saved to {args.save}; last={last_path}; best={best_path} (best_ema={best_loss:.3f})")

    # Luria logging system: Write session summary
    try:
        from utils.logger_luria import append_jsonl
        import json

        # Calculate rolling mean for shadow size if stats file exists
        shadow_avg_size = len(mem.shadow_bank) if hasattr(mem, 'shadow_bank') else 0

        # Get final promote ratio
        promote_ratio_final = 0.0
        if memory_counters["promote_total"] + memory_counters["skip_total"] > 0:
            promote_ratio_final = memory_counters["promote_total"] / (
                memory_counters["promote_total"] + memory_counters["skip_total"]
            )

        summary = {
            "session_started": datetime.utcnow().isoformat(),
            "session_ended": datetime.utcnow().isoformat(),
            "steps": args.steps,
            "ema_min": best_loss,
            "ema_min_step": -1,  # Not tracked in current implementation
            "ppl_end": float(math.exp(min(ema, 20.0))) if ema is not None else 0.0,
            "shadow": {
                "avg_size": shadow_avg_size,
                "promote_ratio_final": promote_ratio_final,
                "decay_events_total": memory_counters["decay_total"],
                "skip_total": memory_counters["skip_total"]
            },
            "dedup": {
                "dup_rate_final": 0.0  # Not tracked in current implementation
            },
            "memory": {
                "final_size": len(mem.items),
                "capacity": mem.max_items,
                "total_writes": len(mem._write_steps) if hasattr(mem, '_write_steps') else 0
            }
        }
        append_jsonl("logus-luria/session_summary.json", summary)
        print(f"[Luria] Session summary written to logus-luria/session_summary.json")
    except Exception as e:
        print(f"[Luria] Warning: Failed to write session summary: {e}")

    # Close CSV file
    if _sum_writer is not None:
        try:
            _sum_file.close()
            print(f"Training log saved to {summary_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
