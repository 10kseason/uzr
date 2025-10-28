import argparse, random, os, math, time, unicodedata
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# NOTE: Keep imports for backwards compatibility, but we won't use ByteTokenizer now.
from .model import UZRModel, ByteTokenizer, seq_ce_loss, soft_threshold
from .memory import CompressedMemory, make_sketch
from .codebook import CodebookManager
from uzr.tasks import sample_task
from uzr.meta_core import load_meta_config, AbstainThresholds, maybe_abstain, ReplayBuffer
from .hooks import abstain_cap_schedule, should_force_write, apply_force_write

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


def inner_adapt_z(model, Xc, Yc, z, lam=1e-3, eta=0.5, steps=3):
    # z-only updates (legacy single-z version)
    z = z.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(Xc, z)
        loss = seq_ce_loss(logits, Yc) + lam * torch.mean(torch.abs(z))
        g = torch.autograd.grad(loss, z, retain_graph=False)[0]
        z = (z - eta * g).detach().requires_grad_(True)
    # proximal soft-threshold (L1)
    with torch.no_grad():
        z = soft_threshold(z, lam * 0.5)
    return z.detach()


def inner_adapt_z_multi(model, Xc, Yc, z_rule, z_think, lang_id, lam_rule=1e-3, lam_think=1e-3, eta=0.5, steps=3):
    """Legacy 2-brain adaptation (rule + think)"""
    z_rule = z_rule.clone().detach().requires_grad_(True)
    z_think = z_think.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = model(Xc, {"rule": z_rule, "think": z_think, "lang_id": lang_id})
        loss = seq_ce_loss(logits, Yc)
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
    cap = min(15, s_max_cap)

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
                          lam_lang=1e-3, lam_logic=1e-3, lam_bridge=1e-3, eta=0.5, steps=3):
    """
    3brains adaptation: language artisan + logic artisan + bridge
    All three brains adapt simultaneously (NOT MoE)
    """
    z_slow_lang = z_slow_lang.clone().detach().requires_grad_(True)
    z_slow_logic = z_slow_logic.clone().detach().requires_grad_(True)
    z_bridge = z_bridge.clone().detach().requires_grad_(True)

    for _ in range(steps):
        logits = model(Xc, {"slow_lang": z_slow_lang, "slow_logic": z_slow_logic, "bridge": z_bridge})
        loss = seq_ce_loss(logits, Yc)
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
    ap.add_argument("--identity_self_dim", type=int, default=2, help="dimension for self-identity embedding (e.g., '루리아')")
    ap.add_argument("--num_langs", type=int, default=len(LANG2ID))
    ap.add_argument("--max_len", type=int, default=512)
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
    ap.add_argument("--warmup_steps", type=int, default=250)
    ap.add_argument("--cosine", action="store_true", help="Use cosine LR decay after warmup")
    ap.add_argument("--single_z", action="store_true", help="Share single z across context batch")
    ap.add_argument("--identity", default="루리아", help="identity string used for identity auxiliary loss")
    ap.add_argument("--id_weight", type=float, default=0.2, help="weight of identity auxiliary loss")

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
    ap.add_argument("--lam_sem", type=float, default=5e-4, help="weight for language-agnostic semantic InfoNCE loss")
    ap.add_argument("--lam_transfer", type=float, default=7e-4, help="weight for cross-language cosine alignment loss")
    ap.add_argument("--aux_baux", type=int, default=4, help="aux mini-batch size per language for semantic alignment")
    ap.add_argument("--sem_dim", type=int, default=12, help="shared semantic projection dim")

    args = ap.parse_args()
    # Propagate self-eval toggle into model construction via environment
    if args.self_eval is not None:
        os.environ["UZR_SELF_EVAL"] = "1" if args.self_eval == "on" else "0"

    if args.lam_rule is None:
        args.lam_rule = args.lam
    if args.lam_think is None:
        args.lam_think = args.lam

    set_seed(args.seed)
    device = torch.device(args.device)
    # Use the new minimal KO+EN tokenizer
    tok = KoEnTokenizer(max_len=args.max_len)

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
            "step","loss","ema","perplexity","id_loss","lr","conf_mean","ent_mean","brier","abstain_ratio",
            "chosen_steps","avg_steps","s_max","inflation_rate","high_step_prob","rule_acc","top5_acc","avg_top1_prob",
            "cloze_transition","gate_pass_60_95","z_lang_norm","z_logic_norm","z_bridge_norm","mem_size",
            "rejector_score","tau_r","coverage","use_autonomous","difficulty_level","composite_score",
            "abstain_cap","force_write_enabled","fw_ce","fw_len_pen"
        ])
        _sum_writer.writeheader()
    except Exception:
        _sum_writer = None

    # Create memory first with learning enabled
    mem = CompressedMemory(
        max_items=8192,
        device=device,
        enable_learning=True,
        learn_hidden=512,
        learn_depth=3,
        learn_rate=1e-3,
        min_train_items=32,
        warmup_steps=args.warmup_steps,
        entropy_floor=0.0,
        softmax_temp=0.8,
    )
    # Memory policy upgrade (lacomi.txt prescription)
    # - Lower similarity threshold by 0.02 for better diversity
    # - Raise near_merge_thr to 0.10 (merge less aggressively)
    # - Increase write budgets for more diverse memory
    mem.set_policy_thresholds(
        write_per_turn=2,
        write_per_100=14,           # Increased from 12 (more frequent writes)
        tail_write_per_100=8,       # Increased from 6
        entropy_floor=0.0,
        stage_mid_low=0.55,         # Lowered similarity threshold
        near_merge_thr=0.10,        # Raised to merge less frequently
        dup_skip_thr=0.90,          # Lowered by 0.02 from typical 0.92
    )
    # Enable entropy check only after bootstrapping
    setattr(mem, "entropy_check_start", 32)

    # Create model with memory reference and 3brains dimensions
    model = UZRModel(
        tok.vocab_size,
        d_model=args.d_model,
        z_dim=args.z_dim,
        max_len=args.max_len,
        z_think_dim=args.z_think_dim,
        z_lang_dim=args.z_lang_dim,
        num_langs=args.num_langs,
        identity_self_dim=args.identity_self_dim,
        memory=mem,
        z_slow_lang_dim=args.z_slow_lang_dim,
        z_slow_logic_dim=args.z_slow_logic_dim,
        z_bridge_dim=args.z_bridge_dim,
    ).to(device)

    # Initialize CodebookManager for text+vector representation
    codebook = CodebookManager.init(
        d=args.d_model,
        t_cfg=dict(Gt=6, Kt=128, dt=384, m=131072, ema_decay=0.995),
        v_cfg=dict(G=6, K=128, beta=0.25, ema_decay=0.995),
        seed=args.seed,
        device=device,
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

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

    # Rejector-Head parameters (5k+ autonomous abstain learning)
    state_dim = 8  # [conf, ent, verifier, seq_len, gray_flag, infl_512, loss_delta, conf_delta]
    rejector_w = torch.nn.Parameter(torch.randn(state_dim, device=device) * 0.01)
    rejector_b = torch.nn.Parameter(torch.zeros(1, device=device))
    tau_r = torch.nn.Parameter(torch.tensor(0.5, device=device))  # learnable threshold
    opt.add_param_group({'params': [rejector_w, rejector_b, tau_r], 'lr': args.lr * 0.5})

    # Autonomous abstain tracking
    abstain_window_100 = []  # 100-step window for runaway detection
    target_coverage = 0.8  # target coverage τ (accept rate)
    prev_loss_val = None
    prev_conf_mean = None

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
        start_step = data.get("step", 0)
        best_loss = data.get("best_loss", best_loss)
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

    ema = None
    z_bridge_history = []  # Track z_bridge for decay detection

    # Coverage EMA tracker (rikomi.txt: 100-step smoothing to prevent LR jitter)
    coverage_ema = 0.8  # Start optimistic (high coverage assumption)

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

    pbar = tqdm(range(start_step, args.steps), desc="3brains-meta-train")
    for step in pbar:
        # rikomi.txt fix: Initialize gate_passed early to prevent UnboundLocalError
        gate_passed = True  # Safe default: allow update unless gate check fails later

        # Determine if autonomous abstain is enabled (5k+ steps)
        use_autonomous_abstain = (step >= 5000)

        # Apply abstain cap schedule (3k→15k linear decay: 1.0→0.2)
        abstain_cap = abstain_cap_schedule(step)

        # Determine if force write is enabled for this batch
        force_write_enabled = should_force_write(step)
        fw_ce, fw_len_pen = 0.0, 0.0  # Initialize force write metrics

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

        C_pairs, Q_pairs, desc = sample_task(n_context=n_context, n_query=n_query, n_tokens=n_tokens)

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

        # Create task key for cooldown
        task_key = f"{lang_key}_{task_type}_{desc[:10]}"

        # Choose adaptive steps using bucket mapping
        chosen_steps = choose_adaptive_steps(
            conf=conf0,
            ent=ent0,
            verifier=verifier0,
            seq_len=seq_len,
            conf_ema=conf_ema_tracker,
            s_max_cap=s_max_adaptive,
            task_key=task_key,
            cooldown_tracker=cooldown_tracker
        )

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

        # 3brains adaptation
        if args.single_z:
            enc_avg = model.avg_embed(Xc).mean(dim=0)
            # Initialize z for 3brains
            z_lang0 = model.init_z_slow_lang(batch_size=1)[0].to(device)
            z_logic0 = model.init_z_slow_logic(batch_size=1)[0].to(device)
            z_bridge0 = model.init_z_bridge(batch_size=1)[0].to(device)

            # Try to get better bridge initialization from memory
            z_from_mem = model.get_z_from_memory(Xc, z_init=z_bridge0, topk=3, blend=0.6)
            if z_from_mem is not None:
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
            )
            z_for_q = {"slow_lang": z_lang_star, "slow_logic": z_logic_star, "bridge": z_bridge_star}
        else:
            # Per-sample adaptation
            z_lang0 = model.init_z_slow_lang(batch_size=Xc.size(0)).to(device)
            z_logic0 = model.init_z_slow_logic(batch_size=Xc.size(0)).to(device)
            z_bridge0 = model.init_z_bridge(batch_size=Xc.size(0)).to(device)

            # Try to get better bridge initialization from memory for each sample
            for b in range(Xc.size(0)):
                z_from_mem = model.get_z_from_memory(Xc[b:b+1], z_init=z_bridge0[b], topk=3, blend=0.6)
                if z_from_mem is not None:
                    # Adapt memory z to bridge dimension
                    if z_from_mem.size(-1) >= z_bridge0[b].size(-1):
                        z_bridge0[b] = z_from_mem[..., :z_bridge0[b].size(-1)].squeeze(0)
                    else:
                        z_bridge0[b, :z_from_mem.size(-1)] = z_from_mem.squeeze(0)

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
                )
                zl_list.append(zl)
                zg_list.append(zg)
                zb_list.append(zb)
            z_lang_star = torch.stack(zl_list, dim=0).mean(dim=0)
            z_logic_star = torch.stack(zg_list, dim=0).mean(dim=0)
            z_bridge_star = torch.stack(zb_list, dim=0).mean(dim=0)
            z_for_q = {"slow_lang": z_lang_star, "slow_logic": z_logic_star, "bridge": z_bridge_star}

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
            loss = seq_ce_loss(logits_q, Yq)
            # Self-eval metrics on query batch
            conf_vec = model.confidence(Xq)
            if conf_vec is None:
                # fallback proxy from logits/targets
                from .model import confidence_from_logits as _proxy_conf
                conf_vec = _proxy_conf(logits_q, Yq)
            ent_vec = model.sequence_entropy(logits_q)
            # Brier regularizer (align scalar conf with token-correctness)
            brier = model.brier_from_logits_conf(logits_q, Yq, conf_vec)

            # Identity aux: use 3brains for identity QA
            Xi, Yi = make_identity_batch(tok, args.identity, batch_lang="mix", n=4)
            Xi, Yi = Xi.to(device), Yi.to(device)
            logits_i = model(Xi, {"slow_lang": z_lang_star, "slow_logic": z_logic_star, "bridge": z_bridge_star})
            id_loss = seq_ce_loss(logits_i, Yi)
            # Combine total loss
            total = loss + args.id_weight * id_loss + float(_meta_cfg.get("lambda_brier", 0.2)) * brier

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
            except Exception:
                conf_mean_current, ent_mean_current = 0.5, 2.0

            # Calculate current inflation rate (needed for state_vec)
            if len(inflation_window) > 0:
                avg_steps_current = sum(inflation_window) / len(inflation_window)
                inflation_rate = avg_steps_current / s_base
            else:
                avg_steps_current, inflation_rate = s_base, 1.0

            # State vector: [conf, ent, verifier, seq_len, gray_flag, infl_512, loss_delta, conf_delta]
            loss_delta = (loss_val - prev_loss_val) if prev_loss_val is not None else 0.0
            conf_delta = (conf_mean_current - prev_conf_mean) if prev_conf_mean is not None else 0.0
            in_gray_zone = (0.90 <= verifier0 < 0.95)

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
                lambda_rej = 0.1 if step < 6000 else 0.05
                loss_rejection = rejector_score * lambda_rej

                # Cost constraint: inflation penalty
                lambda_infl = 0.02
                loss_cost = lambda_infl * max(0.0, inflation_rate - 1.30) if len(inflation_window) > 0 else 0.0

                # Scheduler for weights (5k-6k: supervised only, 6k-8k: 30% bandit, 8k+: 60% bandit)
                if step < 6000:
                    # 5k-6k: supervised only
                    lambda_cov = 2.0
                    lambda_sel = 1.0
                else:
                    # 6k+: reduce coverage constraint, increase selective risk
                    lambda_cov = 1.0
                    lambda_sel = 2.0

                # Add to total loss
                total = total + lambda_cov * loss_coverage + lambda_sel * loss_selective + loss_rejection + loss_cost

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
                # 2. Length cap integration: seq_len > 64 → increase threshold
                # 2.1. Abstain cap schedule: lower cap → higher threshold (harder to abstain)
                # rikomi.txt fix: Clamp adjustment to ±0.25 to prevent excessive swing
                adjustment = (1.0 - abstain_cap) * 0.5
                adjustment = max(-0.25, min(0.25, adjustment))  # Clamp to [-0.25, +0.25]
                tau_r_adjusted = tau_r.item() + adjustment
                if seq_len > 64:
                    tau_r_adjusted -= 0.02  # harder to abstain

                # 3. Hysteresis (separate entry/exit)
                if abstain_hysteresis_state:
                    # Exit: conf > 0.66 and ent < 2.2
                    if conf_mean_current > 0.66 and ent_mean_current < 2.2:
                        abstain_hysteresis_state = False
                        should_abstain = False
                    else:
                        should_abstain = True
                else:
                    # Entry: conf < 0.60 or ent > 2.6
                    if conf_mean_current < 0.60 or ent_mean_current > 2.6:
                        abstain_hysteresis_state = True
                        should_abstain = True
                    else:
                        # Use policy decision
                        should_abstain = (rejector_score > tau_r_adjusted)

            abstain_ratio = 1.0 if should_abstain else 0.0
            abstain_window_100.append(abstain_ratio)
            if len(abstain_window_100) > 100:
                abstain_window_100.pop(0)

            # 4. Runaway protection: if abstain rate > 0.8 in last 100 steps
            if len(abstain_window_100) >= 100:
                recent_abstain_rate = sum(abstain_window_100) / len(abstain_window_100)
                if recent_abstain_rate > 0.8:
                    # Emergency: increase threshold to reduce abstaining
                    with torch.no_grad():
                        tau_r.data.clamp_(min=tau_r.item() + 0.02)

            abstain_enabled = True

        # Coverage-aware LR adjustment (lacomi.txt + rikomi.txt prescription)
        # coverage < 0.4: maintain LR (don't decay too fast during low coverage)
        # coverage > 0.6: apply additional cooling
        # NOTE: This must be done AFTER abstain_ratio is calculated
        # rikomi.txt fix: Use 100-step EMA to prevent LR jitter from single-batch noise
        coverage_raw = 1.0 - abstain_ratio
        alpha_coverage = 2.0 / (100.0 + 1.0)  # ~0.02 for 100-step EMA
        coverage_ema = alpha_coverage * coverage_raw + (1.0 - alpha_coverage) * coverage_ema

        lr_mult = lr_schedule(step)

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
        ema = loss_val if ema is None else ema * 0.95 + loss_val * 0.05

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
        # Skip update if quality gate fails or abstain triggers
        if not gate_passed:
            did_update = False
        elif abstain_enabled and args.abstain and should_abstain:
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
            else:
                total.backward()
                # Gradient clipping (after 2k steps, more aggressive)
                if step >= 2000:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()

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

        # Dynamic s_max recovery (from tune.txt):
        # If p(s≥12) ≤ 1%, increase s_max back to 12 (then 15)
        if len(inflation_window) >= 256:  # Need enough samples
            if high_step_prob <= 0.01 and s_max_adaptive < 12:
                s_max_adaptive = 12  # Recover to 12
            elif high_step_prob <= 0.005 and s_max_adaptive < 15:
                s_max_adaptive = 15  # Fully recover to 15

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
        else:
            cb_t_ent = getattr(pbar, '_cb_t_ent', 0.0)
            cb_v_ent = getattr(pbar, '_cb_v_ent', 0.0)
        pbar._cb_t_ent = cb_t_ent
        pbar._cb_v_ent = cb_v_ent

        pbar.set_postfix({
            "loss": f"{loss_val:.3f}",
            "ema": f"{ema:.3f}",
            "ppl": f"{perplexity:.1f}",
            "conf": f"{conf_mean:.2f}",
            "s": chosen_steps,
            "s_avg": f"{avg_steps_current:.1f}",
            "s_max": s_max_adaptive,
            "infl": f"{inflation_rate:.2f}",
            "p12+": f"{high_step_prob:.2%}",
            "acc": f"{rule_acc:.3f}",
            "gate": f"{gate_pass_60_95:.2f}",
            "abstain": f"{abstain_ratio:.1f}",
            "abs_cap": f"{abstain_cap:.2f}",
            "fw": "Y" if force_write_enabled else "N",
            "rej_r": f"{rejector_display:.2f}" if use_autonomous_abstain else "-",
            "τ_r": f"{tau_r_display:.2f}" if use_autonomous_abstain else "-",
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

            # Continuous gate: surprise (entropy) * z_norm composite score
            surprise = ent_mean  # entropy as surprise metric
            z_quality = z_bridge_norm  # z magnitude as quality indicator
            composite_score = surprise * z_quality

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
                    zc_tokens = codebook.encode(text=query_text, h=enc_avg_q, return_parts=True)
                    zc_str = codebook.format_zc(zc_tokens)

                    # 온라인 업데이트 (shadow에 누적)
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
                    mem.add_with_policy(key, val, step, meta={"lang": int(lang_idx), "desc": desc, "composite_score": float(composite_score)})
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

                _sum_writer.writerow({
                    "step": step + 1,
                    "loss": round(loss_val, 4),
                    "ema": round(ema, 4),
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
                    "cloze_transition": round(cloze_transition, 4),
                    "gate_pass_60_95": round(gate_pass_60_95, 4),
                    "z_lang_norm": round(z_lang_norm, 4),
                    "z_logic_norm": round(z_logic_norm, 4),
                    "z_bridge_norm": round(z_bridge_norm, 4),
                    "mem_size": len(mem.items),
                    "rejector_score": round(rejector_display, 4),
                    "tau_r": round(tau_r_display, 4),
                    "coverage": round(1.0 - abstain_ratio, 4),
                    "use_autonomous": 1 if use_autonomous_abstain else 0,
                    "difficulty_level": difficulty_level,
                    "composite_score": round(composite_score_log, 4),
                    "abstain_cap": round(abstain_cap, 4),
                    "force_write_enabled": 1 if force_write_enabled else 0,
                    "fw_ce": round(fw_ce, 4),
                    "fw_len_pen": round(fw_len_pen, 4),
                })
                # Flush to disk periodically for real-time monitoring
                if (step + 1) % 10 == 0:
                    _sum_file.flush()
            except Exception:
                pass

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
    }, args.save)
    print(f"Saved to {args.save}; last={last_path}; best={best_path} (best_ema={best_loss:.3f})")

    # Close CSV file
    if _sum_writer is not None:
        try:
            _sum_file.close()
            print(f"Training log saved to {summary_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

