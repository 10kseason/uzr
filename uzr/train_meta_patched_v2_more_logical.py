import argparse, random, os, math, time, unicodedata
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# NOTE: Keep imports for backwards compatibility, but we won't use ByteTokenizer now.
from .model import UZRModel, ByteTokenizer, seq_ce_loss, soft_threshold
from .memory import CompressedMemory, make_sketch
from uzr.tasks import sample_task

# -------------------------------
# Identity QA (KO/EN only; JA commented out for future support)
# -------------------------------
IDENTITY_QA_KO = [
    ("너는 누구야?", "나는 {id}입니다."),
    ("자기소개 해 봐.", "나는 {id}입니다."),
    ("이름이 뭐야?", "{id}"),
]
IDENTITY_QA_EN = [
    ("Who are you?", "I am {id}."),
    ("What is your name?", "{id}"),
    ("Introduce yourself.", "I am {id}."),
]
# IDENTITY_QA_JA = [
#     ("あなたはだれ？", "わたしは{id}です。"),
#     ("おなまえは？", "{id}"),
#     ("じこしょうかいして。", "わたしは{id}です。"),
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
    # z-only updates
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
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--inner_steps", type=int, default=6)
    ap.add_argument("--inner_eta", type=float, default=0.425)
    ap.add_argument("--save", default="uzr_ckpt.pt")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--warmup_steps", type=int, default=1500)
    ap.add_argument("--cosine", action="store_true", help="Use cosine LR decay after warmup")
    ap.add_argument("--single_z", action="store_true", help="Share single z across context batch")
    ap.add_argument("--identity", default="루리아", help="identity string used for identity auxiliary loss")
    ap.add_argument("--id_weight", type=float, default=0.2, help="weight of identity auxiliary loss")
    ap.add_argument("--lam", type=float, default=1e-8, help="L1 regularization on z (legacy, used if lam_rule/lam_think unset)")
    ap.add_argument("--lam_rule", type=float, default=8e-4, help="L1 regularization strength for z_rule")
    ap.add_argument("--lam_think", type=float, default=2e-4, help="L1 regularization strength for z_thinking")
    ap.add_argument("--lam_sem", type=float, default=5e-4, help="weight for language-agnostic semantic InfoNCE loss")
    ap.add_argument("--lam_transfer", type=float, default=7e-4, help="weight for cross-language cosine alignment loss")
    ap.add_argument("--aux_baux", type=int, default=4, help="aux mini-batch size per language for semantic alignment")
    ap.add_argument("--sem_dim", type=int, default=12, help="shared semantic projection dim")
    args = ap.parse_args()

    if args.lam_rule is None:
        args.lam_rule = args.lam
    if args.lam_think is None:
        args.lam_think = args.lam

    set_seed(args.seed)
    device = torch.device(args.device)
    # Use the new minimal KO+EN tokenizer
    tok = KoEnTokenizer(max_len=args.max_len)

    # Create memory first with learning enabled
    mem = CompressedMemory(
        max_items=8192,
        device=device,
        enable_learning=True,
        learn_hidden=512,
        learn_depth=3,
        learn_rate=1e-3,
        min_train_items=32,
    )

    # Create model with memory reference
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
    ).to(device)

    def avg_embed(X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = model.encoder(X)
            return h.mean(dim=1)

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

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
    last_path = "uzr_ckpt_last.pt"
    best_path = "uzr_ckpt_best.pt"
    if args.resume and os.path.exists(args.resume):
        data = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(data["model"])
        if "opt" in data:
            opt.load_state_dict(data["opt"])
        if "memory" in data:
            mem.load_state_dict(data["memory"])
            print(f"[resume] restored {len(mem.items)} memory items, learner: {mem.has_learner()}")
        start_step = data.get("step", 0)
        best_loss = data.get("best_loss", best_loss)
        print(f"[resume] loaded {args.resume} from step {start_step} (best_loss={best_loss:.3f})")

    def lr_schedule(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.cosine:
            t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, t))))
        return 1.0

    ema = None
    pbar = tqdm(range(start_step, args.steps), desc="meta-train")
    for step in pbar:
        C_pairs, Q_pairs, desc = sample_task(n_context=6, n_query=8, n_tokens=5)

        example_text = Q_pairs[0][0] + " " + Q_pairs[0][1]
        lang_key = detect_lang_from_text(example_text)
        lang_idx = LANG2ID.get(lang_key, LANG2ID["base"])

        Xc, Yc = batchify(C_pairs, tok)
        Xq, Yq = batchify(Q_pairs, tok)
        Xc, Yc, Xq, Yq = Xc.to(device), Yc.to(device), Xq.to(device), Yq.to(device)

        if args.single_z:
            enc_avg = avg_embed(Xc).mean(dim=0)
            # Use memory predictor for z initialization
            z_init_fallback = model.init_z(batch_size=1)[0]
            z_rule0 = mem.get_or_predict_z(
                avg_emb=enc_avg,
                z_init=z_init_fallback,
                topk=4,
                blend=0.5,
                use_predictor=True,
            ).to(device)
            z_think0 = model.init_z_thinking(batch_size=1)[0].to(device)
            z_rule_star, z_think_star = inner_adapt_z_multi(
                model,
                Xc,
                Yc,
                z_rule0,
                z_think0,
                lang_idx,
                lam_rule=args.lam_rule,
                lam_think=args.lam_think,
                eta=args.inner_eta,
                steps=args.inner_steps,
            )
            z_for_q = {"rule": z_rule_star, "think": z_think_star, "lang_id": lang_idx}
        else:
            z_rule0 = []
            z_think0 = model.init_z_thinking(batch_size=Xc.size(0)).to(device)
            enc = avg_embed(Xc)
            for b in range(Xc.size(0)):
                # Use memory predictor for each sample's z initialization
                z_init_fallback = model.init_z(batch_size=1)[0]
                z0 = mem.get_or_predict_z(
                    avg_emb=enc[b],
                    z_init=z_init_fallback,
                    topk=4,
                    blend=0.5,
                    use_predictor=True,
                )
                z_rule0.append(z0)
            z_rule0 = torch.stack(z_rule0, dim=0).to(device)
            zr_list, zt_list = [], []
            for b in range(Xc.size(0)):
                zr, zt = inner_adapt_z_multi(
                    model,
                    Xc[b:b + 1],
                    Yc[b:b + 1],
                    z_rule0[b],
                    z_think0[b],
                    lang_idx,
                    lam_rule=args.lam_rule,
                    lam_think=args.lam_think,
                    eta=args.inner_eta,
                    steps=args.inner_steps,
                )
                zr_list.append(zr)
                zt_list.append(zt)
            z_rule_star = torch.stack(zr_list, dim=0).mean(dim=0)
            z_think_star = torch.stack(zt_list, dim=0).mean(dim=0)
            z_for_q = {"rule": z_rule_star, "think": z_think_star, "lang_id": lang_idx}

        cur_lr = args.lr * lr_schedule(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits_q = model(Xq, z_for_q)
            loss = seq_ce_loss(logits_q, Yq)

            # Identity aux: pick same z but prepare KO/EN batches
            Xi, Yi = make_identity_batch(tok, args.identity, batch_lang="mix", n=4)
            Xi, Yi = Xi.to(device), Yi.to(device)
            identity_text = tok.decode(Xi[0].tolist())
            id_lang_key = detect_lang_from_text(identity_text)
            id_lang_idx = LANG2ID.get(id_lang_key, LANG2ID["base"])
            logits_i = model(Xi, {"rule": z_rule_star, "think": z_think_star, "lang_id": id_lang_idx})
            id_loss = seq_ce_loss(logits_i, Yi)
            total = loss + args.id_weight * id_loss

            # ---- Language-agnostic semantic & cross-language alignment (KO<->EN) ----
            Baux = args.aux_baux
            if Baux > 0:
                Xko, _ = make_identity_batch(tok, args.identity, batch_lang="ko", n=Baux)
                Xen, _ = make_identity_batch(tok, args.identity, batch_lang="en", n=Baux)
                Xko, Xen = Xko.to(device), Xen.to(device)

                E_ko = avg_embed(Xko); E_en = avg_embed(Xen)
                C_ko = concept_embed(Xko); C_en = concept_embed(Xen)

                S_ko = proj_sem(E_ko, _W_enc2sem); S_en = proj_sem(E_en, _W_enc2sem)
                K_ko = proj_sem(C_ko, _W_con2sem); K_en = proj_sem(C_en, _W_con2sem)

                loss_sem  = (info_nce(S_ko, K_ko) + info_nce(S_en, K_en)) / 2.0
                loss_xfer = (1 - torch.nn.functional.cosine_similarity(S_ko, S_en, dim=-1)).mean()

                total = total + args.lam_sem * loss_sem + args.lam_transfer * loss_xfer

        opt.zero_grad()
        if scaler.is_enabled():
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            opt.step()

        loss_val = float(loss.item())
        ema = loss_val if ema is None else ema * 0.95 + loss_val * 0.05
        pbar.set_postfix({
            "loss": f"{loss_val:.3f}",
            "ema": f"{ema:.3f}",
            "lr": f"{cur_lr:.2e}",
            "task": desc[:32],
            "lang": ID2LANG.get(lang_idx, "base"),
        })

        with torch.no_grad():
            enc_avg_q = avg_embed(Xq).mean(dim=0)
            key, val = make_sketch(enc_avg_q, z_rule_star, meta={"lang": int(lang_idx), "desc": desc})
            mem.add(key, val, step)

        # Train memory predictor periodically
        if (step + 1) % 10 == 0 and len(mem.items) >= mem.min_train_items:
            mem_loss = mem.train_model(steps=5, batch_size=64, shuffle=True)
            if mem_loss is not None and mem.learner_loss is not None:
                pbar.set_postfix({
                    "loss": f"{loss_val:.3f}",
                    "ema": f"{ema:.3f}",
                    "mem_loss": f"{mem.learner_loss:.4f}",
                    "lr": f"{cur_lr:.2e}",
                    "task": desc[:24],
                    "lang": ID2LANG.get(lang_idx, "base"),
                })

        if (step + 1) % max(1, args.save_every) == 0 or step + 1 == args.steps:
            payload = {
                "model": model.state_dict(),
                "args": vars(args),
                "opt": opt.state_dict(),
                "step": step + 1,
                "best_loss": best_loss,
                "memory": mem.state_dict(),  # Save memory state
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
    }, args.save)
    print(f"Saved to {args.save}; last={last_path}; best={best_path} (best_ema={best_loss:.3f})")


if __name__ == "__main__":
    main()
