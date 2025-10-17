
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, pathlib, json, datetime, re
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running next to a local 'uzr' package folder
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "uzr"))

from uzr.model import UZRModel, ByteTokenizer, soft_threshold, seq_ce_loss, confidence_from_logits
from uzr.memory import CompressedMemory, make_sketch
from uzr.infer_longrun import avg_embed, init_from_retrieval, seed_identity

PROMPT_USER = "사용자> "
PROMPT_BOT  = "루리아> "

def clamp_l2(x: torch.Tensor, max_norm: float = 50.0) -> torch.Tensor:
    n = x.norm()
    if n > max_norm:
        x = x * (max_norm / (n + 1e-8))
    return x

# ---------- utilities ----------

def guess_lang(s: str) -> str:
    if any("\u3040" <= ch <= "\u30ff" for ch in s):  # Hiragana/Katakana
        return "ja"
    if any("\uac00" <= ch <= "\ud7af" for ch in s):  # Hangul syllables
        return "ko"
    return "en"

def safe_trim(text: str) -> str:
    # remove non-printables (keep basic whitespace)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t ")
    # collapse excessive punctuation and spaces
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.{3,}", "..", text)
    text = re.sub(r"[!！]{3,}", "!!", text)
    text = re.sub(r"[?？]{3,}", "??", text)
    # strip long garbage tails of repeated symbols
    text = re.sub(r"([\.IHO0Kヨョロ๑오이])\1{4,}.*$", r"\1", text)
    return text.strip()

def tone_filter(s: str, tone: str, lang: str) -> str:
    s = s.strip()
    if tone == "dry":
        return s
    if lang == "ko":
        if not s.endswith(("요","요.","요!","요?")):
            s = s + "요"
    if tone == "friendly":
        if not any(x in s for x in [":)", "😊", "🙂", "☺️"]):
            s = s + " 🙂"
    return s

# ---------- low-rank online adapter (optional) ----------

class LowRankAdapter(nn.Module):
    """
    Produces a delta to logits from pooled encoder features and current z.
    delta_logits = B( ReLU( A * [pool(h); z_proj] ) ), rank r << D
    Only this module is updated online (base model is frozen).
    """
    def __init__(self, d_model: int, z_dim: int, vocab_size: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.z_proj = nn.Linear(z_dim, d_model, bias=False)
        self.A = nn.Linear(d_model*2, rank)
        self.B = nn.Linear(rank, vocab_size, bias=False)
        # small init
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)
        nn.init.zeros_(self.B.weight)

    def forward(self, h_mean: torch.Tensor, z: torch.Tensor):
        # h_mean: [B, D], z: [B, z_dim] or [z_dim]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z_up = self.z_proj(z)             # [B,D]
        hcat = torch.cat([h_mean, z_up], dim=-1)  # [B,2D]
        x = F.relu(self.A(hcat))          # [B,r]
        dlogits = self.B(x)               # [B,V]
        return dlogits

# ---------- main CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--identity", default="루리아")
    # inner adaptation (z)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--prox", type=float, default=1e-3)
    ap.add_argument("--step_coef", type=float, default=0.6, help="single inner step coefficient (if inner_steps is small)")
    # memory / language / tone
    ap.add_argument("--max_items", type=int, default=20000)
    ap.add_argument("--tone", choices=["polite","friendly","dry"], default="polite")
    ap.add_argument("--lang", choices=["auto","ko","en","ja","mix"], default="auto")
    # online adapter options
    ap.add_argument("--online", action="store_true", help="enable online low-rank adapter updates")
    ap.add_argument("--online_lr", type=float, default=3e-4)
    ap.add_argument("--online_steps", type=int, default=1)
    ap.add_argument("--replay_k", type=int, default=8, help="replay buffer size")
    ap.add_argument("--conf_thresh", type=float, default=0.6, help="only update online when confidence is below this")
    ap.add_argument("--revert_tol", type=float, default=0.5, help="revert if online loss spikes by this ratio vs prev")
    # transcript
    ap.add_argument("--save_transcript", default="")
    # stabilization knobs
    ap.add_argument("--decay", type=float, default=0.005)
    ap.add_argument("--zslow_l2max", type=float, default=50.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    data = torch.load(args.ckpt, map_location="cpu")
    cfg = data["args"]
    tok = ByteTokenizer(max_len=cfg["max_len"])
    model = UZRModel(tok.vocab_size, d_model=cfg["d_model"], z_dim=cfg["z_dim"], max_len=cfg["max_len"])
    model.load_state_dict(data["model"])
    model.to(device).eval()

    mem = CompressedMemory(max_items=args.max_items, device=device)
    seed_identity(mem, model, tok, device, identity=args.identity)

    # global slow state
    z_slow = model.init_z(batch_size=1).to(device)[0] * 0.0

    # Online adapter (frozen by default; only this trains when --online)
    adapter = LowRankAdapter(d_model=cfg["d_model"], z_dim=cfg["z_dim"], vocab_size=tok.vocab_size, rank=16).to(device)
    online_opt = torch.optim.AdamW(adapter.parameters(), lr=args.online_lr, betas=(0.9,0.98), weight_decay=0.01)
    last_online_loss = None
    replay = []  # recent (X_in, Y_target) tensors for mini replay

    def decode_argmax_with_adapter(user_text: str, z_total: torch.Tensor):
        X = torch.stack([tok.encode(user_text)], dim=0).to(device)
        with torch.no_grad():
            base_logits = model(X, z_total)           # [1,T,V]
            h = model.encoder(X)                      # [1,T,D]
            h_mean = h.mean(dim=1)                    # [1,D]
            dlogits = adapter(h_mean, z_total)        # [1,V]
            # broadcast delta over all time positions
            dlogits_b = dlogits.unsqueeze(1).expand_as(base_logits)
            logits = base_logits + dlogits_b
            ids = logits.argmax(dim=-1)[0].tolist()
        return tok.decode(ids), base_logits, logits

    def tone_and_trim(out_text: str, lg: str) -> str:
        out_text = safe_trim(out_text)
        out_text = tone_filter(out_text, args.tone, lg)
        return out_text

    # transcript logging
    log = []
    def log_line(role, text):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        log.append({"t": ts, "role": role, "text": text})

    print("인터랙티브 모드 시작. (Ctrl+C로 종료)")
    while True:
        try:
            user = input(PROMPT_USER)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        log_line("user", user)

        # language choice
        if args.lang == "auto":
            lg = guess_lang(user)
        elif args.lang == "mix":
            lg = guess_lang(user)  # simple per-turn auto
        else:
            lg = args.lang

        # retrieval init (language-aware: skip identity-only keys if possible)
        Xu = torch.stack([tok.encode(user)], dim=0).to(device)
        enc_avg = avg_embed(model, Xu)
        z_retrieved = init_from_retrieval(mem, enc_avg, z_slow)
        delta0 = (z_retrieved - z_slow)
        sim = torch.nn.functional.cosine_similarity(z_retrieved, z_slow, dim=0).clamp(-1, 1).item()
        beta = 0.3 * max(0.0, 1.0 - sim)
        z_fast = (beta * delta0).detach().requires_grad_(True)

        # Prepare a light "target" for stable adaptation
        u_low = user.lower()
        if any(q in u_low for q in ["who are you", "your name"]) or "누구" in user or "이름" in user or "だれ" in user or "なまえ" in user:
            if lg == "ko":
                target = f"나는 {args.identity}입니다."
            elif lg == "ja":
                target = f"わたしは{args.identity}です。"
            else:
                target = f"I am {args.identity}."
        else:
            target = user  # weak echo prior

        Xc = torch.stack([tok.encode(user)], dim=0).to(device)
        Yc = torch.stack([tok.encode(target)], dim=0).to(device)

        # inner adaptation on z (confidence-gated step size)
        step_coef = args.step_coef
        for _ in range(max(1, args.inner_steps)):
            logits_c = model(Xc, z_slow + z_fast)
            conf = confidence_from_logits(logits_c, Yc).mean()
            step = 0.4 + 0.6*conf.item()
            step = min(step, step_coef)  # additionally clamp by user coef
            loss_c = seq_ce_loss(logits_c, Yc) + args.lam * torch.mean(torch.abs(z_slow + z_fast))
            g = torch.autograd.grad(loss_c, z_fast, retain_graph=False)[0]
            z_fast = z_fast - step * g

        with torch.no_grad():
            z_fast = soft_threshold(z_fast, args.lam * 0.5)

        # compose state
        z_total = z_slow + z_fast

        # generate with adapter
        out_raw, base_logits, full_logits = decode_argmax_with_adapter(user, z_total)
        out = tone_and_trim(out_raw, lg)
        print(PROMPT_BOT + out)
        log_line("bot", out)

        # slow state + memory update
        with torch.no_grad():
            decay = getattr(args, "decay", 0.005)
            z_slow = (1.0 - decay) * z_slow + args.alpha * z_fast
            z_slow = soft_threshold(z_slow, args.prox)
            z_slow = clamp_l2(z_slow, max_norm=getattr(args, "zslow_l2max", 50.0))
            key, val = make_sketch(enc_avg, z_slow, meta={"lang": lg, "tone": args.tone})
            mem.add(key, val, step=len(log))

        # -------- online adapter update (optional) --------
        if args.online:
            # condition: update when confidence is low (hard turn) OR explicit identity question
            logits_for_conf = base_logits  # confidence from base (stricter)
            conf_now = confidence_from_logits(logits_for_conf, Yc).mean().item()
            need_update = conf_now < args.conf_thresh or target != user
            if need_update:
                adapter.train()
                # build target as ground truth for current turn
                Xgt = Xc
                Ygt = Yc
                # one or two small online steps
                for _ in range(max(1, args.online_steps)):
                    # forward through base + adapter (adapter needs grad)
                    base = model(Xgt, z_total).detach()      # stop grad through base
                    h = model.encoder(Xgt).detach()
                    h_mean = h.mean(dim=1)
                    dlog = adapter(h_mean, z_total)          # [B,V]
                    dlog_b = dlog.unsqueeze(1).expand_as(base)
                    logits = base + dlog_b
                    loss_online = seq_ce_loss(logits, Ygt)

                    online_opt.zero_grad(set_to_none=True)
                    loss_online.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    online_opt.step()

                    # revert if spikes badly
                    nonlocal_last = globals().get("_last_online_loss_cache", None)
                    if nonlocal_last is not None and loss_online.item() > (1.0 + args.revert_tol) * nonlocal_last:
                        # revert by zeroing recent step (simple fallback)
                        for p in adapter.parameters():
                            if hasattr(p, "_prev"):
                                p.data.copy_(p._prev)
                    # stash current params to allow manual revert next turn
                    for p in adapter.parameters():
                        p._prev = p.data.detach().clone()

                    globals()["_last_online_loss_cache"] = loss_online.item()
                    last_online = loss_online.item()

                # add to small replay
                replay.append((Xgt.detach().cpu(), Ygt.detach().cpu()))
                if len(replay) > max(1, args.replay_k):
                    replay.pop(0)

                # tiny replay step (one sample) to keep adapter stable
                if replay:
                    Xr, Yr = replay[0]
                    Xr, Yr = Xr.to(device), Yr.to(device)
                    base = model(Xr, z_total).detach()
                    h = model.encoder(Xr).detach(); h_mean = h.mean(dim=1)
                    dlog = adapter(h_mean, z_total); dlog_b = dlog.unsqueeze(1).expand_as(base)
                    logits = base + dlog_b
                    loss_replay = seq_ce_loss(logits, Yr)
                    online_opt.zero_grad(set_to_none=True)
                    loss_replay.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    online_opt.step()

                adapter.eval()

    # save transcript
    if args.save_transcript:
        with open(args.save_transcript, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"대화 로그 저장: {args.save_transcript}")


if __name__ == "__main__":
    main()
