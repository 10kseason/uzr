
# nv_infer_debug.py
# CPU demo: load your ckpt (uzr_ckpt_best.pt), generate with anti-collapse sampling,
# Korean-friendly decode, and non-volatile z (snapshot + EMA consolidate).

import os, sys, math, time, json, argparse, torch
from typing import List
from model_nonvolatile import (
    ByteTokenizer, UZRModel, PersistentZStore,
    confidence_from_logits, guess_lang_id,
    LANG_EN, LANG_KO, LANG_JA, LANG_BASE
)

def top_p_filtering(logits, top_p=0.95, min_tokens_to_keep=1):
    """Nucleus filtering (from huggingface)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits

def apply_repetition_penalty(logits, generated_ids: List[int], penalty=1.1):
    if penalty <= 1.0 or len(generated_ids) == 0:
        return logits
    # Penalize tokens that appeared in the recent window
    recent = generated_ids[-256:]  # long enough window for byte-level repeat
    unique = list(set(recent))
    if len(unique) == 0:
        return logits
    for tid in unique:
        logits[tid] /= penalty
    return logits

def apply_bad_token_bias(logits, bad_ids=None, bias=-2.0):
    if not bad_ids:
        return logits
    for tid in bad_ids:
        logits[tid] += bias  # lower logit (bias is negative)
    return logits

def entropy(logits):
    p = torch.softmax(logits, dim=-1)
    logp = torch.log(p + 1e-12)
    return float(-(p * logp).sum())

def generate(model, tok, prompt, z, max_new_tokens=200, temperature=0.9, top_p=0.95,
             rep_penalty=1.15, eos_id=257, bad_ids=None, device="cpu"):
    x = tok.encode(prompt).unsqueeze(0).to(device)
    generated = []
    last_id = None
    same_streak = 0

    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(x, z)[0, -1]  # [V]
            # dynamic guards
            H = entropy(logits)
            if H < 0.5:  # extremely peaked
                temperature = max(temperature, 1.2)
                top_p = max(top_p, 0.98)

            logits = apply_repetition_penalty(logits, generated, penalty=rep_penalty)
            logits = apply_bad_token_bias(logits, bad_ids=bad_ids, bias=-2.5)

            logits = logits / max(temperature, 1e-6)
            logits = top_p_filtering(logits, top_p=top_p, min_tokens_to_keep=1)
            probs = torch.softmax(logits, dim=-1)

            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)

            if next_id == eos_id:
                break

            if last_id is not None and next_id == last_id:
                same_streak += 1
                if same_streak >= 16:
                    # ban this token once and resample
                    logits[next_id] = float("-inf")
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                    generated[-1] = next_id
                    same_streak = 0
            else:
                same_streak = 0
            last_id = next_id

            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    text_out = tok.decode(generated)
    return text_out, generated

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="uzr_ckpt_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--text", default="안녕하세요. 한국어로 대답해 주세요.")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep_penalty", type=float, default=1.15)
    ap.add_argument("--ban_colon", action="store_true", help="Temporarily bias against ':' if collapse occurs.")
    args = ap.parse_args()

    device = args.device
    tok = ByteTokenizer(max_len=256)
    model = UZRModel(
        vocab_size=tok.vocab_size,
        d_model=1024, z_dim=512,
        n_head=4, n_layer=4, max_len=256,
        z_think_dim=256, z_lang_dim=128, num_langs=4
    ).to(device)
    model.eval()

    pkg = torch.load(args.ckpt, map_location=device)
    state = pkg["state_dict"] if isinstance(pkg, dict) and "state_dict" in pkg else pkg
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load] missing:", missing, "| unexpected:", unexpected)

    zstore = PersistentZStore("./nv_store")
    zlast = zstore.load_last()

    lang_id = guess_lang_id(args.text)
    B = 1
    if zlast is None:
        z = {"rule": model.init_z(B), "think": model.init_z_thinking(B), "lang_id": lang_id}
    else:
        z = {"rule": zlast["rule"], "think": zlast["think"], "lang_id": int(zlast["lang_id"])}

    bad_ids = [58] if args.ban_colon else []  # ':' byte

    out, ids = generate(
        model, tok, args.text, z,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        bad_ids=bad_ids,
        device=device
    )
    print("\n=== OUTPUT ===")
    print(out)
    print("\n[first 64 token ids]:", ids[:64])

    # consolidate z0 from teacher-forced pass on prompt (simple proxy)
    x = tok.encode(args.text).unsqueeze(0).to(device)
    logits = model(x, z)
    conf = confidence_from_logits(logits, target=x)
    print("[conf]:", float(conf[0]))
    model.consolidate_z0(z["rule"], conf=conf, alpha=0.10, conf_thr=0.60)

    zstore.save(z["rule"], z["think"], z["lang_id"], step=0)
    torch.save({"state_dict": model.state_dict()}, "uzr_ckpt_best_nv.pt")
    print("✓ Saved z snapshot to ./nv_store/z_last.pt and weights to uzr_ckpt_best_nv.pt")

if __name__ == "__main__":
    main()
