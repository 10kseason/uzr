#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filtered long-run inference that **excludes specific BASE rules** (e.g., reverse token order,
# aggressive prefix/suffix wrappers) while leaving everything else untouched.
#
# Usage (example):
#   python /mnt/data/infer_longrun_exclude_rules.py --device cuda --ckpt uzr_ckpt.pt --turns 200 --inner_steps 5
#
# You can edit BAD_RULE_PATTERNS below to refine which rules are filtered.

import argparse, csv, statistics, json, sys, pathlib, re
from typing import Tuple

import torch

# allow running next to a local 'uzr' package folder
HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE / 'uzr'))

from uzr.model import UZRModel, ByteTokenizer, KoEnTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from uzr.memory import CompressedMemory, make_sketch
import uzr.tasks as UZRT

# ------------------------------
# Rule filtering config
# ------------------------------
# Any rule_desc matching these patterns (regex) will be **skipped** when lang == 'base'.
BAD_RULE_PATTERNS = [
    r"^base:\s*reverse token order",          # proven worst-offender (CE spikes)
    r"^base:\s*prefix=.*\s+suffix=.*",        # aggressive wrapper-format rules
    # add more patterns if needed, e.g.:
    # r"^base:\s*caesar shift [45]$",
]

_bad_rule_regexes = [re.compile(p) for p in BAD_RULE_PATTERNS]

def _is_bad_base_rule(desc: str) -> bool:
    if not isinstance(desc, str):
        return False
    if not desc.startswith("base:"):
        return False
    for rgx in _bad_rule_regexes:
        if rgx.search(desc):
            return True
    return False

def sample_task_filtered(*args, **kwargs):
    """Call original sample_task until we get a rule_desc **not** blacklisted for BASE."""
    # We impose a hard cap to avoid infinite loops if the generator only yields bad rules.
    for _ in range(32):
        C, Q, desc = UZRT.sample_task(*args, **kwargs)
        if not _is_bad_base_rule(desc):
            return C, Q, desc
    # Fallback: if we really couldn't get a good one, just return the last sample (don't block run)
    return C, Q, desc

LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}

def lang_of(desc: str) -> str:
    if desc.startswith("ko:"): return "ko"
    if desc.startswith("en:"): return "en"
    if desc.startswith("ja:"): return "ja"
    if desc.startswith("base:"): return "base"
    for tag in ("ko:", "en:", "ja:", "base:"):
        if tag in desc:
            return tag.split(":")[0]
    return "mix"

def avg_embed(model: UZRModel, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h = model.encoder(X)
        return h.mean(dim=1).mean(dim=0)

def encode_str(tok: ByteTokenizer, s: str, device: torch.device) -> torch.Tensor:
    return torch.stack([tok.encode(s)], dim=0).to(device)

def resolve_lang_idx(lang_key: str, num_langs: int) -> int:
    idx = LANG2ID.get(lang_key, LANG2ID["base"])
    if idx >= num_langs:
        return LANG2ID["base"]
    return idx

def init_from_retrieval_multi(mem: CompressedMemory, enc_avg: torch.Tensor, z_rule_ref: torch.Tensor,
                               z_think_ref: torch.Tensor, lang_id: int, topk: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    items = mem.retrieve(enc_avg, topk=topk)
    if not items:
        return z_rule_ref.new_zeros(z_rule_ref.shape), z_think_ref.new_zeros(z_think_ref.shape)

    def collect(candidates, match_lang: bool):
        rules, thinks = [], []
        for it in candidates:
            val = it.val
            if match_lang:
                if val.get("lang_id") is None or int(val["lang_id"]) != int(lang_id):
                    continue
            z_rule = val.get("z_rule")
            z_think = val.get("z_think")
            if isinstance(z_rule, torch.Tensor) and isinstance(z_think, torch.Tensor):
                rules.append(z_rule.to(z_rule_ref.device))
                thinks.append(z_think.to(z_think_ref.device))
        if rules:
            return (torch.stack(rules, dim=0).mean(dim=0), torch.stack(thinks, dim=0).mean(dim=0))
        return None

    match = collect(items, match_lang=True)
    if match is None:
        match = collect(items, match_lang=False)
    if match is None:
        return z_rule_ref.new_zeros(z_rule_ref.shape), z_think_ref.new_zeros(z_think_ref.shape)
    return match

def seed_identity(mem: CompressedMemory, model: UZRModel, tok: ByteTokenizer, device: torch.device, identity: str = "루리아"):
    phrases = [
        (f"나는 {identity}입니다.", "ko"),
        (f"I am {identity}.", "en"),
        (f"わたしは{identity}です。", "ja"),
    ]
    for text, lang_key in phrases:
        lang_idx = resolve_lang_idx(lang_key, model.num_langs)
        X = encode_str(tok, text, device)
        emb = avg_embed(model, X)
        z_rule = model.init_z(batch_size=1).to(device)[0].detach()
        z_rule.zero_()
        z_think = model.init_z_thinking(batch_size=1).to(device)[0].detach()
        z_think.zero_()
        fused = model._fuse_z(z_rule, z_think, lang_idx)[0].detach()
        key, val = make_sketch(emb, fused, meta={"identity": identity, "phrase": text, "lang": lang_key})
        val["z_rule"] = z_rule.clone()
        val["z_think"] = z_think.clone()
        val["lang_id"] = int(lang_idx)
        mem.add(key, val, step=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--lam", type=float, default=3e-3, help="Legacy L1 on fused z (fallback)")
    ap.add_argument("--lam_rule", type=float, default=None, help="L1 penalty for z_rule (defaults to --lam)")
    ap.add_argument("--lam_think", type=float, default=None, help="L1 penalty for z_thinking (defaults to --lam)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--prox", type=float, default=1e-3)
    ap.add_argument("--max_items", type=int, default=20000)
    ap.add_argument("--summary_csv", default="infer_summary.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary.json")
    ap.add_argument("--identity", default="루리아")
    args = ap.parse_args()

    lam_rule = args.lam if args.lam_rule is None else args.lam_rule
    lam_think = args.lam if args.lam_think is None else args.lam_think

    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = data["args"]

    # Auto-detect tokenizer from vocab size
    vocab_size = data["model"]["encoder.tok.weight"].shape[0]
    tok = ByteTokenizer(max_len=cfg["max_len"]) if vocab_size == 258 else KoEnTokenizer(max_len=cfg["max_len"])

    model = UZRModel(
        tok.vocab_size,
        d_model=cfg["d_model"],
        z_dim=cfg["z_dim"],
        max_len=cfg["max_len"],
        z_think_dim=cfg.get("z_think_dim", 64),
        z_lang_dim=cfg.get("z_lang_dim", 32),
        num_langs=cfg.get("num_langs", len(LANG2ID)),
    )
    model.load_state_dict(data["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    mem = CompressedMemory(max_items=args.max_items, device=device)
    seed_identity(mem, model, tok, device, identity=args.identity)

    z_slow_rule = model.init_z(batch_size=1).to(device)[0].detach()
    z_slow_think = model.init_z_thinking(batch_size=1).to(device)[0].detach()

    fieldnames = [
        "turn",
        "lang",
        "lang_id",
        "rule_desc",
        "ce_query",
        "conf_context",
        "zslow_rule_l1",
        "zslow_think_l1",
        "zlang_norm",
        "mem_items",
    ]
    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader()

    ce_hist = []
    per_lang = {"ko": [], "en": [], "ja": [], "base": [], "mix": []}

    last_lang_idx = 0
    last_lang_norm = float(torch.norm(model.lang_embed(torch.tensor([last_lang_idx], device=device))).cpu().item())

    for t in range(args.turns):
        # *** only changed line: use filtered sampler ***
        C, Q, desc = sample_task_filtered(n_context=4, n_query=2, n_tokens=5)

        def enc_batch(pairs):
            X = torch.stack([tok.encode(x) for x,_ in pairs], dim=0).to(device)
            Y = torch.stack([tok.encode(y) for _,y in pairs], dim=0).to(device)
            return X, Y

        Xc, Yc = enc_batch(C)
        Xq, Yq = enc_batch(Q)

        enc_avg = avg_embed(model, Xc)

        lang_key = lang_of(desc)
        lang_idx = resolve_lang_idx(lang_key, model.num_langs)
        z_fast_rule_0, z_fast_think_0 = init_from_retrieval_multi(mem, enc_avg, z_slow_rule, z_slow_think, lang_idx)
        z_fast_rule = z_fast_rule_0.clone().detach().requires_grad_(True)
        z_fast_think = z_fast_think_0.clone().detach().requires_grad_(True)

        conf_val = 0.0
        for _ in range(args.inner_steps):
            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_c = model(Xc, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            loss_c = seq_ce_loss(logits_c, Yc)
            loss_c = loss_c + lam_rule * torch.mean(torch.abs(z_rule_total)) + lam_think * torch.mean(torch.abs(z_think_total))
            grad_rule, grad_think = torch.autograd.grad(loss_c, (z_fast_rule, z_fast_think), retain_graph=False)
            conf = confidence_from_logits(logits_c, Yc).mean()
            conf_val = float(conf.item())
            step = 0.4 + 0.6 * conf_val
            z_fast_rule = z_fast_rule - step * grad_rule
            z_fast_think = z_fast_think - step * grad_think

        with torch.no_grad():
            z_fast_rule = soft_threshold(z_fast_rule, lam_rule * 0.5)
            z_fast_think = soft_threshold(z_fast_think, lam_think * 0.5)

            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_q = model(Xq, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            ce_q = seq_ce_loss(logits_q, Yq).item()

            z_slow_rule = z_slow_rule + args.alpha * (z_fast_rule - z_slow_rule)
            z_slow_think = z_slow_think + args.alpha * (z_fast_think - z_slow_think)
            z_slow_rule = soft_threshold(z_slow_rule, args.prox)
            z_slow_think = soft_threshold(z_slow_think, args.prox)

            fused_slow = model._fuse_z(z_slow_rule, z_slow_think, lang_idx)[0].detach()
            key, val = make_sketch(enc_avg, fused_slow, meta={"desc": desc, "lang": lang_key})
            val["z_rule"] = z_slow_rule.detach().clone()
            val["z_think"] = z_slow_think.detach().clone()
            val["lang_id"] = int(lang_idx)
            mem.add(key, val, step=t)

            z_rule_l1 = float(torch.sum(torch.abs(z_slow_rule)).cpu().item())
            z_think_l1 = float(torch.sum(torch.abs(z_slow_think)).cpu().item())
            lang_norm = float(torch.norm(model.lang_embed(torch.tensor([lang_idx], device=device))).cpu().item())

        ce_hist.append(ce_q)
        if len(ce_hist) > 200:
            ce_hist.pop(0)

        per_lang.setdefault(lang_key, []).append(ce_q)

        row = {
            "turn": t + 1,
            "lang": lang_key,
            "lang_id": int(lang_idx),
            "rule_desc": desc,
            "ce_query": round(ce_q, 4),
            "conf_context": round(conf_val, 4),
            "zslow_rule_l1": round(z_rule_l1, 4),
            "zslow_think_l1": round(z_think_l1, 4),
            "zlang_norm": round(lang_norm, 4),
            "mem_items": len(mem.items),
        }
        w.writerow(row)

        last_lang_idx = int(lang_idx)
        last_lang_norm = lang_norm

        if (t + 1) % args.summary_every == 0:
            med = statistics.median(ce_hist) if ce_hist else float("nan")
            mean = sum(ce_hist) / len(ce_hist)
            print(f"[turn {t + 1}] CE(mean/med last {len(ce_hist)}): {mean:.3f}/{med:.3f} | z_rule_L1={z_rule_l1:.1f} | z_think_L1={z_think_l1:.1f} | mem={len(mem.items)} | rule='{desc}'")

    fcsv.close()

    def agg(vals):
        if not vals:
            return {"count": 0, "mean": None, "median": None}
        return {"count": len(vals), "mean": sum(vals)/len(vals), "median": statistics.median(vals)}

    summary = {
        "overall": agg([v for vs in per_lang.values() for v in vs]),
        "by_lang": {k: agg(vs) for k,vs in per_lang.items()},
        "turns": args.turns,
        "identity_seeded": args.identity,
        "csv_path": args.summary_csv,
        "final_z_rule_l1": z_rule_l1,
        "final_z_think_l1": z_think_l1,
        "final_lang_id": last_lang_idx,
        "final_lang_embed_norm": last_lang_norm,
        "bad_rule_patterns": BAD_RULE_PATTERNS,
    }
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    print(f"Summary CSV saved to {args.summary_csv}")
    print(f"Summary JSON saved to {args.summary_json}")
    print("Done. (Filtered long-run finished)")

if __name__ == "__main__":
    main()
