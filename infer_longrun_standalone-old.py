
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, random, csv, statistics, json, sys, pathlib
# allow running next to a local 'uzr' package folder
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'uzr'))

from uzr.model import UZRModel, ByteTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from uzr.memory import CompressedMemory, make_sketch
from uzr.infer_longrun import avg_embed, init_from_retrieval  # reuse helpers from package version
from uzr.infer_longrun import seed_identity  # identity seeding

def lang_of(desc: str):
    if desc.startswith("ko:"): return "ko"
    if desc.startswith("en:"): return "en"
    if desc.startswith("ja:"): return "ja"
    if desc.startswith("base:"): return "base"
    for tag in ("ko:", "en:", "ja:", "base:"):
        if tag in desc: return tag.split(":")[0]
    return "mix"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--lam", type=float, default=3e-3)   # L1 on z (stability)
    ap.add_argument("--alpha", type=float, default=0.3)  # slow update rate (EMA)
    ap.add_argument("--prox", type=float, default=1e-3)  # extra prox for z_slow
    ap.add_argument("--max_items", type=int, default=20000)
    ap.add_argument("--summary_csv", default="infer_summary.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary.json")
    ap.add_argument("--identity", default="루리아")
    args = ap.parse_args()

    import torch
    data = torch.load(args.ckpt, map_location="cpu")
    cfg = data["args"]
    tok = ByteTokenizer(max_len=cfg["max_len"])
    model = UZRModel(tok.vocab_size, d_model=cfg["d_model"], z_dim=cfg["z_dim"], max_len=cfg["max_len"])
    model.load_state_dict(data["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    from uzr.tasks import sample_task
    mem = CompressedMemory(max_items=args.max_items, device=device)

    # Identity seed (ko/en/ja)
    seed_identity(mem, model, tok, device, identity=args.identity)

    # global slow state
    z_slow = model.init_z(batch_size=1).to(device)[0] * 0.0

    # summary log writer
    fieldnames = ["turn","lang","rule_desc","ce_query","conf_context","zslow_l1","mem_items"]
    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader()

    ce_hist = []
    per_lang = {"ko": [], "en": [], "ja": [], "base": [], "mix": []}

    from tqdm import trange
    for t in trange(args.turns, desc="long-run"):
        C, Q, desc = sample_task(n_context=4, n_query=2, n_tokens=5)

        def enc_batch(pairs):
            X = torch.stack([tok.encode(x) for x,_ in pairs], dim=0).to(device)
            Y = torch.stack([tok.encode(y) for _,y in pairs], dim=0).to(device)
            return X, Y

        Xc, Yc = enc_batch(C)
        Xq, Yq = enc_batch(Q)

        enc_avg = avg_embed(model, Xc)

        z_fast = init_from_retrieval(mem, enc_avg, z_slow).clone().detach().requires_grad_(True)

        for _ in range(args.inner_steps):
            logits_c = model(Xc, z_slow + z_fast)
            loss_c = seq_ce_loss(logits_c, Yc) + args.lam * torch.mean(torch.abs(z_slow + z_fast))
            g = torch.autograd.grad(loss_c, z_fast, retain_graph=False)[0]
            conf = confidence_from_logits(logits_c, Yc).mean()
            step = (0.4 + 0.6*conf.item())
            z_fast = z_fast - step * g

        with torch.no_grad():
            z_fast = soft_threshold(z_fast, args.lam * 0.5)

        with torch.no_grad():
            logits_q = model(Xq, z_slow + z_fast)
            ce_q = seq_ce_loss(logits_q, Yq).item()

        with torch.no_grad():
            z_slow = z_slow + args.alpha * (z_fast - z_slow)
            z_slow = soft_threshold(z_slow, args.prox)

        with torch.no_grad():
            sketch_key, sketch_val = make_sketch(enc_avg, z_slow, meta={"desc": desc})
            mem.add(sketch_key, sketch_val, step=t)

        import math
        z_l1 = float(torch.sum(torch.abs(z_slow)).cpu().item())
        ce_hist.append(ce_q)
        if len(ce_hist) > 200:
            ce_hist.pop(0)

        lg = lang_of(desc)
        per_lang.setdefault(lg, []).append(ce_q)

        row = {
            "turn": t+1,
            "lang": lg,
            "rule_desc": desc,
            "ce_query": round(ce_q, 4),
            "conf_context": round(float(conf.item()), 4),
            "zslow_l1": round(z_l1, 4),
            "mem_items": len(mem.items)
        }
        w.writerow(row)

        if (t+1) % args.summary_every == 0:
            med = statistics.median(ce_hist) if ce_hist else float("nan")
            mean = sum(ce_hist)/len(ce_hist)
            print(f"[turn {t+1}] CE(mean/med over last {len(ce_hist)}): {mean:.3f}/{med:.3f} | zL1={z_l1:.1f} | mem={len(mem.items)} | rule='{desc}'")

    fcsv.close()

    def agg(vals):
        if not vals: return {"count": 0, "mean": None, "median": None}
        return {"count": len(vals), "mean": sum(vals)/len(vals), "median": statistics.median(vals)}

    summary = {
        "overall": agg([v for vs in per_lang.values() for v in vs]),
        "by_lang": {k: agg(vs) for k,vs in per_lang.items()},
        "turns": args.turns,
        "identity_seeded": args.identity,
        "csv_path": args.summary_csv,
    }
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    print(f"Summary CSV saved to {args.summary_csv}")
    print(f"Summary JSON saved to {args.summary_json}")
    print("Done. (Standalone long-run finished)")

if __name__ == "__main__":
    main()
