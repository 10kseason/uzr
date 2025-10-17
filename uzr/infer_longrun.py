
import argparse, os, random, csv, statistics, json
import torch
from tqdm import trange

from .model import UZRModel, ByteTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from .memory import CompressedMemory, make_sketch
from .tasks import sample_task

def init_from_retrieval(mem, enc_avg, z_slow, topk=4):
    items = mem.retrieve(enc_avg, topk=topk)
    if not items:
        return torch.zeros_like(z_slow)
    Z = torch.stack([it.val["z_slow"] for it in items], dim=0)
    return Z.mean(dim=0)

def avg_embed(model, X):
    with torch.no_grad():
        h = model.encoder(X)
        m = h.mean(dim=1)
        return m.mean(dim=0)

def encode_str(tok, s, device):
    return torch.stack([tok.encode(s)], dim=0).to(device)

def lang_of(desc: str):
    if desc.startswith("ko:"): return "ko"
    if desc.startswith("en:"): return "en"
    if desc.startswith("ja:"): return "ja"
    if desc.startswith("base:"): return "base"
    for tag in ("ko:", "en:", "ja:", "base:"):
        if tag in desc: return tag.split(":")[0]
    return "mix"

def seed_identity(mem, model, tok, device, identity="Luria"):
    phrases = [
        f"나는 {identity}입니다.",
        f"I am {identity}.",
        f"わたしは{identity}です。",
    ]
    for ph in phrases:
        X = encode_str(tok, ph, device)
        emb = avg_embed(model, X)
        z0 = model.init_z(batch_size=1).to(device)[0] * 0.0
        key, val = make_sketch(emb, z0, meta={"identity": identity, "phrase": ph})
        mem.add(key, val, step=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--inner_steps", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--prox", type=float, default=5e-4)
    ap.add_argument("--max_items", type=int, default=2048)
    ap.add_argument("--summary_csv", default="infer_summary.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary.json")
    ap.add_argument("--identity", default="루리아", help="Base identity name to seed into memory (e.g., '루리아'/'Luria')")
    args = ap.parse_args()

    data = torch.load(args.ckpt, map_location="cpu")
    cfg = data["args"]
    tok = ByteTokenizer(max_len=cfg["max_len"])
    model = UZRModel(tok.vocab_size, d_model=cfg["d_model"], z_dim=cfg["z_dim"], max_len=cfg["max_len"])
    model.load_state_dict(data["model"])
    model.to(args.device)
    model.eval()

    mem = CompressedMemory(max_items=args.max_items, device=args.device)

    # Seed identity into memory (ko/en/ja)
    seed_identity(mem, model, tok, args.device, identity=args.identity)

    # global slow state
    z_slow = model.init_z(batch_size=1).to(args.device)[0] * 0.0

    # summary log writer
    fieldnames = ["turn","lang","rule_desc","ce_query","conf_context","zslow_l1","mem_items"]
    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader()

    ce_hist = []
    per_lang = {"ko": [], "en": [], "ja": [], "base": [], "mix": []}

    for t in trange(args.turns, desc="long-run"):
        C, Q, desc = sample_task(n_context=4, n_query=2, n_tokens=5)

        def enc_batch(pairs):
            X = torch.stack([tok.encode(x) for x,_ in pairs], dim=0).to(args.device)
            Y = torch.stack([tok.encode(y) for _,y in pairs], dim=0).to(args.device)
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

    # JSON summary aggregates
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
    print("Done. (Prototype long-run finished)")
