
import argparse, os, random, csv, statistics, json
from pathlib import Path
from datetime import datetime
import torch
from tqdm import trange

from .model import UZRModel, ByteTokenizer, KoEnTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from .meta_core import (
    load_meta_config,
    AbstainThresholds,
    maybe_abstain,
    inner_steps_from_conf,
)
from .memory import CompressedMemory, make_sketch
from .tasks import sample_task

def init_from_retrieval(mem, enc_avg, z_slow, topk=None):
    k = mem.topk if (topk is None) else topk
    items = mem.retrieve(enc_avg, topk=k)
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

    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = data["args"]
    rdw = data.get("model", {}).get("readout.weight")
    rows = rdw.size(0) if isinstance(rdw, torch.Tensor) else None
    tok = ByteTokenizer(max_len=cfg["max_len"]) if rows == 258 else KoEnTokenizer(max_len=cfg["max_len"])
    model = UZRModel(
        tok.vocab_size,
        d_model=cfg["d_model"],
        z_dim=cfg["z_dim"],
        max_len=cfg["max_len"],
        z_think_dim=cfg.get("z_think_dim", 64),
        z_lang_dim=cfg.get("z_lang_dim", 32),
        num_langs=cfg.get("num_langs", 4),
        identity_self_dim=cfg.get("identity_self_dim", 2),
        z_slow_lang_dim=cfg.get("z_slow_lang_dim", 96),
        z_slow_logic_dim=cfg.get("z_slow_logic_dim", 96),
        z_bridge_dim=cfg.get("z_bridge_dim", 64),
    )
    model.load_state_dict(data["model"])
    model.to(args.device)
    model.eval()

    # Auto-name CSV under logu/<timestamp>_s{inner}_t{turns}_{model}.csv when using default name
    try:
        if args.summary_csv == "infer_summary.csv":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_base = Path(args.ckpt).stem if args.ckpt else "model"
            auto_name = f"{ts}_s{args.inner_steps}_t{args.turns}_{model_base}.csv"
            out_dir = Path("logu"); out_dir.mkdir(parents=True, exist_ok=True)
            args.summary_csv = str(out_dir / auto_name)
    except Exception:
        pass

    mem = CompressedMemory(max_items=args.max_items, device=args.device)

    # Seed identity into memory (ko/en/ja)
    seed_identity(mem, model, tok, args.device, identity=args.identity)

    # global slow state
    z_slow = model.init_z(batch_size=1).to(args.device)[0] * 0.0

    # summary log writer (extended with self-eval metrics)
    fieldnames = [
        "turn","lang","rule_desc",
        "ce_query","conf_context","zslow_l1","mem_items",
        "conf0","chosen_steps","tries","best_conf",
        "conf_self_c","ent_c","brier_c","abstain_c",
        "conf_self_q","ent_q","brier_q","abstain_q",
        "gate_pass","compute_tokens",
    ]
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

        # Determine inner-step budget from initial confidence (gradient-free self-eval)
        cfg_meta = load_meta_config()
        thr = AbstainThresholds(conf_min=cfg_meta["conf_min"], ent_max=cfg_meta["ent_max"])
        conf0_vec = model.confidence(Xc)
        if conf0_vec is None:
            with torch.no_grad():
                logits0 = model(Xc, z_slow + z_fast)
                conf0_vec = confidence_from_logits(logits0, Yc)
        conf0 = float(conf0_vec.mean().item())
        chosen_steps = inner_steps_from_conf(conf0, s_max=int(args.inner_steps), s_min=0, k=10.0, mid=0.7)
        tries = 0
        best_conf = conf0

        for _ in range(chosen_steps):
            logits_c = model(Xc, z_slow + z_fast)
            loss_c = seq_ce_loss(logits_c, Yc) + args.lam * torch.mean(torch.abs(z_slow + z_fast))
            g = torch.autograd.grad(loss_c, z_fast, retain_graph=False)[0]
            # Self-eval confidence on context
            conf_self_c_vec = model.confidence(Xc)
            if conf_self_c_vec is None:
                conf_self_c_vec = confidence_from_logits(logits_c, Yc)
            conf = conf_self_c_vec.mean()
            step = (0.4 + 0.6 * float(conf.item()))
            z_fast = z_fast - step * g
            tries += 1
            # Early stop if high confidence
            cval = float(conf.item())
            if cval > best_conf:
                best_conf = cval
            if cval >= 0.8:
                break

        with torch.no_grad():
            z_fast = soft_threshold(z_fast, args.lam * 0.5)

        with torch.no_grad():
            logits_q = model(Xq, z_slow + z_fast)
            ce_q = seq_ce_loss(logits_q, Yq).item()

        with torch.no_grad():
            z_slow = z_slow + args.alpha * (z_fast - z_slow)
            z_slow = soft_threshold(z_slow, args.prox)

        with torch.no_grad():
            # Compute self-eval metrics for logging
            conf_self_c_vec = model.confidence(Xc)
            if conf_self_c_vec is None:
                conf_self_c_vec = confidence_from_logits(logits_c, Yc)
            ent_c = model.sequence_entropy(logits_c)
            brier_c = model.brier_from_logits_conf(logits_c, Yc, conf_self_c_vec)
            abstain_c_mask = maybe_abstain(conf_self_c_vec, ent_c, thr)

            conf_self_q_vec = model.confidence(Xq)
            if conf_self_q_vec is None:
                conf_self_q_vec = confidence_from_logits(logits_q, Yq)
            ent_q = model.sequence_entropy(logits_q)
            brier_q = model.brier_from_logits_conf(logits_q, Yq, conf_self_q_vec)
            abstain_q_mask = maybe_abstain(conf_self_q_vec, ent_q, thr)

            # Apply long-term memory policy (3brain manual)
            meta = {
                "desc": desc,
                "ce_q": float(ce_q),
                "conf": float(conf.item()),
                "conf0": conf0,
                "chosen_steps": int(chosen_steps),
                "tries": int(tries),
                "best_conf": float(best_conf),
                "conf_self_c": float(conf_self_c_vec.mean().item()),
                "ent_c": float(ent_c.mean().item()),
                "brier_c": float(brier_c.item()),
                "conf_self_q": float(conf_self_q_vec.mean().item()),
                "ent_q": float(ent_q.mean().item()),
                "brier_q": float(brier_q.item()),
            }
            sketch_key, sketch_val = make_sketch(enc_avg, z_slow, meta=meta)
            if hasattr(mem, "add_with_policy"):
                mem.add_with_policy(sketch_key, sketch_val, step=t, meta=meta)
            else:
                mem.add(sketch_key, sketch_val, step=t)

        z_l1 = float(torch.sum(torch.abs(z_slow)).cpu().item())
        ce_hist.append(ce_q)
        if len(ce_hist) > 200:
            ce_hist.pop(0)

        lg = lang_of(desc)
        per_lang.setdefault(lg, []).append(ce_q)

        # Approx compute tokens consumed by inner loops (context forward per try)
        try:
            Bc, Tc = Xc.shape
            compute_tokens = int(tries * Bc * Tc)
        except Exception:
            compute_tokens = tries

        row = {
            "turn": t+1,
            "lang": lg,
            "rule_desc": desc,
            "ce_query": round(ce_q, 4),
            "conf_context": round(float(conf.item()), 4),
            "zslow_l1": round(z_l1, 4),
            "mem_items": len(mem.items),
            "conf0": round(conf0, 4),
            "chosen_steps": int(chosen_steps),
            "tries": int(tries),
            "best_conf": round(float(best_conf), 4),
            "conf_self_c": round(float(conf_self_c_vec.mean().item()), 4),
            "ent_c": round(float(ent_c.mean().item()), 4),
            "brier_c": round(float(brier_c.item()), 4),
            "abstain_c": int(abstain_c_mask.float().mean().item() > 0),
            "conf_self_q": round(float(conf_self_q_vec.mean().item()), 4),
            "ent_q": round(float(ent_q.mean().item()), 4),
            "brier_q": round(float(brier_q.item()), 4),
            "abstain_q": int(abstain_q_mask.float().mean().item() > 0),
            "gate_pass": int((abstain_c_mask.float().mean().item() <= 0.5) and (abstain_q_mask.float().mean().item() <= 0.5)),
            "compute_tokens": int(compute_tokens),
        }
        w.writerow(row)

        if (t+1) % args.summary_every == 0:
            # Periodic maintenance per manual
            if hasattr(mem, "rebalance"):
                mem.rebalance()
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
