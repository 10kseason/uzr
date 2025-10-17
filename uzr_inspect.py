#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UZR Inspect — Deep inspection utility for early UZR checkpoints.

Features
- Loads a UZR checkpoint (supports both {model,args} and full training payloads).
- Reconstructs ByteTokenizer/UZRModel from saved args.
- Prints architecture & parameter summary (total/ per-module counts).
- Shows z0 stats and builds z via: zero | init | random | from file.
- Computes FiLM (gamma/beta) statistics for the chosen z.
- Runs a forward pass on a sample text and:
  * Captures TinyEncoder layer outputs via forward hooks (mean/std per layer).
  * Reports tokenization round-trip and shapes.
  * Dumps top-K predictions per token (first N positions) with byte-decoded tokens.
- Compares outputs across z variants (zero/init/random) via cosine sim & KL divergence.
- Optionally dumps a JSON file with all summary data.

Usage examples
  python uzr_inspect.py --ckpt uzr_ckpt.pt --text "나는 루리아입니다." --z-mode init --topk 5 --dump-json report.json
  python uzr_inspect.py --ckpt uzr_ckpt_best.pt --text "Hello, world." --z-file my_z.pt --max-positions 12

Note
- This script expects the UZR sources (model.py, etc.) to be importable either as a package (uzr.*) or
  from the same directory where this file lives.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F

# ---- Robust imports: try package ('uzr.*'), then local files ----
def _import_uzr():
    mod = {}
    try:
        from uzr.model import UZRModel, ByteTokenizer, soft_threshold
        mod["UZRModel"] = UZRModel
        mod["ByteTokenizer"] = ByteTokenizer
        mod["soft_threshold"] = soft_threshold
        # Optional helpers if present
        try:
            from uzr.infer_longrun import avg_embed, init_from_retrieval  # noqa: F401
            mod["avg_embed"] = avg_embed
            mod["init_from_retrieval"] = init_from_retrieval
        except Exception:
            pass
        return mod
    except Exception:
        # try local directory where this script lives
        base = os.path.dirname(os.path.abspath(__file__))
        if base not in sys.path:
            sys.path.append(base)
        from model import UZRModel, ByteTokenizer, soft_threshold  # type: ignore
        mod["UZRModel"] = UZRModel
        mod["ByteTokenizer"] = ByteTokenizer
        mod["soft_threshold"] = soft_threshold
        try:
            from infer_longrun import avg_embed, init_from_retrieval  # noqa: F401
            mod["avg_embed"] = avg_embed
            mod["init_from_retrieval"] = init_from_retrieval
        except Exception:
            pass
        return mod

uzr = _import_uzr()
UZRModel = uzr["UZRModel"]
ByteTokenizer = uzr["ByteTokenizer"]
soft_threshold = uzr.get("soft_threshold")

def human_bytes(n: int) -> str:
    s = n
    for unit in ["", "K", "M", "G"]:
        if s < 1024:
            return f"{s:.0f}{unit}B"
        s /= 1024
    return f"{s:.1f}GB"

def count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    if t is None:
        return {}
    t = t.detach().float().view(-1)
    if t.numel() == 0:
        return {"numel": 0}
    return {
        "numel": t.numel(),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "l2": float(t.norm(2).item()),
        "l1": float(t.abs().sum().item()),
    }

def param_table(model: torch.nn.Module) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for name, p in model.named_parameters():
        prefix = name.split(".")[0] if "." in name else name
        # aggregate per top-level prefix
        if prefix not in seen:
            seen.add(prefix)
        rows.append({
            "param": name,
            "shape": list(p.shape),
            "numel": p.numel(),
            "requires_grad": bool(p.requires_grad),
            "mean": float(p.data.mean().item()),
            "std": float(p.data.std(unbiased=False).item()) if p.numel() > 1 else 0.0,
        })
    return rows

def build_model_from_ckpt(ckpt_path: str, device: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    data = torch.load(ckpt_path, map_location="cpu")
    if "model" in data and "args" in data:
        state = data["model"]
        cfg = data["args"]
    elif isinstance(data, dict):
        # support simple {"model":..., "args":...} saved via train_meta.py or similar
        state = data.get("model", data)
        cfg = data.get("args", {})
    else:
        raise ValueError("Unsupported checkpoint format.")
    # Required fields with fallbacks
    d_model = cfg.get("d_model", 256)
    z_dim = cfg.get("z_dim", 128)
    max_len = cfg.get("max_len", 128)

    tok = ByteTokenizer(max_len=max_len)
    model = UZRModel(tok.vocab_size, d_model=d_model, z_dim=z_dim, max_len=max_len)
    model.load_state_dict(state, strict=False)
    device = torch.device(device)
    model.to(device).eval()
    meta = {
        "cfg": cfg,
        "vocab_size": tok.vocab_size,
        "d_model": d_model,
        "z_dim": z_dim,
        "max_len": max_len,
        "device": str(device),
    }
    return model, tok, meta

def make_z(model: torch.nn.Module, z_mode: str, z_file: str = "", z_dim: int = 128, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if z_file:
        if z_file.endswith(".pt") or z_file.endswith(".pth"):
            z = torch.load(z_file, map_location="cpu")
        elif z_file.endswith(".npy"):
            import numpy as np
            z = torch.from_numpy(np.load(z_file))
        else:
            raise ValueError("Unsupported z_file extension. Use .pt/.pth/.npy")
        z = z.to(device).float().view(-1)
        if z.numel() != z_dim:
            raise ValueError(f"Loaded z has dim {z.numel()} but expected {z_dim}")
        return z
    if z_mode == "zero":
        return torch.zeros(z_dim, device=device)
    if z_mode == "init":
        return model.init_z(batch_size=1).to(device)[0]
    if z_mode == "random":
        return torch.randn(z_dim, device=device)
    raise ValueError(f"Unknown z_mode: {z_mode}")

def film_stats(model: torch.nn.Module, z: torch.Tensor) -> Dict[str, Any]:
    # Compute pre-activation (fc) and post-activation (tanh+1)*h + beta stats
    with torch.no_grad():
        fc_out = model.film.fc(z)  # [2*d_model]
        d = fc_out.shape[0] // 2
        gamma_raw, beta = fc_out[:d], fc_out[d:]
        gamma = torch.tanh(gamma_raw)  # applied as (1 + tanh(gamma_raw))
    return {
        "gamma_raw": tensor_stats(gamma_raw),
        "beta": tensor_stats(beta),
        "gamma_tanh": tensor_stats(gamma),
    }

def capture_encoder_layers(model: torch.nn.Module) -> List[Tuple[str, torch.utils.hooks.RemovableHandle]]:
    # Attach hooks to TransformerEncoder layers
    handles = []
    enc = model.encoder.enc
    # enc.layers is a ModuleList
    for idx, layer in enumerate(enc.layers):
        name = f"encoder.layer.{idx}"
        h = layer.register_forward_hook(lambda m, inp, out, name=name: setattr(model, f"_hook_{name.replace('.', '_')}", out.detach().cpu()))
        handles.append((name, h))
    return handles

def topk_per_position(logits: torch.Tensor, k: int, vocab_size: int) -> List[List[Dict[str, Any]]]:
    # logits: [B, T, V] (B=1)
    probs = F.softmax(logits[0], dim=-1)  # [T, V]
    T = probs.size(0)
    result: List[List[Dict[str, Any]]] = []
    for t in range(T):
        pv, pi = torch.topk(probs[t], k=min(k, vocab_size))
        result.append([{"id": int(pi[i].item()), "p": float(pv[i].item())} for i in range(pv.size(0))])
    return result

def decode_ids(ids: List[int]) -> str:
    # Local tokenizer to decode bytes (exclude BOS/EOS/0 pads). We'll rebuild a tiny decoder matching ByteTokenizer
    BOS, EOS = 256, 257
    ids = [i for i in ids if i not in (BOS, EOS, 0)]
    try:
        return bytes(ids).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def compare_z_variants(model: torch.nn.Module, X: torch.Tensor, z_list: List[torch.Tensor]) -> Dict[str, Any]:
    # Returns pairwise cosine similarity and KL divergences across logits distributions
    with torch.no_grad():
        logits = [model(X, z) for z in z_list]  # list of [1,T,V]
        # flatten per-token logits for cosine
        flats = [l.view(-1) for l in logits]
        cos = []
        for i in range(len(flats)):
            row = []
            for j in range(len(flats)):
                a = flats[i]; b = flats[j]
                val = float(F.cosine_similarity(a, b, dim=0).item())
                row.append(val)
            cos.append(row)
        # Average symmetric KL over positions
        def avg_sym_kl(L1, L2):
            P = F.softmax(L1, dim=-1)
            Q = F.softmax(L2, dim=-1)
            kl_pq = (P * (P.clamp_min(1e-8).log() - Q.clamp_min(1e-8).log())).sum(dim=-1)  # [1,T]
            kl_qp = (Q * (Q.clamp_min(1e-8).log() - P.clamp_min(1e-8).log())).sum(dim=-1)
            sym = 0.5 * (kl_pq + kl_qp)  # [1,T]
            return float(sym.mean().item())
        skl = []
        for i in range(len(logits)):
            row = []
            for j in range(len(logits)):
                row.append(avg_sym_kl(logits[i], logits[j]))
            skl.append(row)
    return {"cosine": cos, "sym_kl": skl}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to UZR checkpoint (*.pt)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--text", default="Hello, UZR.", help="Sample text to run through the model")
    ap.add_argument("--z-mode", default="init", choices=["zero","init","random"], help="How to construct z if --z-file is not given")
    ap.add_argument("--z-file", default="", help="Optional: path to .pt/.pth/.npy containing a z vector")
    ap.add_argument("--topk", type=int, default=5, help="Top-K predictions per token to report")
    ap.add_argument("--max-positions", type=int, default=16, help="Limit positions to print in the console report")
    ap.add_argument("--dump-json", default="", help="If set, write a JSON report to this path")
    args = ap.parse_args()

    model, tok, meta = build_model_from_ckpt(args.ckpt, device=args.device)
    device = torch.device(args.device)

    # ---- Architecture summary ----
    total, trainable = count_params(model)
    arch = {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_human": human_bytes(total * 4),  # rough FP32 size
        "modules": param_table(model),
        "cfg": meta["cfg"],
        "vocab_size": meta["vocab_size"],
        "d_model": meta["d_model"],
        "z_dim": meta["z_dim"],
        "max_len": meta["max_len"],
    }

    # ---- z construction & stats ----
    z = make_z(model, args.z_mode, args.z_file, z_dim=meta["z_dim"], device=device)
    z_stats = tensor_stats(z)
    z0_stats = tensor_stats(model.z0.detach()) if hasattr(model, "z0") else {}

    # ---- FiLM stats ----
    fs = film_stats(model, z)

    # ---- Tokenization & encoder probes ----
    X = torch.stack([tok.encode(args.text)], dim=0).to(device)  # [1, T]
    # Pre-encoder embeddings (tok + pos) for reference
    with torch.no_grad():
        tok_emb = model.encoder.tok(X)
        B, T, D = tok_emb.shape
        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = model.encoder.pos(pos_idx)
        h0 = tok_emb + pos_emb

    # Attach hooks to capture per-layer outputs
    handles = capture_encoder_layers(model)

    # ---- Forward pass ----
    with torch.no_grad():
        logits = model(X, z)  # [1,T,V]
    # Remove hooks
    for _, h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # Collect layer outputs saved on the model attributes
    layer_summaries = []
    # Find attributes _hook_encoder_layer_*
    for k in sorted([k for k in dir(model) if k.startswith("_hook_encoder_layer_")]):
        t = getattr(model, k)
        layer_summaries.append({
            "name": k.replace("_hook_", ""),
            "shape": list(t.shape),
            "stats": tensor_stats(t),
        })

    # ---- Predictions per position ----
    topk = topk_per_position(logits, k=args.topk, vocab_size=meta["vocab_size"])
    ids = logits.argmax(dim=-1)[0].tolist()
    decoded = decode_ids(ids)

    # ---- Compare z variants ----
    z_zero = make_z(model, "zero", "", meta["z_dim"], device)
    z_init = make_z(model, "init", "", meta["z_dim"], device)
    z_rand = make_z(model, "random", "", meta["z_dim"], device)
    z_comp = compare_z_variants(model, X, [z_zero, z_init, z_rand])

    # ---- Build report ----
    report = {
        "checkpoint": os.path.abspath(args.ckpt),
        "device": meta["device"],
        "arch": arch,
        "z_stats": {"chosen": z_stats, "z0_param": z0_stats},
        "film": fs,
        "input": {"text": args.text, "encoded_len": int(X.shape[1])},
        "encoder": {"h0_shape": list(h0.shape), "h0_stats": tensor_stats(h0), "layers": layer_summaries},
        "logits": {"shape": list(logits.shape)},
        "topk_per_pos": topk,
        "decoded_argmax": decoded,
        "z_variant_compare": z_comp,
    }

    # ---- Pretty console output ----
    def print_kv(k, v): print(f"{k:>22}: {v}")
    print("\n=== UZR Inspect Report ===")
    print_kv("ckpt", report["checkpoint"])
    print_kv("device", report["device"])
    print_kv("vocab_size", arch["vocab_size"])
    print_kv("d_model", arch["d_model"])
    print_kv("z_dim", arch["z_dim"])
    print_kv("max_len", arch["max_len"])
    print_kv("params (total/train)", f"{arch['total_params']}/{arch['trainable_params']} (~{arch['total_params_human']})")
    print("\n[z] stats (chosen):", z_stats)
    if z0_stats: print("[z0] parameter stats:", z0_stats)
    print("\n[FiLM] gamma_tanh stats:", fs["gamma_tanh"])
    print("[FiLM] beta stats:", fs["beta"])
    print(f"\n[input] '{args.text}' len={report['input']['encoded_len']} tokens")
    print("[encoder] h0:", report["encoder"]["h0_shape"], report["encoder"]["h0_stats"])
    for layer in report["encoder"]["layers"]:
        print(f"  - {layer['name']} -> {layer['shape']} | mean={layer['stats'].get('mean', 'n/a'):.4f} std={layer['stats'].get('std','nan'):.4f}")
    print(f"\n[logits] shape: {report['logits']['shape']}")
    print("[decode argmax]:", report["decoded_argmax"][:120].replace("\n"," ")+("..." if len(report["decoded_argmax"])>120 else ""))
    # Print first N positions' top-k
    T = len(topk)
    N = min(T, args.max_positions)
    print(f"\n[top-{args.topk}] predictions per position (first {N}/{T} tokens):")
    for t in range(N):
        items = ", ".join([f"{d['id']}:{d['p']:.3f}" for d in topk[t]])
        print(f"  pos {t:02d}: {items}")
    # z variant compare summary
    print("\n[z variants] cosine similarity matrix (zero/init/random):")
    for row in report["z_variant_compare"]["cosine"]:
        print("  ", " ".join(f"{v:+.3f}" for v in row))
    print("[z variants] avg symmetric KL (lower ~ closer):")
    for row in report["z_variant_compare"]["sym_kl"]:
        print("  ", " ".join(f"{v:.4f}" for v in row))

    # ---- Optional JSON dump ----
    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved to: {os.path.abspath(args.dump_json)}")

if __name__ == "__main__":
    main()
