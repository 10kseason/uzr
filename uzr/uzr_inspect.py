#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
uzr_inspect

Checkpoint inspector with automatic dimension detection for UZR.

Usage examples:
  python uzr_inspect.py --ckpt uzr_ckpt_best.pt
  python uzr_inspect.py --ckpt uzr_ckpt_last.pt --json summary.json
  python uzr_inspect.py --ckpt uzr_ckpt_best.pt --text "안녕 Hello" --max_len 128
"""

from __future__ import annotations

import argparse, json, os, sys, pathlib
from typing import Dict, Any, Tuple, Optional

import torch

# Allow running next to a local 'uzr' package folder
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "uzr"))


def _unique_layer_indices(sd_keys):
    idx = set()
    for k in sd_keys:
        # typical: encoder.enc.layers.0.self_attn.in_proj_weight
        if k.startswith("encoder.enc.layers."):
            parts = k.split(".")
            if len(parts) >= 4 and parts[3].isdigit():
                # 'encoder','enc','layers','0', ...
                idx.add(int(parts[3]))
            elif len(parts) >= 3 and parts[2].isdigit():
                # fallback if layout differs
                idx.add(int(parts[2]))
    return sorted(idx)


def detect_dimensions(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def shape(name: str) -> Optional[Tuple[int, ...]]:
        t = state.get(name)
        if isinstance(t, torch.Tensor):
            return tuple(t.shape)
        return None

    # vocab_size, d_model
    ro = shape("readout.weight")
    if ro:
        out["vocab_size"] = ro[0]
        out["d_model"] = ro[1]

    tok_w = shape("encoder.tok.weight")
    if tok_w:
        out.setdefault("vocab_size_tok", tok_w[0])
        out.setdefault("d_model", tok_w[1])

    # max_len
    pos_w = shape("encoder.pos.weight")
    if pos_w:
        out["max_len"] = pos_w[0]

    # z_dim
    z0 = shape("z0")
    if z0:
        out["z_dim"] = z0[0]
    film_w = shape("film.fc.weight")
    if film_w:
        # film.fc: Linear(z_dim -> 2*d_model) => weight [2*d_model, z_dim]
        out.setdefault("z_dim", film_w[1])
        out.setdefault("d_model", film_w[0] // 2)

    # lang embeddings
    lang_w = shape("lang_embed.weight")
    if lang_w:
        out["num_langs"] = lang_w[0]
        out["z_lang_dim"] = lang_w[1]

    # identity_self
    ident = shape("identity_self")
    if ident:
        out["identity_self_dim"] = ident[0]

    # fuse proj
    fuse_w = shape("fuse_proj.weight")
    fuse_in = fuse_w[1] if fuse_w else None
    if fuse_w:
        out.setdefault("z_dim", fuse_w[0])  # Linear(in -> z_dim) => weight [z_dim, in]

    # n_layers (TransformerEncoder layers)
    layer_idx = _unique_layer_indices(state.keys())
    if layer_idx:
        out["n_layers"] = len(set(layer_idx))

    # Try to solve z_think_dim from fuse input if possible
    if fuse_in is not None and all(k in out for k in ("z_dim", "z_lang_dim", "identity_self_dim")):
        R = int(out["z_dim"])          # z_dim
        L = int(out["z_lang_dim"])     # z_lang_dim
        I = int(out["identity_self_dim"])  # identity_self_dim
        F = int(fuse_in)
        # F = 3*R + 3*T + L + I  -> T = (F - 3R - L - I)/3
        num = F - 3 * R - L - I
        if num % 3 == 0 and num >= 0:
            out["z_think_dim"] = num // 3
        else:
            out["z_think_dim"] = None

    return out


def summarize_checkpoint(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)

    # model state
    state: Dict[str, torch.Tensor] = data["model"] if isinstance(data, dict) and "model" in data else data
    dims = detect_dimensions(state)

    # args (if present)
    args = data.get("args", {}) if isinstance(data, dict) else {}

    # tokenizer guess
    vocab = int(dims.get("vocab_size", -1))
    tokenizer = "ByteTokenizer" if vocab == 258 else "KoEnTokenizer"

    # memory summary
    mem_state = data.get("memory") if isinstance(data, dict) else None
    mem_info = None
    if isinstance(mem_state, dict):
        items = mem_state.get("items") or []
        learner = mem_state.get("learner_state") is not None
        ema_loss = mem_state.get("ema_loss", None)
        mem_info = {
            "items": len(items),
            "learner": bool(learner),
            "ema_loss": float(ema_loss) if isinstance(ema_loss, (int, float)) else ema_loss,
            "learn_fields": mem_state.get("learn_fields"),
        }

    # layers index list for reference
    layers_idx = _unique_layer_indices(state.keys())

    return {
        "path": path,
        "exists": os.path.exists(path),
        "tokenizer": tokenizer,
        "dims": dims,
        "args": args,
        "memory": mem_info,
        "layer_indices": layers_idx,
        "keys_total": len(state.keys()),
        "param_tensors": len([k for k in state.keys() if isinstance(state[k], torch.Tensor)]),
    }


def print_summary(obj: Dict[str, Any]):
    print("== UZR Checkpoint Summary ==")
    print(f"path        : {obj['path']}")
    print(f"tokenizer   : {obj['tokenizer']}")
    d = obj.get("dims", {})
    print("-- dimensions --")
    for k in (
        "vocab_size","d_model","max_len","z_dim","z_think_dim","z_lang_dim","num_langs","identity_self_dim","n_layers",
    ):
        if k in d:
            print(f"{k:>15}: {d[k]}")

    if obj.get("memory") is not None:
        m = obj["memory"]
        print("-- memory --")
        print(f"{'items':>15}: {m['items']}")
        print(f"{'learner':>15}: {m['learner']}")
        if m.get("ema_loss") is not None:
            print(f"{'ema_loss':>15}: {m['ema_loss']:.6f}")
        if m.get("learn_fields") is not None:
            print(f"{'learn_fields':>15}: {m['learn_fields']}")

    # Show mismatch hints vs args if available
    a = obj.get("args", {})
    if a:
        hints = []
        for key, dkey in [
            ("d_model","d_model"), ("z_dim","z_dim"), ("max_len","max_len"),
            ("z_think_dim","z_think_dim"), ("z_lang_dim","z_lang_dim"), ("num_langs","num_langs"), ("identity_self_dim","identity_self_dim"),
        ]:
            if dkey in d and key in a and a[key] != d[dkey]:
                hints.append((key, a[key], d[dkey]))
        if hints:
            print("-- config hints (args vs detected) --")
            for k, av, dv in hints:
                print(f"{k:>15}: args={av} detected={dv}")


def maybe_tokenize_sample(obj: Dict[str, Any], text: Optional[str], max_len: Optional[int]):
    if not text:
        return
    dims = obj.get("dims", {})
    vocab = int(dims.get("vocab_size", -1))
    # Lazy import to avoid dependency if not needed
    from uzr.model import ByteTokenizer, KoEnTokenizer  # type: ignore
    tok = ByteTokenizer(max_len=max_len or dims.get("max_len", 128)) if vocab == 258 else KoEnTokenizer(max_len=max_len or dims.get("max_len", 128))
    ids = tok.encode(text)
    rec = tok.decode(ids)
    print("-- sample encode/decode --")
    print(f"input  : {text}")
    print(f"ids(n) : {len(ids)}  first16={ids[:16].tolist() if hasattr(ids,'tolist') else str(ids)[:64]}")
    print(f"decode : {rec}")


def main():
    ap = argparse.ArgumentParser(description="Inspect UZR checkpoints with automatic dimension detection")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--json", default="", help="Optional path to write JSON summary")
    ap.add_argument("--text", default="", help="Optional sample text to tokenize/decode")
    ap.add_argument("--max_len", type=int, default=0, help="Override tokenizer max_len for sample")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    summary = summarize_checkpoint(args.ckpt)
    print_summary(summary)
    maybe_tokenize_sample(summary, args.text, args.max_len if args.max_len > 0 else None)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[ok] wrote JSON: {args.json}")


if __name__ == "__main__":
    main()

