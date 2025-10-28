#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UZR OOD Predictor — uzr-style loader (matches infer_longrun_standalone_logged.py)

- Loads ckpt: torch.load(...) → cfg = data["args"] → ByteTokenizer(cfg["max_len"]) → UZRModel(...).load_state_dict(data["model"])
- Adds script_dir and script_dir/uzr to sys.path so a local 'uzr' package folder works without install
- Reads OOD CSV (columns: input, lang), predicts with zero-initialized z_rule/z_think and provided lang_id
- Writes CSV: input,pred

Usage (Windows example):
  python uzr_ood_predict_uzrstyle.py ^
    --ckpt uzr_ckpt_best.pt ^
    --in ood_tests.csv ^
    --out ood_preds_uzr.csv ^
    --device cuda ^
    --batch_size 64

Notes:
- This is a minimal forward-only predictor (no inner-loop adaptation/memory). It mirrors the *loading*
  pattern of your longrun script and uses greedy argmax decoding via tokenizer.decode().
"""
import argparse, csv, sys, pathlib
from typing import List

import torch

# Allow running next to a local 'uzr' package folder (same as your longrun script)
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "uzr"))

from uzr.model import UZRModel, ByteTokenizer, KoEnTokenizer  # type: ignore

LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}

def resolve_lang_idx(lang_key: str, num_langs: int) -> int:
    idx = LANG2ID.get(lang_key, LANG2ID["base"])
    if idx >= num_langs:
        return LANG2ID["base"]
    return idx

def encode_batch(tok: ByteTokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
    X = torch.stack([tok.encode(t) for t in texts], dim=0).to(device)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="CSV with at least 'input' column (and optional 'lang')")
    ap.add_argument("--out", required=True, help="CSV to write predictions: input,pred")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # ---- Load checkpoint exactly like your longrun script ----
    data = torch.load(args.ckpt, map_location="cpu")
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
        num_langs=cfg.get("num_langs", len(LANG2ID)),
    )
    model.load_state_dict(data["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    # ---- Read inputs ----
    rows = []
    with open(args.inp, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)

    inputs = [r["input"] for r in rows]
    langs  = [r.get("lang", "base") for r in rows]

    # ---- Predict in batches (zero-initialized z vectors, lang from CSV or 'base') ----
    preds_all: List[str] = []
    B = max(1, int(args.batch_size))
    with torch.no_grad():
        for i in range(0, len(inputs), B):
            batch_inputs = inputs[i:i+B]
            batch_langs  = langs[i:i+B]
            X = encode_batch(tok, batch_inputs, device)

            # z init (zeros) like your longrun's starting point
            z_rule = model.init_z(batch_size=X.shape[0]).to(device)
            z_think = model.init_z_thinking(batch_size=X.shape[0]).to(device)

            # Per-item lang ids
            lang_ids = [resolve_lang_idx(lk, model.num_langs) for lk in batch_langs]

            # Forward & greedy argmax
            logits = model(X, {"rule": z_rule, "think": z_think, "lang_id": lang_ids})
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu()

            # Decode each row
            for j in range(pred_ids.shape[0]):
                try:
                    text = tok.decode(pred_ids[j])
                except Exception:
                    text = " ".join(map(str, pred_ids[j].tolist()))
                preds_all.append(text)

    # ---- Write predictions ----
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["input", "pred"])
        for t, p in zip(inputs, preds_all):
            wr.writerow([t, p])

    print(f"[OK] wrote predictions: {args.out}  (n={len(preds_all)})")

if __name__ == "__main__":
    main()
