#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UZR meta-train (logged, with generalization losses) — single file
-----------------------------------------------------------------
- Adds InfoNCE(semantic) + cross-lingual transfer losses
- Supports inner thinking loop hyperparams
- CSV logging every N steps (default: 50)
- Optional lambda annealing and spike-skip logic

This script expects the same environment as your previous train_meta:
- UZRModel (encoder + forward returning {'loss', ... or 'loss_main', 'z_lang', 'rule_emb'})
- make_identity_batch, tokenizer `tok`, task sampling, etc.
Adapt the import section to your project if names differ.
"""

import os, sys, math, csv, time, random, contextlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------
# Project-specific imports
# -------------------------
# Adjust these to match your codebase. The names below reflect common patterns
try:
    from uzr.model import UZRModel  # your model class
except Exception:
    # Fallback: if model is defined elsewhere
    from uzr import UZRModel  # type: ignore

try:
    from uzr.data import make_identity_batch, load_tokenizer  # typical util names
except Exception:
    # Provide light shims or expect caller to replace
    def make_identity_batch(tok, identity, batch_lang="en", n=4):
        raise RuntimeError("Please import your project's make_identity_batch")
    def load_tokenizer():
        raise RuntimeError("Please import your project's load_tokenizer")

# -------------------------
# Argparse
# -------------------------
import argparse
def build_parser():
    ap = argparse.ArgumentParser("UZR Meta-Train (logged)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--identity", type=str, default="루리아")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # Inner thinking
    ap.add_argument("--inner_steps", type=int, default=8)
    ap.add_argument("--inner_eta", type=float, default=0.38)
    ap.add_argument("--id_weight", type=float, default=0.15)
    ap.add_argument("--lam_think", type=float, default=None,
                    help="L1 regularization strength for z_thinking (optional)")

    # Generalization auxiliaries
    ap.add_argument("--lam_sem", type=float, default=5e-4,
                    help="weight for language-agnostic semantic InfoNCE loss")
    ap.add_argument("--lam_transfer", type=float, default=7e-4,
                    help="weight for cross-language cosine alignment loss")
    ap.add_argument("--aux_baux", type=int, default=4,
                    help="aux mini-batch size per language for semantic alignment")
    ap.add_argument("--sem_dim", type=int, default=256,
                    help="shared semantic projection dim")

    # Logging
    ap.add_argument("--log_file", type=str, default="train_log.csv",
                    help="per-STEP CSV log path")
    ap.add_argument("--log_every", type=int, default=50,
                    help="log every N steps")
    ap.add_argument("--print_every", type=int, default=50,
                    help="print every N steps")

    # Lambda annealing (optional)
    ap.add_argument("--anneal_from", type=int, default=1200)
    ap.add_argument("--anneal_to", type=int, default=3000)
    ap.add_argument("--anneal_floor", type=float, default=0.3)

    # Spike handling
    ap.add_argument("--spike_jump", type=float, default=0.6,
                    help="skip update if loss > ema + spike_jump")

    return ap

# -------------------------
# Utils
# -------------------------
def set_deterministic(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("high")

@contextlib.contextmanager
def maybe_autocast(use_amp: bool, device: str):
    if use_amp and device == "cuda":
        from torch import amp
        with amp.autocast("cuda"):
            yield
    else:
        yield

def _init_csv_log(path, fieldnames):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        w.writeheader()
    return f, w

def _safe(v, default=float("nan")):
    try:
        if v is None: return default
        if isinstance(v, (float, int)): return v
        if isinstance(v, torch.Tensor):
            v = v.detach()
            return float(v.mean().item())
        return float(v)
    except Exception:
        return default

def cosine_warmup(step:int, total:int, warmup:int, base_lr:float)->float:
    if step < warmup:
        return base_lr * (step+1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))

def anneal_lambda(lmbd: float, step: int, s0: int, s1: int, floor: float)->float:
    if step < s0: return lmbd
    t = min(1.0, max(0.0, (step - s0) / max(1, s1 - s0)))
    return lmbd * (1 - t*(1 - floor))

# -------------------------
# Concept helpers (language-agnostic projection)
# -------------------------
class ConceptBank(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.vocab_size = vocab_size
        self.dim = dim

    @staticmethod
    def _hash_bytes(bs: torch.Tensor, mod: int, nmin: int = 2, nmax: int = 5):
        ints = bs.tolist()
        MOD = 2**61 - 1
        base = 257
        idx = []
        for n in range(nmin, nmax+1):
            h = 0
            p = pow(base, n-1, MOD)
            for i, ch in enumerate(ints):
                ch = int(ch)
                if i < n:
                    h = (h*base + ch + 1) % MOD
                    if i == n-1:
                        idx.append(h % mod)
                else:
                    h = (h - (int(ints[i-n]) + 1) * p) % MOD
                    h = (h*base + ch + 1) % MOD
                    idx.append(h % mod)
        if not idx: idx = [0]
        return idx

    def forward(self, x_bytes: torch.Tensor) -> torch.Tensor:
        B, T = x_bytes.shape
        device = x_bytes.device
        outs = []
        for b in range(B):
            ids = self._hash_bytes(x_bytes[b], self.vocab_size)
            ids = torch.tensor(ids, device=device, dtype=torch.long)
            em = self.emb(ids).mean(dim=0)
            outs.append(em)
        out = torch.stack(outs, dim=0)
        return F.normalize(out, dim=-1)

def info_nce(q: torch.Tensor, k: torch.Tensor, tau: float = 0.07)->torch.Tensor:
    logits = (q @ k.t()) / tau
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)

# -------------------------
# Main training
# -------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    set_deterministic(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tok = load_tokenizer()

    model = UZRModel()
    model.to(device)

    # Expose inner thinking hyperparams if available
    if hasattr(model, "inner_steps"):
        model.inner_steps = args.inner_steps
    if hasattr(model, "inner_eta"):
        model.inner_eta = args.inner_eta

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Concept projection params
    _CONCEPT_VOCAB = 65536
    concept_bank = ConceptBank(_CONCEPT_VOCAB, args.sem_dim).to(device)
    W_enc2sem = nn.Parameter(torch.empty(model.d_model if hasattr(model, "d_model") else 768, args.sem_dim, device=device))
    W_con2sem = nn.Parameter(torch.empty(args.sem_dim, args.sem_dim, device=device))
    nn.init.xavier_uniform_(W_enc2sem); nn.init.xavier_uniform_(W_con2sem)
    opt.add_param_group({"params": [W_enc2sem, W_con2sem, concept_bank.parameters()], "lr": args.lr})

    # Logging
    fieldnames = ["time", "step", "loss", "ema", "lr", "loss_sem", "loss_xfer", "grad_norm", "task"]
    csv_f, csv_w = _init_csv_log(args.log_file, fieldnames)
    start_wall = time.time()

    ema = 0.0
    grad_clip = 1.0

    def avg_embed(x_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = model.encoder(x_tokens)           # (B,T,d_model)
            return h.mean(dim=1)                  # (B,d_model)

    def proj_sem(x: torch.Tensor, W: torch.Tensor)->torch.Tensor:
        return F.normalize(x @ W, dim=-1)

    def step_task():
        # Replace with your task sampling logic; here we make an identity batch for demo
        X, Y = make_identity_batch(tok, args.identity, batch_lang="en", n=8)
        X = X.to(device)
        Y = Y.to(device) if torch.is_tensor(Y) else None
        return {"input_ids": X, "labels": Y, "bytes": X}

    # Training loop
    for step in range(args.steps):
        # LR schedule
        lr = cosine_warmup(step, args.steps, args.warmup_steps, args.lr) if args.cosine else args.lr
        for g in opt.param_groups:
            g["lr"] = lr

        batch = step_task()
        task = "meta"

        x = batch["input_ids"]
        y = batch["labels"]
        x_bytes = batch.get("bytes", x)

        # Forward
        with maybe_autocast(args.amp, device.type):
            out = model(input_ids=x, labels=y)
            # expected: out['loss'] or ('loss_main','z_lang','rule_emb')
            if "loss" in out:
                loss_main = out["loss"]
                z_lang = out.get("z_lang", None)
                rule_emb = out.get("rule_emb", None)
            else:
                loss_main = out["loss_main"]
                z_lang = out["z_lang"]
                rule_emb = out.get("rule_emb", None)

            total = loss_main + args.id_weight * 0.0  # keep hook consistent; set to 0 if no id_loss

            # Aux generalization
            Baux = args.aux_baux
            loss_sem = torch.tensor(0.0, device=device)
            loss_xfer = torch.tensor(0.0, device=device)
            if Baux > 0:
                # three langs for alignment
                Xko, _ = make_identity_batch(tok, args.identity, batch_lang="ko", n=Baux)
                Xen, _ = make_identity_batch(tok, args.identity, batch_lang="en", n=Baux)
                Xja, _ = make_identity_batch(tok, args.identity, batch_lang="ja", n=Baux)
                Xko, Xen, Xja = Xko.to(device), Xen.to(device), Xja.to(device)

                E_ko = avg_embed(Xko); E_en = avg_embed(Xen); E_ja = avg_embed(Xja)
                C_ko = concept_bank(Xko); C_en = concept_bank(Xen); C_ja = concept_bank(Xja)

                S_ko = proj_sem(E_ko, W_enc2sem); S_en = proj_sem(E_en, W_enc2sem); S_ja = proj_sem(E_ja, W_enc2sem)
                K_ko = proj_sem(C_ko, W_con2sem); K_en = proj_sem(C_en, W_con2sem); K_ja = proj_sem(C_ja, W_con2sem)

                loss_sem  = (info_nce(S_ko, K_ko) + info_nce(S_en, K_en) + info_nce(S_ja, K_ja)) / 3.0
                loss_xfer = ((1 - F.cosine_similarity(S_ko, S_en, dim=-1)).mean() +
                             (1 - F.cosine_similarity(S_ko, S_ja, dim=-1)).mean() +
                             (1 - F.cosine_similarity(S_en, S_ja, dim=-1)).mean()) / 3.0

                # anneal lambdas (optional)
                lam_sem_t = anneal_lambda(args.lam_sem, step, args.anneal_from, args.anneal_to, args.anneal_floor)
                lam_xfer_t = anneal_lambda(args.lam_transfer, step, args.anneal_from, args.anneal_to, args.anneal_floor)

                total = total + lam_sem_t * loss_sem + lam_xfer_t * loss_xfer

        # Spike guard
        if (ema > 0) and (float(total.detach().cpu()) > (ema + args.spike_jump)):
            # skip this update
            for g in opt.param_groups: g["lr"] *= 0.8
            continue

        # Backward
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        # EMA (simple)
        val = float(total.detach().cpu())
        if step == 0: ema = val
        else: ema = 0.97 * ema + 0.03 * val

        # --- per-N logging ---
        if (step % args.log_every) == 0:
            try:
                lr_out = opt.param_groups[0]["lr"]
            except Exception:
                lr_out = float("nan")
            row = {
                "time": round(time.time() - start_wall, 3),
                "step": int(step),
                "loss": _safe(total),
                "ema":  round(ema, 6),
                "lr":   _safe(lr_out),
                "loss_sem": _safe(loss_sem),
                "loss_xfer": _safe(loss_xfer),
                "grad_norm": float("nan"),
                "task": task,
            }
            csv_w.writerow(row); csv_f.flush()
            if (step % args.print_every) == 0:
                print(f"[log/{step}] loss={row['loss']:.3f} ema={row['ema']:.3f} lr={row['lr']:.3e} sem={row['loss_sem']:.3f} xfer={row['loss_xfer']:.3f}")

    try: csv_f.close()
    except Exception: pass

if __name__ == "__main__":
    main()
