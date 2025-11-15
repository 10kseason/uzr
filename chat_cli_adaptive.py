
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import time
import json
import datetime
import os
from typing import Dict, Tuple, List, Any

import torch
import torch.nn.functional as F

"""
Import model/tokenizer with robust fallbacks.
Prefer uzr.model for both UZRModel and ByteTokenizer to ensure interface compatibility.
"""
# --- Optional local imports; fallbacks are handled below ---
try:
    from uzr.model import UZRModel  # Your project model
except Exception as e:
    print("[fatal] Could not import UZRModel from uzr.model:", e, file=sys.stderr)
    raise

# Try to import ByteTokenizer and KoEnTokenizer from uzr.model first, then other common locations.
CKPT_VOCAB_GUESS = None
ByteTokenizer = None
KoEnTokenizer = None
try:
    from uzr.model import ByteTokenizer as _UZRByteTokenizer  # preferred
    from uzr.model import KoEnTokenizer as _UZRKoEnTokenizer
    ByteTokenizer = _UZRByteTokenizer
    KoEnTokenizer = _UZRKoEnTokenizer
except Exception:
    pass

if ByteTokenizer is None:
    for _imp in ("uzr.tokenizer", "uzr.data", "tokenizer"):
        try:
            mod = __import__(_imp, fromlist=["ByteTokenizer"])
            if hasattr(mod, "ByteTokenizer"):
                ByteTokenizer = getattr(mod, "ByteTokenizer")
                break
        except Exception:
            pass

if ByteTokenizer is None:
    print("[warn] ByteTokenizer not found; defining a smarter fallback (byte-level with optional specials).", file=sys.stderr)
    class ByteTokenizer:
        def __init__(self, max_len: int = 2048, vocab_size: int = 256):
            # If ckpt suggests a vocab, use it
            vs = globals().get("CKPT_VOCAB_GUESS", None)
            if isinstance(vs, int) and vs >= 256:
                vocab_size = vs
            self.max_len = max_len
            self.vocab_size = vocab_size
            # Reserve specials when vocab_size > 256
            self._use_specials = self.vocab_size > 256
            # Define BOS/EOS ids as last two tokens when available
            self.BOS = self.vocab_size - 2 if self._use_specials else None
            self.EOS = self.vocab_size - 1 if self._use_specials else None

        def _override_vocab(self, new_vs: int):
            # Allow runtime override from ckpt inspection
            if new_vs >= 256:
                self.vocab_size = new_vs
                self._use_specials = self.vocab_size > 256
                self.BOS = self.vocab_size - 2 if self._use_specials else None
                self.EOS = self.vocab_size - 1 if self._use_specials else None

        def encode(self, s: str, add_special: bool = False) -> List[int]:
            b = s.encode("utf-8", errors="ignore")[: self.max_len]
            ids = list(b)  # 0..255
            if add_special and self._use_specials and self.BOS is not None:
                return [self.BOS] + ids + ([self.EOS] if self.EOS is not None else [])
            return ids

        def decode(self, ids: List[int]) -> str:
            # Drop specials when present
            if self._use_specials:
                ids = [i for i in ids if i < 256]
            b = bytes([i % 256 for i in ids])
            return b.decode("utf-8", errors="ignore")


def pick_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    if name in ("cuda", "cuda:0"):
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available; falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("[warn] MPS requested but not available; falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device(name)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    data = torch.load(path, map_location=device)
    if "model" not in data:
        raise RuntimeError("Checkpoint missing 'model' state_dict.")
    return data


def build_model_from_cfg(tok, cfg: Dict[str, Any]) -> Any:
    """
    Instantiate UZRModel using ckpt args when possible.
    Falls back to minimal required args if keys are missing.
    """
    import inspect
    sig = inspect.signature(UZRModel)
    params = set(sig.parameters.keys())

    base = dict(
        vocab_size=getattr(tok, "vocab_size", cfg.get("vocab_size", 256)),
        d_model=cfg.get("d_model", 1024),
        z_dim=cfg.get("z_dim", 512),
        max_len=cfg.get("max_len", 2048),
    )
    # Merge any optional args present in the checkpoint cfg that match constructor
    extra = {k: v for k, v in cfg.items() if k in params and k not in base}
    # If tokenizer-driven sizes exist in cfg (e.g., lang_dim), they will be picked via 'extra'
    model = UZRModel(**base, **extra)
    return model


def import_symbol(mod_path: str, name: str):
    mod = __import__(mod_path, fromlist=[name])
    if not hasattr(mod, name):
        raise ImportError(f"Module {mod_path} has no symbol {name}")
    return getattr(mod, name)


def parse_kwargs(s: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if not s:
        return d
    parts = [p for p in s.split(",") if p.strip()]
    for kv in parts:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip(); v = v.strip()
        vl = v.lower()
        if vl in ("true", "false"):
            d[k] = (vl == "true")
            continue
        try:
            if v.startswith("0x"):
                d[k] = int(v, 16)
            elif "." in v:
                d[k] = float(v)
            else:
                d[k] = int(v)
        except Exception:
            d[k] = v
    return d


def adapt_state_dict(sd: Dict[str, torch.Tensor],
                     msd: Dict[str, torch.Tensor],
                     prefer_tail: bool = False) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]:
    """
    Given source state_dict (sd) and model's target (msd), crop/pad tensors to match shapes.
    Returns (adapted_sd, changes_report).

    Strategy:
      - For every key present in model.state_dict():
        - If key found in sd with identical shape -> copy as-is.
        - If shapes differ but ranks equal -> create zeros_like(target), copy the overlapping slice.
          Overlap is computed as min along each dimension.
        - If key missing in sd -> leave it to model default (not overriding), load with strict=False.

      - This is deterministic and safe; performance may differ from exact match but works robustly.
    """
    adapted = dict(sd)  # start from original
    changes = []

    for name, target in msd.items():
        if name not in sd:
            continue
        src = sd[name]
        if src.shape == target.shape:
            continue

        # Only handle matching ranks; otherwise fall back to zeros_like target
        if src.dim() != target.dim():
            new_param = torch.zeros_like(target)
            adapted[name] = new_param
            changes.append((name, tuple(src.shape), tuple(target.shape)))
            continue

        # Build slices for overlap copy
        overlap = [min(a, b) for a, b in zip(src.shape, target.shape)]
        src_slices = []
        tgt_slices = []
        for i, (osz, tsz, olp) in enumerate(zip(src.shape, target.shape, overlap)):
            if prefer_tail:
                src_start = max(0, osz - olp)
                tgt_start = max(0, tsz - olp)
            else:
                src_start = 0
                tgt_start = 0
            src_slices.append(slice(src_start, src_start + olp))
            tgt_slices.append(slice(tgt_start, tgt_start + olp))

        new_param = torch.zeros_like(target)
        new_param[tuple(tgt_slices)] = src[tuple(src_slices)]
        adapted[name] = new_param
        changes.append((name, tuple(src.shape), tuple(target.shape)))

    return adapted, changes


@torch.inference_mode()
def _tok_encode(tok, s: str, add_special: bool = False):
    """Call tokenizer.encode with a tolerant signature and return a 1D LongTensor."""
    try:
        x = tok.encode(s, add_special=add_special)
    except TypeError:
        x = tok.encode(s)
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            x = x.view(1)
        return x.to(dtype=torch.long)
    return torch.tensor(x, dtype=torch.long)


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95, device: torch.device = torch.device("cpu"), add_bos: bool=False, eos_stop: bool=False, z: Any = None, include_meta: bool = False):
    """
    Minimal autoregressive generator assuming model(input_ids) -> logits.
    Tokenizer must provide encode/decode for integer ids.
    """
    model.eval()
    x = _tok_encode(tok, prompt, add_special=add_bos)
    # Convert possible fixed-length padded encoding to variable-length prefix
    max_len = int(getattr(tok, "max_len", x.numel() if isinstance(x, torch.Tensor) else 2048))
    if isinstance(x, torch.Tensor):
        arr = x.view(-1).tolist()
    else:
        arr = list(x)
    # Determine effective length: up to EOS if present; otherwise trim trailing zeros
    eos = getattr(tok, "EOS", None)
    t_eff = None
    if eos is not None:
        try:
            t_eff = arr.index(int(eos)) + 1  # position just after EOS
        except ValueError:
            t_eff = None
    if t_eff is None:
        # rstrip zeros
        t_eff = len(arr)
        while t_eff > 0 and arr[t_eff - 1] == 0:
            t_eff -= 1
        if t_eff == 0:
            t_eff = min(1, len(arr))
    arr = arr[:t_eff]
    # If trailing EOS exists, drop it so generation can continue beyond it
    if eos is not None and len(arr) > 0 and arr[-1] == int(eos):
        arr = arr[:-1]
    # If already at or beyond max_len, keep the tail and leave room for one token
    if len(arr) >= max_len:
        keep = max_len - 1
        if keep <= 0:
            keep = max_len
        arr = arr[-keep:]
    ids = torch.tensor([arr], dtype=torch.long, device=device)
    # default z (rule channel) if not provided
    if z is None:
        try:
            z = model.init_z(batch_size=1).to(device)[0]
        except Exception:
            z = torch.zeros(getattr(model, "z_dim", 128), device=device)
    start_len = int(ids.size(1))
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        # avoid exceeding positional embedding length
        if ids.size(1) >= max_len:
            stop_reason = "length"
            break
        logits = model(ids, z)  # expected shape [1, T, vocab]
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        next_logits = logits[:, -1, :]  # [1, V]
        if temperature > 0:
            next_logits = next_logits / max(temperature, 1e-6)
            # top-p (nucleus) sampling
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            # ensure at least one token kept
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(sorted_probs, num_samples=1)  # [1,1]
            next_token = sorted_idx.gather(-1, next_id)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1,1]

        ids = torch.cat([ids, next_token], dim=1)
        if eos_stop and hasattr(tok, "EOS") and tok.EOS is not None:
            if int(next_token.item()) == int(tok.EOS):
                stop_reason = "eos"
                break
    # Robust decode (accept list or tensor)
    sample = ids[0]
    sample_list = sample.tolist() if isinstance(sample, torch.Tensor) else sample
    try:
        out = tok.decode(sample_list)
    except TypeError:
        out = tok.decode(sample_list)
    if include_meta:
        meta = {
            "start_len": start_len,
            "end_len": int(ids.size(1)),
            "generated": int(ids.size(1) - start_len),
            "max_len": int(max_len),
            "stop_reason": stop_reason,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "device": str(device),
        }
        return out, meta
    return out


def main():
    ap = argparse.ArgumentParser(description="Adaptive UZR Chat CLI — loads ckpt with shape adaptation")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g., uzr_ckpt_best.pt)")
    ap.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto (default: auto)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--prefer_tail", action="store_true", help="When cropping, take the tail slice instead of the head")
    ap.add_argument("--dry_run", action="store_true", help="Only load and report adaptations, then exit")
    ap.add_argument("--prewarm", type=int, default=0, help="Generate N burn-in tokens from system prompt before first turn")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    ap.add_argument("--system", type=str, default="You are an assistant.", help="Optional system prompt prefix")
    # Tokenizer injection options
    ap.add_argument("--tok_mod", type=str, default="", help="Optional: module path to import tokenizer class from (e.g., mypkg.mytok)")
    ap.add_argument("--tok_factory", type=str, default="ByteTokenizer", help="Class or factory name in --tok_mod (default: ByteTokenizer)")
    ap.add_argument("--tok_kwargs", type=str, default="", help="Optional kwargs for tokenizer factory, k=v comma-separated (e.g., max_len=256)")
    ap.add_argument("--tok_py", type=str, default="", help="Optional: path to a .py file containing the tokenizer class; module name inferred from filename")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[info] device = {device}")

    # Logging (JSONL), enabled by default
    log_path = "chat_cli_log.jsonl"
    try:
        log_fp = open(log_path, "a", encoding="utf-8")
    except Exception:
        log_fp = None
    def log_event(event: str, payload: Dict[str, Any]):
        if not log_fp:
            return
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "event": event}
        rec.update(payload)
        try:
            log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_fp.flush()
        except Exception:
            pass

    data = load_checkpoint(args.ckpt, device)
    cfg = data.get("args", {})
    print("[info] ckpt keys:", list(data.keys()))
    print("[info] ckpt args:", {k: cfg[k] for k in list(cfg.keys())[:32]})
    log_event("startup", {
        "device": str(device),
        "ckpt": args.ckpt,
        "ckpt_keys": list(data.keys()),
        "ckpt_args_head": {k: cfg[k] for k in list(cfg.keys())[:32]},
    })

    # Tokenizer (injectable)
    max_len = int(cfg.get("max_len", 2048))
    tok = None
    injected = False
    if args.tok_py:
        p = os.path.abspath(args.tok_py)
        d, fname = os.path.dirname(p), os.path.basename(p)
        mod_name = os.path.splitext(fname)[0]
        if d not in sys.path:
            sys.path.append(d)
        try:
            Tok = import_symbol(mod_name, args.tok_factory)
            kw = parse_kwargs(args.tok_kwargs)
            kw.setdefault("max_len", max_len)
            tok = Tok(**kw) if callable(Tok) else Tok
            injected = True
        except Exception as e:
            print(f"[warn] tok_py injection failed: {e}; falling back.", file=sys.stderr)
    if tok is None and args.tok_mod:
        try:
            Tok = import_symbol(args.tok_mod, args.tok_factory)
            kw = parse_kwargs(args.tok_kwargs)
            kw.setdefault("max_len", max_len)
            tok = Tok(**kw) if callable(Tok) else Tok
            injected = True
        except Exception as e:
            print(f"[warn] tok_mod injection failed: {e}; falling back.", file=sys.stderr)
    if tok is None:
        # Auto-detect from CKPT_VOCAB_GUESS or default to KoEnTokenizer
        if CKPT_VOCAB_GUESS == 258 and ByteTokenizer is not None:
            tok = ByteTokenizer(max_len=max_len)
        elif KoEnTokenizer is not None:
            tok = KoEnTokenizer(max_len=max_len)
        elif ByteTokenizer is not None:
            tok = ByteTokenizer(max_len=max_len)
        else:
            raise RuntimeError("No tokenizer available")
    print(f"[info] tokenizer: {type(tok).__name__}, vocab_size={getattr(tok, 'vocab_size', 'NA')}, max_len={getattr(tok, 'max_len', max_len)}")
    log_event("tokenizer", {
        "name": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "max_len": getattr(tok, "max_len", max_len),
        "injected": injected,
        "tok_mod": args.tok_mod or (os.path.splitext(os.path.basename(args.tok_py))[0] if args.tok_py else None),
        "tok_factory": args.tok_factory,
        "tok_kwargs": parse_kwargs(args.tok_kwargs),
    })

    # Model init (try to respect ckpt args via introspection)
    model = build_model_from_cfg(tok, cfg)
    model.to(device)

    # Shape adaptation + load
    sd_src = data["model"]
    msd = model.state_dict()
    adapted_sd, changes = adapt_state_dict(sd_src, msd, prefer_tail=args.prefer_tail)

    if changes:
        print("[info] Adapted tensors (src_shape -> dst_shape):")
        for name, sshape, dshape in changes:
            print(f"  - {name}: {sshape} -> {dshape}")
        log_event("adapt", {"count": len(changes), "items": [{"name": n, "src": s, "dst": d} for (n,s,d) in changes[:50]]})
    else:
        print("[info] No shape adaptations required.")
        log_event("adapt", {"count": 0})

    missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
        for k in missing[:30]:
            print("   ·", k)
        if len(missing) > 30: print("   · ...")
        log_event("load_state", {"missing": len(missing)})
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
        for k in unexpected[:30]:
            print("   ·", k)
        if len(unexpected) > 30: print("   · ...")
        log_event("load_state", {"unexpected": len(unexpected)})

    print("[info] model loaded.")
    log_event("ready", {"model": "UZRModel"})

    if args.dry_run:
        print("[info] dry_run complete. Exiting.")
        return

    model.eval()
    # Prepare a persistent z state (simple constant rule vector)
    try:
        z_const = model.init_z(batch_size=1).to(device)[0]
    except Exception:
        z_const = torch.zeros(getattr(model, "z_dim", 128), device=device)

    # Optional prewarm (burn-in tokens) from system prompt only
    if args.prewarm > 0:
        _, meta = generate(model, tok, prompt=args.system, max_new_tokens=args.prewarm, temperature=0.0, top_p=1.0, device=device, add_bos=True, eos_stop=False, z=z_const, include_meta=True)
        print(f"[info] prewarm complete: {args.prewarm} tokens")
        log_event("prewarm", meta)

    # Chat loop
    print("===========================================================")
    print(" Adaptive UZR Chat — type your message and press Enter")
    print("   • Ctrl+C or empty line to exit")
    print("===========================================================")

    history = []
    try:
        while True:
            try:
                user = input("\nYou: ").strip()
            except EOFError:
                break
            if not user:
                break
            log_event("user", {"text": user, "len": len(user)})
            prompt = args.system + "\n"
            for u, a in history:
                prompt += f"User: {u}\nAssistant: {a}\n"
            prompt += f"User: {user}\nAssistant:"

            try:
                out, meta = generate(
                model,
                tok,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                z=z_const,
                include_meta=True,
                )
            except Exception as e:
                log_event("error", {"where": "generate", "msg": str(e)})
                raise
            # Heuristic: take text after the last "Assistant:" to the end.
            cut = out.rfind("Assistant:")
            reply = out[cut + len("Assistant:") :].strip() if cut != -1 else out
            print("\nAssistant:", reply)
            history.append((user, reply))
            meta_update = dict(meta)
            meta_update.update({"reply_len": len(reply)})
            log_event("assistant", meta_update)
            # Keep context length in check
            if len(history) > 20:
                history = history[-20:]
    except KeyboardInterrupt:
        print("\n[info] bye.")
        log_event("exit", {"reason": "keyboard"})
        if log_fp:
            try:
                log_fp.close()
            except Exception:
                pass
        return


if __name__ == "__main__":
    main()
