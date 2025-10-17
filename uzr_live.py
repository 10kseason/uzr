#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UZR Live — real-time console dashboard for early UZR models.

Update (chat mode):
- Toggle **chat typing** with key **[t]**.
- While chat mode is ON, typed characters are captured into a line buffer.
  - [Enter] to commit the line into the running transcript.
  - [Backspace] edits the line; [Esc] cancels current line.
- On commit:
  - The line is appended to the base text (with optional prefix/suffix).
  - The dashboard recomputes on the new context.
  - If --log-jsonl is set, an event with event="chat_commit" is appended.

Other hotkeys remain the same:
  [0]=z_zero  [i]=z_init  [r]=z_random  [g]=toggle gen(loop)  [s]=step 1
  [c]=clear tail  [q]=quit  [t]=chat mode on/off

Usage
  python uzr_live.py --ckpt uzr_ckpt.pt --text "Hello UZR." --interval 0.5
  python uzr_live.py --ckpt uzr_ckpt.pt --text-file live.txt --interval 0.25 --topk 8 --device cuda
  python uzr_live.py --ckpt uzr_ckpt.pt --text "..." --log-jsonl live_events.jsonl --chat-prefix "USER: " --chat-suffix "\\n"
"""

import argparse
import os
import sys
import time
import json
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

# ---------------------------- Robust import of UZR ----------------------------
def _import_uzr():
    mod = {}
    try:
        from uzr.model import UZRModel, ByteTokenizer  # noqa: F401
        mod["UZRModel"] = UZRModel
        mod["ByteTokenizer"] = ByteTokenizer
        try:
            from uzr.infer_longrun import avg_embed, init_from_retrieval  # noqa: F401
            mod["avg_embed"] = avg_embed
            mod["init_from_retrieval"] = init_from_retrieval
        except Exception:
            pass
        return mod
    except Exception:
        base = os.path.dirname(os.path.abspath(__file__))
        if base not in sys.path:
            sys.path.append(base)
        from model import UZRModel, ByteTokenizer  # type: ignore
        mod["UZRModel"] = UZRModel
        mod["ByteTokenizer"] = ByteTokenizer
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

# ---------------------------- Helpers ----------------------------------------
def clear_screen():
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def human(n: float) -> str:
    try:
        return f"{float(n):.4f}"
    except Exception:
        return "nan"

def tensor_stats(t: torch.Tensor):
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
    }

def safe_ByteTokenizer(max_len: int):
    try:
        return ByteTokenizer(max_len=max_len)  # preferred signature
    except TypeError:
        return ByteTokenizer()  # fallback for early versions

def build_model(ckpt: str, device: str = "cpu"):
    data = torch.load(ckpt, map_location="cpu")
    if "model" in data and "args" in data:
        state = data["model"]; cfg = data["args"]
    else:
        state = data.get("model", data); cfg = data.get("args", {})
    d_model = cfg.get("d_model", 256)
    z_dim   = cfg.get("z_dim", 128)
    max_len = cfg.get("max_len", 128)
    tok = safe_ByteTokenizer(max_len=max_len)
    model = UZRModel(tok.vocab_size, d_model=d_model, z_dim=z_dim, max_len=max_len)
    model.load_state_dict(state, strict=False)
    dev = torch.device(device)
    model.to(dev).eval()
    meta = {"d_model": d_model, "z_dim": z_dim, "max_len": max_len, "vocab_size": tok.vocab_size, "cfg": cfg, "device": str(dev)}
    return model, tok, meta

def make_z(model, mode: str, z_dim: int, device: torch.device):
    if mode == "zero":
        return torch.zeros(z_dim, device=device)
    if mode == "init":
        with torch.no_grad():
            return model.init_z(batch_size=1).to(device)[0]
    if mode == "random":
        return torch.randn(z_dim, device=device)
    raise ValueError("unknown z mode")

def film_stats(model, z):
    with torch.no_grad():
        fc = model.film.fc(z)  # [2*d_model]
        d = fc.shape[0] // 2
        g_raw, b = fc[:d], fc[d:]
        g = torch.tanh(g_raw)
        return {
            "gamma_tanh": tensor_stats(g),
            "beta": tensor_stats(b),
        }

def capture_layers(model):
    handles = []
    enc = model.encoder.enc
    for idx, layer in enumerate(enc.layers):
        name = f"encoder.layer.{idx}"
        h = layer.register_forward_hook(lambda m, inp, out, name=name: setattr(model, f"_hook_{name.replace('.', '_')}", out.detach().cpu()))
        handles.append(h)
    return handles

def get_layer_summaries(model):
    layers = []
    for k in sorted([k for k in dir(model) if k.startswith("_hook_encoder_layer_")]):
        t = getattr(model, k)
        st = tensor_stats(t)
        layers.append((k.replace("_hook_", ""), st["mean"], st["std"], list(t.shape)))
    return layers

def topk_for_last(logits, k=5):
    probs = torch.softmax(logits[0, -1], dim=-1)
    pv, pi = torch.topk(probs, k=min(k, probs.numel()))
    return [(int(pi[i]), float(pv[i])) for i in range(pv.size(0))]

def decode_ids(ids: List[int]) -> str:
    BOS, EOS = 256, 257
    ids = [i for i in ids if i not in (BOS, EOS, 0)]
    try:
        return bytes(ids).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def cosine_kl_triplet(model, X, z_zero, z_init, z_rand):
    with torch.no_grad():
        L0 = model(X, z_zero)
        L1 = model(X, z_init)
        L2 = model(X, z_rand)
        flats = [L0.view(-1), L1.view(-1), L2.view(-1)]
        cos = [[float(torch.cosine_similarity(a, b, dim=0)) for b in flats] for a in flats]
        def avg_sym_kl(A, B):
            P = torch.softmax(A, dim=-1); Q = torch.softmax(B, dim=-1)
            kl_pq = (P * (P.clamp_min(1e-8).log() - Q.clamp_min(1e-8).log())).sum(dim=-1)
            kl_qp = (Q * (Q.clamp_min(1e-8).log() - P.clamp_min(1e-8).log())).sum(dim=-1)
            return float((0.5 * (kl_pq + kl_qp)).mean().item())
        KL = [[avg_sym_kl(A, B) for B in [L0, L1, L2]] for A in [L0, L1, L2]]
        return cos, KL

# ---------------------------- Keyboard (non-blocking) -------------------------
class KeyPoller:
    def __init__(self):
        self.win = os.name == "nt"
        if not self.win:
            import termios, tty, select
            self.termios = termios; self.tty = tty; self.select = select
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)

    def __enter__(self):
        if not self.win:
            self.tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.win:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old)

    def poll(self):
        try:
            if self.win:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    return ch
                return None
            else:
                dr, _, _ = self.select.select([sys.stdin], [], [], 0)
                if dr:
                    ch = sys.stdin.read(1)
                    return ch
                return None
        except Exception:
            return None

# ---------------------------- Generation -------------------------------------
def generate_next(model, tok, X, z, temperature=1.0, top_k=0, top_p=0.0):
    with torch.no_grad():
        logits = model(X, z)  # [1,T,V]
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        V = probs.size(0)
        if top_k and top_k < V:
            v, i = torch.topk(probs, k=top_k)
            p = v / v.sum()
            idx = i[torch.multinomial(p, 1)].item()
        elif top_p and 0 < top_p < 1:
            v, i = torch.sort(probs, descending=True)
            c = torch.cumsum(v, dim=0)
            mask = c <= top_p
            cutoff = int(torch.nonzero(mask, as_tuple=False).numel()) or 1
            v = v[:cutoff]; i = i[:cutoff]
            p = v / v.sum()
            idx = i[torch.multinomial(p, 1)].item()
        else:
            idx = int(torch.argmax(probs).item())
        return idx

# ---------------------------- Main loop --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--text", default="Hello, UZR.")
    ap.add_argument("--text-file", default="", help="If set, watch this file and recompute on changes")
    ap.add_argument("--interval", type=float, default=0.5, help="UI refresh seconds")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--z-mode", default="init", choices=["zero","init","random"])
    ap.add_argument("--log-jsonl", default="", help="Append JSON snapshot per refresh")
    # Chat options
    ap.add_argument("--chat-prefix", default="USER: ", help="Prefix added before each committed chat line")
    ap.add_argument("--chat-suffix", default="\n", help="Suffix appended after each committed chat line (use \\n for newline)")
    args = ap.parse_args()

    model, tok, meta = build_model(args.ckpt, args.device)
    device = torch.device(args.device)
    z_mode = args.z_mode
    z = make_z(model, z_mode, meta["z_dim"], device)

    base_text = args.text
    # Optional file watcher
    watcher = None
    if args.text_file:
        class FileWatcher:
            def __init__(self, path: str):
                self.path = path; self.last_mtime = 0.0
            def read_if_changed(self):
                if not self.path: return (False, "")
                try:
                    st = os.stat(self.path)
                    if st.st_mtime != self.last_mtime:
                        self.last_mtime = st.st_mtime
                        with open(self.path, "r", encoding="utf-8") as f:
                            return (True, f.read())
                except FileNotFoundError:
                    pass
                return (False, "")
        watcher = FileWatcher(args.text_file)
        changed, txt = watcher.read_if_changed()
        if changed and txt.strip():
            base_text = txt

    def enc(text: str):
        return torch.stack([tok.encode(text)], dim=0).to(device)

    X = enc(base_text)
    generated_ids: List[int] = []

    # Chat typing state
    chat_mode = False
    chat_buf: List[str] = []  # collect characters for current line

    # Utility: log a JSONL record
    def log_jsonl(obj: dict):
        if not args.log_jsonl: return
        try:
            with open(args.log_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Loop
    with KeyPoller() as kp:
        while True:
            # File change -> reset tail
            if watcher:
                changed, txt = watcher.read_if_changed()
                if changed:
                    base_text = txt
                    X = enc(base_text)
                    generated_ids = []

            # Effective sequence (append generated tail)
            if generated_ids:
                tail = torch.tensor([generated_ids], device=device, dtype=X.dtype)
                X_eff = torch.cat([X, tail], dim=1)
                max_len = meta["max_len"]
                if X_eff.size(1) > max_len:
                    X_eff = X_eff[:, -max_len:]
            else:
                X_eff = X

            # Forward + hooks
            handles = capture_layers(model)
            with torch.no_grad():
                logits = model(X_eff, z)
            for h in handles:
                try: h.remove()
                except Exception: pass

            # Stats
            fs = film_stats(model, z)
            layers = get_layer_summaries(model)
            last_topk = topk_for_last(logits, k=args.topk)
            ids_argmax = logits.argmax(dim=-1)[0].tolist()
            decoded_preview = decode_ids(ids_argmax)[-120:]

            # z probe (cos/KL)
            z_zero = make_z(model, "zero", meta["z_dim"], device)
            z_init = make_z(model, "init", meta["z_dim"], device)
            z_rand = make_z(model, "random", meta["z_dim"], device)
            cos, KL = cosine_kl_triplet(model, X_eff, z_zero, z_init, z_rand)

            # UI
            clear_screen()
            print(f"UZR Live  |  ckpt={os.path.abspath(args.ckpt)}  |  device={meta['device']}  |  d={meta['d_model']}  z={meta['z_dim']}  L={meta['max_len']}")
            print(f"z_mode={z_mode}  |  text_len={X_eff.shape[1]}  |  gen_len={len(generated_ids)}  |  interval={args.interval}s  |  chat={'ON' if chat_mode else 'OFF'}")
            print("-"*100)
            print(f"[FiLM] gamma(mean={human(fs['gamma_tanh']['mean'])}, std={human(fs['gamma_tanh']['std'])})  |  beta(mean={human(fs['beta']['mean'])}, std={human(fs['beta']['std'])})")
            print(f"[Last Top-{args.topk}] " + ", ".join([f"{tokid}:{prob:.3f}" for tokid, prob in last_topk]))
            print(f"[Argmax preview] ...{decoded_preview}")
            print("-"*100)
            # Layer table
            print("Encoder layers (mean/std) — first 12 shown")
            for i, (name, m, s, shp) in enumerate(layers[:12]):
                print(f"  {name:<22} | mean={human(m)}  std={human(s)}  shape={shp}")
            if len(layers) > 12:
                print(f"  ... ({len(layers)-12} more layers)")
            print("-"*100)
            print("[z variants] cosine (rows/cols = zero, init, rand):")
            for row in cos:
                print("  " + " ".join(f"{v:+.3f}" for v in row))
            print("[z variants] avg sym KL:")
            for row in KL:
                print("  " + " ".join(f"{v:.4f}" for v in row))
            print("-"*100)
            print("Hotkeys: [0]=z_zero  [i]=z_init  [r]=z_random  [g]=toggle gen  [s]=step  [c]=clear tail  [t]=chat on/off  [q]=quit")
            if chat_mode:
                print(f"Chat> {''.join(chat_buf)}", end="", flush=True)

            # Per-frame JSON log (snapshot)
            log_jsonl({
                "ts": time.time(),
                "text_len": int(X_eff.shape[1]),
                "gen_len": len(generated_ids),
                "film": {"gamma_mean": fs["gamma_tanh"]["mean"], "gamma_std": fs["gamma_tanh"]["std"], "beta_mean": fs["beta"]["mean"], "beta_std": fs["beta"]["std"]},
                "last_topk": last_topk,
                "cosine": cos, "sym_kl": KL
            })

            # Keys / interval
            t_end = time.time() + args.interval
            auto_gen = getattr(main, "_auto_gen", False)
            while time.time() < t_end:
                ch = kp.poll()
                if not ch:
                    time.sleep(0.02); continue

                # If chat mode, accumulate text
                if chat_mode:
                    if ch in ("\r", "\n"):  # commit
                        line = "".join(chat_buf)
                        chat_buf = []
                        # Append to base text with prefix/suffix
                        base_text += (args.chat_prefix + line + args.chat_suffix)
                        X = enc(base_text)
                        generated_ids = []
                        # Dump a chat event record
                        log_jsonl({
                            "ts": time.time(),
                            "event": "chat_commit",
                            "chat_text": line,
                            "after_text_len": int(X.shape[1]),
                            "film": {"gamma_mean": fs["gamma_tanh"]["mean"], "gamma_std": fs["gamma_tanh"]["std"], "beta_mean": fs["beta"]["mean"], "beta_std": fs["beta"]["std"]},
                            "cosine": cos, "sym_kl": KL,
                            "last_topk": last_topk
                        })
                    elif ch in ("\x08", "\x7f"):  # backspace (win/unix)
                        if chat_buf: chat_buf.pop()
                    elif ch == "\x1b":  # ESC -> cancel current line
                        chat_buf = []
                    else:
                        # Basic filter to avoid control chars
                        if ord(ch) >= 32 or ch == "\t":
                            chat_buf.append(ch)
                    continue  # eat key here

                # Not in chat mode: global hotkeys
                if ch == "q":
                    clear_screen(); return
                elif ch == "0":
                    z = make_z(model, "zero", meta["z_dim"], device); z_mode = "zero"
                elif ch == "i":
                    z = make_z(model, "init", meta["z_dim"], device); z_mode = "init"
                elif ch == "r":
                    z = make_z(model, "random", meta["z_dim"], device); z_mode = "random"
                elif ch == "c":
                    generated_ids = []
                elif ch == "s":
                    idx = generate_next(model, tok, X_eff, z, args.temperature, args.top_k, args.top_p)
                    generated_ids.append(idx)
                elif ch == "g":
                    auto_gen = not auto_gen; setattr(main, "_auto_gen", auto_gen)
                elif ch == "t":
                    chat_mode = not chat_mode
                # ignore other keys

            # auto-gen step
            if getattr(main, "_auto_gen", False) and not chat_mode:
                idx = generate_next(model, tok, X_eff, z, args.temperature, args.top_k, args.top_p)
                generated_ids.append(idx)

if __name__ == "__main__":
    main()
