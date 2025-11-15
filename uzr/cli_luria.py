# -*- coding: utf-8 -*-
import os
import sys
import argparse
import readline  # ?몄쓽
import csv
import re
import glob
import atexit
import signal
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch

# 濡쒖뺄 ?⑦궎吏 ?꾪룷??(?⑦궎吏 紐⑤뱶 沅뚯옣)
# ?꾨줈?앺듃 猷⑦듃?먯꽌: python -m uzr.cli_luria --device cpu --resume ckpt.pt
from .model import UZRModel, KoEnTokenizer  # tokenizer/model
try:
    from .npu import OrtEngine
except Exception:
    OrtEngine = None  # optional
from .meta_core import AbstainThresholds, maybe_abstain
from .train_meta_3brains import inner_adapt_z_3brains  # z ?곸쓳 猷⑦떞
from .memory import CompressedMemory

# -----------------------------
# ?붿쭊 ?섑띁: 異붾줎(NPU/ORT/CPU) vs ?숈뒿(CPU)
# -----------------------------

def f3(x):
    """None-safe 3?먮- ?щ㎎??""
    return "nan" if x is None else f"{x:.3f}"

def _intent_force_from_model(model):
    try:
        _, toggle = model.identity_intent_control()
    except Exception:
        return None
    if toggle <= -0.5:
        return False
    if toggle >= 0.5:
        return True
    return None

def default_save_path(path):
    """??꾩뒪?ы봽 湲곕낯 ?뚯씪紐??앹꽦"""
    if path and path != "luria_session.pt":
        return path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"luria_session_{ts}.pt"

@dataclass
class HistoryItem:
    """????덉뒪?좊- ??ぉ"""
    step: int
    user_input: str
    luria_response: str
    user_tokens: int
    response_tokens: int
    conf: Optional[float]
    entropy: Optional[float]

@dataclass
class SessionState:
    z_for_q: Dict[str, torch.Tensor]  # {"slow_lang":..., "slow_logic":..., "bridge":...}
    step: int = 0
    history: List[HistoryItem] = field(default_factory=list)

def build_model(device: torch.device, mem: Optional[CompressedMemory], ckpt_args: Optional[Dict[str, Any]] = None) -> UZRModel:
    # ?섏씠?쇳뙆?쇰??곕뒗 泥댄겕?ъ씤???먮뒗 湲곕낯媛??ъ슜
    max_len = ckpt_args.get("max_len", 128) if ckpt_args else 128
    tok = KoEnTokenizer(max_len=max_len)

    # 泥댄겕?ъ씤?몄뿉???섏씠?쇳뙆?쇰???濡쒕뱶 ?먮뒗 湲곕낯媛??ъ슜
    if ckpt_args:
        d_model = ckpt_args.get("d_model", 256)
        z_dim = ckpt_args.get("z_dim", 128)
        z_think_dim = ckpt_args.get("z_think_dim", 64)
        z_lang_dim = ckpt_args.get("z_lang_dim", 32)
        num_langs = ckpt_args.get("num_langs", 4)
        identity_self_dim = ckpt_args.get("identity_self_dim", 32)
        identity_intent_dim = ckpt_args.get("identity_intent_dim", min(16, max(0, identity_self_dim // 2)))
        z_slow_lang_dim = ckpt_args.get("z_slow_lang_dim", 96)
        z_slow_logic_dim = ckpt_args.get("z_slow_logic_dim", 96)
        z_bridge_dim = ckpt_args.get("z_bridge_dim", 64)
    else:
        d_model = 256
        z_dim = 128
        z_think_dim = 64
        z_lang_dim = 32
        num_langs = 4
        identity_self_dim = 32
        identity_intent_dim = 16
        z_slow_lang_dim = 96
        z_slow_logic_dim = 96
        z_bridge_dim = 64

    m = UZRModel(
        vocab_size=tok.vocab_size,
        d_model=d_model, n_head=4, n_layer=4,
        z_dim=z_dim, z_think_dim=z_think_dim, z_lang_dim=z_lang_dim,
        num_langs=num_langs, identity_self_dim=identity_self_dim, identity_intent_dim=identity_intent_dim,
        z_slow_lang_dim=z_slow_lang_dim, z_slow_logic_dim=z_slow_logic_dim, z_bridge_dim=z_bridge_dim,
        memory=mem, use_self_eval=True
    )
    return m.to(device)

def encode_pair(tok: KoEnTokenizer, x: str, y: Optional[str] = None, device="cpu"):
    X = tok.encode(x).unsqueeze(0).to(device)
    Y = tok.encode(y).unsqueeze(0).to(device) if y is not None else None
    return X, Y

def _extract_prompt_ids(tok: KoEnTokenizer, text: str) -> List[int]:
    ids = tok.encode(text).tolist()
    out = []
    for i in ids:
        if i == tok.PAD:
            break
        if i == tok.EOS:
            break
        out.append(i)
    if not out or out[0] != tok.BOS:
        out = [tok.BOS] + out
    return out

def _entropy_from_probs(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-9)
    ent = -(p * p.log()).sum().item()
    return float(ent)

def generate_tokens(model: UZRModel, tok: KoEnTokenizer, prompt_ids: List[int], z_for_q: Dict[str, torch.Tensor], device: torch.device,
                    max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40,
                    rep_penalty: float = 1.10, min_eos_len: int = 8, ort_engine=None) -> Dict[str, Any]:
    last_entropy: Optional[float] = None
    # Respect encoder maximum length
    try:
        max_len_model = int(getattr(model.encoder, "max_len", 128))
    except Exception:
        max_len_model = 128
    budget = max(0, max_len_model - ids.size(1) - 1)
    steps = min(int(max_new_tokens), int(budget))
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for _ in range(steps):
            # Use ORT/QNN engine if available to compute logits; else PyTorch model
            if ort_engine is not None:
                try:
                    out = ort_engine.run(input_ids=ids.detach().cpu().numpy())
                    logits_np = out.get("logits", None)
                    if logits_np is None:
                        logits_np = list(out.values())[0]
                    logits_full = torch.from_numpy(logits_np).to(device)
                    logits = logits_full[:, -1, :]
                except Exception:
                    logits = model(ids, z_for_q)[:, -1, :]
            else:
                logits = model(ids, z_for_q)[:, -1, :]
                probs_filtered = torch.zeros_like(probs)
                probs_filtered.scatter_(0, top_k_idx, top_k_probs)
                probs = probs_filtered
                probs = probs / probs.sum().clamp_min(1e-9)  # Renormalize

            # Apply top-p (nucleus sampling)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= top_p
                if keep.numel() > 0:
                    keep[0] = True
                cutoff = int(keep.sum().item())
                filtered = sorted_probs.clone()
                if cutoff < filtered.numel():
                    filtered[cutoff:] = 0
                total = filtered.sum().clamp_min(1e-9)
                filtered = filtered / total
                next_sorted_idx = torch.multinomial(filtered, num_samples=1)
                next_id = sorted_idx.gather(-1, next_sorted_idx)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            last_entropy = _entropy_from_probs(probs)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
            if ids.size(1) - len(prompt_ids) >= int(min_eos_len) and int(next_id.item()) == tok.EOS:
                break
    gen_ids = ids.squeeze(0).tolist()[len(prompt_ids):]
    return {"gen_ids": gen_ids, "last_entropy": last_entropy}

def predict(model: UZRModel, tok: KoEnTokenizer, prompt: str, state: SessionState, device: torch.device,
            gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
    prompt_ids = _extract_prompt_ids(tok, prompt)
    out = generate_tokens(
        model, tok, prompt_ids, state.z_for_q, device,
        max_new_tokens=int(gen_cfg.get("max_gen_len", 128)),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        top_k=int(gen_cfg.get("top_k", 40)),
        rep_penalty=float(gen_cfg.get("rep_penalty", 1.10)),
        min_eos_len=int(gen_cfg.get("min_eos_len", 8)), ort_engine=gen_cfg.get("ort_engine", None))
    gen_ids = out["gen_ids"]
    text = tok.decode(gen_ids).strip()
    user_tokens = len(prompt_ids)
    response_tokens = len(gen_ids)
    conf = None
    try:
        Xq = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        conf_t = model.confidence(Xq)
        if conf_t is not None:
            conf = float(conf_t[0].item())
    except Exception:
        conf = None
    ent = out.get("last_entropy", None)
    # Abstain gating (optional)
    if gen_cfg.get("abstain_enabled") and (conf is not None) and (ent is not None):
        thr = AbstainThresholds(conf_min=float(gen_cfg.get("abstain_conf_min", 0.65)),
                                 ent_max=float(gen_cfg.get("abstain_ent_max", 2.2)))
        cm = torch.tensor([conf], dtype=torch.float32)
        em = torch.tensor([ent], dtype=torch.float32)
        mask = maybe_abstain(cm, em, thr)
        if bool(mask[0].item()):
            text = str(gen_cfg.get("abstain_message", "[蹂대쪟] ?뺤떊????븘 ?듬???蹂대쪟?⑸땲??"))
            # Do not change token counts; preserve logging consistency
    return {
        "text": text,
        "user_tokens": user_tokens,
        "response_tokens": response_tokens,
        "conf": conf,
        "entropy": ent,
    }

def adapt_z(model_cpu: UZRModel, tok: KoEnTokenizer, Xc_txt: str, Yc_txt: str, state: SessionState, steps:int=6,
            lam=(1e-3,1e-3,1e-3), eta=0.5, device=torch.device("cpu")) -> Dict[str, torch.Tensor]:
    # inner_adapt??CPU 寃쎈줈?먯꽌 ?섑뻾
    model_cpu.eval().to(device)
    Xc, Yc = encode_pair(tok, Xc_txt, Yc_txt, device=device)

    # 珥덇린 z?????몄뀡 媛믪뿉??媛?몄샂
    z_lang0 = state.z_for_q["slow_lang"].detach().to(device)
    z_logic0 = state.z_for_q["slow_logic"].detach().to(device)
    z_bridge0 = state.z_for_q["bridge"].detach().to(device)

    zl, zg, zb = inner_adapt_z_3brains(
        model_cpu, Xc, Yc, z_lang0, z_logic0, z_bridge0,
        lam_lang=lam[0], lam_logic=lam[1], lam_bridge=lam[2],
        eta=eta, steps=steps
    )
    return {"slow_lang": zl, "slow_logic": zg, "bridge": zb}  # ??z

def init_session(model: UZRModel, device: torch.device) -> SessionState:
    # 踰좎씠??z 珥덇린?? 湲곕낯 ?대땲?쒕씪?댁? ?ъ슜
    z_lang0 = model.init_z_slow_lang(batch_size=1)[0].to(device)
    z_logic0 = model.init_z_slow_logic(batch_size=1)[0].to(device)
    z_bridge0 = model.init_z_bridge(batch_size=1)[0].to(device)
    return SessionState(z_for_q={"slow_lang": z_lang0, "slow_logic": z_logic0, "bridge": z_bridge0})

def init_log_csv(log_dir: str) -> str:
    """濡쒓렇 ?붾젆?좊-? CSV ?뚯씪 珥덇린??""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"luria_chat_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "step", "user_input", "user_tokens",
            "luria_response", "response_tokens", "confidence", "entropy",
            "action", "correct_answer"
        ])

    return csv_path

def log_interaction(csv_path: str, step: int, user_input: str, user_tokens: int,
                   luria_response: str, response_tokens: int,
                   conf: Optional[float], ent: Optional[float],
                   action: str, correct_answer: str = ""):
    """?곹샇?묒슜 濡쒓렇瑜?CSV??湲곕줉"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, step, user_input, user_tokens,
            luria_response, response_tokens,
            conf if conf is not None else "",
            ent if ent is not None else "",
            action, correct_answer
        ])

def save_checkpoint(model: UZRModel, mem: CompressedMemory, state: SessionState,
                   ckpt_args: Optional[Dict], save_path: str):
    """?꾩옱 ?몄뀡??泥댄겕?ъ씤?몃줈 ???(?먯옄?????"""
    path = default_save_path(save_path)
    tmp = path + ".tmp"

    ckpt = {
        "model": model.state_dict(),
        "args": ckpt_args,
        "step": state.step,
        "z_for_q": {k: v.cpu() for k, v in state.z_for_q.items()},
        "memory": mem.state_dict(),
    }
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    print(f"[????꾨즺] {path}")

def save_train_compatible_last(model: UZRModel, mem: CompressedMemory, ckpt_args: Optional[Dict], path: str = "uzr_3brains_ckpt_last.pt"):
    """?몃젅???ш컻? ?명솚?섎뒗 理쒖냼 ?ㅻ깄?룹쓣 ??ν븳??"""
    payload = {
        "model": model.state_dict(),
        "args": ckpt_args if ckpt_args else {},
        "memory": mem.state_dict(),
        "step": 0,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[留덉?留??ㅻ깄?? {path}")

def parse_saved_command(s: str) -> Optional[tuple]:
    """'/saved "吏덈Ц" = "?뺣떟"' ?뺤떇 ?뚯떛 (寃ш퀬??踰꾩쟾)"""
    if not s.startswith("/saved"):
        return None

    parts = s.split(" ", 1)
    if len(parts) < 2:
        return None

    rest = parts[1].strip()
    if "=" not in rest:
        return None

    # 泥?踰덉㎏ = 湲곗??쇰줈 split (?뺣떟??=媛 ?ㅼ뼱媛????덉쓬)
    q, a = [t.strip().strip("\"'") for t in rest.split("=", 1)]
    return (q, a) if q and a else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])  # ORT/QNN 遺숈씪 ??蹂꾨룄 遺꾧린 異붽?
    ap.add_argument("--resume", default=None, help="checkpoint .pt path")
    ap.add_argument("--mem_dir", default="./", help="memory log dir (writes.csv, entropy.csv ??")
    ap.add_argument("--log_dir", default="./luria-log", help="chat log directory")
    ap.add_argument("--steps", type=int, default=6, help="inner_adapt steps")
    ap.add_argument("--save_path", default="luria_session.pt", help="session save path")
    ap.add_argument("--autosave_steps", type=int, default=0, help="autosave every N steps (0=disabled)")
    # decoding options (lacomi.txt: temp=0.7, top-p=0.9, top-k=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.10)
    ap.add_argument("--max_gen_len", type=int, default=128)
    ap.add_argument("--min_eos_len", type=int, default=8)
    # abstain options
    ap.add_argument("--abstain", choices=["on", "off"], default="off", help="enable abstain gating on low confidence/high entropy")
    ap.add_argument("--abstain_conf_min", type=float, default=0.65)
    ap.add_argument("--abstain_ent_max", type=float, default=2.2)
    ap.add_argument("--abstain_message", default="[蹂대쪟] ?뺤떊????븘 ?듬???蹂대쪟?⑸땲??")
    # ORT/QNN engine options
    ap.add_argument("--ort_model", type=str, default="", help="ONNX(QDQ) model path for ORT/QNN engine")
    ap.add_argument("--engine", type=str, default="torch", choices=["torch","qnn","qnn_strict","ort_fallback"], help="Inference engine selection")
args = ap.parse_args()

# ORT/QNN engine (optional)
ort_engine = None
if args.engine != "torch" and args.ort_model:
    if OrtEngine is None:
        print("[경고] onnxruntime-qnn 미설치로 ORT/QNN 엔진 비활성화. PyTorch 경로로 진행합니다.")
    else:
        backend = "htp"
        mode = "qnn" if args.engine == "qnn" else ("qnn_strict" if args.engine == "qnn_strict" else "ort_fallback")
        try:
            ort_engine = OrtEngine(args.ort_model, mode=mode, backend=backend)
            print(f"[engine] ORT engine ready: mode={mode}, backend={backend}")
        except Exception as e:
            print(f"[경고] ORT 엔진 초기화 실패: {e}. PyTorch 경로로 진행합니다.")
    # ?μ튂
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[寃쎄퀬] CUDA ?ъ슜 遺덇?. CPU濡??泥댄빀?덈떎.")
        args.device = "cpu"
    device = torch.device(args.device)

    # 泥댄겕?ъ씤??濡쒕뱶 (紐⑤뜽 ?앹꽦 ?꾩뿉 args ?뺤씤)
    ckpt_args = None
    ckpt = None
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            # PyTorch 援щ쾭???명솚??(weights_only ?뚮씪誘명꽣 ?놁쓬)
            ckpt = torch.load(args.resume, map_location=device)
        ckpt_args = ckpt.get("args", None)

    # 硫붾え由?
    mem = CompressedMemory(device=str(device), log_dir=args.mem_dir)

    # 紐⑤뜽 濡쒕뱶 (泥댄겕?ъ씤??args ?ъ슜)
    model = build_model(device, mem, ckpt_args)
    max_len = ckpt_args.get("max_len", 128) if ckpt_args else 128
    tok = KoEnTokenizer(max_len=max_len)

    # 泥댄겕?ъ씤??state_dict 蹂듭썝
    if ckpt is not None:
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        if "memory" in ckpt and isinstance(mem, CompressedMemory):
            try:
                mem.load_state_dict(ckpt["memory"])
                print(f"[硫붾え由?蹂듭썝] items={len(mem.items)} learner={mem.has_learner()}")
            except Exception as e:
                print(f"[寃쎄퀬] 硫붾え由?蹂듭썝 ?ㅽ뙣: {e}")

    # ?몄뀡 ?곹깭
    state = init_session(model, device)

    # 濡쒓렇 ?뚯씪 珥덇린??
    csv_path = init_log_csv(args.log_dir)
    print(f"[濡쒓렇 ?쒖옉] {csv_path}")

    print("=== Luria Chat CLI (infer=NPU/CPU, adapt=CPU) ===")
    print("紐낅졊:")
    print("  /no [踰덊샇]         - 吏곸쟾 ?먮뒗 ?덉뒪?좊- 踰덊샇???묐떟????몄쓬 (?뺣떟 ?낅젰 ?꾩슂)")
    print("  /yes [踰덊샇]        - 吏곸쟾 ?먮뒗 ?덉뒪?좊- 踰덊샇???묐떟??留욎븯??")
    print("  /set_y <?뺣떟>      - ?뺣떟 ?깅줉")
    print("  /saved \"吏덈Ц\" = \"?뺣떟\" - 利됱떆 ?숈뒿")
    print("  /history           - 理쒓렐 ????덉뒪?좊- ?쒖떆")
    print("  /save, /load, /reset, /quit")
    if ort_engine is not None:
        print("  /lora_npz <path> - Load adapter/FiLM params (.npz) and swap into engine")
        print("  /hot_swap - Recreate ORT session (shadow→active) and warmup")
    print(f"?붿퐫?? T={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, rep_penalty={args.rep_penalty}, max_len={args.max_gen_len}")
    if args.abstain == "on":
        print(f"蹂대쪟 寃뚯씠?? conf_min={args.abstain_conf_min}, ent_max={args.abstain_ent_max}")
    if args.autosave_steps:
        print(f"?ㅽ넗?몄씠釉? {args.autosave_steps} ?ㅽ뀦留덈떎 ?먮룞 ???)
    pending_x: Optional[str] = None
    pending_pred: Optional[Dict[str, Any]] = None
    staged_y: Optional[str] = None

    # ?먮룞 ????좏떥
    def maybe_autosave():
        if args.autosave_steps and state.step % args.autosave_steps == 0:
            save_checkpoint(model, mem, state, ckpt_args, args.save_path)
            save_train_compatible_last(model, mem, ckpt_args, "uzr_3brains_ckpt_last.pt")
            log_interaction(csv_path, state.step, "[autosave]", 0, "", 0, None, None, "autosave")

    # 醫낅즺 ?? ?뺤긽 醫낅즺/?쒓렇?먯뿉????긽 理쒖떊 ?ㅻ깄???④?
    def _graceful_exit():
        try:
            save_checkpoint(model, mem, state, ckpt_args, args.save_path)
            save_train_compatible_last(model, mem, ckpt_args, "uzr_3brains_ckpt_last.pt")
        finally:
            pass

    def _sig_handler(signum, frame):
        print(f"\n[?좏샇 {signum}] ?덉쟾 ?????醫낅즺")
        _graceful_exit()
        sys.exit(0)

    atexit.register(_graceful_exit)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            # ?쇰? ?섍꼍(?? Jupyter)?먯꽌??signal ?깅줉???쒗븳?????덉쓬
            pass

    while True:
        try:
            s = input("?좎?> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n醫낅즺")
            break

        if not s:
            continue
        if s == "/quit":
            break

        # /save 紐낅졊?? ?꾩옱 ?몄뀡 ???
        if s == "/save":
            save_checkpoint(model, mem, state, ckpt_args, args.save_path)
            save_train_compatible_last(model, mem, ckpt_args, "uzr_3brains_ckpt_last.pt")
            log_interaction(csv_path, state.step, s, 0, "[?몄뀡 ??λ맖]", 0, None, None, "save")
            continue

        # /load 紐낅졊?? ?몄뀡 蹂듭썝
        if s.startswith("/load"):
            # 寃쎈줈 ?뚯떛
            if s.strip() == "/load":
                path = args.save_path
            else:
                user_input = s.split(" ", 1)[1].strip()

                # ?뚯씪??議댁옱?섎㈃ 洹몃?濡??ъ슜
                if os.path.exists(user_input):
                    path = user_input
                else:
                    # ?⑦꽩?쇰줈 寃??(??꾩뒪?ы봽 ?ы븿)
                    # luria_session_*{user_input}*.pt ?뺥깭濡?寃??
                    patterns = [
                        f"luria_session_*{user_input}*.pt",
                        f"*{user_input}*.pt",
                        user_input if user_input.endswith(".pt") else f"{user_input}.pt"
                    ]

                    matched_files = []
                    for pattern in patterns:
                        matched_files.extend(glob.glob(pattern))

                    # 以묐났 ?쒓굅
                    matched_files = list(set(matched_files))

                    if len(matched_files) == 0:
                        print(f"[?먮윭] ?뚯씪??李얠쓣 ???놁쓬: {user_input}")
                        print(f"寃???⑦꽩: luria_session_*{user_input}*.pt")
                        continue
                    elif len(matched_files) == 1:
                        path = matched_files[0]
                        print(f"[?먮룞 ?좏깮] {path}")
                    else:
                        print(f"[{len(matched_files)}媛??뚯씪 諛쒓껄]")
                        for i, f in enumerate(matched_files, 1):
                            # ?뚯씪 ?섏젙 ?쒓컙 ?쒖떆
                            mtime = os.path.getmtime(f)
                            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                            print(f"  {i}. {f} (?섏젙: {mtime_str})")
                        try:
                            choice = input("踰덊샇 ?좏깮> ").strip()
                            idx = int(choice) - 1
                            if 0 <= idx < len(matched_files):
                                path = matched_files[idx]
                            else:
                                print("[?먮윭] ?섎せ??踰덊샇")
                                continue
                        except (ValueError, EOFError, KeyboardInterrupt):
                            print("[痍⑥냼??")
                            continue

            # ?뚯씪 濡쒕뱶
            if not os.path.exists(path):
                print(f"[?먮윭] ?뚯씪??李얠쓣 ???놁쓬: {path}")
                continue

            try:
                ck = torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                ck = torch.load(path, map_location=device)

            model.load_state_dict(ck["model"], strict=False)
            state.step = ck.get("step", 0)
            state.z_for_q = {k: v.to(device) for k, v in ck["z_for_q"].items()}
            if "memory" in ck and isinstance(mem, CompressedMemory):
                try:
                    mem.load_state_dict(ck["memory"])
                    print(f"[硫붾え由?蹂듭썝] items={len(mem.items)} learner={mem.has_learner()}")
                except Exception as e:
                    print(f"[寃쎄퀬] 硫붾え由?蹂듭썝 ?ㅽ뙣: {e}")
            print(f"[濡쒕뱶 ?꾨즺] {path} (step={state.step})")
            log_interaction(csv_path, state.step, s, 0, f"[濡쒕뱶:{path}]", 0, None, None, "load")
            continue

        # /reset 紐낅졊?? ?몄뀡 珥덇린??
        if s == "/reset":
            state = init_session(model, device)
            pending_x = None; pending_pred = None; staged_y = None
            print("?몄뀡 珥덇린??)
            log_interaction(csv_path, state.step, s, 0, "[?몄뀡 珥덇린?붾맖]", 0, None, None, "reset")
            continue

        # /history 紐낅졊?? ????덉뒪?좊- ?쒖떆
        if s == "/history":
            if not state.history:
                print("?덉뒪?좊-媛 鍮꾩뼱?덉뒿?덈떎.")
                continue
            print("\n=== 理쒓렐 ????덉뒪?좊- (理쒕? 20媛? ===")
            # 理쒓렐 20媛쒕쭔 ?쒖떆
            recent = state.history[-20:]
            for idx, item in enumerate(recent, 1):
                print(f"{idx:2d}. [step={item.step}] {item.user_input[:40]}")
                print(f"    -> {item.luria_response[:60]}")
                print(f"    (?좏겙: {item.user_tokens}/{item.response_tokens}, "
                      f"conf={f3(item.conf)}, ent={f3(item.entropy)})")
            print("?ъ슜踰? /no <踰덊샇> ?먮뒗 /yes <踰덊샇>\n")
            continue

        # /saved "吏덈Ц" = "?뺣떟" ?뺤떇 ?뚯떛
        saved_pair = parse_saved_command(s)
        if saved_pair:
            question, answer = saved_pair
            print(f"[?숈뒿 ?곗씠???깅줉] Q: {question} ??A: {answer}")
            # 利됱떆 ?숈뒿 ?ㅽ뻾
            print("[?숈뒿] inner_adapt_z_3brains ?ㅽ뻾 以?..")
            new_z = adapt_z(model, tok, question, answer, state, steps=args.steps, device=device)
            state.z_for_q = {k: v.to(device) for k, v in new_z.items()}

            # 硫붾え由?而ㅻ컠
            enc_avg, _ = encode_pair(tok, question, None, device=device)
            enc_avg = model.avg_embed(enc_avg).mean(dim=0).detach()
            key = enc_avg
            val = {"z_slow": state.z_for_q["bridge"].detach()}
            intent_force = _intent_force_from_model(model)
            mem.add_with_policy(key, val, state.step, meta={"luria_intent_force": intent_force}, bench_callback=lambda: True)

            print("[?숈뒿 ?꾨즺] z 媛깆떊 諛?硫붾え由?泥섎-")
            log_interaction(csv_path, state.step, question, len(tok.encode(question)),
                          answer, len(tok.encode(answer)), None, None, "saved_learning", answer)
            state.step += 1
            continue

        # ?뺣떟 ?명똿
        if s.startswith("/set_y "):
            staged_y = s[len("/set_y "):].strip()
            print(f"[?뺣떟 ?꾨낫 ?깅줉] Yc = {staged_y}")
            log_interaction(csv_path, state.step, s, 0, f"[?뺣떟 ?깅줉: {staged_y}]", 0, None, None, "set_y", staged_y)
            continue

        # ?먯젙 而ㅻ㎤?? /no [踰덊샇]
        if s.startswith("/no"):
            # 踰덊샇 ?뚯떛
            hist_idx = None
            if len(s.split()) > 1:
                try:
                    hist_idx = int(s.split()[1]) - 1  # 1-based to 0-based
                except ValueError:
                    print("[?먮윭] ?щ컮瑜?踰덊샇瑜??낅젰?섏꽭?? ?? /no 3")
                    continue

            # ?덉뒪?좊-?먯꽌 ?좏깮 ?먮뒗 吏곸쟾 ????ъ슜
            if hist_idx is not None:
                recent = state.history[-20:]
                if hist_idx < 0 or hist_idx >= len(recent):
                    print(f"[?먮윭] ?섎せ??踰덊샇?낅땲?? 1-{len(recent)} ?ъ씠濡??낅젰?섏꽭??")
                    continue
                hist_item = recent[hist_idx]
                target_x = hist_item.user_input
                target_pred = {
                    "user_tokens": hist_item.user_tokens,
                    "response_tokens": hist_item.response_tokens,
                    "conf": hist_item.conf,
                    "entropy": hist_item.entropy
                }
                print(f"[?좏깮] {hist_idx+1}踰? {target_x[:40]} -> {hist_item.luria_response[:40]}")
            else:
                # 吏곸쟾 ????ъ슜
                if pending_x is None or pending_pred is None:
                    print("吏곸쟾 吏덉쓽/?묐떟???놁뒿?덈떎. /history濡??뺤씤 ??踰덊샇瑜?吏?뺥븯?몄슂.")
                    continue
                target_x = pending_x
                target_pred = pending_pred

            # ?뺣떟 ?뺤씤
            if not staged_y:
                print("?뺣떟???놁뒿?덈떎. 癒쇱? `/set_y <?뺣떟>`?쇰줈 ?뚮젮二쇱꽭??")
                continue

            # ?숈뒿(adapt) 寃쎈줈
            print("[?숈뒿] inner_adapt_z_3brains ?ㅽ뻾 以?..")
            new_z = adapt_z(model, tok, target_x, staged_y, state, steps=args.steps, device=device)
            # ?몄뀡 z 媛깆떊
            state.z_for_q = {k: v.to(device) for k, v in new_z.items()}
            # 硫붾え由?而ㅻ컠(2PC 踰ㅼ튂 肄쒕갚 ?덉떆)
            enc_avg, _ = encode_pair(tok, target_x, None, device=device)
            enc_avg = model.avg_embed(enc_avg).mean(dim=0).detach()
            key = enc_avg
            val = {"z_slow": state.z_for_q["bridge"].detach()}
            step = state.step
            def bench():
                ent = target_pred.get("entropy", 9.9) or 9.9
                conf = target_pred.get("conf", 0.0) or 0.0
                return (conf >= 0.62) or (ent <= 3.0)
            intent_force = _intent_force_from_model(model)
            mem.add_with_policy(key, val, step, meta={"luria_intent_force": intent_force}, bench_callback=bench)
            print("[?숈뒿 ?꾨즺] z 媛깆떊 諛?硫붾え由?泥섎-")

            # 濡쒓렇 湲곕줉
            log_interaction(csv_path, state.step, target_x, target_pred.get("user_tokens", 0),
                          staged_y, len(tok.encode(staged_y)),
                          target_pred.get("conf"), target_pred.get("entropy"), "no_adapt", staged_y)

            staged_y = None
            state.step += 1
            maybe_autosave()
            continue

        # ?먯젙 而ㅻ㎤?? /yes [踰덊샇]
        if s.startswith("/yes"):
            # 踰덊샇 ?뚯떛
            hist_idx = None
            if len(s.split()) > 1:
                try:
                    hist_idx = int(s.split()[1]) - 1  # 1-based to 0-based
                except ValueError:
                    print("[?먮윭] ?щ컮瑜?踰덊샇瑜??낅젰?섏꽭?? ?? /yes 3")
                    continue

            # ?덉뒪?좊-?먯꽌 ?좏깮 ?먮뒗 吏곸쟾 ????ъ슜
            if hist_idx is not None:
                recent = state.history[-20:]
                if hist_idx < 0 or hist_idx >= len(recent):
                    print(f"[?먮윭] ?섎せ??踰덊샇?낅땲?? 1-{len(recent)} ?ъ씠濡??낅젰?섏꽭??")
                    continue
                hist_item = recent[hist_idx]
                target_x = hist_item.user_input
                target_pred = {
                    "text": hist_item.luria_response,
                    "user_tokens": hist_item.user_tokens,
                    "response_tokens": hist_item.response_tokens,
                    "conf": hist_item.conf,
                    "entropy": hist_item.entropy
                }
                print(f"[?좏깮] {hist_idx+1}踰? {target_x[:40]} -> {hist_item.luria_response[:40]}")
            else:
                # 吏곸쟾 ????ъ슜
                if pending_x is None or pending_pred is None:
                    print("吏곸쟾 吏덉쓽/?묐떟???놁뒿?덈떎. /history濡??뺤씤 ??踰덊샇瑜?吏?뺥븯?몄슂.")
                    continue
                target_x = pending_x
                target_pred = pending_pred

            # ?뺣떟?쇰줈 媛꾩＜ ??硫붾え由??뺤콉???곕씪 ?ㅽ뀒?댁?/而ㅻ컠
            enc_avg, _ = encode_pair(tok, target_x, None, device=device)
            enc_avg = model.avg_embed(enc_avg).mean(dim=0).detach()
            key = enc_avg
            val = {"z_slow": state.z_for_q["bridge"].detach()}
            step = state.step
            intent_force = _intent_force_from_model(model)
            mem.add_with_policy(key, val, step, meta={"luria_intent_force": intent_force}, bench_callback=lambda: True)
            print("[硫붾え由? ?뺣떟 ?섑뵆 湲곕줉 ?쒕룄")

            # 濡쒓렇 湲곕줉
            log_interaction(csv_path, state.step, target_x, target_pred.get("user_tokens", 0),
                          target_pred.get("text", ""), target_pred.get("response_tokens", 0),
                          target_pred.get("conf"), target_pred.get("entropy"), "yes_confirm")

            state.step += 1
            maybe_autosave()
            continue

        # ?쇰컲 吏덉쓽: 異붾줎
        # ORT/QNN engine commands
        if s.lower().startswith("/lora_npz") and ort_engine is not None:
            parts = s.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /lora_npz path/to/params.npz")
                continue
            path = parts[1].strip().strip('"')
            try:
                import numpy as np
                data = np.load(path)
                A = data.get("adapter_A"); B = data.get("adapter_B")
                gamma = data.get("film_gamma"); beta = data.get("film_beta")
                if A is None and B is None and gamma is None and beta is None:
                    print("No adapter_A/B or film_gamma/beta found in npz.")
                else:
                    ort_engine.swap_adapters(A=A, B=B, gamma=gamma, beta=beta)
                    print(f"[engine] adapters swapped from {os.path.basename(path)}")
            except Exception as e:
                print(f"[engine] load failed: {e}")
            continue

        if s.lower() == "/hot_swap" and ort_engine is not None:
            try:
                ort_engine.hot_swap()
                print("[engine] hot swapped and warmed up.")
            except Exception as e:
                print(f"[engine] hot swap failed: {e}")
            continue
        pending_x = s
        gen_cfg = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "rep_penalty": args.rep_penalty,
            "max_gen_len": args.max_gen_len,
            "min_eos_len": args.min_eos_len,
            "abstain_enabled": (args.abstain == "on"),
            "abstain_conf_min": args.abstain_conf_min,
            "abstain_ent_max": args.abstain_ent_max,
            "abstain_message": args.abstain_message,
            "ort_engine": ort_engine,
        }
        out = predict(model, tok, pending_x, state, device, gen_cfg)
        pending_pred = out

        # ?꾩옱 ?덉뒪?좊- 踰덊샇 (理쒓렐 20媛?湲곗?)
        hist_num = len(state.history) + 1
        recent_start = max(1, hist_num - 19)  # 理쒓렐 20媛?踰붿쐞???쒖옉 踰덊샇
        display_num = hist_num - recent_start + 1  # ?쒖떆??踰덊샇 (1-20)

        print(f"[{display_num}] 猷⑤-?? {out['text']}")
        print(f"(?좏겙: ?낅젰={out.get('user_tokens')}, 異쒕젰={out.get('response_tokens')}, "
              f"conf={f3(out.get('conf'))}, ent={f3(out.get('entropy'))})")

        # 濡쒓렇 湲곕줉
        log_interaction(csv_path, state.step, pending_x, out.get("user_tokens", 0),
                      out.get("text", ""), out.get("response_tokens", 0),
                      out.get("conf"), out.get("entropy"), "predict")

        # ?덉뒪?좊-??異붽?
        state.history.append(HistoryItem(
            step=state.step,
            user_input=pending_x,
            luria_response=out.get("text", ""),
            user_tokens=out.get("user_tokens", 0),
            response_tokens=out.get("response_tokens", 0),
            conf=out.get("conf"),
            entropy=out.get("entropy")
        ))

        state.step += 1

        # ?ㅽ넗?몄씠釉?
        maybe_autosave()

if __name__ == "__main__":
    main()




