import os
import sys
import argparse
import readline  # 편의
import csv
import re
import glob
import atexit
import signal
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch

# 로컬 패키지 임포트 (패키지 모드 권장)
# 프로젝트 루트에서: python -m uzr.cli_luria --device cpu --resume ckpt.pt
from .model import UZRModel, KoEnTokenizer  # 토크나이저/모델 정의 있음
from .meta_core import AbstainThresholds, maybe_abstain
from .train_meta_3brains import inner_adapt_z_3brains  # z 적응 루틴
from .memory import CompressedMemory

# -----------------------------
# 엔진 래퍼: 추론(NPU/ORT/CPU) vs 학습(CPU)
# -----------------------------

def f3(x):
    """None-safe 3자리 포맷팅"""
    return "nan" if x is None else f"{x:.3f}"

def default_save_path(path):
    """타임스탬프 기본 파일명 생성"""
    if path and path != "luria_session.pt":
        return path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"luria_session_{ts}.pt"

@dataclass
class HistoryItem:
    """대화 히스토리 항목"""
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
    # 하이퍼파라미터는 체크포인트 또는 기본값 사용
    max_len = ckpt_args.get("max_len", 128) if ckpt_args else 128
    tok = KoEnTokenizer(max_len=max_len)

    # 체크포인트에서 하이퍼파라미터 로드 또는 기본값 사용
    if ckpt_args:
        d_model = ckpt_args.get("d_model", 256)
        z_dim = ckpt_args.get("z_dim", 128)
        z_think_dim = ckpt_args.get("z_think_dim", 64)
        z_lang_dim = ckpt_args.get("z_lang_dim", 32)
        num_langs = ckpt_args.get("num_langs", 4)
        identity_self_dim = ckpt_args.get("identity_self_dim", 2)
        z_slow_lang_dim = ckpt_args.get("z_slow_lang_dim", 96)
        z_slow_logic_dim = ckpt_args.get("z_slow_logic_dim", 96)
        z_bridge_dim = ckpt_args.get("z_bridge_dim", 64)
    else:
        d_model = 256
        z_dim = 128
        z_think_dim = 64
        z_lang_dim = 32
        num_langs = 4
        identity_self_dim = 2
        z_slow_lang_dim = 96
        z_slow_logic_dim = 96
        z_bridge_dim = 64

    m = UZRModel(
        vocab_size=tok.vocab_size,
        d_model=d_model, n_head=4, n_layer=4,
        z_dim=z_dim, z_think_dim=z_think_dim, z_lang_dim=z_lang_dim,
        num_langs=num_langs, identity_self_dim=identity_self_dim,
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
                    rep_penalty: float = 1.10, min_eos_len: int = 8) -> Dict[str, Any]:
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
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
            logits = model(ids, z_for_q)[:, -1, :]
            if rep_penalty and rep_penalty != 1.0:
                uniq = ids.unique()
                logits[:, uniq] = logits[:, uniq] / float(rep_penalty)
            logits = logits / max(float(temperature), 1e-5)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            # lacomi.txt prescription: top-k + top-p sampling (NO argmax)
            # Apply top-k filtering first, then top-p
            if top_k > 0 and top_k < probs.numel():
                # Keep only top-k tokens
                top_k_probs, top_k_idx = torch.topk(probs, k=min(top_k, probs.numel()))
                # Zero out non-top-k tokens
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
        min_eos_len=int(gen_cfg.get("min_eos_len", 8)),
    )
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
            text = str(gen_cfg.get("abstain_message", "[보류] 확신이 낮아 답변을 보류합니다."))
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
    # inner_adapt는 CPU 경로에서 수행
    model_cpu.eval().to(device)
    Xc, Yc = encode_pair(tok, Xc_txt, Yc_txt, device=device)

    # 초기 z는 현 세션 값에서 가져옴
    z_lang0 = state.z_for_q["slow_lang"].detach().to(device)
    z_logic0 = state.z_for_q["slow_logic"].detach().to(device)
    z_bridge0 = state.z_for_q["bridge"].detach().to(device)

    zl, zg, zb = inner_adapt_z_3brains(
        model_cpu, Xc, Yc, z_lang0, z_logic0, z_bridge0,
        lam_lang=lam[0], lam_logic=lam[1], lam_bridge=lam[2],
        eta=eta, steps=steps
    )
    return {"slow_lang": zl, "slow_logic": zg, "bridge": zb}  # 새 z

def init_session(model: UZRModel, device: torch.device) -> SessionState:
    # 베이스 z 초기화: 기본 이니셜라이저 사용
    z_lang0 = model.init_z_slow_lang(batch_size=1)[0].to(device)
    z_logic0 = model.init_z_slow_logic(batch_size=1)[0].to(device)
    z_bridge0 = model.init_z_bridge(batch_size=1)[0].to(device)
    return SessionState(z_for_q={"slow_lang": z_lang0, "slow_logic": z_logic0, "bridge": z_bridge0})

def init_log_csv(log_dir: str) -> str:
    """로그 디렉토리와 CSV 파일 초기화"""
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
    """상호작용 로그를 CSV에 기록"""
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
    """현재 세션을 체크포인트로 저장 (원자적 저장)"""
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
    print(f"[저장 완료] {path}")

def save_train_compatible_last(model: UZRModel, mem: CompressedMemory, ckpt_args: Optional[Dict], path: str = "uzr_3brains_ckpt_last.pt"):
    """트레인 재개와 호환되는 최소 스냅샷을 저장한다."""
    payload = {
        "model": model.state_dict(),
        "args": ckpt_args if ckpt_args else {},
        "memory": mem.state_dict(),
        "step": 0,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[마지막 스냅샷] {path}")

def parse_saved_command(s: str) -> Optional[tuple]:
    """'/saved "질문" = "정답"' 형식 파싱 (견고한 버전)"""
    if not s.startswith("/saved"):
        return None

    parts = s.split(" ", 1)
    if len(parts) < 2:
        return None

    rest = parts[1].strip()
    if "=" not in rest:
        return None

    # 첫 번째 = 기준으로 split (정답에 =가 들어갈 수 있음)
    q, a = [t.strip().strip("\"'") for t in rest.split("=", 1)]
    return (q, a) if q and a else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])  # ORT/QNN 붙일 땐 별도 분기 추가
    ap.add_argument("--resume", default=None, help="checkpoint .pt path")
    ap.add_argument("--mem_dir", default="./", help="memory log dir (writes.csv, entropy.csv 등)")
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
    ap.add_argument("--abstain_message", default="[보류] 확신이 낮아 답변을 보류합니다.")
    args = ap.parse_args()

    # 장치
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[경고] CUDA 사용 불가. CPU로 대체합니다.")
        args.device = "cpu"
    device = torch.device(args.device)

    # 체크포인트 로드 (모델 생성 전에 args 확인)
    ckpt_args = None
    ckpt = None
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            # PyTorch 구버전 호환성 (weights_only 파라미터 없음)
            ckpt = torch.load(args.resume, map_location=device)
        ckpt_args = ckpt.get("args", None)

    # 메모리
    mem = CompressedMemory(device=str(device), log_dir=args.mem_dir)

    # 모델 로드 (체크포인트 args 사용)
    model = build_model(device, mem, ckpt_args)
    max_len = ckpt_args.get("max_len", 128) if ckpt_args else 128
    tok = KoEnTokenizer(max_len=max_len)

    # 체크포인트 state_dict 복원
    if ckpt is not None:
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        if "memory" in ckpt and isinstance(mem, CompressedMemory):
            try:
                mem.load_state_dict(ckpt["memory"])
                print(f"[메모리 복원] items={len(mem.items)} learner={mem.has_learner()}")
            except Exception as e:
                print(f"[경고] 메모리 복원 실패: {e}")

    # 세션 상태
    state = init_session(model, device)

    # 로그 파일 초기화
    csv_path = init_log_csv(args.log_dir)
    print(f"[로그 시작] {csv_path}")

    print("=== Luria Chat CLI (infer=NPU/CPU, adapt=CPU) ===")
    print("명령:")
    print("  /no [번호]         - 직전 또는 히스토리 번호의 응답이 틀렸음 (정답 입력 필요)")
    print("  /yes [번호]        - 직전 또는 히스토리 번호의 응답이 맞았음")
    print("  /set_y <정답>      - 정답 등록")
    print("  /saved \"질문\" = \"정답\" - 즉시 학습")
    print("  /history           - 최근 대화 히스토리 표시")
    print("  /save, /load, /reset, /quit")
    print(f"디코딩: T={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, rep_penalty={args.rep_penalty}, max_len={args.max_gen_len}")
    if args.abstain == "on":
        print(f"보류 게이트: conf_min={args.abstain_conf_min}, ent_max={args.abstain_ent_max}")
    if args.autosave_steps:
        print(f"오토세이브: {args.autosave_steps} 스텝마다 자동 저장")
    pending_x: Optional[str] = None
    pending_pred: Optional[Dict[str, Any]] = None
    staged_y: Optional[str] = None

    # 자동 저장 유틸
    def maybe_autosave():
        if args.autosave_steps and state.step % args.autosave_steps == 0:
            save_checkpoint(model, mem, state, ckpt_args, args.save_path)
            save_train_compatible_last(model, mem, ckpt_args, "uzr_3brains_ckpt_last.pt")
            log_interaction(csv_path, state.step, "[autosave]", 0, "", 0, None, None, "autosave")

    # 종료 훅: 정상 종료/시그널에서 항상 최신 스냅샷 남김
    def _graceful_exit():
        try:
            save_checkpoint(model, mem, state, ckpt_args, args.save_path)
            save_train_compatible_last(model, mem, ckpt_args, "uzr_3brains_ckpt_last.pt")
        finally:
            pass

    def _sig_handler(signum, frame):
        print(f"\n[신호 {signum}] 안전 저장 후 종료")
        _graceful_exit()
        sys.exit(0)

    atexit.register(_graceful_exit)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            # 일부 환경(예: Jupyter)에서는 signal 등록이 제한될 수 있음
            pass

    while True:
        try:
            s = input("유저> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break

        if not s:
            continue
        if s == "/quit":
            break

        # /save 명령어: 현재 세션 저장
        if s == "/save":
            save_checkpoint(model, mem, state, ckpt_args, args.save_path)
            save_train_compatible_last(model, mem, ckpt_args, "uzr_3brains_ckpt_last.pt")
            log_interaction(csv_path, state.step, s, 0, "[세션 저장됨]", 0, None, None, "save")
            continue

        # /load 명령어: 세션 복원
        if s.startswith("/load"):
            # 경로 파싱
            if s.strip() == "/load":
                path = args.save_path
            else:
                user_input = s.split(" ", 1)[1].strip()

                # 파일이 존재하면 그대로 사용
                if os.path.exists(user_input):
                    path = user_input
                else:
                    # 패턴으로 검색 (타임스탬프 포함)
                    # luria_session_*{user_input}*.pt 형태로 검색
                    patterns = [
                        f"luria_session_*{user_input}*.pt",
                        f"*{user_input}*.pt",
                        user_input if user_input.endswith(".pt") else f"{user_input}.pt"
                    ]

                    matched_files = []
                    for pattern in patterns:
                        matched_files.extend(glob.glob(pattern))

                    # 중복 제거
                    matched_files = list(set(matched_files))

                    if len(matched_files) == 0:
                        print(f"[에러] 파일을 찾을 수 없음: {user_input}")
                        print(f"검색 패턴: luria_session_*{user_input}*.pt")
                        continue
                    elif len(matched_files) == 1:
                        path = matched_files[0]
                        print(f"[자동 선택] {path}")
                    else:
                        print(f"[{len(matched_files)}개 파일 발견]")
                        for i, f in enumerate(matched_files, 1):
                            # 파일 수정 시간 표시
                            mtime = os.path.getmtime(f)
                            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                            print(f"  {i}. {f} (수정: {mtime_str})")
                        try:
                            choice = input("번호 선택> ").strip()
                            idx = int(choice) - 1
                            if 0 <= idx < len(matched_files):
                                path = matched_files[idx]
                            else:
                                print("[에러] 잘못된 번호")
                                continue
                        except (ValueError, EOFError, KeyboardInterrupt):
                            print("[취소됨]")
                            continue

            # 파일 로드
            if not os.path.exists(path):
                print(f"[에러] 파일을 찾을 수 없음: {path}")
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
                    print(f"[메모리 복원] items={len(mem.items)} learner={mem.has_learner()}")
                except Exception as e:
                    print(f"[경고] 메모리 복원 실패: {e}")
            print(f"[로드 완료] {path} (step={state.step})")
            log_interaction(csv_path, state.step, s, 0, f"[로드:{path}]", 0, None, None, "load")
            continue

        # /reset 명령어: 세션 초기화
        if s == "/reset":
            state = init_session(model, device)
            pending_x = None; pending_pred = None; staged_y = None
            print("세션 초기화")
            log_interaction(csv_path, state.step, s, 0, "[세션 초기화됨]", 0, None, None, "reset")
            continue

        # /history 명령어: 대화 히스토리 표시
        if s == "/history":
            if not state.history:
                print("히스토리가 비어있습니다.")
                continue
            print("\n=== 최근 대화 히스토리 (최대 20개) ===")
            # 최근 20개만 표시
            recent = state.history[-20:]
            for idx, item in enumerate(recent, 1):
                print(f"{idx:2d}. [step={item.step}] {item.user_input[:40]}")
                print(f"    -> {item.luria_response[:60]}")
                print(f"    (토큰: {item.user_tokens}/{item.response_tokens}, "
                      f"conf={f3(item.conf)}, ent={f3(item.entropy)})")
            print("사용법: /no <번호> 또는 /yes <번호>\n")
            continue

        # /saved "질문" = "정답" 형식 파싱
        saved_pair = parse_saved_command(s)
        if saved_pair:
            question, answer = saved_pair
            print(f"[학습 데이터 등록] Q: {question} → A: {answer}")
            # 즉시 학습 실행
            print("[학습] inner_adapt_z_3brains 실행 중...")
            new_z = adapt_z(model, tok, question, answer, state, steps=args.steps, device=device)
            state.z_for_q = {k: v.to(device) for k, v in new_z.items()}

            # 메모리 커밋
            enc_avg, _ = encode_pair(tok, question, None, device=device)
            enc_avg = model.avg_embed(enc_avg).mean(dim=0).detach()
            key = enc_avg
            val = {"z_slow": state.z_for_q["bridge"].detach()}
            mem.add_with_policy(key, val, state.step, bench_callback=lambda: True)

            print("[학습 완료] z 갱신 및 메모리 처리")
            log_interaction(csv_path, state.step, question, len(tok.encode(question)),
                          answer, len(tok.encode(answer)), None, None, "saved_learning", answer)
            state.step += 1
            continue

        # 정답 세팅
        if s.startswith("/set_y "):
            staged_y = s[len("/set_y "):].strip()
            print(f"[정답 후보 등록] Yc = {staged_y}")
            log_interaction(csv_path, state.step, s, 0, f"[정답 등록: {staged_y}]", 0, None, None, "set_y", staged_y)
            continue

        # 판정 커맨드: /no [번호]
        if s.startswith("/no"):
            # 번호 파싱
            hist_idx = None
            if len(s.split()) > 1:
                try:
                    hist_idx = int(s.split()[1]) - 1  # 1-based to 0-based
                except ValueError:
                    print("[에러] 올바른 번호를 입력하세요. 예: /no 3")
                    continue

            # 히스토리에서 선택 또는 직전 대화 사용
            if hist_idx is not None:
                recent = state.history[-20:]
                if hist_idx < 0 or hist_idx >= len(recent):
                    print(f"[에러] 잘못된 번호입니다. 1-{len(recent)} 사이로 입력하세요.")
                    continue
                hist_item = recent[hist_idx]
                target_x = hist_item.user_input
                target_pred = {
                    "user_tokens": hist_item.user_tokens,
                    "response_tokens": hist_item.response_tokens,
                    "conf": hist_item.conf,
                    "entropy": hist_item.entropy
                }
                print(f"[선택] {hist_idx+1}번: {target_x[:40]} -> {hist_item.luria_response[:40]}")
            else:
                # 직전 대화 사용
                if pending_x is None or pending_pred is None:
                    print("직전 질의/응답이 없습니다. /history로 확인 후 번호를 지정하세요.")
                    continue
                target_x = pending_x
                target_pred = pending_pred

            # 정답 확인
            if not staged_y:
                print("정답이 없습니다. 먼저 `/set_y <정답>`으로 알려주세요.")
                continue

            # 학습(adapt) 경로
            print("[학습] inner_adapt_z_3brains 실행 중...")
            new_z = adapt_z(model, tok, target_x, staged_y, state, steps=args.steps, device=device)
            # 세션 z 갱신
            state.z_for_q = {k: v.to(device) for k, v in new_z.items()}
            # 메모리 커밋(2PC 벤치 콜백 예시)
            enc_avg, _ = encode_pair(tok, target_x, None, device=device)
            enc_avg = model.avg_embed(enc_avg).mean(dim=0).detach()
            key = enc_avg
            val = {"z_slow": state.z_for_q["bridge"].detach()}
            step = state.step
            def bench():
                ent = target_pred.get("entropy", 9.9) or 9.9
                conf = target_pred.get("conf", 0.0) or 0.0
                return (conf >= 0.62) or (ent <= 3.0)
            mem.add_with_policy(key, val, step, bench_callback=bench)
            print("[학습 완료] z 갱신 및 메모리 처리")

            # 로그 기록
            log_interaction(csv_path, state.step, target_x, target_pred.get("user_tokens", 0),
                          staged_y, len(tok.encode(staged_y)),
                          target_pred.get("conf"), target_pred.get("entropy"), "no_adapt", staged_y)

            staged_y = None
            state.step += 1
            maybe_autosave()
            continue

        # 판정 커맨드: /yes [번호]
        if s.startswith("/yes"):
            # 번호 파싱
            hist_idx = None
            if len(s.split()) > 1:
                try:
                    hist_idx = int(s.split()[1]) - 1  # 1-based to 0-based
                except ValueError:
                    print("[에러] 올바른 번호를 입력하세요. 예: /yes 3")
                    continue

            # 히스토리에서 선택 또는 직전 대화 사용
            if hist_idx is not None:
                recent = state.history[-20:]
                if hist_idx < 0 or hist_idx >= len(recent):
                    print(f"[에러] 잘못된 번호입니다. 1-{len(recent)} 사이로 입력하세요.")
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
                print(f"[선택] {hist_idx+1}번: {target_x[:40]} -> {hist_item.luria_response[:40]}")
            else:
                # 직전 대화 사용
                if pending_x is None or pending_pred is None:
                    print("직전 질의/응답이 없습니다. /history로 확인 후 번호를 지정하세요.")
                    continue
                target_x = pending_x
                target_pred = pending_pred

            # 정답으로 간주 → 메모리 정책에 따라 스테이지/커밋
            enc_avg, _ = encode_pair(tok, target_x, None, device=device)
            enc_avg = model.avg_embed(enc_avg).mean(dim=0).detach()
            key = enc_avg
            val = {"z_slow": state.z_for_q["bridge"].detach()}
            step = state.step
            mem.add_with_policy(key, val, step, bench_callback=lambda: True)
            print("[메모리] 정답 샘플 기록 시도")

            # 로그 기록
            log_interaction(csv_path, state.step, target_x, target_pred.get("user_tokens", 0),
                          target_pred.get("text", ""), target_pred.get("response_tokens", 0),
                          target_pred.get("conf"), target_pred.get("entropy"), "yes_confirm")

            state.step += 1
            maybe_autosave()
            continue

        # 일반 질의: 추론
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
        }
        out = predict(model, tok, pending_x, state, device, gen_cfg)
        pending_pred = out

        # 현재 히스토리 번호 (최근 20개 기준)
        hist_num = len(state.history) + 1
        recent_start = max(1, hist_num - 19)  # 최근 20개 범위의 시작 번호
        display_num = hist_num - recent_start + 1  # 표시할 번호 (1-20)

        print(f"[{display_num}] 루리아> {out['text']}")
        print(f"(토큰: 입력={out.get('user_tokens')}, 출력={out.get('response_tokens')}, "
              f"conf={f3(out.get('conf'))}, ent={f3(out.get('entropy'))})")

        # 로그 기록
        log_interaction(csv_path, state.step, pending_x, out.get("user_tokens", 0),
                      out.get("text", ""), out.get("response_tokens", 0),
                      out.get("conf"), out.get("entropy"), "predict")

        # 히스토리에 추가
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

        # 오토세이브
        maybe_autosave()

if __name__ == "__main__":
    main()
