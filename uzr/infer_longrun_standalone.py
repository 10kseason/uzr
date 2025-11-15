#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, statistics, json, sys, pathlib
from pathlib import Path
from datetime import datetime
from collections import deque
import random
from typing import Tuple, List, Callable, Optional

import torch

# allow running next to a local 'uzr' package folder
HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE / 'uzr'))

from uzr.model import UZRModel, ByteTokenizer, KoEnTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from uzr.memory import CompressedMemory, make_sketch
from uzr.meta_core import load_meta_config, AbstainThresholds, maybe_abstain, inner_steps_from_conf
import uzr.tasks as tasklib

LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def lang_of(desc: str) -> str:
    if desc.startswith("ko:"): return "ko"
    if desc.startswith("en:"): return "en"
    if desc.startswith("ja:"): return "ja"
    if desc.startswith("base:"): return "base"
    for tag in ("ko:", "en:", "ja:", "base:"):
        if tag in desc:
            return tag.split(":")[0]
    return "mix"


def avg_embed(model: UZRModel, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h = model.encoder(X)
        return h.mean(dim=1).mean(dim=0)

def _intent_force_from_model(model: UZRModel) -> Optional[bool]:
    try:
        _, toggle = model.identity_intent_control()
    except Exception:
        return None
    if toggle <= -0.5:
        return False
    if toggle >= 0.5:
        return True
    return None


def encode_str(tok: ByteTokenizer, s: str, device: torch.device) -> torch.Tensor:
    return torch.stack([tok.encode(s)], dim=0).to(device)


def resolve_lang_idx(lang_key: str, num_langs: int) -> int:
    idx = LANG2ID.get(lang_key, LANG2ID["base"])
    if idx >= num_langs:
        return LANG2ID["base"]
    return idx




# ---- Challenge suite helpers ----

def make_transformation_task(desc: str, fn: Callable[[str], str], inputs: List[str], n_context: int = 4):
    if len(inputs) < n_context + 2:
        raise ValueError(f"need at least {n_context + 2} inputs, got {len(inputs)} for '{desc}'")
    pairs = [(s, fn(s)) for s in inputs]
    return pairs[:n_context], pairs[n_context:], desc


def append_index_tokens(text: str) -> str:
    tokens = text.split()
    return " ".join(f"{tok}#{i + 1}" for i, tok in enumerate(tokens))


def acronym_with_count(text: str) -> str:
    tokens = [tok for tok in text.split() if tok]
    acronym = "".join(tok[0].upper() for tok in tokens)
    return f"{acronym} ({len(tokens)})"


def sum_of_squares(text: str) -> str:
    parts = text.split()
    nums = [int(p) for p in parts[1:]]
    total = sum(n * n for n in nums)
    return f"sum={total}"


def factorial_rule(text: str) -> str:
    parts = text.split()
    if len(parts) < 2:
        return text
    n = int(parts[1])
    result = 1
    for i in range(2, n + 1):
        result *= i
    return str(result)


def range_span(text: str) -> str:
    parts = text.split()
    nums = [int(p) for p in parts[1:]]
    if not nums:
        return "span=0"
    return f"span={max(nums) - min(nums)}"


def reverse_tokens_imnida(text: str) -> str:
    tokens = list(reversed(text.split()))
    if not tokens or tokens[-1] != "입니다":
        tokens.append("입니다")
    return " ".join(tokens)


def reverse_upper_ascii(text: str) -> str:
    tokens = list(reversed(text.split()))
    out = []
    for tok in tokens:
        out.append(tok.upper() if tok.isascii() else tok)
    return " ".join(out)


def sort_by_length(text: str) -> str:
    tokens = text.split()
    enumerated = list(enumerate(tokens))
    ordered = [tok for _, tok in sorted(enumerated, key=lambda item: (len(item[1]), item[0]))]
    return " ".join(ordered)


def rich_vowel_tokens(text: str) -> str:
    vowels = set("aeiouAEIOU")
    tokens = [tok for tok in text.split() if sum((ch in vowels) for ch in tok) >= 2]
    return " ".join(tokens)


def build_challenge_suite() -> List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]]:
    suite: List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]] = []

    reverse_fn, _ = tasklib.rule_reverse_tokens()
    uppercase2_fn, _ = tasklib.rule_uppercase_every_k(k=2)
    suite.append(make_transformation_task(
        "en: reverse tokens then uppercase every 2nd token",
        tasklib.compose([reverse_fn, uppercase2_fn]),
        [
            "quantum flux valley core",
            "digital nebula memory lane",
            "silent orbit around sun",
            "fractured mirror reflects sky",
            "hidden lantern guides travelers",
            "autumn breeze wakes valley",
        ],
    ))

    dedupe_fn, _ = tasklib.rule_dedupe_preserve_order()
    surround_fn, _ = tasklib.rule_surround_numbers(l="[", r="]")
    suite.append(make_transformation_task(
        "en: dedupe tokens then wrap digits with []",
        tasklib.compose([dedupe_fn, surround_fn]),
        [
            "code code 12 12 sparks",
            "bridge 7 7 lights 7 align",
            "level 3 sensor 3 readings",
            "signal 42 42 18 noise",
            "memory 8 pulse 8 pulse",
            "matrix 1 1 1 reboot",
        ],
    ))

    caesar_fn, _ = tasklib.rule_caesar(shift=2)
    toggle_fn, _ = tasklib.rule_toggle_case_ascii(step=3)
    suite.append(make_transformation_task(
        "en: caesar shift by 2 then toggle ASCII case every 3 tokens",
        tasklib.compose([caesar_fn, toggle_fn]),
        [
            "orbit gate signal node",
            "titan forge armor shield",
            "cyber dream micro frame",
            "hidden layer tensor code",
            "vector field gamma delta",
            "alpha beta gamma delta",
        ],
    ))

    prefix_fn, _ = tasklib.rule_prefix_suffix(prefix="<<", suffix=">>")
    sort_desc_fn, _ = tasklib.rule_sort_numeric_tokens(ascending=False)
    suite.append(make_transformation_task(
        "base: add << >> then sort numeric tokens descending",
        tasklib.compose([prefix_fn, sort_desc_fn]),
        [
            "task 1 5 2 9",
            "batch 3 7 4 4",
            "queue 8 2 6 1",
            "stack 5 5 5 2",
            "memory 9 0 3 3",
            "signal 4 4 9 7",
        ],
    ))

    suite.append(make_transformation_task(
        "en: append #index (1-based) to each token",
        append_index_tokens,
        [
            "merge states to stable output",
            "align cached gradients quickly",
            "plan verify deploy iterate",
            "store encrypted payload fragments",
            "gather labeled events nightly",
            "render summary table view",
        ],
    ))

    suite.append(make_transformation_task(
        "en: build acronym and append token count",
        acronym_with_count,
        [
            "guided latent space exploration",
            "dynamic response routing graph",
            "multi agent rollout evaluation",
            "prompt repair audit module",
            "neural task alignment process",
            "secure channel handshake sequence",
        ],
    ))

    eval_fn, _ = tasklib.rule_eval_simple_math()
    suite.append(make_transformation_task(
        "base: evaluate arithmetic expression",
        eval_fn,
        [
            "calc(12 + 5 * 3)",
            "7*8+9",
            "calc(45-18/3)",
            "63/7+5*2",
            "calc(9*(4+3))",
            "120/(4+6)",
        ],
    ))

    suite.append(make_transformation_task(
        "base: sumsq -> output sum of squares as sum=VALUE",
        sum_of_squares,
        [
            "sumsq 3 4 5",
            "sumsq 10 2",
            "sumsq 7 1 2",
            "sumsq 5 5 5",
            "sumsq 9",
            "sumsq 2 3 6 1",
        ],
    ))

    suite.append(make_transformation_task(
        "base: factorial -> return n!",
        factorial_rule,
        [
            "fact 5",
            "fact 3",
            "fact 6",
            "fact 4",
            "fact 7",
            "fact 2",
        ],
    ))

    suite.append(make_transformation_task(
        "base: range -> output span between max and min",
        range_span,
        [
            "range 3 9 4 12",
            "range -5 7 0",
            "range 10 10 10",
            "range 2 8 -2 5",
            "range 100 60 80",
            "range -3 -7 -1",
        ],
    ))

    josa_fn, _ = tasklib.rule_ko_emphasize_josa()
    append_yo_fn, _ = tasklib.rule_ko_append_yo()
    suite.append(make_transformation_task(
        "ko: 강조된 조사 후 문장 끝에 요 추가",
        tasklib.compose([josa_fn, append_yo_fn]),
        [
            "나 는 모델 을 훈련 하고 있다",
            "우리 는 결과 를 함께 검토 했다",
            "데이터 가 부족 해서 보강 이 필요 하다",
            "오늘 은 평가 를 두 번 진행 했다",
            "연구 자 들 은 회의 에서 계획 을 공유 했다",
            "엔지니어 가 새로운 기능 을 배포 했다",
        ],
    ))

    date_ko_fn, _ = tasklib.rule_ko_date_to_iso()
    suite.append(make_transformation_task(
        "ko: 날짜를 ISO 형태로 변환",
        date_ko_fn,
        [
            "회의 2024년 5월 7일 일정 공유",
            "배포 2023년 12월 3일 완료",
            "보고서 2025.08.19 제출 예정",
            "실험 2022년 1월 9일 시작",
            "점검 2021년 7월 21일 완료",
            "데모 2020.11.30 준비",
        ],
    ))

    box_fn, _ = tasklib.rule_ko_kdigit_box()
    suite.append(make_transformation_task(
        "ko: 숫자를 괄호로 감싸기",
        box_fn,
        [
            "버전 3 업데이트 2 번 진행",
            "테스트 5 단계 1 순위",
            "시도 4 회차 3 성공",
            "수집 7 차례 8 실패",
            "라운드 9 에서 2 득점",
            "피드백 6 건 5 처리",
        ],
    ))

    suite.append(make_transformation_task(
        "ko: 토큰 순서를 뒤집고 마지막에 입니다 추가",
        reverse_tokens_imnida,
        [
            "지금 모델 상태 점검 중",
            "새로운 실험 결과 공유 예정",
            "데이터 전처리 작업 완료",
            "기술 지원 절차 업데이트",
            "환경 설정 매뉴얼 검토",
            "학습 파이프라인 재구성",
        ],
    ))

    hira2kata_fn, _ = tasklib.rule_ja_hira2kata()
    suite.append(make_transformation_task(
        "ja: ひらがな を カタカナ に 変換",
        hira2kata_fn,
        [
            "みずたまり が ひかる",
            "せかい の おんど を はかる",
            "みらい の けいかく を たてる",
            "しすてむ を さいこう ちょうせい する",
            "ほうこく を まとめる",
            "れんしゅう を つづける",
        ],
    ))

    kata2hira_fn, _ = tasklib.rule_ja_kata2hira()
    suite.append(make_transformation_task(
        "ja: カタカナ を ひらがな に 変換",
        kata2hira_fn,
        [
            "テスト データ を チェック",
            "システム ガ オンライン",
            "メモリ を スキャン",
            "アップデート を シェア",
            "モデル ガ スタート",
            "プラン を コンファーム",
        ],
    ))

    fullwidth_fn, _ = tasklib.rule_ja_fullwidth_digits()
    suite.append(make_transformation_task(
        "ja: 半角数字 を 全角 に",
        fullwidth_fn,
        [
            "結果 2024 05 07",
            "ログ 12 30",
            "データ 4096 点",
            "番号 17 42",
            "進捗 88 99",
            "チェック 321",
        ],
    ))

    suite.append(make_transformation_task(
        "ko:en mix - reverse tokens and uppercase ASCII tokens",
        reverse_upper_ascii,
        [
            "모델 update 진행 중",
            "로그 분석 pipeline 점검",
            "데이터 sync 완료 status",
            "시스템 monitor ready 상태",
            "배포 schedule final 확정",
            "사용자 feedback queue 정리",
        ],
    ))

    suite.append(make_transformation_task(
        "base: sort tokens by length ascending",
        sort_by_length,
        [
            "delta alpha sigma beta",
            "graph tree node edge",
            "process queue job task",
            "mirror light sound tone",
            "agent plan code spec",
            "shift align scale move",
        ],
    ))

    suite.append(make_transformation_task(
        "en: keep tokens containing at least two vowels",
        rich_vowel_tokens,
        [
            "aerial roots capture ozone gently",
            "stone paths curve around lakes",
            "audible tones reveal anomalies",
            "unique clouds over silent oceans",
            "vivid aurora covers entire horizon",
            "shallow river carries smooth pebbles",
        ],
    ))

    # --- Additional Korean challenges ---
    suite.append(make_transformation_task(
        "ko: 각 토큰에 #index 붙이기",
        append_index_tokens,
        [
            "빨리 정확히 단단히",
            "데이터 품질 지표 확인",
            "모델 출력 예시 정리",
            "로그 오류 원인 분석",
            "배포 전 최종 점검",
            "성과 지표 보고 준비",
        ],
    ))

    suite.append(make_transformation_task(
        "ko: 토큰 길이순 정렬(오름차순)",
        sort_by_length,
        [
            "지표 품질 점검 완료",
            "모델 성능 보고 준비",
            "데이터셋 샘플 추출",
            "문서 업데이트 계획",
            "결과 재현 실험 구성",
            "추론 파이프라인 정리",
        ],
    ))

    if len(suite) < 20:
        raise AssertionError(f"expected at least 20 challenges, got {len(suite)}")

    return suite


def init_from_retrieval_multi(mem: Optional[CompressedMemory], enc_avg: torch.Tensor, z_rule_ref: torch.Tensor,
                               z_think_ref: torch.Tensor, lang_id: int, topk: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if mem is None:
        return z_rule_ref.new_zeros(z_rule_ref.shape), z_think_ref.new_zeros(z_think_ref.shape)
    k = mem.topk if (topk is None) else topk
    items = mem.retrieve(enc_avg, topk=k)
    if not items:
        return z_rule_ref.new_zeros(z_rule_ref.shape), z_think_ref.new_zeros(z_think_ref.shape)

    def collect(candidates, match_lang: bool):
        rules, thinks = [], []
        for it in candidates:
            val = it.val
            if match_lang:
                if val.get("lang_id") is None or int(val["lang_id"]) != int(lang_id):
                    continue
            z_rule = val.get("z_rule")
            z_think = val.get("z_think")
            if isinstance(z_rule, torch.Tensor) and isinstance(z_think, torch.Tensor):
                rules.append(z_rule.to(z_rule_ref.device))
                thinks.append(z_think.to(z_think_ref.device))
        if rules:
            return (torch.stack(rules, dim=0).mean(dim=0), torch.stack(thinks, dim=0).mean(dim=0))
        return None

    match = collect(items, match_lang=True)
    if match is None:
        match = collect(items, match_lang=False)
    if match is None:
        return z_rule_ref.new_zeros(z_rule_ref.shape), z_think_ref.new_zeros(z_think_ref.shape)
    return match


def seed_identity(mem: CompressedMemory, model: UZRModel, tok: ByteTokenizer, device: torch.device, identity: str = "루리아"):
    phrases = [
        (f"나는 {identity}입니다.", "ko"),
        (f"I am {identity}.", "en"),
        (f"わたしは{identity}です。", "ja"),
    ]
    for text, lang_key in phrases:
        lang_idx = resolve_lang_idx(lang_key, model.num_langs)
        X = encode_str(tok, text, device)
        emb = avg_embed(model, X)
        z_rule = model.init_z(batch_size=1).to(device)[0].detach()
        z_rule.zero_()
        z_think = model.init_z_thinking(batch_size=1).to(device)[0].detach()
        z_think.zero_()
        fused = model._fuse_z(z_rule, z_think, lang_idx)[0].detach()
        key, val = make_sketch(emb, fused, meta={"identity": identity, "phrase": text, "lang": lang_key})
        val["z_rule"] = z_rule.clone()
        val["z_think"] = z_think.clone()
        val["lang_id"] = int(lang_idx)
        mem.add(key, val, step=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--lam", type=float, default=3e-3, help="Legacy L1 on fused z (fallback)")
    ap.add_argument("--lam_rule", type=float, default=None, help="L1 penalty for z_rule (defaults to --lam)")
    ap.add_argument("--lam_think", type=float, default=None, help="L1 penalty for z_thinking (defaults to --lam)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--prox", type=float, default=1e-3)
    ap.add_argument("--max_items", type=int, default=32000)
    ap.add_argument("--summary_csv", default="infer_summary.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary.json")
    ap.add_argument("--identity", default="루리아", help="identity string used for identity seeding")
    ap.add_argument("--offmem", choices=["off", "on"], default="off", help="disable episodic memory when set to on")
    args = ap.parse_args()

    lam_rule = args.lam if args.lam_rule is None else args.lam_rule
    lam_think = args.lam if args.lam_think is None else args.lam_think

    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = data["args"]
    rdw = data.get("model", {}).get("readout.weight")
    rows = rdw.size(0) if isinstance(rdw, torch.Tensor) else None
    tok = ByteTokenizer(max_len=cfg["max_len"]) if rows == 258 else KoEnTokenizer(max_len=cfg["max_len"])
    def _build_model(override_dims=None):
        od = override_dims or {}
        return UZRModel(
            tok.vocab_size,
            d_model=cfg["d_model"],
            z_dim=cfg["z_dim"],
            max_len=cfg["max_len"],
            z_think_dim=cfg.get("z_think_dim", 64),
            z_lang_dim=cfg.get("z_lang_dim", 32),
            num_langs=cfg.get("num_langs", len(LANG2ID)),
            identity_self_dim=cfg.get("identity_self_dim", 32),
            identity_intent_dim=cfg.get("identity_intent_dim"),
            z_slow_lang_dim=od.get("z_slow_lang_dim", cfg.get("z_slow_lang_dim", 96)),
            z_slow_logic_dim=od.get("z_slow_logic_dim", cfg.get("z_slow_logic_dim", 96)),
            z_bridge_dim=od.get("z_bridge_dim", cfg.get("z_bridge_dim", 64)),
        )

    model = _build_model()
    try:
        model.load_state_dict(data["model"])
    except RuntimeError as e:
        msg = str(e)
        if "fuse_proj_3brains.weight" in msg:
            w = data["model"].get("fuse_proj_3brains.weight")
            if isinstance(w, torch.Tensor):
                in_dim = int(w.shape[1])
                half = in_dim // 2
                override = {
                    "z_slow_lang_dim": half,
                    "z_slow_logic_dim": half,
                    "z_bridge_dim": in_dim - half,
                }
                model = _build_model(override_dims=override)
                model.load_state_dict(data["model"])
            else:
                raise
        else:
            raise
    device = torch.device(args.device)
    model.to(device).eval()

    use_memory = args.offmem != "on"
    mem = CompressedMemory(max_items=args.max_items, device=device) if use_memory else None
    tail_q = deque(maxlen=2000) if use_memory else None
    if use_memory:
        seed_identity(mem, model, tok, device, identity=args.identity)

    challenges = build_challenge_suite()

    z_slow_rule = model.init_z(batch_size=1).to(device)[0].detach()
    z_slow_think = model.init_z_thinking(batch_size=1).to(device)[0].detach()

    fieldnames = [
        "turn",
        "lang",
        "lang_id",
        "rule_desc",
        "ce_query",
        "conf_context",
        # Dynamic inner-steps logging
        "conf0","chosen_steps","tries","best_conf",
        # Self-eval extended metrics
        "conf_self_c","ent_c","brier_c","abstain_c",
        "conf_self_q","ent_q","brier_q","abstain_q",
        "zslow_rule_l1",
        "zslow_think_l1",
        "zlang_norm",
        "mem_items",
        "gate_pass","compute_tokens",
    ]
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

    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader()

    ce_hist = []
    per_lang = {"ko": [], "en": [], "ja": [], "base": [], "mix": []}

    last_lang_idx = resolve_lang_idx("base", model.num_langs)
    last_lang_norm = float(torch.norm(model.lang_embed(torch.tensor([last_lang_idx], device=device))).cpu().item())

    total_challenges = len(challenges)
    cfg_meta = load_meta_config()
    thr = AbstainThresholds(conf_min=cfg_meta["conf_min"], ent_max=cfg_meta["ent_max"])

    for t in range(args.turns):
        C, Q, desc = challenges[t % total_challenges]

        def enc_batch(pairs):
            X = torch.stack([tok.encode(x) for x,_ in pairs], dim=0).to(device)
            Y = torch.stack([tok.encode(y) for _,y in pairs], dim=0).to(device)
            return X, Y

        Xc, Yc = enc_batch(C)
        Xq, Yq = enc_batch(Q)

        enc_avg = avg_embed(model, Xc)
        fused_slow = None

        lang_key = lang_of(desc)
        lang_idx = resolve_lang_idx(lang_key, model.num_langs)
        z_fast_rule_0, z_fast_think_0 = init_from_retrieval_multi(mem, enc_avg, z_slow_rule, z_slow_think, lang_idx)
        z_fast_rule = z_fast_rule_0.clone().detach().requires_grad_(True)
        z_fast_think = z_fast_think_0.clone().detach().requires_grad_(True)

        conf_val = 0.0
        # Dynamic inner-step budget from initial confidence
        conf0_vec = model.confidence(Xc)
        if conf0_vec is None:
            with torch.no_grad():
                logits0 = model(Xc, {"rule": z_slow_rule + z_fast_rule, "think": z_slow_think + z_fast_think, "lang_id": lang_idx})
                conf0_vec = confidence_from_logits(logits0, Yc)
        conf0 = float(conf0_vec.mean().item())
        chosen_steps = inner_steps_from_conf(conf0, s_max=int(args.inner_steps), s_min=0, k=10.0, mid=0.7)
        tries = 0
        best_conf = conf0
        conf_self_c_vec = None
        ent_c = None
        brier_c_val = None
        abstain_c_mask = None
        for _ in range(chosen_steps):
            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_c = model(Xc, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            loss_c = seq_ce_loss(logits_c, Yc)
            loss_c = loss_c + lam_rule * torch.mean(torch.abs(z_rule_total)) + lam_think * torch.mean(torch.abs(z_think_total))
            grad_rule, grad_think = torch.autograd.grad(loss_c, (z_fast_rule, z_fast_think), retain_graph=False)
            # Self-eval confidence (fallback to logits proxy if head missing)
            conf_self_c_vec = model.confidence(Xc)
            if conf_self_c_vec is None:
                conf_self_c_vec = confidence_from_logits(logits_c, Yc)
            conf = conf_self_c_vec.mean()
            conf_val = float(conf.item())
            step = 0.4 + 0.6 * conf_val
            z_fast_rule = z_fast_rule - step * grad_rule
            z_fast_think = z_fast_think - step * grad_think
            tries += 1
            if conf_val > best_conf:
                best_conf = conf_val
            if conf_val >= 0.8:
                break

        with torch.no_grad():
            z_fast_rule = soft_threshold(z_fast_rule, lam_rule * 0.5)
            z_fast_think = soft_threshold(z_fast_think, lam_think * 0.5)

            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_q = model(Xq, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            ce_q = seq_ce_loss(logits_q, Yq).item()
            # Compute self-eval/query metrics
            if conf_self_c_vec is None:
                conf_self_c_vec = model.confidence(Xc)
                if conf_self_c_vec is None:
                    conf_self_c_vec = confidence_from_logits(logits_c, Yc)
            ent_c = model.sequence_entropy(logits_c)
            brier_c_val = model.brier_from_logits_conf(logits_c, Yc, conf_self_c_vec)
            abstain_c_mask = maybe_abstain(conf_self_c_vec, ent_c, thr)

            conf_self_q_vec = model.confidence(Xq)
            if conf_self_q_vec is None:
                conf_self_q_vec = confidence_from_logits(logits_q, Yq)
            ent_q = model.sequence_entropy(logits_q)
            brier_q_val = model.brier_from_logits_conf(logits_q, Yq, conf_self_q_vec)
            abstain_q_mask = maybe_abstain(conf_self_q_vec, ent_q, thr)

            z_slow_rule = z_slow_rule + args.alpha * (z_fast_rule - z_slow_rule)
            z_slow_think = z_slow_think + args.alpha * (z_fast_think - z_slow_think)
            z_slow_rule = soft_threshold(z_slow_rule, args.prox)
            z_slow_think = soft_threshold(z_slow_think, args.prox)

            if use_memory:
                fused_slow = model._fuse_z(z_slow_rule, z_slow_think, lang_idx)[0].detach()
                meta = {
                    "desc": desc, "lang": lang_key, "ce_q": float(ce_q), "conf": float(conf_val),
                    "conf0": conf0, "chosen_steps": int(chosen_steps), "tries": int(tries), "best_conf": float(best_conf),
                    "conf_self_c": float(conf_self_c_vec.mean().item()),
                    "ent_c": float(ent_c.mean().item()),
                    "brier_c": float(brier_c_val.item()),
                    "conf_self_q": float(conf_self_q_vec.mean().item()),
                    "ent_q": float(ent_q.mean().item()),
                    "brier_q": float(brier_q_val.item()),
                }
                key, val = make_sketch(enc_avg, fused_slow, meta=meta)
                val["z_rule"] = z_slow_rule.detach().clone()
                val["z_think"] = z_slow_think.detach().clone()
                val["lang_id"] = int(lang_idx)
                if hasattr(mem, "add_with_policy"):
                    mem_meta = dict(meta)
                    mem_meta["luria_intent_force"] = _intent_force_from_model(model_cpu)
                    mem.add_with_policy(key, val, step=t, meta=mem_meta)
                else:
                    mem.add(key, val, step=t)

            z_rule_l1 = float(torch.sum(torch.abs(z_slow_rule)).cpu().item())
            z_think_l1 = float(torch.sum(torch.abs(z_slow_think)).cpu().item())
            lang_norm = float(torch.norm(model.lang_embed(torch.tensor([lang_idx], device=device))).cpu().item())

        ce_hist.append(ce_q)
        if len(ce_hist) > 200:
            ce_hist.pop(0)
        # Tail-bucket collection: add top-10% CE samples into tail queue
        if use_memory and fused_slow is not None and tail_q is not None:
            window = ce_hist[-200:]
            if window:
                srt = sorted(window)
                k = max(0, min(len(srt) - 1, int(round(0.9 * (len(srt) - 1)))))
                p90 = srt[k]
                if ce_q >= p90:
                    try:
                        tail_q.append({
                            "enc_avg": enc_avg.detach().cpu(),
                            "z": fused_slow.detach().cpu(),
                            "desc": desc,
                            "lang": lg,
                            "ce": float(ce_q),
                            "conf": float(conf_val),
                        })
                    except Exception:
                        pass

        lg = lang_key
        per_lang.setdefault(lg, []).append(ce_q)

        mem_item_count = len(mem.items) if use_memory else 0

        # Approx compute tokens consumed by inner loops (context forward per try)
        try:
            Bc, Tc = Xc.shape
            compute_tokens = int(tries * Bc * Tc)
        except Exception:
            compute_tokens = tries

        row = {
            "turn": t + 1,
            "lang": lg,
            "lang_id": int(lang_idx),
            "rule_desc": desc,
            "ce_query": round(ce_q, 4),
            "conf_context": round(conf_val, 4),
            "conf0": round(conf0, 4),
            "chosen_steps": int(chosen_steps),
            "tries": int(tries),
            "best_conf": round(float(best_conf), 4),
            "conf_self_c": round(float(conf_self_c_vec.mean().item()), 4) if conf_self_c_vec is not None else None,
            "ent_c": round(float(ent_c.mean().item()), 4) if ent_c is not None else None,
            "brier_c": round(float(brier_c_val.item()), 4) if brier_c_val is not None else None,
            "abstain_c": int(abstain_c_mask.float().mean().item() > 0) if abstain_c_mask is not None else 0,
            "conf_self_q": round(float(conf_self_q_vec.mean().item()), 4),
            "ent_q": round(float(ent_q.mean().item()), 4),
            "brier_q": round(float(brier_q_val.item()), 4),
            "abstain_q": int(abstain_q_mask.float().mean().item() > 0),
            "zslow_rule_l1": round(z_rule_l1, 4),
            "zslow_think_l1": round(z_think_l1, 4),
            "zlang_norm": round(lang_norm, 4),
            "mem_items": mem_item_count,
            "gate_pass": int((abstain_c_mask.float().mean().item() <= 0.5) and (abstain_q_mask.float().mean().item() <= 0.5)),
            "compute_tokens": int(compute_tokens),
        }
        w.writerow(row)

        last_lang_idx = int(lang_idx)
        last_lang_norm = lang_norm

        if (t + 1) % args.summary_every == 0:
            # Promote a few tail-bucket samples periodically
            if use_memory and tail_q and len(tail_q) > 0:
                k = min(5, len(tail_q))
                for s in random.sample(list(tail_q), k=k):
                    meta_tail = {"desc": s["desc"], "lang": s["lang"], "ce_q": s["ce"], "conf": s["conf"], "bucket": "tail"}
                    key_t, val_t = make_sketch(s["enc_avg"], s["z"], meta=meta_tail)
                    if hasattr(mem, "add_with_policy"):
                        tail_meta = dict(meta_tail)
                        tail_meta["luria_intent_force"] = _intent_force_from_model(model_cpu)
                        mem.add_with_policy(key_t, val_t, step=t, meta=tail_meta)
                    else:
                        mem.add(key_t, val_t, step=t)
            med = statistics.median(ce_hist) if ce_hist else float("nan")
            mean = sum(ce_hist) / len(ce_hist)
            print(f"[turn {t + 1}] CE(mean/med over last {len(ce_hist)}): {mean:.3f}/{med:.3f} | z_rule_L1={z_rule_l1:.1f} | z_think_L1={z_think_l1:.1f} | mem={mem_item_count} | rule='{desc}'")

    fcsv.close()

    def agg(vals):
        if not vals:
            return {"count": 0, "mean": None, "median": None}
        return {"count": len(vals), "mean": sum(vals)/len(vals), "median": statistics.median(vals)}

    summary = {
        "overall": agg([v for vs in per_lang.values() for v in vs]),
        "by_lang": {k: agg(vs) for k,vs in per_lang.items()},
        "turns": args.turns,
        "identity_seeded": args.identity,
        "offmem_mode": args.offmem,
        "csv_path": args.summary_csv,
        "final_z_rule_l1": z_rule_l1,
        "final_z_think_l1": z_think_l1,
        "final_lang_id": last_lang_idx,
        "final_lang_embed_norm": last_lang_norm,
    }
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    print(f"Summary CSV saved to {args.summary_csv}")
    print(f"Summary JSON saved to {args.summary_json}")
    print("Done. (Standalone long-run finished)")

if __name__ == "__main__":
    main()
