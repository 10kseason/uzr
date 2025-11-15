#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_longrun_writer_5000_llm.py

LLM comparison version using Ollama (qwen3:8b by default).
Runs the same transformation tasks as infer_longrun_writer_5000.py
but uses a commercial/open LLM via Ollama API instead of UZR.

This allows direct comparison between:
- UZR (meta-learned transformation model)
- LLM (general-purpose language model with few-shot prompting)

Metrics:
- Exact match accuracy (instead of CE loss)
- Token-level accuracy
- Response time
- Memory/context usage
"""

import argparse
import csv
import statistics
import json
import sys
import pathlib
import os
import atexit
import random
import time
import requests
from typing import Tuple, List, Callable, Optional, Dict

# allow running next to a local 'uzr' package folder
HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE / 'uzr'))

import uzr.tasks as tasklib

LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}

# ---------------------------------------------------------------------------------
# Ollama API helpers
# ---------------------------------------------------------------------------------
class OllamaClient:
    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                print(f"[Ollama] Connected. Available models: {model_names}")
                if not any(model in name for name in model_names):
                    print(f"[Warning] Model '{model}' not found. Available: {model_names}")
            else:
                print(f"[Warning] Could not list Ollama models: {response.status_code}")
        except Exception as e:
            print(f"[Warning] Could not connect to Ollama at {base_url}: {e}")
            print("Make sure Ollama is running with: ollama serve")

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 200) -> Tuple[str, float]:
        """Generate response from Ollama.

        Returns:
            (response_text, response_time_ms)
        """
        start_time = time.time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "").strip()

            elapsed_ms = (time.time() - start_time) * 1000
            return text, elapsed_ms

        except requests.exceptions.Timeout:
            print(f"[Error] Ollama request timed out")
            return "", -1.0
        except Exception as e:
            print(f"[Error] Ollama generation failed: {e}")
            return "", -1.0

# ---------------------------------------------------------------------------------
# Language detection from context tokens
# ---------------------------------------------------------------------------------
def detect_lang_from_texts(texts: List[str]) -> str:
    has_hangul = any(any('가' <= ch <= '힣' for ch in s) for s in texts)
    has_hira   = any(any('\u3040' <= ch <= '\u309F' for ch in s) for s in texts)
    has_kata   = any(any('\u30A0' <= ch <= '\u30FF' for ch in s) for s in texts)
    has_ascii  = any(any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in s) for s in texts)

    if has_hangul: return "ko"
    if has_hira or has_kata: return "ja"
    if has_ascii: return "en"
    return "base"

# ---------------------------------------------------------------------------------
# Simple "writer-ish" deterministic rules (same as UZR version)
# ---------------------------------------------------------------------------------
def rule_en_template_expand(text: str) -> str:
    toks = text.strip().strip(".")
    if not toks:
        return "We observed nothing, and we wrote nothing."
    return f"We observed {toks}, and we wrote them down carefully."

def rule_ko_template_expand(text: str) -> str:
    toks = text.strip().strip(".")
    if not toks:
        return "우리는 아무것도 관찰하지 못했고, 기록도 하지 못했습니다."
    return f"우리는 {toks}를 확인했고, 그 흐름을 정리했습니다."

def rule_ja_template_expand(text: str) -> str:
    toks = text.strip().strip("。")
    if not toks:
        return "わたしたちは何も観測できず、記録もできませんでした。"
    return f"わたしたちは{toks}を観測し、その流れを整理しました。"

# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def make_transformation_task(desc: str, fn: Callable[[str], str], inputs: List[str], n_context: int = 4):
    if len(inputs) < n_context + 2:
        raise ValueError(f"need at least {n_context + 2} inputs, got {len(inputs)} for '{desc}'")
    pairs = [(s, fn(s)) for s in inputs]
    return pairs[:n_context], pairs[n_context:], desc

# ---------------------------------------------------------------------------------
# Challenge suite (same as UZR version)
# ---------------------------------------------------------------------------------
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

    suite.append(make_transformation_task(
        "en: writer template expansion",
        rule_en_template_expand,
        [
            "latent space patterns",
            "structured memory retrieval",
            "symbolic prompts",
            "compositional generalization",
            "token cascade",
            "phase offset coupling",
        ],
    ))

    suite.append(make_transformation_task(
        "ko: 문장 템플릿 확장",
        rule_ko_template_expand,
        [
            "패턴 전이 양상",
            "메모리 재사용 결과",
            "언어 조합 신호",
            "위상차 추세",
            "맥락 적응 정도",
            "추론 루프 안정성",
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

    suite.append(make_transformation_task(
        "ja: 文 を 丁寧体 に 展開",
        rule_ja_template_expand,
        [
            "しゅうごう ぱたーん の けんしょう",
            "きおく の さいりよう",
            "げんご こうせい の しさ",
            "そうおん の きざし",
            "ぶんみゃく おうよう",
            "ないぶ さくご の へいじょう",
        ],
    ))

    sort_desc_fn, _ = tasklib.rule_sort_numeric_tokens(ascending=False)
    suite.append(make_transformation_task(
        "base: sort numeric tokens descending",
        sort_desc_fn,
        [
            "1 5 2 9",
            "3 7 4 4",
            "8 2 6 1",
            "5 5 5 2",
            "9 0 3 3",
            "4 4 9 7",
        ],
    ))

    toggle_fn, _ = tasklib.rule_toggle_case_ascii(step=3)
    suite.append(make_transformation_task(
        "en: toggle ASCII case every 3 tokens",
        toggle_fn,
        [
            "orbit gate signal node",
            "titan forge armor shield",
            "cyber dream micro frame",
            "hidden layer tensor code",
            "vector field gamma delta",
            "alpha beta gamma delta",
        ],
    ))

    return suite

# ---------------------------------------------------------------------------------
# Few-shot prompt construction
# ---------------------------------------------------------------------------------
def build_few_shot_prompt(context_pairs: List[Tuple[str, str]],
                          query_input: str,
                          desc: str = "") -> str:
    """Build a few-shot prompt for the LLM.

    Format:
        Task: [description]

        Examples:
        Input: [x1]
        Output: [y1]

        Input: [x2]
        Output: [y2]

        Now complete this:
        Input: [query]
        Output:
    """
    prompt_parts = []

    if desc:
        prompt_parts.append(f"Task: {desc}\n")

    prompt_parts.append("Examples:")
    for inp, out in context_pairs:
        prompt_parts.append(f"Input: {inp}")
        prompt_parts.append(f"Output: {out}")
        prompt_parts.append("")  # blank line

    prompt_parts.append("Now complete this:")
    prompt_parts.append(f"Input: {query_input}")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)

# ---------------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------------
def exact_match(pred: str, target: str) -> bool:
    """Check if prediction exactly matches target (after stripping)."""
    return pred.strip() == target.strip()

def token_accuracy(pred: str, target: str) -> float:
    """Calculate token-level accuracy."""
    pred_tokens = pred.strip().split()
    target_tokens = target.strip().split()

    if not target_tokens:
        return 1.0 if not pred_tokens else 0.0

    matches = sum(1 for p, t in zip(pred_tokens, target_tokens) if p == t)
    # Penalize length mismatch
    max_len = max(len(pred_tokens), len(target_tokens))
    return matches / max_len if max_len > 0 else 0.0

# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    ap.add_argument("--base_url", default="http://localhost:11434", help="Ollama server URL")
    ap.add_argument("--turns", type=int, default=5000)
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (0=deterministic)")
    ap.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    ap.add_argument("--summary_csv", default="infer_summary_llm.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary_llm.json")
    ap.add_argument("--genlog_csv", default="generation_log_llm.csv")

    args = ap.parse_args()

    # Initialize Ollama client
    client = OllamaClient(model=args.model, base_url=args.base_url)

    challenges = build_challenge_suite()

    fieldnames = [
        "turn",
        "lang",
        "guessed_lang",
        "rule_desc",
        "exact_match_rate",
        "token_accuracy",
        "avg_response_time_ms",
    ]
    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader()

    # Generation log (one row per Q item)
    gen_fields = [
        'turn', 'q_index', 'lang', 'guessed_lang', 'rule_desc',
        'input', 'target', 'pred',
        'exact_match', 'token_acc', 'response_time_ms', 'timestamp'
    ]
    genlog_abspath = os.path.abspath(args.genlog_csv)
    os.makedirs(os.path.dirname(genlog_abspath) or '.', exist_ok=True)
    print(f"[genlog] Writing generation log to: {genlog_abspath}")
    gcsv = open(genlog_abspath, 'w', newline='', encoding='utf-8', buffering=1)
    gw = csv.DictWriter(gcsv, fieldnames=gen_fields)
    gw.writeheader()
    atexit.register(lambda: (gcsv.flush(), gcsv.close()))

    exact_match_hist = []
    token_acc_hist = []
    per_lang = {"ko": [], "en": [], "ja": [], "base": []}

    total_challenges = len(challenges)
    rnd = random.Random(2025)

    for t in range(args.turns):
        # Cycle through challenges
        idx = t % total_challenges
        if idx == 0 and t > 0:
            rnd.shuffle(challenges)
        C, Q, desc = challenges[idx]

        # Detect language from context
        context_inputs = [x for x, _ in C]
        guessed_lang = detect_lang_from_texts(context_inputs)
        lang_id = LANG2ID.get(guessed_lang, LANG2ID["base"])

        # Track metrics for this turn
        turn_exact_matches = []
        turn_token_accs = []
        turn_response_times = []

        # Process each query item
        now_ts = __import__('datetime').datetime.now().isoformat(timespec='seconds')

        for qi, (q_input, q_target) in enumerate(Q):
            # Build prompt
            prompt = build_few_shot_prompt(C, q_input, desc)

            # Generate response
            pred, response_time = client.generate(
                prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            # Clean up prediction (sometimes LLM adds extra text)
            # Try to extract just the output line
            pred_lines = pred.strip().split('\n')
            pred_clean = pred_lines[0].strip() if pred_lines else ""

            # Evaluate
            is_exact = exact_match(pred_clean, q_target)
            tok_acc = token_accuracy(pred_clean, q_target)

            turn_exact_matches.append(1.0 if is_exact else 0.0)
            turn_token_accs.append(tok_acc)
            turn_response_times.append(response_time)

            # Log generation
            gw.writerow({
                'turn': t + 1,
                'q_index': qi,
                'lang': ID2LANG.get(lang_id, 'base'),
                'guessed_lang': guessed_lang,
                'rule_desc': desc,
                'input': q_input,
                'target': q_target,
                'pred': pred_clean,
                'exact_match': int(is_exact),
                'token_acc': round(tok_acc, 4),
                'response_time_ms': round(response_time, 2),
                'timestamp': now_ts,
            })

        gcsv.flush()

        # Aggregate turn metrics
        exact_match_rate = sum(turn_exact_matches) / len(turn_exact_matches) if turn_exact_matches else 0.0
        avg_token_acc = sum(turn_token_accs) / len(turn_token_accs) if turn_token_accs else 0.0
        avg_response_time = sum(turn_response_times) / len(turn_response_times) if turn_response_times else 0.0

        exact_match_hist.append(exact_match_rate)
        token_acc_hist.append(avg_token_acc)

        per_lang.setdefault(guessed_lang, []).append(exact_match_rate)

        # Write summary row
        row = {
            "turn": t + 1,
            "lang": ID2LANG.get(lang_id, 'base'),
            "guessed_lang": guessed_lang,
            "rule_desc": desc,
            "exact_match_rate": round(exact_match_rate, 4),
            "token_accuracy": round(avg_token_acc, 4),
            "avg_response_time_ms": round(avg_response_time, 2),
        }
        w.writerow(row)

        if (t + 1) % args.summary_every == 0:
            recent_exact = exact_match_hist[-min(200, len(exact_match_hist)):]
            recent_token = token_acc_hist[-min(200, len(token_acc_hist)):]

            mean_exact = sum(recent_exact) / len(recent_exact) if recent_exact else 0.0
            mean_token = sum(recent_token) / len(recent_token) if recent_token else 0.0

            print(f"[turn {t + 1}] ExactMatch(mean): {mean_exact:.3f} | TokenAcc(mean): {mean_token:.3f} | ResponseTime: {avg_response_time:.1f}ms | '{desc}' | lang={guessed_lang}")

    fcsv.close()

    def agg(vals):
        if not vals:
            return {"count": 0, "mean": None, "median": None}
        return {"count": len(vals), "mean": sum(vals)/len(vals), "median": statistics.median(vals)}

    summary = {
        "model": args.model,
        "overall_exact_match": agg(exact_match_hist),
        "overall_token_accuracy": agg(token_acc_hist),
        "by_lang": {k: agg(vs) for k, vs in per_lang.items()},
        "turns": args.turns,
        "temperature": args.temperature,
        "csv_path": args.summary_csv,
    }
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    print(f"Summary CSV saved to {args.summary_csv}")
    print(f"Summary JSON saved to {args.summary_json}")
    print(f"Done. LLM ({args.model}) long-run finished")

if __name__ == "__main__":
    main()
