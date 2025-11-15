#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_longrun_writer_5000.py

What's new vs infer_longrun_standalone_logged.py:
- Default 5000 turns, stronger episodic memory, and "anticipatory" init:
  * Language is inferred from context tokens (not from desc string).
  * Retrieval mixes: episodic nearest-neighbors + per-desc prototype + per-lang prototype.
  * Online consolidation into prototypes every turn.
- Focus on "pattern → writing" ability:
  * Adds two simple but writer-ish deterministic tasks (template expansions).
  * Keeps existing transformation challenges but cycles through a wider loop for 5000 turns.
- Logging unchanged in spirit but adds guessed_lang and fused-init diagnostics.

This script stays fully deterministic in targets to allow CE-based monitoring.
"""

import argparse, csv, statistics, json, sys, pathlib, math, os, atexit, random
from typing import Tuple, List, Callable, Optional, Dict

import torch

# allow running next to a local 'uzr' package folder
HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE / 'uzr'))

from uzr.model import UZRModel, ByteTokenizer, KoEnTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from uzr.memory import CompressedMemory, make_sketch
import uzr.tasks as tasklib

LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}

# ---------------------------------------------------------------------------------
# Language detection from context tokens (anticipatory; independent from desc)
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

def resolve_lang_idx(lang_key: str, num_langs: int) -> int:
    idx = LANG2ID.get(lang_key, LANG2ID["base"])
    if idx >= num_langs:
        return LANG2ID["base"]
    return idx

# ---------------------------------------------------------------------------------
# Simple "writer-ish" deterministic rules to emphasize prose patterning
# ---------------------------------------------------------------------------------
def rule_en_template_expand(text: str) -> str:
    # turn a short phrase into a two-clause sentence with light rhythm
    # ex) "latent space patterns" -> "We observed latent space patterns, and we wrote them down carefully."
    toks = text.strip().strip(".")
    if not toks:
        return "We observed nothing, and we wrote nothing."
    return f"We observed {toks}, and we wrote them down carefully."

def rule_ko_template_expand(text: str) -> str:
    # ex) "패턴 인식 결과" -> "우리는 패턴 인식 결과를 확인했고, 그 흐름을 정리했습니다."
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

def avg_embed(model: UZRModel, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h = model.encoder(X)
        return h.mean(dim=1).mean(dim=0)

def encode_str(tok: ByteTokenizer, s: str, device: torch.device) -> torch.Tensor:
    return torch.stack([tok.encode(s)], dim=0).to(device)

# ---------------------------------------------------------------------------------
# Memory init: episodic + prototypes (desc/lang) for anticipatory z
# ---------------------------------------------------------------------------------
class ProtoBank:
    def __init__(self, device: torch.device):
        self.device = device
        self.by_desc: Dict[str, Dict[str, torch.Tensor]] = {}
        self.by_lang: Dict[str, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _ema(prev: Optional[torch.Tensor], new: torch.Tensor, beta: float = 0.9) -> torch.Tensor:
        if prev is None:
            return new.detach().clone()
        return (beta * prev + (1.0 - beta) * new).detach().clone()

    def update_desc(self, desc: str, z_rule: torch.Tensor, z_think: torch.Tensor):
        slot = self.by_desc.setdefault(desc, {})
        slot["z_rule"]  = self._ema(slot.get("z_rule"),  z_rule)
        slot["z_think"] = self._ema(slot.get("z_think"), z_think)

    def update_lang(self, lang_key: str, z_rule: torch.Tensor, z_think: torch.Tensor):
        slot = self.by_lang.setdefault(lang_key, {})
        slot["z_rule"]  = self._ema(slot.get("z_rule"),  z_rule)
        slot["z_think"] = self._ema(slot.get("z_think"), z_think)

    def get_desc(self, desc: str):
        slot = self.by_desc.get(desc)
        if not slot: return None, None
        return slot.get("z_rule"), slot.get("z_think")

    def get_lang(self, lang_key: str):
        slot = self.by_lang.get(lang_key)
        if not slot: return None, None
        return slot.get("z_rule"), slot.get("z_think")

def init_from_memory(mem: Optional[CompressedMemory],
                     enc_avg: torch.Tensor,
                     z_rule_ref: torch.Tensor,
                     z_think_ref: torch.Tensor,
                     lang_id: int,
                     desc: str,
                     proto: ProtoBank,
                     topk_ep: int = 8,
                     w_ep: float = 0.5,
                     w_desc: float = 0.3,
                     w_lang: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Anticipatory init z = weighted mix(episodic avg, desc proto, lang proto).
    Fallback to zeros when components missing.
    """
    device = z_rule_ref.device

    z_ep_rule = z_rule_ref.new_zeros(z_rule_ref.shape)
    z_ep_think = z_think_ref.new_zeros(z_think_ref.shape)

    if mem is not None:
        items = mem.retrieve(enc_avg, topk=topk_ep)
        rules, thinks = [], []
        for it in items:
            val = it.val
            if val.get("lang_id") is not None and int(val["lang_id"]) != int(lang_id):
                continue
            zr = val.get("z_rule"); zt = val.get("z_think")
            if isinstance(zr, torch.Tensor) and isinstance(zt, torch.Tensor):
                rules.append(zr.to(device)); thinks.append(zt.to(device))
        if rules:
            z_ep_rule  = torch.stack(rules,  dim=0).mean(dim=0)
            z_ep_think = torch.stack(thinks, dim=0).mean(dim=0)

    z_dp_rule, z_dp_think = proto.get_desc(desc)
    z_lp_rule, z_lp_think = proto.get_lang(ID2LANG.get(int(lang_id), "base"))

    # Replace None with zeros
    z_dp_rule  = z_ep_rule.new_zeros(z_ep_rule.shape)  if z_dp_rule  is None else z_dp_rule.to(device)
    z_dp_think = z_ep_think.new_zeros(z_ep_think.shape) if z_dp_think is None else z_dp_think.to(device)
    z_lp_rule  = z_ep_rule.new_zeros(z_ep_rule.shape)  if z_lp_rule  is None else z_lp_rule.to(device)
    z_lp_think = z_ep_think.new_zeros(z_ep_think.shape) if z_lp_think is None else z_lp_think.to(device)

    z0_rule  = w_ep*z_ep_rule  + w_desc*z_dp_rule  + w_lang*z_lp_rule
    z0_think = w_ep*z_ep_think + w_desc*z_dp_think + w_lang*z_lp_think
    return z0_rule, z0_think

# ---------------------------------------------------------------------------------
# Challenge suite (extends the original one with writer templates)
# ---------------------------------------------------------------------------------
def build_challenge_suite() -> List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]]:
    suite: List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]] = []

    # Reuse a subset of uzr.tasks factories to keep deterministic targets
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

    # Writer-like templates (deterministic; evaluate CE on transformation to "expanded" sentence)
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

    # Another writer-like for JA to balance
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

    # Numeric / base tasks
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

    # Keep total manageable but diverse
    return suite

# ---------------------------------------------------------------------------------
# Identity seed for memory
# ---------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--turns", type=int, default=5000)                 # 5000 by default
    ap.add_argument("--inner_steps", type=int, default=6)              # slightly more inner steps for adaptation
    ap.add_argument("--lam", type=float, default=2.5e-3, help="Legacy L1 on fused z (fallback)")
    ap.add_argument("--lam_rule", type=float, default=None, help="L1 penalty for z_rule (defaults to --lam)")
    ap.add_argument("--lam_think", type=float, default=None, help="L1 penalty for z_thinking (defaults to --lam)")
    ap.add_argument("--alpha", type=float, default=0.35)               # EMA on slow z
    ap.add_argument("--prox", type=float, default=8e-4)                # proximal shrink
    ap.add_argument("--max_items", type=int, default=120000)           # larger memory
    ap.add_argument("--summary_csv", default="infer_summary.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary.json")
    ap.add_argument("--genlog_csv", default="generation_log.csv")
    ap.add_argument("--identity", default="루리아", help="identity string used for identity seeding")
    ap.add_argument("--offmem", choices=["off", "on"], default="off", help="disable episodic memory when set to on")

    # Anticipation / prototypes
    ap.add_argument("--topk_mem", type=int, default=8)
    ap.add_argument("--w_ep", type=float, default=0.5)
    ap.add_argument("--w_desc", type=float, default=0.3)
    ap.add_argument("--w_lang", type=float, default=0.2)

    args = ap.parse_args()

    lam_rule = args.lam if args.lam_rule is None else args.lam_rule
    lam_think = args.lam if args.lam_think is None else args.lam_think

    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = data["args"]

    # Auto-detect tokenizer from vocab size
    vocab_size = data["model"]["encoder.tok.weight"].shape[0]
    tok = ByteTokenizer(max_len=cfg["max_len"]) if vocab_size == 258 else KoEnTokenizer(max_len=cfg["max_len"])

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

    use_memory = args.offmem != "on"
    mem = CompressedMemory(max_items=args.max_items, device=device) if use_memory else None
    proto = ProtoBank(device=device)
    if use_memory:
        seed_identity(mem, model, tok, device, identity=args.identity)

    challenges = build_challenge_suite()

    z_slow_rule = model.init_z(batch_size=1).to(device)[0].detach()
    z_slow_think = model.init_z_thinking(batch_size=1).to(device)[0].detach()

    fieldnames = [
        "turn",
        "lang",
        "guessed_lang",
        "lang_id",
        "rule_desc",
        "ce_query",
        "conf_context",
        "zslow_rule_l1",
        "zslow_think_l1",
        "zlang_norm",
        "mem_items",
        "init_mix_norm",
    ]
    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader()

    # Generation log (one row per Q item)
    gen_fields = [
        'turn','q_index','lang','guessed_lang','lang_id','rule_desc',
        'input','target','pred',
        'ce_query','conf_context','zslow_rule_l1','zslow_think_l1','zlang_norm','mem_items','init_mix_norm','timestamp'
    ]
    genlog_abspath = os.path.abspath(args.genlog_csv)
    os.makedirs(os.path.dirname(genlog_abspath) or '.', exist_ok=True)
    print(f"[genlog] Writing generation log to: {genlog_abspath}")
    gcsv = open(genlog_abspath, 'w', newline='', encoding='utf-8', buffering=1)
    gw = csv.DictWriter(gcsv, fieldnames=gen_fields)
    gw.writeheader()
    atexit.register(lambda: (gcsv.flush(), gcsv.close()))

    ce_hist = []
    per_lang = {"ko": [], "en": [], "ja": [], "base": [], "mix": []}

    last_lang_idx = resolve_lang_idx("base", model.num_langs)
    last_lang_norm = float(torch.norm(model.lang_embed(torch.tensor([last_lang_idx], device=device))).cpu().item())

    total_challenges = len(challenges)
    rnd = random.Random(2025)

    for t in range(args.turns):
        # Cycle but also permute within each epoch for variety across 5000 turns
        idx = t % total_challenges
        if idx == 0 and t > 0:
            rnd.shuffle(challenges)
        C, Q, desc = challenges[idx]

        # ---- Language: anticipate from context inputs (not from desc) ----
        context_inputs = [x for x,_ in C]
        guessed_lang = detect_lang_from_texts(context_inputs)
        lang_idx = resolve_lang_idx(guessed_lang, model.num_langs)

        def enc_batch(pairs):
            X = torch.stack([tok.encode(x) for x,_ in pairs], dim=0).to(device)
            Y = torch.stack([tok.encode(y) for _,y in pairs], dim=0).to(device)
            return X, Y

        Xc, Yc = enc_batch(C)
        Xq, Yq = enc_batch(Q)

        enc_avg = avg_embed(model, Xc)

        # ---- Anticipatory init from memory + prototypes ----
        z0_rule, z0_think = init_from_memory(mem, enc_avg, z_slow_rule, z_slow_think, lang_idx, desc, proto,
                                             topk_ep=args.topk_mem, w_ep=args.w_ep, w_desc=args.w_desc, w_lang=args.w_lang)
        init_mix_norm = float((torch.norm(z0_rule) + torch.norm(z0_think)).detach().cpu().item())

        z_fast_rule = z0_rule.clone().detach().requires_grad_(True)
        z_fast_think = z0_think.clone().detach().requires_grad_(True)

        conf_val = 0.0
        for _ in range(args.inner_steps):
            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_c = model(Xc, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            loss_c = seq_ce_loss(logits_c, Yc)
            loss_c = loss_c + lam_rule * torch.mean(torch.abs(z_rule_total)) + lam_think * torch.mean(torch.abs(z_think_total))
            grad_rule, grad_think = torch.autograd.grad(loss_c, (z_fast_rule, z_fast_think), retain_graph=False)
            conf = confidence_from_logits(logits_c, Yc).mean()
            conf_val = float(conf.item())
            step = 0.4 + 0.6 * conf_val
            z_fast_rule = z_fast_rule - step * grad_rule
            z_fast_think = z_fast_think - step * grad_think

        with torch.no_grad():
            z_fast_rule = soft_threshold(z_fast_rule, lam_rule * 0.5)
            z_fast_think = soft_threshold(z_fast_think, lam_think * 0.5)

            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_q = model(Xq, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            ce_q = seq_ce_loss(logits_q, Yq).item()

            # ---- Slow Z update
            z_slow_rule = z_slow_rule + args.alpha * (z_fast_rule - z_slow_rule)
            z_slow_think = z_slow_think + args.alpha * (z_fast_think - z_slow_think)
            z_slow_rule = soft_threshold(z_slow_rule, args.prox)
            z_slow_think = soft_threshold(z_slow_think, args.prox)

            # ---- Episodic memory add
            if use_memory:
                fused_slow = model._fuse_z(z_slow_rule, z_slow_think, lang_idx)[0].detach()
                key, val = make_sketch(enc_avg, fused_slow, meta={"desc": desc, "lang": guessed_lang})
                val["z_rule"] = z_slow_rule.detach().clone()
                val["z_think"] = z_slow_think.detach().clone()
                val["lang_id"] = int(lang_idx)
                mem.add(key, val, step=t)

            # ---- Prototype consolidation
            proto.update_desc(desc, z_slow_rule, z_slow_think)
            proto.update_lang(guessed_lang, z_slow_rule, z_slow_think)

            z_rule_l1 = float(torch.sum(torch.abs(z_slow_rule)).cpu().item())
            z_think_l1 = float(torch.sum(torch.abs(z_slow_think)).cpu().item())
            lang_norm = float(torch.norm(model.lang_embed(torch.tensor([lang_idx], device=device))).cpu().item())

        ce_hist.append(ce_q)
        if len(ce_hist) > 200:
            ce_hist.pop(0)

        lg = guessed_lang
        per_lang.setdefault(lg, []).append(ce_q)

        mem_item_count = len(mem.items) if use_memory else 0

        # --- Generation logging (decode predictions per query item) ---
        pred_ids = torch.argmax(logits_q, dim=-1).detach().cpu()
        try:
            pred_texts = [tok.decode(pred_ids[i]) for i in range(pred_ids.shape[0])]
        except Exception:
            pred_texts = [' '.join(map(str, pred_ids[i].tolist())) for i in range(pred_ids.shape[0])]
        now_ts = __import__('datetime').datetime.now().isoformat(timespec='seconds')
        for qi, (qin, qout) in enumerate(Q):
            pred = pred_texts[qi] if qi < len(pred_texts) else ''
            gw.writerow({
                'turn': t + 1,
                'q_index': qi,
                'lang': ID2LANG.get(lang_idx, 'base'),
                'guessed_lang': guessed_lang,
                'lang_id': int(lang_idx),
                'rule_desc': desc,
                'input': qin,
                'target': qout,
                'pred': pred,
                'ce_query': round(ce_q, 4),
                'conf_context': round(conf_val, 4),
                'zslow_rule_l1': round(z_rule_l1, 4),
                'zslow_think_l1': round(z_think_l1, 4),
                'zlang_norm': round(lang_norm, 4),
                'mem_items': mem_item_count,
                'init_mix_norm': round(init_mix_norm, 4),
                'timestamp': now_ts,
            })
        gcsv.flush()

        row = {
            "turn": t + 1,
            "lang": ID2LANG.get(lang_idx, 'base'),
            "guessed_lang": guessed_lang,
            "lang_id": int(lang_idx),
            "rule_desc": desc,
            "ce_query": round(ce_q, 4),
            "conf_context": round(conf_val, 4),
            "zslow_rule_l1": round(z_rule_l1, 4),
            "zslow_think_l1": round(z_think_l1, 4),
            "zlang_norm": round(lang_norm, 4),
            "mem_items": mem_item_count,
            "init_mix_norm": round(init_mix_norm, 4),
        }
        w.writerow(row)

        last_lang_idx = int(lang_idx)
        last_lang_norm = lang_norm

        if (t + 1) % args.summary_every == 0:
            med = statistics.median(ce_hist) if ce_hist else float("nan")
            mean = sum(ce_hist) / len(ce_hist)
            print(f"[turn {t + 1}] CE(mean/med last {len(ce_hist)}): {mean:.3f}/{med:.3f} | z_rule_L1={z_rule_l1:.1f} | z_think_L1={z_think_l1:.1f} | mem={mem_item_count} | init_norm={init_mix_norm:.2f} | '{desc}' | lang={guessed_lang}")

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
        "topk_mem": args.topk_mem,
        "w_ep": args.w_ep,
        "w_desc": args.w_desc,
        "w_lang": args.w_lang,
    }
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    print(f"Summary CSV saved to {args.summary_csv}")
    print(f"Summary JSON saved to {args.summary_json}")
    print("Done. (Writer-focused long-run finished)")

if __name__ == "__main__":
    main()
