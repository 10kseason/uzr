#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_longrun_standalone_ko.py
Korean-only long-run driver for UZR ("Luria").
"""

import argparse, csv, statistics, json, sys, pathlib, os, atexit, random, hashlib
from typing import Tuple, List, Callable, Optional, Dict

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE / 'uzr'))

from uzr.model import UZRModel, ByteTokenizer, KoEnTokenizer, seq_ce_loss, soft_threshold, confidence_from_logits
from uzr.memory import CompressedMemory, make_sketch
import uzr.tasks as tasklib

LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}

def _stable_pick(options: List[str], key: str) -> str:
    import hashlib
    h = hashlib.sha1(key.encode('utf-8')).hexdigest()
    idx = int(h[:8], 16) % max(1, len(options))
    return options[idx]

def rule_ko_paragraph_expand(text: str, n_sentences: int = 4) -> str:
    topic = text.strip().strip(" .,!?:;\"'()[]{}")
    if not topic:
        topic = "관찰한 현상"
    leads = [
        f"오늘 우리는 {topic}에 대해 차분히 살펴보았습니다.",
        f"우리는 먼저 {topic}의 윤곽을 가만히 그려보았습니다.",
        f"{topic}을(를) 중심에 두고, 조심스럽게 맥락을 정리했습니다.",
    ]
    bridges = [
        f"무엇보다 이 주제는 주변 맥락과 맞물려 서서히 의미를 드러냈고, 우리는 그 흐름을 기록했습니다.",
        f"관련된 신호들을 겹겹이 확인하면서, 핵심과 주변을 구분하려고 애썼습니다.",
        f"세부를 급히 확정하지 않고, 작게 반복하며 조용히 윤곽을 다듬었습니다.",
    ]
    insights = [
        f"그 과정에서 드러난 핵심은 {topic}이(가) 점진적으로 안정화된다는 점이었습니다.",
        f"관찰을 이어가 보니, {topic}은(는) 맥락이 바뀌어도 일관된 결을 유지했습니다.",
        f"여러 조건을 바꿔 보았지만, {topic}의 중심 의미는 크게 흔들리지 않았습니다.",
    ]
    finals = [
        f"따라서 우리는 {topic}을(를) 기준으로 조금 더 천천히, 그러나 분명한 방향으로 글을 이어가기로 했습니다.",
        f"결론적으로 {topic}을(를) 둘러싼 작은 차이들을 존중하면서, 안정된 서술을 계속하겠습니다.",
        f"이제 {topic}을(를) 출발점으로 삼아, 다음 구간의 생각을 차분히 전개하겠습니다.",
    ]
    s1 = _stable_pick(leads, topic)
    s2 = _stable_pick(bridges, topic + "#b")
    s3 = _stable_pick(insights, topic + "#i")
    s4 = _stable_pick(finals, topic + "#f")
    out = " ".join([s for s in [s1, s2, s3, s4][:max(1, n_sentences)]])
    return out

def rule_ko_reason_result(text: str) -> str:
    topic = text.strip().strip(" .,!?:;\"'()[]{}")
    if not topic:
        topic = "관찰된 패턴"
    because = [
        f"{topic}은(는) 반복 관찰에서 같은 양상을 보였습니다.",
        f"{topic}에 대한 신호가 시간에 따라 고르게 누적되었습니다.",
        f"{topic}과(와) 관련된 근거가 서로 모순되지 않았습니다.",
    ]
    therefore = [
        f"그래서 우리는 {topic}을(를) 임시 기준으로 채택해도 무방하다고 판단했습니다.",
        f"따라서 이후 단계에서도 {topic}을(를) 중심 축으로 사용하겠습니다.",
        f"결과적으로 {topic}을(를) 설명의 첫 단추로 두겠습니다.",
    ]
    s1 = _stable_pick(because, topic + "#r1")
    s2 = _stable_pick(therefore, topic + "#r2")
    return f"{s1} {s2}"

def compose(fs: List[Callable[[str], str]]):
    def f(x: str):
        y = x
        for g in fs:
            try:
                y = g(y)
            except Exception:
                pass
        return y
    return f

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
                     topk_ep: int = 12,
                     w_ep: float = 0.35,
                     w_desc: float = 0.45,
                     w_lang: float = 0.20):
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
    z_lp_rule, z_lp_think = proto.get_lang("ko")

    z_dp_rule  = z_ep_rule.new_zeros(z_ep_rule.shape)  if z_dp_rule  is None else z_dp_rule.to(device)
    z_dp_think = z_ep_think.new_zeros(z_ep_think.shape) if z_dp_think is None else z_dp_think.to(device)
    z_lp_rule  = z_ep_rule.new_zeros(z_ep_rule.shape)  if z_lp_rule  is None else z_lp_rule.to(device)
    z_lp_think = z_ep_think.new_zeros(z_ep_think.shape) if z_lp_think is None else z_lp_think.to(device)

    z0_rule  = w_ep*z_ep_rule  + w_desc*z_dp_rule  + w_lang*z_lp_rule
    z0_think = w_ep*z_ep_think + w_desc*z_dp_think + w_lang*z_lp_think
    return z0_rule, z0_think

def build_challenge_suite_ko():
    suite = []
    suite.append(make_transformation_task(
        "ko: 단락 템플릿 확장",
        rule_ko_paragraph_expand,
        [
            "패턴 전이 양상", "메모리 재사용 결과", "언어 조합 신호", "위상차 추세", "맥락 적응 정도", "추론 루프 안정성",
            "장기 관찰의 함의", "개념 농도 변화", "표현의 매끄러움", "의미적 일관성"
        ],
        n_context=4
    ))
    suite.append(make_transformation_task(
        "ko: 이유-결과 서술",
        rule_ko_reason_result,
        [
            "패턴 전이 양상", "메모리 재사용 결과", "언어 조합 신호", "위상차 추세", "맥락 적응 정도", "추론 루프 안정성",
            "장기 관찰의 함의", "개념 농도 변화", "표현의 매끄러움", "의미적 일관성"
        ],
        n_context=4
    ))
    # Optional KO punct normalization if provided by uzr.tasks
    if hasattr(tasklib, "rule_normalize_korean_punct"):
        norm_ko_fn, _ = tasklib.rule_normalize_korean_punct()
        suite.append(make_transformation_task(
            "ko: 한글 문장부호 정리",
            norm_ko_fn,
            [
                "우리는 관찰을 이어가며 , 의미를 분명히 했다 .",
                "작은 노이즈 를 천천히 걷어내며 , 중심을 확인했다 .",
                "반복 측정 을 통해 흐름 을 기록 했다 .",
                "결론 은 서두르지 않고 , 차분히 적었다 .",
                "조사 와 어미 를 자연스럽게 맞췄다 .",
                "문장 구조 를 길고 섬세 하게 유지했다 .",
            ],
            n_context=4
        ))
    return suite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="uzr_ckpt.pt")
    ap.add_argument("--turns", type=int, default=4000)
    ap.add_argument("--inner_steps", type=int, default=6)
    ap.add_argument("--lam", type=float, default=2.5e-3)
    ap.add_argument("--lam_rule", type=float, default=None)
    ap.add_argument("--lam_think", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--prox", type=float, default=8e-4)
    ap.add_argument("--max_items", type=int, default=140000)
    ap.add_argument("--summary_csv", default="infer_summary_ko.csv")
    ap.add_argument("--summary_every", type=int, default=50)
    ap.add_argument("--summary_json", default="infer_summary_ko.json")
    ap.add_argument("--genlog_csv", default="generation_log_ko.csv")
    ap.add_argument("--identity", default="루리아")
    ap.add_argument("--offmem", choices=["off", "on"], default="off")
    ap.add_argument("--topk_mem", type=int, default=12)
    ap.add_argument("--w_ep", type=float, default=0.35)
    ap.add_argument("--w_desc", type=float, default=0.45)
    ap.add_argument("--w_lang", type=float, default=0.20)

    args = ap.parse_args()

    lam_rule = args.lam if args.lam_rule is None else args.lam_rule
    lam_think = args.lam if args.lam_think is None else args.lam_think

    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = data["args"]
    ckpt_vocab = data["model"]["encoder.tok.weight"].shape[0]

    # Auto-detect tokenizer from vocab size
    tok = ByteTokenizer(max_len=cfg["max_len"]) if ckpt_vocab == 258 else KoEnTokenizer(max_len=cfg["max_len"])

    model = UZRModel(
        ckpt_vocab,
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

    # Fixed Korean lang
    lang_idx = min(LANG2ID["ko"], model.num_langs - 1)

    # Seed identity
    if use_memory:
        Xid = encode_str(tok, f"나는 {args.identity}입니다.", device)
        emb = avg_embed(model, Xid)
        z_rule = model.init_z(batch_size=1).to(device)[0].detach(); z_rule.zero_()
        z_think = model.init_z_thinking(batch_size=1).to(device)[0].detach(); z_think.zero_()
        fused = model._fuse_z(z_rule, z_think, lang_idx)[0].detach()
        key, val = make_sketch(emb, fused, meta={"identity": args.identity, "lang": "ko"})
        val["z_rule"] = z_rule.clone(); val["z_think"] = z_think.clone(); val["lang_id"] = int(lang_idx)
        mem.add(key, val, step=-1)

    challenges = build_challenge_suite_ko()

    z_slow_rule = model.init_z(batch_size=1).to(device)[0].detach()
    z_slow_think = model.init_z_thinking(batch_size=1).to(device)[0].detach()

    fieldnames = [
        "turn","lang","lang_id","rule_desc","ce_query","conf_context",
        "zslow_rule_l1","zslow_think_l1","zlang_norm","mem_items","init_mix_norm","n_q"
    ]
    fcsv = open(args.summary_csv, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=fieldnames); w.writeheader()

    gen_fields = [
        'turn','q_index','lang','lang_id','rule_desc',
        'input','target','pred',
        'ce_query','conf_context','zslow_rule_l1','zslow_think_l1','zlang_norm','mem_items','init_mix_norm','timestamp'
    ]
    genlog_abspath = os.path.abspath(args.genlog_csv)
    os.makedirs(os.path.dirname(genlog_abspath) or '.', exist_ok=True)
    print(f"[genlog] Writing generation log to: {genlog_abspath}")
    gcsv = open(genlog_abspath, 'w', newline='', encoding='utf-8', buffering=1)
    gw = csv.DictWriter(gcsv, fieldnames=gen_fields); gw.writeheader()
    atexit.register(lambda: (gcsv.flush(), gcsv.close()))

    ce_hist = []
    total_challenges = len(challenges)
    rnd = random.Random(2025)

    for t in range(args.turns):
        idx = t % total_challenges
        if idx == 0 and t > 0:
            rnd.shuffle(challenges)
        C, Q, desc = challenges[idx]

        def enc_batch(pairs):
            X = torch.stack([tok.encode(x) for x,_ in pairs], dim=0).to(device)
            Y = torch.stack([tok.encode(y) for _,y in pairs], dim=0).to(device)
            return X, Y

        Xc, Yc = enc_batch(C)
        Xq, Yq = enc_batch(Q)

        enc_avg = avg_embed(model, Xc)

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
            loss_c = loss_c + (args.lam if args.lam_rule is None else args.lam_rule) * torch.mean(torch.abs(z_rule_total))                            + (args.lam if args.lam_think is None else args.lam_think) * torch.mean(torch.abs(z_think_total))
            grad_rule, grad_think = torch.autograd.grad(loss_c, (z_fast_rule, z_fast_think), retain_graph=False)
            conf = confidence_from_logits(logits_c, Yc).mean()
            conf_val = float(conf.item())
            step = 0.4 + 0.6 * conf_val
            z_fast_rule = z_fast_rule - step * grad_rule
            z_fast_think = z_fast_think - step * grad_think

        with torch.no_grad():
            lam_rule = args.lam if args.lam_rule is None else args.lam_rule
            lam_think = args.lam if args.lam_think is None else args.lam_think
            z_fast_rule = soft_threshold(z_fast_rule, lam_rule * 0.5)
            z_fast_think = soft_threshold(z_fast_think, lam_think * 0.5)

            z_rule_total = z_slow_rule + z_fast_rule
            z_think_total = z_slow_think + z_fast_think
            logits_q = model(Xq, {"rule": z_rule_total, "think": z_think_total, "lang_id": lang_idx})
            ce_q = seq_ce_loss(logits_q, Yq).item()

            z_slow_rule = z_slow_rule + args.alpha * (z_fast_rule - z_slow_rule)
            z_slow_think = z_slow_think + args.alpha * (z_fast_think - z_slow_think)
            z_slow_rule = soft_threshold(z_slow_rule, args.prox)
            z_slow_think = soft_threshold(z_slow_think, args.prox)

            if mem is not None:
                fused_slow = model._fuse_z(z_slow_rule, z_slow_think, lang_idx)[0].detach()
                key, val = make_sketch(enc_avg, fused_slow, meta={"desc": desc, "lang": "ko"})
                val["z_rule"] = z_slow_rule.detach().clone()
                val["z_think"] = z_slow_think.detach().clone()
                val["lang_id"] = int(lang_idx)
                mem.add(key, val, step=t)

            proto.update_desc(desc, z_slow_rule, z_slow_think)
            proto.update_lang("ko", z_slow_rule, z_slow_think)

            z_rule_l1 = float(torch.sum(torch.abs(z_slow_rule)).cpu().item())
            z_think_l1 = float(torch.sum(torch.abs(z_slow_think)).cpu().item())
            lang_norm = float(torch.norm(model.lang_embed(torch.tensor([lang_idx], device=device))).cpu().item())

        ce_hist.append(ce_q)
        if len(ce_hist) > 200:
            ce_hist.pop(0)

        mem_item_count = len(mem.items) if mem is not None else 0

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
                'lang': 'ko',
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
            "lang": 'ko',
            "lang_id": int(lang_idx),
            "rule_desc": desc,
            "ce_query": round(ce_q, 4),
            "conf_context": round(conf_val, 4),
            "zslow_rule_l1": round(z_rule_l1, 4),
            "zslow_think_l1": round(z_think_l1, 4),
            "zlang_norm": round(lang_norm, 4),
            "mem_items": mem_item_count,
            "init_mix_norm": round(init_mix_norm, 4),
            "n_q": len(Q),
        }
        w.writerow(row)

        if (t + 1) % args.summary_every == 0:
            med = statistics.median(ce_hist) if ce_hist else float("nan")
            mean = sum(ce_hist) / len(ce_hist)
            print(f"[turn {t + 1}] CE(mean/med last {len(ce_hist)}): {mean:.3f}/{med:.3f} | z_rule_L1={z_rule_l1:.1f} | z_think_L1={z_think_l1:.1f} | mem={mem_item_count} | init_norm={init_mix_norm:.2f} | '{desc}' | lang=ko")

    fcsv.close()

    summary = {
        "turns": args.turns,
        "lang_mode": "ko-fixed",
        "topk_mem": args.topk_mem,
        "weights": {"w_ep": args.w_ep, "w_desc": args.w_desc, "w_lang": args.w_lang},
        "final": {
            "z_rule_l1": z_rule_l1,
            "z_think_l1": z_think_l1,
            "lang_id": int(lang_idx),
            "mem_items": len(mem.items) if mem is not None else 0
        },
        "csv_path": args.summary_csv,
        "genlog_path": args.genlog_csv
    }
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    print(f"Summary CSV saved to {args.summary_csv}")
    print(f"Summary JSON saved to {args.summary_json}")
    print("Done. (KO-only long-run finished)")

if __name__ == "__main__":
    main()
