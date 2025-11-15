import argparse
import os
import json as _json
from typing import List, Optional, Tuple
from urllib import request as _urlreq, error as _urlerr
import hashlib, hmac

import torch

from .chat_cli import load_checkpoint, KoEnTokenizer, ByteTokenizer
from .exaone_adapter import ExaoneGenerator
try:
    from utils.logger_luria import append_jsonl as _append_jsonl
except Exception:
    _append_jsonl = None


def _http_json(base_url: str, path: str, payload: Optional[dict] = None, timeout: float = 4.0) -> Optional[dict]:
    url = f"{base_url.rstrip('/')}{path}"
    data = None
    if payload is not None:
        data = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    tok = os.environ.get("UZR_MEM_TOKEN", "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    key = os.environ.get("UZR_MEM_HMAC_KEY", "")
    if key and data is not None:
        try:
            headers["X-Signature"] = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
        except Exception:
            pass
    req = _urlreq.Request(url, data=data, headers=headers, method=("POST" if payload is not None else "GET"))
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if not body:
                return None
            return _json.loads(body)
    except (_urlerr.URLError, _urlerr.HTTPError, TimeoutError):
        return None
    except Exception:
        return None


def mem_search(mem_url: str, query: str, k: int, project: Optional[str]) -> List[dict]:
    filters = {"type": ["Decision", "Preference", "Fact", "Episode"]}
    if project and project.lower() != "all":
        filters["project"] = [project]
    res = _http_json(mem_url, "/mem/search", payload={"q": query, "k": int(k), "filters": filters})
    if not res or "candidates" not in res:
        return []
    return list(res.get("candidates") or [])


def mem_prepare(mem_url: str, ops: List[dict], author: str) -> Tuple[Optional[str], Optional[dict]]:
    res = _http_json(mem_url, "/mem/prepare", payload={"ops": ops, "author": author, "two_pc": True})
    if not res or res.get("status") != "prepared":
        return None, res
    return str(res.get("ticket")), None


def mem_commit(mem_url: str, ticket: str) -> Tuple[bool, Optional[dict]]:
    res = _http_json(mem_url, "/mem/commit", payload={"ticket": ticket})
    if not res:
        return False, None
    return True, res


def _intent_force_from_model(model) -> Optional[bool]:
    try:
        _, toggle = model.identity_intent_control()
    except Exception:
        return None
    if toggle <= -0.5:
        return False
    if toggle >= 0.5:
        return True
    return None


def build_context(mem_url: str, prompt: str, k: int, project: Optional[str], primer: Optional[str]) -> Tuple[str, List[str]]:
    used_ids: List[str] = []
    lines: List[str] = []
    if primer:
        lines.append(primer)
        lines.append("")
    guide = [
        "[MEMORY GUIDELINES]",
        "- 아래 MEMORY CONTEXT의 항목을 우선 활용해 톤/사실/정책 일관성을 유지.",
        "- 유형 우선순위: Decision > Preference > Fact > Episode.",
        "- 근거가 있을 때 답변 말미에 [MEM:<id>] 형태로 간단히 표기.",
        "- 모순되는 항목은 함께 언급하고 우선순위를 명시.",
        "- 무관하면 간결히 무시하고 일반 답변.",
    ]
    lines.extend(["\n".join(guide), ""]) 
    cands = mem_search(mem_url, prompt, k, project)
    if cands:
        lines.append("[MEMORY CONTEXT]")
        for c in cands:
            cid = c.get("id", "")
            ttl = c.get("title", "").strip()
            sn = (c.get("snippet", "") or "").strip()
            lines.append(f"- ({cid}) {ttl} — {sn}")
            if cid:
                used_ids.append(cid)
        lines.append("")
    return "\n".join(lines), used_ids

def build_internal_context(model, memory, tokenizer, prompt: str, k: int, max_lines: Optional[int] = None,
                           project: Optional[str] = None,
                           recent_n: int = 0, project_n: int = 0, nearest_n: Optional[int] = None) -> Tuple[str, List[str]]:
    """Build context lines from internal UZR memory (CompressedMemory).

    Returns: (context_text, internal_ids)
      - internal_ids: ['int:<step>', ...] up to listed items
    Sections (optional): RECENT, PROJECT, NEAREST
    """
    try:
        if memory is None or not getattr(memory, "items", None):
            return "", []
        # Encode and average-embed
        X = tokenizer.encode(prompt).unsqueeze(0).to(model.readout.weight.device)
        with torch.no_grad():
            enc_avg = model.avg_embed(X)[0].detach()
        # Sections
        lines: List[str] = ["[MEMORY CONTEXT — INTERNAL]"]
        internal_ids: List[str] = []
        cap = max_lines if (isinstance(max_lines, int) and max_lines > 0) else None

        def _fmt(it):
            meta = (it.val or {}).get("meta", {}) if hasattr(it, "val") else {}
            title = str(meta.get("title") or meta.get("source") or meta.get("project") or "uzr")
            snip = str((it.val or {}).get("repr_text") or meta.get("repr") or "")
            if not snip:
                snip = f"z|{(it.val or {}).get('z_slow').shape if (it.val or {}).get('z_slow') is not None else 'n/a'}"
            sid = f"int:{getattr(it,'step',0)}"; internal_ids.append(sid)
            return f"- ({sid}) {title} — {snip[:200]}"

        total_emitted = 0

        # RECENT
        if recent_n and recent_n > 0 and getattr(memory, "items", None):
            try:
                recent_sorted = sorted(memory.items, key=lambda t: getattr(t, 'step', 0), reverse=True)
                lines.append("[INTERNAL — RECENT]")
                for it in recent_sorted[: recent_n]:
                    lines.append(_fmt(it))
                    total_emitted += 1
                    if cap and total_emitted >= cap:
                        return "\n".join(lines) + "\n", internal_ids
            except Exception:
                pass

        # PROJECT
        if project and project_n and project_n > 0 and getattr(memory, "items", None):
            try:
                proj_items = [it for it in memory.items if ((it.val or {}).get('meta', {}) or {}).get('project') == project]
                if proj_items:
                    lines.append("[INTERNAL — PROJECT]")
                    for it in proj_items[: project_n]:
                        lines.append(_fmt(it))
                        total_emitted += 1
                        if cap and total_emitted >= cap:
                            return "\n".join(lines) + "\n", internal_ids
            except Exception:
                pass

        # NEAREST
        nn_k = nearest_n if (isinstance(nearest_n, int) and nearest_n is not None) else max(1, int(k))
        try:
            nns = memory.retrieve(enc_avg, topk=nn_k)
            if nns:
                lines.append("[INTERNAL — NEAREST]")
                for it in nns:
                    lines.append(_fmt(it))
                    total_emitted += 1
                    if cap and total_emitted >= cap:
                        return "\n".join(lines) + "\n", internal_ids
        except Exception:
            pass

        return "\n".join(lines) + "\n", internal_ids
    except Exception:
        return "", []

def build_consciousness_frame(user: str, int_ctx: str, ext_ctx: str, intent_text: Optional[str], uzr_first: bool = True) -> str:
    """Compose a Consciousness Frame with SENSORY/WM/EXT/INTENT blocks.

    LLM은 의식(Consciousness), UZR는 물리 뇌(Brain)라는 기조에 따라
    내부(UZR) 컨텍스트를 우선 배치하고, 외부(L2) 컨텍스트를 보조로 섞습니다.
    """
    parts: List[str] = []

    # SENSORY
    parts.append("[SENSORY]")
    parts.append(user.strip())
    parts.append("")

    # Memory contexts
    if uzr_first:
        if int_ctx:
            parts.append(int_ctx.strip())
        if ext_ctx:
            parts.append(ext_ctx.strip())
    else:
        if ext_ctx:
            parts.append(ext_ctx.strip())
        if int_ctx:
            parts.append(int_ctx.strip())

    # INTENT
    if intent_text and intent_text.strip():
        parts.append("[INTENT]")
        parts.append(intent_text.strip())

    return "\n".join(p for p in parts if p).strip()


def build_primer(mem_url: str, k: int, project: Optional[str]) -> Optional[str]:
    cands = mem_search(mem_url, "session bootstrap primer", max(4, k), project)
    if not cands:
        return None
    lines = ["[PRIMER]"]
    for c in cands[: max(3, k)]:
        cid = c.get("id", "")
        title = c.get("title", "").strip()
        snip = (c.get("snippet", "") or "").strip()
        lines.append(f"- ({cid}) {title} — {snip}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="UZR Orchestrator: route any LLM via UZR path + external memory")
    # UZR
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--conf_min", type=float, default=0.65, help="Self-eval confidence threshold to commit 2PC")
    # Generator (EXAONE/LM Studio)
    ap.add_argument("--backend", type=str, default="http", choices=["http","hf"])
    ap.add_argument("--exa_url", type=str, default=os.environ.get("EXA_URL","http://127.0.0.1:1234/v1/chat/completions"))
    ap.add_argument("--http_mode", type=str, default=os.environ.get("EXA_HTTP_MODE","openai-chat"), choices=["raw","generate","openai-chat","oai-chat","chat","openai-completion","oai-completion","completion","tgi"])
    ap.add_argument("--exa_model", type=str, default=os.environ.get("EXA_MODEL","default"))
    ap.add_argument("--model_dir", type=str, default=os.environ.get("EXA_MODEL_DIR","EXAONE-4.0-1.2B"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=15000)
    # Memory
    ap.add_argument("--mem_on", action="store_true")
    ap.add_argument("--mem_url", type=str, default=os.environ.get("UZR_MEM_URL","http://127.0.0.1:8088"))
    ap.add_argument("--mem_k", type=int, default=int(os.environ.get("UZR_MEM_K","6")))
    ap.add_argument("--mem_project", type=str, default=os.environ.get("UZR_MEM_PROJECT","uzr"))
    ap.add_argument("--primer", action="store_true")
    # UZR-first autonomous routing: 내부 메모리 컨텍스트 우선/혼합
    ap.add_argument("--uzr_first", dest="uzr_first", action="store_true", default=True, help="Prefer internal UZR memory context before external (default: on)")
    ap.add_argument("--no_uzr_first", dest="uzr_first", action="store_false", help="Place internal context after external or disable by empty internal")
    ap.add_argument("--mix_ctx_k", type=int, default=0, help="Internal memory top-k for context mixing (0→use mem_k)")
    ap.add_argument("--int_ctx_max_lines", type=int, default=int(os.environ.get("UZR_INT_CTX_MAX_LINES","6")), help="Max lines for internal WM context (default 6)")
    ap.add_argument("--int_ctx_recent", type=int, default=int(os.environ.get("UZR_INT_CTX_RECENT","6")), help="Recent internal items to include (default 6)")
    ap.add_argument("--int_ctx_project", type=int, default=int(os.environ.get("UZR_INT_CTX_PROJECT","6")), help="Project-matching internal items to include (default 6)")
    ap.add_argument("--int_ctx_nearest", type=int, default=int(os.environ.get("UZR_INT_CTX_NEAREST","6")), help="Nearest internal items to include (default 6)")
    ap.add_argument("--ext_ctx_max_lines", type=int, default=int(os.environ.get("UZR_EXT_CTX_MAX_LINES","6")), help="Max lines for external memory context (default 6; applied upstream by mem server filters where possible)")
    # Conscious Intent (의식의 목표/정책/톤 등)
    ap.add_argument("--intent", type=str, default=os.environ.get("UZR_INTENT", ""), help="Conscious intent text to guide LLM (optional)")
    ap.add_argument("--intent_off", action="store_true", help="Disable auto-intent when not provided")
    ap.add_argument("--trace_dir", type=str, default=os.environ.get("UZR_TRACE_DIR", "logu"), help="Directory to write consciousness trace JSONL")
    args = ap.parse_args()

    # Load UZR model/tokenizer/memory
    model, memory, tokenizer, ckpt_args = load_checkpoint(args.ckpt, device=args.device)
    # Ensure memory learning is on with warmup=1
    if memory is not None:
        try:
            memory.enable_learning = True
            memory.warmup_steps = 1
            if hasattr(memory, "write_threshold_warmup_end"):
                memory.write_threshold_warmup_end = 1
        except Exception:
            pass
    # External generator
    gen = ExaoneGenerator(backend=args.backend, exa_url=args.exa_url, http_mode=args.http_mode, exa_model=args.exa_model, model_dir=args.model_dir, device=args.device)

    primer_text = None
    if args.mem_on and args.primer:
        primer_text = build_primer(args.mem_url, args.mem_k, args.mem_project)

    identity_name = ckpt_args.get("identity", "루리아")
    print("\n" + "="*60)
    print(f"  Orchestrator via {identity_name} + External LLM")
    print("="*60)
    print("Commands: /quit, /exit, /help")
    print("\n")

    mem_step = 0
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"/quit","/exit"}:
                print("Bye")
                break
            if user.lower()=="/help":
                print("Commands: /quit, /exit, /help")
                continue

            # Build context via memory
            # Build external context (opt-in)
            ext_ctx, used_ids = ("", [])
            if args.mem_on:
                ext_ctx, used_ids = build_context(args.mem_url, user, args.mem_k, args.mem_project, primer_text)
            # Build internal UZR context (autonomous)
            k_internal = args.mix_ctx_k if args.mix_ctx_k and args.mix_ctx_k > 0 else args.mem_k
            int_ctx, int_refs = build_internal_context(
                model, memory, tokenizer, user, k_internal,
                max_lines=args.int_ctx_max_lines,
                project=args.mem_project,
                recent_n=args.int_ctx_recent,
                project_n=args.int_ctx_project,
                nearest_n=args.int_ctx_nearest,
            )
            # Build intent (default guideline if not provided and not disabled)
            intent_text = args.intent
            if not intent_text and not args.intent_off:
                intent_text = "\n".join([
                    "- 내부(UZR) 작업기억을 먼저 참고해 톤/정책/사실 일관성 유지",
                    "- 외부(L2) 컨텍스트는 보조로 활용하고 충돌 시 내부를 우선",
                    "- 정보 출처를 간결히 암시하되 답변은 자연스럽게",
                ])
            # Consciousness Frame
            prompt = build_consciousness_frame(user, int_ctx, ext_ctx, intent_text, uzr_first=args.uzr_first)

            # Route through UZR path: compute z and self-eval conf/entropy on input
            # Prepare minimal input ids for conf/entropy
            try:
                input_ids = tokenizer.encode(user).unsqueeze(0).to(model.readout.weight.device)
                conf = model.confidence(input_ids)
                conf_val = float(conf[0].item()) if conf is not None else 1.0
                # entropy on teacher-forced input (proxy)
                try:
                    z_rule_local = None
                    try:
                        z_rule_local = model.get_z_from_memory(input_ids, z_init=None, topk=None, blend=0.5)
                    except Exception:
                        z_rule_local = None
                    z_think_local = model.init_z_thinking(batch_size=1)
                    logits_local = model(input_ids, {"rule": (z_rule_local if z_rule_local is not None else model.init_z(1)), "think": z_think_local, "lang_id": 0})
                    ent_val = float(model.sequence_entropy(logits_local).mean().item())
                except Exception:
                    ent_val = None
            except Exception:
                conf_val = 1.0
                ent_val = None

            # Derive memory vectors via UZR for internal learning/logging
            z_rule = None
            avg_emb = None
            try:
                z_rule = model.get_z_from_memory(input_ids, z_init=None, topk=None, blend=0.5)
            except Exception:
                z_rule = None
            try:
                avg_emb = model.avg_embed(input_ids)[0]
            except Exception:
                avg_emb = None

            # Generate via external LLM with UZR fallback
            fallback_used = False
            try:
                text = gen.generate(prompt=prompt, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_tokens)
                resp = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
            except Exception as e:
                try:
                    fb = ChatSession(model=model, memory=memory, tokenizer=tokenizer, device=args.device, temperature=args.temperature, top_p=args.top_p, max_new_tokens=min(1024, args.max_tokens), use_memory=True, mem_on=False)
                    resp = fb.chat(prompt, verbose=False)
                    print(f"[fallback:uzr] external generate failed: {e}")
                    fallback_used = True
                except Exception as e2:
                    resp = f"[error] generation failed: {e2}"
            print(f"LLM: {resp}\n")

            # 2PC write: prepare then commit if conf passes (ERN style)
            if args.mem_on:
                nid_title = (user or "").strip()[:60]
                # Include short ERN summary: conf/entropy (if available)
                meta_line = f"[ERN] conf={conf_val:.3f}" + (f", ent={ent_val:.3f}" if ent_val is not None else "")
                body_main = (resp or "").strip()[:240]
                # provenance signature: hash of sensory+int/ext contexts and used ids
                try:
                    src_blob = (user or "") + "\n" + (int_ctx or "") + (ext_ctx or "") + "\n" + ",".join(used_ids or [])
                    src_hash = hashlib.sha256(src_blob.encode("utf-8")).hexdigest()[:16]
                    src_line = f"\n[SRC] {src_hash}"
                except Exception:
                    src_line = ""
                nid_body = (meta_line + "\n" + body_main + src_line)[:240]
                client_id = os.environ.get("UZR_CLIENT_ID", "orchestrator").strip() or "orchestrator"
                tags = ([f"proj:{args.mem_project}"] if args.mem_project else []) + ["orchestrator","chat", f"client:{client_id}"]
                # Mark as experience record
                tags.append("ern")
                # include brief internal ref ids in body tail if present
                if int_refs:
                    tail = "\n[INTREF] " + ",".join(int_refs[:12])
                    nid_body = (nid_body + tail)[:240]
                ops = [{"op":"UPSERT_NODE","type":"Fact","id":f"fact:or-{os.getpid()}","title":nid_title,"body":nid_body,"tags":tags,"trust": 0.7, "etag":"v1"}]
                for rid in used_ids[:6]:
                    ops.append({"op":"ADD_EDGE","src":ops[0]["id"],"dst":rid,"rel":"relates_to","weight":0.7})
                ticket, err = mem_prepare(args.mem_url, ops, author="orchestrator")
                if ticket and conf_val >= args.conf_min:
                    ok, _ = mem_commit(args.mem_url, ticket)
                    if not ok:
                        print("[warn] commit failed; prepared only.")
                else:
                    print(f"[info] 2PC not committed (conf={conf_val:.2f} < {args.conf_min}); prepared={bool(ticket)}")
                # Optional: reflection note as a separate ERN node (when committed)
                if ticket and conf_val >= args.conf_min:
                    try:
                        refl = []
                        if conf_val < 0.7:
                            refl.append("자신감이 낮음 → 근거 강화 필요")
                        if ent_val is not None and ent_val > 2.2:
                            refl.append("엔트로피 높음 → 답변 분산 큼")
                        if not refl:
                            refl.append("정상 응답으로 기록")
                        ref_title = f"Reflection: {user[:40]}"
                        ref_body = ("; ".join(refl))[:240]
                        ref_ops = [{"op":"UPSERT_NODE","type":"Fact","id":f"fact:or-ref-{os.getpid()}","title":ref_title,"body":ref_body,"tags":([f"proj:{args.mem_project}"] if args.mem_project else []) + ["orchestrator","reflection","ern", f"client:{client_id}"],"trust":0.65,"etag":"v1"}, {"op":"ADD_EDGE","src":"fact:or-ref-{os.getpid()}","dst":ops[0]["id"],"rel":"relates_to","weight":0.7}]
                        t2, _ = mem_prepare(args.mem_url, ref_ops, author="orchestrator")
                        if t2:
                            mem_commit(args.mem_url, t2)
                    except Exception:
                        pass

            # Keep UZR memory learner warm (adaptive budget)
            if memory is not None:
                try:
                    # Track state for retrieval statistics
                    if z_rule is not None:
                        model.update_memory_state(input_ids, z_rule)
                    # Write into internal long-term memory with policy
                    if avg_emb is not None and z_rule is not None:
                        meta = {
                            "project": args.mem_project,
                            "source": "orchestrator",
                            "repr": model.build_text_repr(input_ids, tokenizer=tokenizer),
                        }
                        mem_meta = {"luria_intent_force": _intent_force_from_model(model)}
                        _ = memory.add_with_policy(
                            key=avg_emb.detach().cpu(),
                            val={"z_slow": z_rule.detach().cpu(), "avg_emb": avg_emb.detach().cpu(), "meta": meta},
                            step=mem_step,
                            meta=mem_meta,
                            bench_callback=lambda: conf_val >= args.conf_min,
                        )
                        mem_step += 1
                    # Adaptive tiny training: more when conf 중간~상, 덜할 때는 1
                    tr_steps = 1
                    if conf_val >= (args.conf_min + 0.05):
                        tr_steps = 2
                    _ = memory.train_model(steps=tr_steps, batch_size=16)
                except Exception:
                    pass

            # Consciousness trace JSONL
            if _append_jsonl is not None:
                try:
                    os.makedirs(args.trace_dir, exist_ok=True)
                    _append_jsonl(os.path.join(args.trace_dir, "conc_trace.jsonl"), {
                        "ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                        "sensory": user,
                        "intent": intent_text or "",
                        "uzr_first": bool(args.uzr_first),
                        "int_ctx_lines": int(len([ln for ln in (int_ctx or "").splitlines() if ln.strip()])),
                        "ext_used_ids": used_ids,
                        "int_refs": int_refs,
                        "conf": conf_val,
                        "ent": ent_val,
                        "resp_len": len(resp or ""),
                        "fallback": fallback_used,
                        "mem_project": args.mem_project,
                    })
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\nBye")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
