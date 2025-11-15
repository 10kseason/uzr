import argparse
import os
import json as _json
from typing import List, Optional
from urllib import request as _urlreq, error as _urlerr
import hashlib, hmac
import time as _time

from .exaone_adapter import ExaoneGenerator


def _http_json(base_url: str, path: str, payload: Optional[dict] = None, timeout: float = 3.5) -> Optional[dict]:
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


def mem_search(mem_url: str, query: str, k: int, project: Optional[str] = None) -> List[dict]:
    filters = {"type": ["Decision", "Preference", "Fact", "Episode"]}
    if project and project.lower() != "all":
        filters["project"] = [project]
    res = _http_json(mem_url, "/mem/search", payload={"q": query, "k": int(k), "filters": filters})
    if not res or "candidates" not in res:
        return []
    return list(res.get("candidates") or [])


def mem_write(mem_url: str, user_text: str, assistant_text: str, used_ids: Optional[List[str]] = None, project: Optional[str] = None) -> None:
    stamp = _time.strftime("%Y%m%d-%H%M%S", _time.gmtime())
    nid = f"fact:exa-cli-{stamp}"
    title = (user_text or "").strip()[:60]
    body = (assistant_text or "").strip()[:240]
    client_id = os.environ.get("UZR_CLIENT_ID", "exaone-cli").strip() or "exaone-cli"
    tags = ["exaone", "cli", f"client:{client_id}"]
    if project and project.strip():
        tags = [f"proj:{project.strip()}"] + tags
    ops = [{"op": "UPSERT_NODE", "type": "Fact", "id": nid, "title": title, "body": body, "tags": tags, "trust": 0.7, "etag": "v1"}]
    for rid in (used_ids or [])[:6]:
        ops.append({"op": "ADD_EDGE", "src": nid, "dst": rid, "rel": "relates_to", "weight": 0.7})
    _ = _http_json(mem_url, "/mem/write", payload={"ops": ops, "author": "exaone-cli", "justification": "chat session log", "sig": "ed25519:exaone-cli"})


def build_context(mem_url: str, prompt: str, k: int, primer: Optional[str], project: Optional[str]) -> (str, List[str]):
    used_ids: List[str] = []
    ctx_lines: List[str] = []
    if primer:
        ctx_lines.append(primer)
        ctx_lines.append("")
    # Memory usage guidelines to help the model utilize context
    guide = [
        "[MEMORY GUIDELINES]",
        "- 아래 MEMORY CONTEXT의 항목을 우선 활용해 톤/사실/정책 일관성을 유지.",
        "- 유형 우선순위: Decision > Preference > Fact > Episode.",
        "- 근거가 있을 때 답변 말미에 [MEM:<id>] 형태로 간단히 표기.",
        "- 모순되는 항목은 함께 언급하고 우선순위를 명시.",
        "- 무관하면 간결히 무시하고 일반 답변.",
    ]
    ctx_lines.extend(["\n".join(guide), ""])
    cands = mem_search(mem_url, prompt, k, project)
    if cands:
        ctx_lines.append("[MEMORY CONTEXT]")
        for c in cands:
            cid = c.get("id", "")
            ttl = c.get("title", "").strip()
            snip = (c.get("snippet", "") or "").strip()
            ctx_lines.append(f"- ({cid}) {ttl} — {snip}")
            if cid:
                used_ids.append(cid)
        ctx_lines.append("")
    return ("\n".join(ctx_lines), used_ids)


def build_primer(mem_url: str, k: int, project: Optional[str]) -> Optional[str]:
    cands = mem_search(mem_url, "session bootstrap primer", max(k, 4), project)
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
    ap = argparse.ArgumentParser(description="EXAONE chat with UZR external memory (gateway)")
    ap.add_argument("--backend", type=str, default="http", choices=["http", "hf"], help="EXAONE backend type")
    ap.add_argument("--exa_url", type=str, default=os.environ.get("EXA_URL", "http://127.0.0.1:9000/v1/generate"), help="EXAONE/LM endpoint (for backend=http)")
    ap.add_argument("--http_mode", type=str, default=os.environ.get("EXA_HTTP_MODE", "raw"), choices=["raw","generate","openai-chat","oai-chat","chat","openai-completion","oai-completion","completion","tgi"], help="HTTP payload/response mode")
    ap.add_argument("--exa_model", type=str, default=os.environ.get("EXA_MODEL", "default"), help="Model name (OpenAI-style endpoints)")
    ap.add_argument("--model_dir", type=str, default=os.environ.get("EXA_MODEL_DIR", "EXAONE-4.0-1.2B"), help="Local model dir (for backend=hf)")
    ap.add_argument("--device", type=str, default=os.environ.get("EXA_DEVICE", "cpu"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=15000)
    # memory options
    ap.add_argument("--mem_on", action="store_true", help="Use external memory hydration/write")
    ap.add_argument("--mem_url", type=str, default=os.environ.get("UZR_MEM_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--mem_k", type=int, default=int(os.environ.get("UZR_MEM_K", "6")))
    ap.add_argument("--primer", action="store_true", help="Build session primer from memory")
    ap.add_argument("--mem_project", type=str, default=os.environ.get("UZR_MEM_PROJECT", "exaone"), help="Project tag for memory isolation (e.g., 'exaone','uzr','all')")
    args = ap.parse_args()

    exa = ExaoneGenerator(backend=args.backend, exa_url=args.exa_url, http_mode=args.http_mode, exa_model=args.exa_model, model_dir=args.model_dir, device=args.device)

    primer_text = None
    if args.mem_on and args.primer:
        primer_text = build_primer(args.mem_url, args.mem_k, args.mem_project)

    print("\n" + "=" * 60)
    print("  EXAONE × UZR Memory")
    print("=" * 60)
    print("Commands:")
    print("  /quit, /exit - Exit the chat")
    print("  /help - Show commands")
    print("\n" + "=" * 60 + "\n")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"/quit", "/exit"}:
                print("Bye")
                break
            if user.lower() == "/help":
                print("Commands:")
                print("  /quit, /exit - Exit the chat")
                print("  /help - Show commands")
                continue

            ctx, used_ids = ("", [])
            if args.mem_on:
                ctx, used_ids = build_context(args.mem_url, user, args.mem_k, primer_text, args.mem_project)
            prompt = (ctx + user).strip()

            text = exa.generate(prompt=prompt, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_tokens)
            # Try to extract only the assistant part if both prompt+response exist
            # As a simple heuristic, drop the prompt prefix when present
            if text.startswith(prompt):
                resp = text[len(prompt):].strip()
            else:
                resp = text.strip()
            print(f"EXAONE: {resp}\n")

            if args.mem_on:
                mem_write(args.mem_url, user_text=user, assistant_text=resp, used_ids=used_ids, project=args.mem_project)

        except KeyboardInterrupt:
            print("\nBye")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
