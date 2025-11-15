import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

from .chat_cli import ChatSession, load_checkpoint


class UzrHttpServer(HTTPServer):
    """HTTP server carrying a preloaded UZR ChatSession."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        session: ChatSession,
        default_temperature: float,
        default_top_p: float,
        default_max_new_tokens: int,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.session = session
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.default_max_new_tokens = int(default_max_new_tokens)


class UzrRestHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-style REST adapter for UZR ChatSession.

    Supported endpoints:
      - POST /v1/chat/completions
      - POST /v1/completions
    """

    server: UzrHttpServer  # for type checkers

    def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Read JSON body from the request."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = raw.decode("utf-8") if raw else "{}"
            return json.loads(body or "{}"), None
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"invalid_json: {exc}"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        """Send JSON response."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs minimal; rely on outer supervisor if needed.
        return

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        if self.path in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._handle_chat_completions()
            return
        if self.path in ("/v1/completions", "/v1/completions/"):
            self._handle_completions()
            return
        self._send_json(404, {"error": "not_found", "path": self.path})

    def _extract_prompt_from_messages(self, messages: Any) -> str:
        """Flatten OpenAI-style messages into a single prompt string."""
        if not isinstance(messages, list):
            return ""
        system_parts: List[str] = []
        user_parts: List[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                # For stateless benchmarks, assistant history is optional context.
                user_parts.append(f"[ASSISTANT_PREV] {content}")
        parts = system_parts + user_parts
        return "\n\n".join(parts).strip()

    def _run_uzr_completion(
        self,
        prompt: str,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """Run a single-turn completion through UZR and return only the new text."""
        session = self.server.session

        # Apply per-request sampling overrides.
        session.temperature = float(
            self.server.default_temperature if temperature is None else temperature
        )
        session.top_p = float(self.server.default_top_p if top_p is None else top_p)
        session.max_new_tokens = int(
            self.server.default_max_new_tokens if max_tokens is None else max_tokens
        )

        full = session.generate(prompt, verbose=False)
        # Best-effort: strip the prompt prefix from decoded text.
        idx = full.find(prompt)
        if idx >= 0:
            return full[idx + len(prompt) :].strip()
        return full.strip()

    def _handle_chat_completions(self) -> None:
        body, err = self._read_json()
        if body is None:
            self._send_json(400, {"error": err or "invalid_json"})
            return

        prompt = self._extract_prompt_from_messages(body.get("messages"))
        if not prompt:
            self._send_json(400, {"error": "empty_prompt"})
            return

        temperature = body.get("temperature")
        top_p = body.get("top_p")
        max_tokens = body.get("max_tokens") or body.get("max_new_tokens")

        try:
            completion = self._run_uzr_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(
                500,
                {"error": "generation_failed", "detail": str(exc)},
            )
            return

        resp = {
            "id": "uzr-chat-1",
            "object": "chat.completion",
            "model": body.get("model") or "uzr-3brains",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": "stop",
                }
            ],
        }
        self._send_json(200, resp)

    def _handle_completions(self) -> None:
        body, err = self._read_json()
        if body is None:
            self._send_json(400, {"error": err or "invalid_json"})
            return

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(400, {"error": "empty_prompt"})
            return

        temperature = body.get("temperature")
        top_p = body.get("top_p")
        max_tokens = body.get("max_tokens") or body.get("max_new_tokens")

        try:
            completion = self._run_uzr_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(
                500,
                {"error": "generation_failed", "detail": str(exc)},
            )
            return

        resp = {
            "id": "uzr-completion-1",
            "object": "text_completion",
            "model": body.get("model") or "uzr-3brains",
            "choices": [
                {
                    "index": 0,
                    "text": completion,
                    "finish_reason": "stop",
                }
            ],
        }
        self._send_json(200, resp)


def main() -> None:
    """Entry point: launch REST server around a UZR checkpoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="uzr_3brains_ckpt_best.pt",
        help="Path to UZR checkpoint (best.ckpt recommended).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="Device for UZR model (cpu/cuda).",
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for HTTP server.",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=1234,
        help="Bind port for HTTP server.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Default sampling temperature.",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Default top-p (nucleus) sampling.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Default max_new_tokens per completion.",
    )
    ap.add_argument(
        "--mem_on",
        action="store_true",
        help="Enable external memory server hydration (mem_server.py).",
    )
    ap.add_argument(
        "--mem_url",
        default="http://127.0.0.1:8088",
        help="External memory server URL.",
    )
    ap.add_argument(
        "--mem_k",
        type=int,
        default=6,
        help="Top-k for external memory search.",
    )
    ap.add_argument(
        "--mem_project",
        default="uzr",
        help="Project tag for external memory.",
    )
    ap.add_argument(
        "--mem_primer",
        action="store_true",
        help="Fetch primer from external memory at session start.",
    )
    args = ap.parse_args()

    model, memory, tokenizer, _ckpt_args = load_checkpoint(args.ckpt, device=args.device)

    session = ChatSession(
        model=model,
        memory=memory,
        tokenizer=tokenizer,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        use_memory=True,
        mem_on=args.mem_on,
        mem_url=args.mem_url,
        mem_k=args.mem_k,
        mem_project=args.mem_project,
        mem_primer=args.mem_primer,
    )

    server = UzrHttpServer(
        server_address=(args.host, args.port),
        handler_cls=UzrRestHandler,
        session=session,
        default_temperature=args.temperature,
        default_top_p=args.top_p,
        default_max_new_tokens=args.max_new_tokens,
    )
    print(
        f"[UZR REST] Serving {args.ckpt} on http://{args.host}:{args.port} "
        f"(device={args.device}, mem_on={args.mem_on})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[UZR REST] Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

