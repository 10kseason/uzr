import argparse
import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Union
import json as _json
import time as _time
from urllib import request as _urlreq, error as _urlerr
import hashlib as _hashlib, hmac as _hmac

from .model import UZRModel, ByteTokenizer, KoEnTokenizer
try:
    from .npu import OrtEngine
except Exception:
    OrtEngine = None  # optional dependency
from .memory import CompressedMemory


LANG2ID = {"base": 0, "en": 1, "ko": 2, "ja": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def detect_lang_from_text(text: str) -> str:
    """Detect language from input text."""
    has_hangul = False
    has_kana_or_cjk = False
    has_latin = False
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            has_hangul = True
        elif 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0x4E00 <= code <= 0x9FFF:
            has_kana_or_cjk = True
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            has_latin = True
    if has_hangul:
        return "ko"
    if has_kana_or_cjk:
        return "ja"
    if has_latin:
        return "en"
    return "base"


class ChatSession:
    """Interactive chat session with UZR model."""

    def __init__(
        self,
        model: UZRModel,
        memory: CompressedMemory,
        tokenizer: Union[ByteTokenizer, KoEnTokenizer],
        device: str = "cuda",
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        use_memory: bool = True,
        mem_on: bool = False,
        mem_url: str = "http://127.0.0.1:8088",
        mem_k: int = 6,
        mem_project: str = "uzr",
        mem_primer: bool = False,
        ort_engine: Optional["OrtEngine"] = None,
    ):
        self.model = model
        self.memory = memory
        self.tok = tokenizer
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.use_memory = use_memory
        self.ort_engine = ort_engine
        # External memory gateway config
        self.mem_on = mem_on
        self.mem_url = mem_url.rstrip("/")
        self.mem_k = int(mem_k)
        self.mem_primer = mem_primer
        self.mem_project = (mem_project or "").strip()

        self.model.eval()
        self.history: List[str] = []
        self._primer_text: Optional[str] = None

        # Build primer on session start if requested
        if self.mem_on and self.mem_primer:
            self._primer_text = self._build_primer()

    # ---------------- External memory helpers ----------------
    def _http_json(self, path: str, payload: Optional[dict] = None, timeout: float = 3.0) -> Optional[dict]:
        url = f"{self.mem_url}{path}"
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
                headers["X-Signature"] = _hmac.new(key.encode("utf-8"), data, _hashlib.sha256).hexdigest()
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

    def _mem_search(self, query: str, k: Optional[int] = None) -> List[dict]:
        if not self.mem_on:
            return []
        kk = int(k or self.mem_k or 6)
        filters = {"type": ["Decision", "Preference", "Fact", "Episode"]}
        if self.mem_project and self.mem_project.lower() != "all":
            filters["project"] = [self.mem_project]
        payload = {"q": query, "k": kk, "filters": filters}
        res = self._http_json("/mem/search", payload=payload, timeout=3.0)
        if not res or "candidates" not in res:
            return []
        return list(res.get("candidates") or [])

    def _mem_write_interaction(self, user_text: str, assistant_text: str, used_ids: Optional[List[str]] = None) -> None:
        if not self.mem_on:
            return
        # minimal MAL UPSERT + optional edges
        stamp = _time.strftime("%Y%m%d-%H%M%S", _time.gmtime())
        nid = f"fact:cli-{stamp}"
        title = (user_text or "").strip()[:60]
        body = (assistant_text or "").strip()[:240]
        client_id = os.environ.get("UZR_CLIENT_ID", "chat-cli").strip() or "chat-cli"
        ops = [
            {
                "op": "UPSERT_NODE",
                "type": "Fact",
                "id": nid,
                "title": title,
                "body": body,
                "tags": ([f"proj:{self.mem_project}"] if self.mem_project else []) + ["cli", "chat", f"client:{client_id}"],
                "trust": 0.7,
                "etag": "v1",
            }
        ]
        for rid in (used_ids or [])[:6]:
            ops.append({"op": "ADD_EDGE", "src": nid, "dst": rid, "rel": "relates_to", "weight": 0.7})
        payload = {"ops": ops, "author": "cli", "justification": "chat session log", "sig": "ed25519:cli"}
        _ = self._http_json("/mem/write", payload=payload, timeout=3.0)

    def _build_primer(self) -> Optional[str]:
        """Fetch top memory items to build a session primer."""
        cands = self._mem_search("session bootstrap primer", k=max(3, self.mem_k))
        if not cands:
            return None
        lines = ["[PRIMER]"]
        for c in cands[: max(3, self.mem_k)]:
            cid = c.get("id", "")
            title = c.get("title", "").strip()
            snip = (c.get("snippet", "") or "").strip()
            lines.append(f"- ({cid}) {title} — {snip}")
        return "\n".join(lines)

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample next token from logits using temperature and top-p."""
        if self.temperature <= 0:
            return logits.argmax().item()

        probs = F.softmax(logits / self.temperature, dim=-1)

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumsum_probs > self.top_p
        # Keep at least one token
        sorted_indices_to_remove[0] = False

        # Zero out removed probabilities
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample from filtered distribution
        token_idx = torch.multinomial(sorted_probs, 1).item()
        return sorted_indices[token_idx].item()

    def generate(self, prompt: str, verbose: bool = False) -> str:
        """Generate response for the given prompt."""
        # External memory hydration
        mem_used_ids: List[str] = []
        if self.mem_on:
            # Memory usage guidelines to help the model utilize context
            guide = [
                "[MEMORY GUIDELINES]",
                "- 아래 MEMORY CONTEXT의 항목을 우선 활용해 톤/사실/정책 일관성을 유지.",
                "- 유형 우선순위: Decision > Preference > Fact > Episode.",
                "- 근거가 있을 때 답변 말미에 [MEM:<id>] 형태로 간단히 표기.",
                "- 모순되는 항목은 함께 언급하고 우선순위를 명시.",
                "- 무관하면 간결히 무시하고 일반 답변.",
            ]
            ctx_lines = ["\n".join(guide), ""]
            cands = self._mem_search(prompt, k=self.mem_k)
            if cands:
                ctx_lines.append("[MEMORY CONTEXT]")
                for c in cands:
                    cid = c.get("id", "")
                    ttl = c.get("title", "").strip()
                    sn = (c.get("snippet", "") or "").strip()
                    ctx_lines.append(f"- ({cid}) {ttl} — {sn}")
                    if cid:
                        mem_used_ids.append(cid)
                ctx = "\n".join(ctx_lines) + "\n\n"
            else:
                ctx = ""
        else:
            ctx = ""

        # Primer prepend (if available)
        prefix = (self._primer_text + "\n\n") if self._primer_text else ""
        prompt = prefix + ctx + prompt
        # Detect language
        lang_key = detect_lang_from_text(prompt)
        lang_idx = LANG2ID.get(lang_key, LANG2ID["base"])

        # Tokenize input
        input_ids = self.tok.encode(prompt).unsqueeze(0).to(self.device)  # [1, T]

        # Get z initialization from memory
        with torch.no_grad():
            if self.use_memory and self.memory is not None:
                z_rule = self.model.get_z_from_memory(
                    input_ids,
                    z_init=None,
                    topk=4,
                    blend=0.5,
                )
                if z_rule is None:
                    z_rule = self.model.init_z(batch_size=1)[0].to(self.device)
            else:
                z_rule = self.model.init_z(batch_size=1)[0].to(self.device)

            z_think = self.model.init_z_thinking(batch_size=1)[0].to(self.device)

            if verbose:
                print(f"[Language: {lang_key}] [Memory: {self.memory.has_learner() if self.memory else False}]")

        # Generate tokens
        generated_ids = input_ids[0].tolist()
        eos_token = self.tok.EOS

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                # Prepare current sequence
                curr_ids = torch.tensor(generated_ids, dtype=torch.long, device=self.device).unsqueeze(0)

                # Forward pass: NPU(ORT/QNN) 경로 우선, 실패 시 PyTorch 경로
                if self.ort_engine is not None:
                    try:
                        out = self.ort_engine.run(
                            input_ids=curr_ids.detach().cpu().numpy(),  # ORT feeds: numpy
                        )
                        # 'logits'가 있으면 사용, 없으면 첫 출력 사용
                        if "logits" in out:
                            logits_np = out["logits"]
                        else:
                            # take first value
                            logits_np = list(out.values())[0]
                        logits = torch.from_numpy(logits_np).to(self.device)
                    except Exception:
                        logits = self.model(curr_ids, {"rule": z_rule, "think": z_think, "lang_id": lang_idx})
                else:
                    logits = self.model(curr_ids, {"rule": z_rule, "think": z_think, "lang_id": lang_idx})

                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :]  # [vocab_size]

                # Sample next token
                next_token = self._sample_token(next_token_logits)

                # Stop if EOS
                if next_token == eos_token:
                    break

                generated_ids.append(next_token)

        # Decode generated sequence
        response = self.tok.decode(generated_ids)

        # Update internal UZR memory and external gateway write
        if self.use_memory and self.memory is not None:
            self.model.update_memory_state(input_ids, z_rule)
            # Online learning: small step to keep predictor fresh
            try:
                _ = self.memory.train_model(steps=1, batch_size=16)
            except Exception:
                pass
        if self.mem_on:
            try:
                self._mem_write_interaction(prompt, response, used_ids=mem_used_ids)
            except Exception:
                pass

        return response

    def chat(self, user_input: str, verbose: bool = False) -> str:
        """Process user input and return response."""
        self.history.append(f"User: {user_input}")

        # Generate response
        full_response = self.generate(user_input, verbose=verbose)

        # Extract the response part (after the input)
        # The model generates continuation, so we need to extract only the new part
        response = full_response[len(user_input):].strip()

        self.history.append(f"Assistant: {response}")
        return response

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        if self.memory is not None:
            self.memory.clear_state()


def _pick_tokenizer_for_ckpt(data, max_len: int):
    try:
        rdw = data.get("model", {}).get("readout.weight")
        if isinstance(rdw, torch.Tensor):
            rows = rdw.size(0)
        else:
            rows = None
    except Exception:
        rows = None
    # Byte tokenizer has fixed vocab 258. Otherwise prefer KoEn.
    if rows == 258:
        return ByteTokenizer(max_len=max_len)
    tok = KoEnTokenizer(max_len=max_len)
    # If the checkpoint vocab is smaller than the current tokenizer vocab,
    # truncate the tokenizer tables so ids stay within [0, rows).
    try:
        if isinstance(rows, int) and rows > 0 and rows < tok.vocab_size:
            tok.itos = tok.itos[:rows]
            tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
            tok.vocab_size = len(tok.itos)
    except Exception:
        # Fallback to the original tokenizer if truncation fails.
        pass
    return tok


def load_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load model and memory from checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract args
    args = data.get("args", {})

    # Create tokenizer (prefer KoEn for KO/EN checkpoints; fallback to Byte)
    max_len = args.get("max_len", 512)
    tok = _pick_tokenizer_for_ckpt(data, max_len)

    # Create memory
    mem = None
    if "memory" in data:
        mem = CompressedMemory(
            max_items=32000,
            device=device,
            enable_learning=True,
            learn_hidden=512,
            learn_depth=3,
            warmup_steps=1,
        )
        mem.load_state_dict(data["memory"])
        # Enforce continuous learning with minimal warmup
        try:
            mem.enable_learning = True
            mem.warmup_steps = 1
            # Disable write-threshold warmup to apply gates immediately
            if hasattr(mem, "write_threshold_warmup_end"):
                mem.write_threshold_warmup_end = 1
        except Exception:
            pass
        print(f"Loaded {len(mem.items)} memory items, learner: {mem.has_learner()}")

    # Create model
    # Derive vocab_size from checkpoint readout if present to avoid size mismatch.
    vocab_size = tok.vocab_size
    try:
        rdw = data.get("model", {}).get("readout.weight")
        if isinstance(rdw, torch.Tensor):
            vocab_size = int(rdw.size(0))
    except Exception:
        pass
    identity_self_dim = args.get("identity_self_dim", 32)
    identity_intent_dim = args.get("identity_intent_dim")
    model = UZRModel(
        vocab_size=vocab_size,
        d_model=args.get("d_model", 512),
        z_dim=args.get("z_dim", 1024),
        max_len=max_len,
        z_think_dim=args.get("z_think_dim", 128),
        z_lang_dim=args.get("z_lang_dim", 64),
        num_langs=args.get("num_langs", 4),
        identity_self_dim=identity_self_dim,
        identity_intent_dim=identity_intent_dim,
        memory=mem,
    )

    # Load model weights
    # Allow extra keys (e.g., newer transition/EMA modules) for backward/forward compatibility.
    model.load_state_dict(data["model"], strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  - d_model: {args.get('d_model', 512)}")
    print(f"  - z_dim: {args.get('z_dim', 1024)}")
    print(f"  - identity_self_dim: {identity_self_dim}")

    return model, mem, tok, args


def main():
    parser = argparse.ArgumentParser(description="Chat with UZR model (루리아)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--no_memory", action="store_true", help="Disable memory usage")
    # ORT/QNN 엔진 옵션 (선택)
    parser.add_argument("--ort_model", type=str, default="", help="ONNX(QDQ) model path for ORT/QNN engine")
    parser.add_argument("--engine", type=str, default="torch", choices=["torch", "qnn", "qnn_strict", "ort_fallback"], help="Inference engine selection")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    # External memory gateway options (opt-in)
    parser.add_argument("--mem_on", action="store_true", help="Use external memory gateway (/mem/search, /mem/write)")
    parser.add_argument("--mem_url", type=str, default=os.environ.get("UZR_MEM_URL", "http://127.0.0.1:8088"), help="Memory server base URL")
    parser.add_argument("--mem_k", type=int, default=int(os.environ.get("UZR_MEM_K", "6")), help="Number of items to hydrate")
    parser.add_argument("--mem_project", type=str, default=os.environ.get("UZR_MEM_PROJECT", "uzr"), help="Project tag for memory isolation (e.g., 'uzr','exaone','all')")
    parser.add_argument("--primer", action="store_true", help="Build and prepend a session primer from memory")
    args = parser.parse_args()

    # Load checkpoint
    model, memory, tokenizer, ckpt_args = load_checkpoint(args.ckpt, device=args.device)

    # Create chat session
    # Optional ORT/QNN engine
    ort_engine = None
    if args.engine != "torch" and args.ort_model:
        if OrtEngine is None:
            print("[warn] onnxruntime-qnn 미설치로 ORT/QNN 엔진 비활성화. PyTorch 경로로 진행합니다.")
        else:
            backend = "htp"
            mode = "qnn" if args.engine == "qnn" else ("qnn_strict" if args.engine == "qnn_strict" else "ort_fallback")
            try:
                ort_engine = OrtEngine(args.ort_model, mode=mode, backend=backend)
                print(f"[engine] ORT engine ready: mode={mode}, backend={backend}")
            except Exception as e:
                print(f"[warn] ORT 엔진 초기화 실패: {e}. PyTorch 경로로 진행합니다.")

    session = ChatSession(
        model=model,
        memory=memory,
        tokenizer=tokenizer,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        use_memory=not args.no_memory,
        mem_on=args.mem_on,
        mem_url=args.mem_url,
        mem_k=args.mem_k,
        mem_project=args.mem_project,
        mem_primer=args.primer,
        ort_engine=ort_engine,
    )

    # Print welcome message
    identity_name = ckpt_args.get("identity", "루리아")
    print("\n" + "=" * 60)
    print(f"  Chat with {identity_name}")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit, /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /help - Show this help message")
    print("  /stats - Show memory statistics")
    if ort_engine is not None:
        print("  /lora_npz <path> - Load adapter/FiLM params (.npz) and swap into engine")
        print("  /hot_swap - Recreate ORT session (shadow→active) and warmup")
    print("\n" + "=" * 60 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                print(f"\nGoodbye! {identity_name}와의 대화를 종료합니다.")
                break

            if user_input.lower() == "/clear":
                session.clear_history()
                print("Conversation history cleared.")
                continue

            if user_input.lower() == "/help":
                print("\nCommands:")
                print("  /quit, /exit - Exit the chat")
                print("  /clear - Clear conversation history")
                print("  /help - Show this help message")
                print("  /stats - Show memory statistics")
                if ort_engine is not None:
                    print("  /lora_npz <path> - Load adapter/FiLM params (.npz) and swap into engine")
                    print("  /hot_swap - Recreate ORT session (shadow→active) and warmup")
                continue

            if user_input.lower() == "/stats":
                if memory is not None:
                    print(f"\nMemory Statistics:")
                    print(f"  - Items: {len(memory.items)}/{memory.max_items}")
                    print(f"  - Learner active: {memory.has_learner()}")
                    if memory.learner_loss is not None:
                        print(f"  - Learner loss: {memory.learner_loss:.4f}")
                else:
                    print("\nMemory is not available.")
                continue

            # ORT/QNN engine commands
            if user_input.lower().startswith("/lora_npz") and ort_engine is not None:
                parts = user_input.split(maxsplit=1)
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

            if user_input.lower() == "/hot_swap" and ort_engine is not None:
                try:
                    ort_engine.hot_swap()
                    print("[engine] hot swapped and warmed up.")
                except Exception as e:
                    print(f"[engine] hot swap failed: {e}")
                continue

            # Generate response
            response = session.chat(user_input, verbose=args.verbose)
            print(f"{identity_name}: {response}\n")

        except KeyboardInterrupt:
            print(f"\n\nGoodbye! {identity_name}와의 대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
