from typing import Optional, Dict, Any
import json


class ExaoneGenerator:
    """EXAONE text generator adapter with pluggable backends.

    Backends:
    - http: POST to an HTTP endpoint that returns generated text
      * Expected request: {"prompt", "temperature", "top_p", "max_new_tokens"}
      * Expected response: one of {"text", "generated_text", "generated"}
    - hf: (optional) load local HF model from a directory (requires transformers)
      * model_dir: path containing tokenizer/model files (e.g., EXAONE-4.0-1.2B)
    """

    def __init__(
        self,
        backend: str = "http",
        exa_url: str = "http://127.0.0.1:9000/v1/generate",
        http_mode: str = "raw",
        exa_model: Optional[str] = None,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        default_max_new_tokens: int = 15000,
    ) -> None:
        self.backend = backend
        self.exa_url = exa_url
        self.http_mode = (http_mode or "raw").lower()
        self.exa_model = exa_model
        self.model_dir = model_dir
        self.device = device
        self._hf = None  # lazy-initialized HF model/tokenizer pair
        self.default_max_new_tokens = int(default_max_new_tokens)

    # ---------------- HTTP backend ----------------
    def _http_generate(self, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
        from urllib import request, error
        mode = self.http_mode
        # Build payload per mode
        if mode in ("raw", "generate"):
            payload = {
                "prompt": prompt,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
            }
        elif mode in ("openai-chat", "oai-chat", "chat"):
            payload = {
                "model": self.exa_model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_new_tokens),
            }
        elif mode in ("openai-completion", "oai-completion", "completion"):
            payload = {
                "model": self.exa_model or "default",
                "prompt": prompt,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_new_tokens),
            }
        elif mode in ("tgi", "text-generation-inference"):
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_new_tokens": int(max_new_tokens),
                },
            }
        else:
            payload = {"prompt": prompt, "temperature": float(temperature), "top_p": float(top_p), "max_new_tokens": int(max_new_tokens)}

        req = request.Request(self.exa_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=30.0) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(body)
        except (error.HTTPError, error.URLError, TimeoutError) as e:
            raise RuntimeError(f"EXAONE HTTP backend error: {e}")
        except Exception as e:
            raise RuntimeError(f"EXAONE HTTP backend decode error: {e}")

        # Flexible response parsing
        # 1) direct fields
        for key in ("text", "generated_text", "generated"):
            if isinstance(data, dict) and key in data and isinstance(data[key], str):
                return data[key]
        # 2) OpenAI chat/completion
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            c0 = data["choices"][0]
            # chat: message.content
            if isinstance(c0, dict) and isinstance(c0.get("message"), dict) and isinstance(c0["message"].get("content"), str):
                return c0["message"]["content"]
            # completion: text
            if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                return c0["text"]
        # 3) generic list fallback
        if isinstance(data, dict):
            for key in ("choices", "outputs", "results"):
                v = data.get(key)
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    for k2 in ("text", "generated_text"):
                        if k2 in v[0]:
                            return str(v[0][k2])
        return ""

    # ---------------- HF backend ----------------
    def _ensure_hf(self):
        if self._hf is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except Exception as e:
            raise RuntimeError("HF backend requires 'transformers' and 'torch' to be installed") from e
        if not self.model_dir:
            raise RuntimeError("HF backend requires model_dir to be set")
        tok = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype="auto")
        model = model.to(self.device)
        model.eval()
        self._hf = (tok, model)

    def _hf_generate(self, prompt: str, temperature: float, top_p: float, max_new_tokens: Optional[int]) -> str:
        from torch import no_grad
        import torch
        self._ensure_hf()
        tok, model = self._hf
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        # Dynamic cap based on model context
        try:
            # Try common config fields
            max_ctx = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None) or getattr(model.config, "max_seq_len", None)
            if isinstance(max_ctx, int) and max_ctx > 0:
                used = int(inputs["input_ids"].shape[-1])
                # Leave some headroom (5%)
                avail = max(16, int(max_ctx * 0.95) - used)
            else:
                avail = self.default_max_new_tokens
        except Exception:
            avail = self.default_max_new_tokens
        gen_max = int(self.default_max_new_tokens if max_new_tokens is None else max_new_tokens)
        gen_max = max(1, min(gen_max, avail))
        with no_grad():
            out = model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(1e-4, float(temperature)),
                top_p=float(top_p),
                max_new_tokens=gen_max,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        return text

    # ---------------- Public API ----------------
    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: Optional[int] = None) -> str:
        if self.backend == "http":
            # For HTTP backends we cannot know model context; send requested or default
            max_send = int(self.default_max_new_tokens if max_new_tokens is None else max_new_tokens)
            return self._http_generate(prompt, temperature, top_p, max_send)
        if self.backend in {"hf", "transformers"}:
            return self._hf_generate(prompt, temperature, top_p, max_new_tokens)
        raise ValueError(f"Unknown backend: {self.backend}")
