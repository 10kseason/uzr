import argparse
import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Union

from .model import UZRModel, ByteTokenizer, KoEnTokenizer
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
    ):
        self.model = model
        self.memory = memory
        self.tok = tokenizer
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.use_memory = use_memory

        self.model.eval()
        self.history: List[str] = []

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

                # Forward pass
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

        # Update memory with this interaction
        if self.use_memory and self.memory is not None:
            self.model.update_memory_state(input_ids, z_rule)

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
    return KoEnTokenizer(max_len=max_len)


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
            max_items=8192,
            device=device,
            enable_learning=True,
            learn_hidden=512,
            learn_depth=3,
        )
        mem.load_state_dict(data["memory"])
        print(f"Loaded {len(mem.items)} memory items, learner: {mem.has_learner()}")

    # Create model
    model = UZRModel(
        vocab_size=tok.vocab_size,
        d_model=args.get("d_model", 512),
        z_dim=args.get("z_dim", 1024),
        max_len=max_len,
        z_think_dim=args.get("z_think_dim", 128),
        z_lang_dim=args.get("z_lang_dim", 64),
        num_langs=args.get("num_langs", 4),
        identity_self_dim=args.get("identity_self_dim", 2),
        memory=mem,
    )

    # Load model weights
    model.load_state_dict(data["model"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  - d_model: {args.get('d_model', 512)}")
    print(f"  - z_dim: {args.get('z_dim', 1024)}")
    print(f"  - identity_self_dim: {args.get('identity_self_dim', 2)}")

    return model, mem, tok, args


def main():
    parser = argparse.ArgumentParser(description="Chat with UZR model (루리아)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--no_memory", action="store_true", help="Disable memory usage")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load checkpoint
    model, memory, tokenizer, ckpt_args = load_checkpoint(args.ckpt, device=args.device)

    # Create chat session
    session = ChatSession(
        model=model,
        memory=memory,
        tokenizer=tokenizer,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        use_memory=not args.no_memory,
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
