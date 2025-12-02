"""
TCodebook sliding-window tokenizer.

- Uses TCodebook as the structural tokenizer (Gt x Kt symbols).
- Sliding window over raw text to avoid collapsing long strings into few tokens.
- One-way: ids->text is intentionally unsupported (structure-only hash).
"""

from typing import List, Sequence
import torch

from codebook import TCodebook


class TCodebookTokenizer:
    """Structural tokenizer wrapping TCodebook with sliding window slicing."""

    def __init__(
        self,
        max_len: int = 512,
        window_size: int = 8,
        stride: int = 4,
        Gt: int = 4,
        Kt: int = 256,
        dt: int = 384,
        m: int = 131072,
        ema_decay: float = 0.995,
        seed: int = 42,
        add_bos: bool = True,
        add_eos: bool = True,
        struct_offset: int = 4,  # Reserve PAD/BOS/EOS/UNK at 0-3
    ):
        self.max_len = max_len
        self.window_size = window_size
        self.stride = stride
        self.Gt = Gt
        self.Kt = Kt
        self.dt = dt
        self.m = m
        self.ema_decay = ema_decay
        self.seed = seed
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.struct_offset = max(0, struct_offset)

        # Special tokens (kept minimal; structural ids are offset by struct_offset)
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3

        self.vocab_size = self.struct_offset + (self.Gt * self.Kt)
        self.tokenizer_type = "TCodebook-Sliding"

        # Engine (TCodebook) with structural defaults
        self.engine = TCodebook(
            Gt=self.Gt,
            Kt=self.Kt,
            dt=self.dt,
            m=self.m,
            ema_decay=self.ema_decay,
            seed=self.seed,
        )

        # Prefix mapping for subspaces
        self.prefix_map = {chr(ord("A") + i): i for i in range(self.Gt)}
        self.int_to_prefix = {v: k for k, v in self.prefix_map.items()}

    # ------------------ Public API ------------------
    def encode(self, text: str) -> torch.Tensor:
        """
        Text -> token ids (padded to max_len).

        Note: This tokenizer is one-way. ids->text reconstruction is not supported.
        """
        ids: List[int] = []
        if self.add_bos:
            ids.append(self.BOS)

        ids.extend(self.text_to_ids(text))

        if self.add_eos:
            ids.append(self.EOS)

        # Truncate and pad
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids.extend([self.PAD] * (self.max_len - len(ids)))

        return torch.tensor(ids, dtype=torch.long)

    def text_to_ids(self, text: str) -> List[int]:
        """Raw text -> structural token ids (unpadded)."""
        if not text:
            return []
        ids: List[int] = []
        for chunk in self._chunk_text(text):
            tokens = self.engine.encode(chunk)
            ids.extend(self._tokens_to_ids(tokens))
        return ids

    def ids_to_struct_tokens(self, ids: Sequence[int]) -> List[str]:
        """
        ids -> structural tokens (debug only).
        Specials (PAD/BOS/EOS/UNK) are skipped.
        """
        out: List[str] = []
        for raw in ids:
            idx = int(raw) if isinstance(raw, (int, float)) or hasattr(raw, "item") else raw
            if hasattr(idx, "item"):
                idx = int(idx.item())
            if idx < self.struct_offset:
                continue
            rel = idx - self.struct_offset
            subspace = rel // self.Kt
            code_idx = rel % self.Kt
            prefix = self.int_to_prefix.get(subspace, "A")
            out.append(f"K{prefix}{code_idx:02d}")
        return out

    def decode(self, ids: Sequence[int]) -> str:
        """
        Structural decode: returns space-separated structural tokens.
        Natural-language reconstruction is intentionally unsupported.
        """
        return " ".join(self.ids_to_struct_tokens(ids))

    # ------------------ Internal helpers ------------------
    def _chunk_text(self, text: str) -> List[str]:
        """Sliding-window chunking with optional overlap."""
        if len(text) <= self.window_size:
            return [text] if text.strip() else []
        chunks: List[str] = []
        for i in range(0, len(text), self.stride):
            chunk = text[i : i + self.window_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for t in tokens:
            try:
                prefix_char = t[1]
                idx_str = t[2:]
                subspace_idx = self.prefix_map.get(prefix_char, 0)
                code_idx = int(idx_str)
                token_id = self.struct_offset + (subspace_idx * self.Kt) + code_idx
                # Clamp to vocab bounds
                token_id = max(0, min(token_id, self.vocab_size - 1))
                ids.append(token_id)
            except Exception:
                ids.append(self.UNK)
        return ids


# Convenience alias
TCodeTokenizer = TCodebookTokenizer
