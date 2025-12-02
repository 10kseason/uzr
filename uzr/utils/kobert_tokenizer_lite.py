import unicodedata
from pathlib import Path
from typing import List, Optional

import torch


class KoBertTokenizerLite:
    """Lightweight KoBERT-compatible tokenizer.

    - Uses SentencePiece model if `sentencepiece` python package is available.
    - Falls back to a simple whitespace + SPIECE_UNDERLINE heuristic otherwise.
    - Exposes PAD/BOS/EOS/UNK ids compatible with training loop expectations.
    """

    SPIECE_UNDERLINE = "\u2581"  # "▁"

    def __init__(self, kobert_dir: Path, max_len: int = 512):
        self.kobert_dir = Path(kobert_dir)
        self.max_len = int(max_len)

        vocab_txt = self.kobert_dir / "vocab.txt"
        spm_model = self.kobert_dir / "tokenizer_78b3253a26.model"

        # Build vocab
        self.itos: List[str] = []
        self.stoi = {}
        with open(vocab_txt, "r", encoding="utf-8") as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.itos.append(token)
                self.stoi[token] = idx

        # Resolve special tokens
        self.PAD = self.stoi.get("[PAD]", 0)
        self.UNK = self.stoi.get("[UNK]", 0)
        self.CLS = self.stoi.get("[CLS]", self.PAD)
        self.SEP = self.stoi.get("[SEP]", self.PAD)
        # Map training-time BOS/EOS to BERT CLS/SEP for compatibility
        self.BOS = self.CLS
        self.EOS = self.SEP

        self.vocab_size = len(self.itos)

        # Optional sentencepiece
        self._sp: Optional["SentencePieceProcessor"] = None
        try:
            import sentencepiece as spm  # type: ignore

            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(spm_model))
        except Exception:
            self._sp = None

    def _pieces(self, text: str) -> List[str]:
        s = unicodedata.normalize("NFKC", str(text))
        if self._sp is not None:
            try:
                return list(self._sp.encode(s, out_type=str))
            except Exception:
                pass

        # Fallback heuristic: whitespace split with SPIECE_UNDERLINE prefix per word
        pieces: List[str] = []
        for word in s.strip().split():
            candidate = f"{self.SPIECE_UNDERLINE}{word}"
            if candidate in self.stoi:
                pieces.append(candidate)
            elif word in self.stoi:
                pieces.append(word)
            else:
                # very naive char fallback
                added = False
                for ch in word:
                    cand_ch = f"{self.SPIECE_UNDERLINE}{ch}" if not added else ch
                    if cand_ch in self.stoi:
                        pieces.append(cand_ch)
                        added = True
                    elif ch in self.stoi:
                        pieces.append(ch)
                        added = True
                if not added:
                    pieces.append("[UNK]")
        return pieces

    def encode(self, text: str) -> torch.Tensor:
        pieces = self._pieces(text)
        ids = [self.CLS]
        for p in pieces:
            ids.append(self.stoi.get(p, self.UNK))
            if len(ids) >= self.max_len - 1:
                break
        ids.append(self.SEP)
        if len(ids) < self.max_len:
            ids.extend([self.PAD] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        toks: List[str] = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = int(idx.item())
            if idx in (self.PAD, self.CLS):
                continue
            if idx == self.SEP:
                break
            if 0 <= idx < len(self.itos):
                toks.append(self.itos[idx])
        # merge SPIECE_UNDERLINE as space
        text = "".join(toks)
        text = text.replace(self.SPIECE_UNDERLINE, " ").strip()
        return text


def load_kobert_tokenizer(kobert_dir: Path, max_len: int = 512) -> KoBertTokenizerLite:
    """Factory for KoBertTokenizerLite.

    Args:
        kobert_dir: Path to local kobert assets directory containing vocab.txt and tokenizer_*.model
        max_len: sequence length for padding/truncation
    """
    return KoBertTokenizerLite(kobert_dir=kobert_dir, max_len=max_len)

