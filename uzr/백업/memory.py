
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import random
import torch
import torch.nn.functional as F

@dataclass
class MemoryItem:
    key: torch.Tensor   # [D]
    val: Dict[str, Any] # sketch: {'z_slow': Tensor[D], 'avg_emb': Tensor[D], 'meta': ...}
    step: int = 0

class CompressedMemory:
    def __init__(self, max_items: int = 4096, device="cpu"):
        self.items: List[MemoryItem] = []
        self.max_items = max_items
        self.device = device

    def _prune(self):
        if len(self.items) > self.max_items:
            # simple FIFO prune
            self.items = self.items[-self.max_items:]

    def add(self, key: torch.Tensor, val: Dict[str, Any], step: int):
        self.items.append(MemoryItem(key.detach().to(self.device), {k:(v.detach().to(self.device) if torch.is_tensor(v) else v) for k,v in val.items()}, step))
        self._prune()

    def retrieve(self, query: torch.Tensor, topk: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        keys = torch.stack([it.key for it in self.items], dim=0)  # [N, D]
        q = query.unsqueeze(0)                                    # [1, D]
        sim = F.cosine_similarity(keys, q.expand_as(keys), dim=-1)  # [N]
        vals, idx = torch.topk(sim, k=min(topk, keys.size(0)))
        return [self.items[i] for i in idx.tolist()]

    def sample(self, batch: int = 4) -> List[MemoryItem]:
        return random.sample(self.items, k=min(batch, len(self.items)))

def make_sketch(avg_emb: torch.Tensor, z_slow: torch.Tensor, meta: Dict[str, Any] = None):
    key = F.normalize(avg_emb, dim=-1)
    val = {"avg_emb": avg_emb, "z_slow": z_slow}
    if meta:
        val["meta"] = meta
    return key, val
