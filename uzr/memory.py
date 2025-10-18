from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class MemoryItem:
    key: torch.Tensor   # [D]
    val: Dict[str, Any] # sketch: {'z_slow': Tensor[D], 'avg_emb': Tensor[D], 'meta': ...}
    step: int = 0


class MemoryLearner(nn.Module):
    """Simple MLP regressor that maps averaged embeddings to stored latent codes."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        prev_dim = in_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompressedMemory:
    def __init__(
        self,
        max_items: int = 4096,
        device: str = "cpu",
        enable_learning: bool = True,
        learn_hidden: int = 256,
        learn_depth: int = 2,
        learn_rate: float = 1e-3,
        learn_weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        ema_decay: float = 0.9,
        min_train_items: int = 16,
    ):
        self.items: List[MemoryItem] = []
        self.max_items = max_items
        self.device = device

        # Learning-related configuration/state
        self.enable_learning = enable_learning
        self.learn_hidden = learn_hidden
        self.learn_depth = learn_depth
        self.learn_rate = learn_rate
        self.learn_weight_decay = learn_weight_decay
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.min_train_items = min_train_items

        self._learner: Optional[MemoryLearner] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._ema_loss: Optional[float] = None
        self._learn_fields: Optional[Tuple[str, str]] = None  # (key_field, target_field)

        # Real-time state tracking
        self._session_state: Dict[str, Any] = {}
        self._input_history: List[torch.Tensor] = []
        self._state_history: List[Dict[str, Any]] = []
        self._max_history: int = 100

    # -------------------------
    # Core memory operations
    # -------------------------
    def _prune(self):
        if len(self.items) > self.max_items:
            # simple FIFO prune
            self.items = self.items[-self.max_items:]

    def add(self, key: torch.Tensor, val: Dict[str, Any], step: int):
        packed_val = {}
        for k, v in val.items():
            if torch.is_tensor(v):
                packed_val[k] = v.detach().to(self.device)
            else:
                packed_val[k] = v
        self.items.append(
            MemoryItem(
                key=key.detach().to(self.device),
                val=packed_val,
                step=step,
            )
        )
        self._prune()
        if self.enable_learning:
            self._ensure_learner()

    def retrieve(self, query: torch.Tensor, topk: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        keys = torch.stack([it.key for it in self.items], dim=0)  # [N, D]
        q = query.unsqueeze(0)                                    # [1, D]
        sim = F.cosine_similarity(keys, q.expand_as(keys), dim=-1)  # [N]
        vals, idx = torch.topk(sim, k=min(topk, keys.size(0)))
        return [self.items[i] for i in idx.tolist()]

    def sample(self, batch: int = 4) -> List[MemoryItem]:
        if not self.items:
            return []
        return random.sample(self.items, k=min(batch, len(self.items)))

    # -------------------------
    # Learning helpers
    # -------------------------
    def _iter_items_with_fields(self, key_field: str, target_field: str):
        for it in self.items:
            src = it.val.get(key_field)
            tgt = it.val.get(target_field)
            if torch.is_tensor(src) and torch.is_tensor(tgt):
                yield it

    def _ensure_learner(self, key_field: str = "avg_emb", target_field: str = "z_slow"):
        if not self.enable_learning or self._learner is not None:
            return
        for it in reversed(self.items):
            src = it.val.get(key_field)
            tgt = it.val.get(target_field)
            if torch.is_tensor(src) and torch.is_tensor(tgt):
                in_dim = src.numel()
                out_dim = tgt.numel()
                self._learn_fields = (key_field, target_field)
                self._learner = MemoryLearner(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=self.learn_hidden,
                    depth=self.learn_depth,
                ).to(self.device)
                self._optimizer = optim.Adam(
                    self._learner.parameters(),
                    lr=self.learn_rate,
                    weight_decay=self.learn_weight_decay,
                )
                break

    def has_learner(self) -> bool:
        return self._learner is not None

    @property
    def learner_loss(self) -> Optional[float]:
        """Exponential moving average of training loss (if learning is enabled)."""
        return self._ema_loss

    def train_model(
        self,
        steps: int = 1,
        batch_size: int = 32,
        key_field: str = "avg_emb",
        target_field: str = "z_slow",
        shuffle: bool = True,
    ) -> Optional[float]:
        """Train the internal learner to map stored embeddings to latent codes.

        Returns the average loss over the executed steps (MSE) or ``None`` if no
        update was applied (e.g. insufficient data or learning disabled).
        """
        if not self.enable_learning:
            raise RuntimeError("Learning is disabled for this memory instance.")

        available = [it for it in self._iter_items_with_fields(key_field, target_field)]
        if not available:
            return None
        if len(available) < min(self.min_train_items, batch_size) and len(available) < 2:
            return None

        if self._learn_fields is None:
            self._learn_fields = (key_field, target_field)
        elif self._learn_fields != (key_field, target_field):
            raise ValueError(
                f"Learner already initialised for fields {self._learn_fields},"
                f" cannot retrain with ({key_field}, {target_field})."
            )

        self._ensure_learner(key_field=key_field, target_field=target_field)
        if self._learner is None or self._optimizer is None:
            return None

        self._learner.train()
        losses = []
        for _ in range(steps):
            if shuffle:
                batch = random.sample(available, k=min(batch_size, len(available)))
            else:
                batch = available[:min(batch_size, len(available))]
            inputs = torch.stack([it.val[key_field] for it in batch], dim=0).to(self.device)
            targets = torch.stack([it.val[target_field] for it in batch], dim=0).to(self.device)

            preds = self._learner(inputs)
            loss = F.mse_loss(preds, targets)

            self._optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._learner.parameters(), self.grad_clip)
            self._optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            if self._ema_loss is None:
                self._ema_loss = loss_val
            else:
                self._ema_loss = self.ema_decay * self._ema_loss + (1 - self.ema_decay) * loss_val
            losses.append(loss_val)

        if not losses:
            return None
        return sum(losses) / len(losses)

    def predict(
        self,
        avg_emb: torch.Tensor,
        topk: int = 4,
        blend: float = 0.5,
        key_field: str = "avg_emb",
        target_field: str = "z_slow",
        return_components: bool = False,
    ):
        """Infer a latent vector from memory using learned mapping + nearest neighbours.

        Args:
            avg_emb: Un-normalised average embedding vector of shape [D].
            topk: Number of neighbours to use for the retrieval component.
            blend: Weight for the learner prediction when both sources are available.
            key_field: Field name that stores the embedding in memory items.
            target_field: Field name for the latent/slow vector to recover.
            return_components: If True, also return the raw components and items.

        Returns:
            prediction (Tensor or None) if ``return_components`` is False.
            Otherwise ``(prediction, neighbours, info_dict)`` where ``info_dict``
            exposes the individual contributions.
        """
        if avg_emb.dim() != 1:
            raise ValueError("avg_emb must be a 1D tensor")

        device = torch.device(self.device)
        avg_emb = avg_emb.to(device)
        normed = F.normalize(avg_emb.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)

        neighbours = self.retrieve(normed, topk=topk) if topk > 0 else []
        knn_vec = None
        if neighbours:
            cand = [it.val.get(target_field) for it in neighbours if torch.is_tensor(it.val.get(target_field))]
            if cand:
                knn_vec = torch.stack(cand, dim=0).mean(dim=0)

        model_vec = None
        if self._learner is not None and self.enable_learning:
            self._learner.eval()
            with torch.no_grad():
                model_vec = self._learner(avg_emb.unsqueeze(0)).squeeze(0)

        prediction = None
        if model_vec is not None and knn_vec is not None:
            prediction = blend * model_vec + (1.0 - blend) * knn_vec
        elif model_vec is not None:
            prediction = model_vec
        else:
            prediction = knn_vec

        info = {
            "model_pred": model_vec.detach() if isinstance(model_vec, torch.Tensor) else None,
            "knn_pred": knn_vec.detach() if isinstance(knn_vec, torch.Tensor) else None,
            "ema_loss": self._ema_loss,
            "topk": topk,
            "blend": blend,
        }

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach()
        if not return_components:
            return prediction
        return prediction, neighbours, info

    # -------------------------
    # Real-time state tracking
    # -------------------------
    def update_state(self, key: str, value: Any):
        """Update session state with a key-value pair."""
        self._session_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        return self._session_state.get(key, default)

    def clear_state(self):
        """Clear all session state."""
        self._session_state.clear()
        self._input_history.clear()
        self._state_history.clear()

    def track_input(self, avg_emb: torch.Tensor, z_result: Optional[torch.Tensor] = None):
        """Track input embedding and optionally its result for history."""
        self._input_history.append(avg_emb.detach().cpu())
        if len(self._input_history) > self._max_history:
            self._input_history = self._input_history[-self._max_history:]

        if z_result is not None:
            state = {"avg_emb": avg_emb.detach().cpu(), "z": z_result.detach().cpu()}
            self._state_history.append(state)
            if len(self._state_history) > self._max_history:
                self._state_history = self._state_history[-self._max_history:]

    def get_or_predict_z(
        self,
        avg_emb: torch.Tensor,
        z_init: Optional[torch.Tensor] = None,
        topk: int = 4,
        blend: float = 0.5,
        use_predictor: bool = True,
        key_field: str = "avg_emb",
        target_field: str = "z_slow",
    ) -> torch.Tensor:
        """Get or predict z from memory using learned predictor and retrieval.

        Args:
            avg_emb: Average embedding tensor [D]
            z_init: Initial z to use if memory has no prediction (optional)
            topk: Number of neighbours for retrieval
            blend: Blend factor between learned and retrieval predictions
            use_predictor: Whether to use the learned predictor
            key_field: Field name for embeddings in memory
            target_field: Field name for z vectors in memory

        Returns:
            Predicted or retrieved z tensor
        """
        if avg_emb.dim() != 1:
            raise ValueError("avg_emb must be a 1D tensor")

        # Try to predict from memory if learning is enabled
        if use_predictor and self.enable_learning and self.has_learner():
            prediction = self.predict(
                avg_emb=avg_emb,
                topk=topk,
                blend=blend,
                key_field=key_field,
                target_field=target_field,
                return_components=False,
            )
            if prediction is not None:
                self.track_input(avg_emb, prediction)
                return prediction.to(self.device)

        # Fallback to retrieval-only
        device = torch.device(self.device)
        avg_emb_dev = avg_emb.to(device)
        normed = F.normalize(avg_emb_dev.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0)

        neighbours = self.retrieve(normed, topk=topk) if topk > 0 else []
        if neighbours:
            cand = [it.val.get(target_field) for it in neighbours if torch.is_tensor(it.val.get(target_field))]
            if cand:
                z_pred = torch.stack(cand, dim=0).mean(dim=0)
                self.track_input(avg_emb, z_pred)
                return z_pred.to(device)

        # Final fallback to provided init or zero
        if z_init is not None:
            self.track_input(avg_emb, z_init)
            return z_init.to(device)

        # Return zeros if nothing else available
        if neighbours and len(neighbours) > 0:
            dim = neighbours[0].val.get(target_field).numel()
            z_zero = torch.zeros(dim, device=device)
        else:
            z_zero = torch.zeros(128, device=device)  # default dim

        self.track_input(avg_emb, z_zero)
        return z_zero

    def get_recent_context(self, n: int = 5) -> List[Dict[str, torch.Tensor]]:
        """Get the n most recent state history entries."""
        return self._state_history[-n:] if self._state_history else []

    # -------------------------
    # Serialization
    # -------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Export memory state for saving to checkpoint."""
        state = {
            "items": self.items,
            "config": {
                "max_items": self.max_items,
                "device": self.device,
                "enable_learning": self.enable_learning,
                "learn_hidden": self.learn_hidden,
                "learn_depth": self.learn_depth,
                "learn_rate": self.learn_rate,
                "learn_weight_decay": self.learn_weight_decay,
                "grad_clip": self.grad_clip,
                "ema_decay": self.ema_decay,
                "min_train_items": self.min_train_items,
            },
            "learner_state": self._learner.state_dict() if self._learner is not None else None,
            "optimizer_state": self._optimizer.state_dict() if self._optimizer is not None else None,
            "ema_loss": self._ema_loss,
            "learn_fields": self._learn_fields,
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore memory state from checkpoint."""
        self.items = state.get("items", [])

        # Restore learner if it was saved
        if state.get("learner_state") is not None and state.get("learn_fields") is not None:
            self._learn_fields = state["learn_fields"]
            key_field, target_field = self._learn_fields

            # Infer dimensions from saved items
            for it in reversed(self.items):
                src = it.val.get(key_field)
                tgt = it.val.get(target_field)
                if torch.is_tensor(src) and torch.is_tensor(tgt):
                    in_dim = src.numel()
                    out_dim = tgt.numel()

                    self._learner = MemoryLearner(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        hidden_dim=self.learn_hidden,
                        depth=self.learn_depth,
                    ).to(self.device)
                    self._learner.load_state_dict(state["learner_state"])

                    self._optimizer = optim.Adam(
                        self._learner.parameters(),
                        lr=self.learn_rate,
                        weight_decay=self.learn_weight_decay,
                    )
                    if state.get("optimizer_state") is not None:
                        self._optimizer.load_state_dict(state["optimizer_state"])

                    break

        self._ema_loss = state.get("ema_loss")


def make_sketch(avg_emb: torch.Tensor, z_slow: torch.Tensor, meta: Dict[str, Any] = None):
    key = F.normalize(avg_emb, dim=-1)
    val = {"avg_emb": avg_emb, "z_slow": z_slow}
    if meta:
        val["meta"] = meta
    return key, val
