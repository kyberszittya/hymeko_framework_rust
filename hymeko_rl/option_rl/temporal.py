"""Temporal encoding + fusion for the assimilation runtime (ARCHITECTURAL_ASSIMILATION_V1, tracks A2 + A3).

LSTM (not GRU — prior evidence) over a causal stream of per-step feature vectors (the fused structured-state ⊕ executed
action). Only the GENERIC recurrent contract is here — no CIP node ordering, no LiNGAM attributes, no task loss, no proposal
head. The hidden state is RETURNED (part of runtime state), never a global side effect, so episodes cannot leak into each
other and a mid-episode checkpoint round-trips exactly.

Fusion (A3): ``fused = structured_encoder(structured_state) ⊕ history ⊕ lstm_temporal_embedding`` → the multimodal proposal
consumes THIS, not a raw observation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMTemporalEncoder(nn.Module):
    """Generic causal temporal encoder. Streaming: ``update(x_t, hidden) → (emb_t, hidden)`` one step. Batch:
    ``forward(X, hidden) → (embs, hidden)`` over a padded sequence. STREAM ≡ BATCH by construction (nn.LSTM is causal); the
    equivalence is the closing test. ``initial_hidden`` gives fresh zeros (no cross-episode leakage). Serializable via
    ``state_dict`` (weights) + the returned ``(h, c)`` tuple (mid-episode checkpoint)."""

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.in_dim, self.hidden_dim, self.out_dim = int(in_dim), int(hidden), int(out_dim)
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def initial_hidden(self, batch: int = 1) -> "tuple[torch.Tensor, torch.Tensor]":
        z = torch.zeros(1, batch, self.hidden_dim)
        return (z, z.clone())

    def update(self, x_t: torch.Tensor, hidden: "tuple[torch.Tensor, torch.Tensor] | None" = None):
        """One causal step. ``x_t``: (in_dim,) or (B, in_dim). Returns (emb_t (B,out_dim), hidden). Pure — hidden is
        threaded, not stored on self."""
        x = torch.as_tensor(x_t).float()
        if x.dim() == 1:
            x = x[None, :]
        b = x.shape[0]
        h = self.initial_hidden(b) if hidden is None else hidden
        out, h2 = self.lstm(x[:, None, :], h)                      # (B,1,hidden)
        return self.proj(out[:, 0, :]), h2

    def forward(self, X: torch.Tensor, hidden: "tuple[torch.Tensor, torch.Tensor] | None" = None,
                lengths: "torch.Tensor | None" = None):
        """Batch over a (B, T, in_dim) sequence (padded). ``lengths`` (B,) masks padding in the RETURNED embeddings (rows
        past a sequence's length are zeroed). Returns (embs (B,T,out_dim), hidden)."""
        X = torch.as_tensor(X).float()
        b = X.shape[0]
        h = self.initial_hidden(b) if hidden is None else hidden
        out, h2 = self.lstm(X, h)                                  # (B,T,hidden)
        embs = self.proj(out)
        if lengths is not None:
            t = torch.arange(X.shape[1])[None, :]
            mask = (t < torch.as_tensor(lengths)[:, None]).float()[..., None]
            embs = embs * mask
        return embs, h2


def fuse_state(structured_embedding: np.ndarray, temporal_embedding: np.ndarray,
               history_embedding: "np.ndarray | None" = None) -> np.ndarray:
    """A3 fusion contract: concatenate the structured-state embedding, the LSTM temporal embedding, and (optionally) an
    explicit-history embedding into the vector the multimodal proposal consumes. Concatenation (not sum) keeps the three
    channels linearly separable for the proposal; a learnable projection can follow in a task encoder. Deterministic."""
    parts = [np.asarray(structured_embedding, np.float32).reshape(-1),
             np.asarray(temporal_embedding, np.float32).reshape(-1)]
    if history_embedding is not None:
        parts.append(np.asarray(history_embedding, np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)
