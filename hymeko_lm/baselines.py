"""Matched-parameter baselines for the Phase-1 A/B — the controls the FSR-LM must beat (or match).

``CausalTransformerLM`` is a standard pre-norm causal transformer (MHA + GELU MLP, learned positional
embedding). It shares the ``forward``/``loss``/``n_parameters`` surface with ``FSRLanguageModel`` so the
two are drop-in swappable in any training loop. Kept deliberately minimal: the point is a fair, honest
control at the same depth and parameter budget, not a tuned SOTA transformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerLM(nn.Module):
    """Standard causal transformer LM (the attention control). ``forward(ids:(B,T)) -> (B,T,vocab)``.

    # Preconditions ``d_model % n_heads == 0``; ``T <= max_seq_len``; ids ``long`` in ``[0,vocab)``.
    # Postconditions logits ``(B,T,vocab)``; position ``i`` depends only on ``ids[:, :i+1]``.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 max_seq_len: int, dim_ff: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, batch_first=True,
                                           activation="gelu", norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.max_seq_len}")
        pos = torch.arange(t, device=ids.device)
        h = self.tok(ids) + self.pos(pos).unsqueeze(0)
        mask = torch.triu(torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask, is_causal=True)
        logits: torch.Tensor = self.head(h)
        return logits

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(ids)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
