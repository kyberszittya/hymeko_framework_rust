"""The autoregressive FSR language model: Gömb embed -> L FSR blocks -> token head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_lm.block import FSRBlock
from hymeko_lm.config import FSRConfig, ResidualMode
from hymeko_lm.sphere import SphereEmbedding, l2_normalize


class _ScaledNorm(nn.Module):
    """Unit-normalise then apply a learnable per-dim scale — gives the sphere readout a temperature so
    logits can sharpen past the unigram (nGPT-style). Without it a normalised stream is pinned to ~unigram."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(x) * self.scale


class FSRLanguageModel(nn.Module):
    """Causal LM. ``forward(ids:(B,T)) -> logits (B,T,vocab)``.

    # Preconditions ``ids`` are ``long`` in ``[0, vocab)``; ``T <= cfg.max_seq_len``.
    # Postconditions logits ``(B,T,vocab)``; position ``i`` depends only on ``ids[:, :i+1]``.
    """

    def __init__(self, cfg: FSRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sphere = cfg.residual_mode is ResidualMode.SPHERE
        self.embed = SphereEmbedding(cfg.vocab_size, cfg.d_model, normalize=sphere)
        self.blocks = nn.ModuleList(FSRBlock(cfg) for _ in range(cfg.n_layers))
        # A normalised (sphere) stream needs a learnable readout scale to sharpen logits past the
        # unigram; pre-norm needs the standard final LayerNorm. Identity would pin sphere at unigram.
        self.final_norm: nn.Module = nn.LayerNorm(cfg.d_model) if not sphere else _ScaledNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for block in self.blocks:
            h = block(h)
        logits: torch.Tensor = self.head(self.final_norm(h))
        return logits

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean next-token cross-entropy over all positions.

        # Preconditions ``ids`` and ``targets`` are ``(B,T)``; ``targets`` in ``[0, vocab)``.
        """
        logits = self(ids)
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
