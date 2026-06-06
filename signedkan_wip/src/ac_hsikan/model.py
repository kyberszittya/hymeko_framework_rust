"""AcHsikanClassifier — stack of AcHsikanLayer + classification head.

Iso-parameter-matched to ``signedkan_wip/src/sequence/iso_param_transformer
.py::IMDBTransformerBaseline`` at its default config (~321 k params)
when ``d_model=16, n_layers=2`` (sign-head + embeddings dominate).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import AcHsikanConfig
from .layer import AcHsikanLayer


class _SinusoidalPositional(nn.Module):
    """Standard sinusoidal positional encoding (same as transformer)."""
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * denom)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * denom)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * denom)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class AcHsikanClassifier(nn.Module):
    """Token embedding + N AcHsikanLayer + classification head.

    Inputs
    ------
    token_ids : (B, L) int64 in [0, vocab_size).
    mask      : (B, L) bool; True = real token, False = padding.

    Outputs
    -------
    logits : (B, n_classes).

    The model also exposes ``alpha()`` listing per-layer α_k for
    interpretability (regime compass).
    """

    def __init__(self, vocab_size: int, cfg: AcHsikanConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        # Embedding: full-rank by default; ALBERT-style factorized
        # (Embedding(V, r) @ Linear(r, d, bias=False)) when
        # cfg.embedding_rank ∈ [1, d_model). At V=10000, d=16, r=8:
        # 80,128 params vs 160,000 (−49% on the whole model).
        if 0 < cfg.embedding_rank < cfg.d_model:
            self.embedding = nn.Embedding(vocab_size, cfg.embedding_rank)
            self.emb_proj = nn.Linear(cfg.embedding_rank, cfg.d_model,
                                       bias=False)
        else:
            self.embedding = nn.Embedding(vocab_size, cfg.d_model)
            self.emb_proj = None
        self.positional = _SinusoidalPositional(cfg.d_model,
                                                max_len=cfg.n_positions)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [AcHsikanLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """token_ids (B, L) -> hidden (B, L, d_model)."""
        h = self.embedding(token_ids)
        if self.emb_proj is not None:
            h = self.emb_proj(h)        # (B, L, r) -> (B, L, d_model)
        h = self.positional(h)
        h = self.input_dropout(h)
        for lyr in self.layers:
            h = lyr(h)
        return h

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(token_ids, mask=mask)
        # Pool over the sequence dim.
        if self.cfg.pool == "mean":
            if mask is None:
                pooled = h.mean(dim=1)
            else:
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        elif self.cfg.pool == "max":
            pooled = h.max(dim=1).values
        elif self.cfg.pool == "cls":
            pooled = h[:, 0, :]
        else:
            raise ValueError(f"unknown pool: {self.cfg.pool}")
        return self.head(pooled)

    # ── Regime-compass accessor ─────────────────────────────────────

    def alpha(self) -> list[torch.Tensor]:
        """Per-layer α_k vectors (no_grad, on cpu)."""
        with torch.no_grad():
            return [lyr.alpha().detach().cpu() for lyr in self.layers]

    # ── Convenience ─────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
