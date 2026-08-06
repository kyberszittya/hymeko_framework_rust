"""Iso-parameter Transformer baseline for the IMDB architecture-fairness test.

Companion to ``text_classifier.IMDBClassifier`` (Sequential HSiKAN).
Wires up the same input pipeline (token IDs + boolean mask) into a
PyTorch ``nn.TransformerEncoder`` of matched parameter count, then
mask-aware mean-pools and linear-projects to class logits.

Parameter-count target: 321,137 (the IMDBClassifier number at
|V|=20k, C=4, K=4, enc_depth=3, max_len=200). The embedding row
(|V| × d_model) dominates; with d_model=16 the embedding alone is
320k, leaving room for ~1-7k of transformer + head.

Plan: docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/
(implementation extension: 2026-05-18 architectural-fairness probe).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPositional(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani 2017).

    Stored as a non-learnable buffer; broadcast-added to the
    embedding output.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class IMDBTransformerBaseline(nn.Module):
    """Small TransformerEncoder for IMDB binary sentiment.

    Designed to be parameter-count-matched to
    :class:`text_classifier.IMDBClassifier` at its default config
    (~321 k params). The embedding row dominates; ``d_model=16`` is
    deliberately small so the embedding (|V|=20k × 16 = 320k)
    leaves a budget for a thin transformer stack.

    Args
    ----
    vocab_size : int
        Vocabulary size.
    d_model : int, default 16
        Embedding / hidden width.
    n_heads : int, default 2
        Attention head count (must divide d_model).
    dim_ff : int, default 64
        Feedforward inner width.
    n_layers : int, default 2
        Number of TransformerEncoderLayer blocks.
    max_len : int, default 200
        Maximum sequence length (positional cache cap).
    n_classes : int, default 2
        Output class count.
    dropout : float, default 0.1
        Dropout applied inside attention + feedforward.

    Forward shapes
    --------------
    token_ids : (B, L) int64, padding == 0.
    mask      : (B, L) bool — True for real tokens, False for pad.
                If None, no masking is applied.

    Returns (B, n_classes) class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 16,
        n_heads: int = 2,
        dim_ff: int = 64,
        n_layers: int = 2,
        max_len: int = 200,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.n_classes = int(n_classes)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.embed.weight, std=0.05)
        with torch.no_grad():
            self.embed.weight[0].zero_()  # zero pad row
        self.positional = _SinusoidalPositional(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_head = nn.Linear(d_model, n_classes)

    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-token hidden states (B, L, d_model).

        Used by the MLM pretrain path; the forward path adds pooling
        and a class head on top.
        """
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be (B, L); got {tuple(token_ids.shape)}"
            )
        if token_ids.shape[1] > self.max_len:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds max_len="
                f"{self.max_len}"
            )
        x = self.embed(token_ids)
        x = self.positional(x)
        if mask is not None:
            # TransformerEncoder src_key_padding_mask: True at masked positions.
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(token_ids, mask=mask)  # (B, L, d_model)
        if mask is not None:
            if mask.shape != token_ids.shape:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must equal token_ids "
                    f"{tuple(token_ids.shape)}"
                )
            mask_f = mask.to(h.dtype).unsqueeze(-1)
            h = h * mask_f
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = h.sum(dim=1) / denom
        else:
            pooled = h.mean(dim=1)
        return self.cls_head(pooled)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
