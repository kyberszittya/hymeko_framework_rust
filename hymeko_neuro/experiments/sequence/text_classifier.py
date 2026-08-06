"""Sequential HSiKAN text classifier.

Wires up TokenMultivectorEmbedding + DualPathEncoder (encode path of
DualPathEncoderDecoder, decoder unused) + mask-aware mean pool +
linear classification head. The encoder is exactly the Clifford-FIR
multichannel architecture validated on the synthetic copy task
(99.83% TF-acc / 99.31% greedy-decode after scheduled-sampling fix).

Plan: docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .clifford import CL2_DIM
from .encoder_decoder import DualPathEncoderDecoder


class IMDBClassifier(nn.Module):
    """Sequential HSiKAN encoder + mean pool + linear classifier.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size; the embedding table has shape (V, C * CL2_DIM).
    enc_depth : int, default 3
        Number of DualPath encoder blocks.
    K : int, default 4
        FIR window length per block.
    n_channels : int, default 4
        Number of Cl(2, 0) channels per token (v2 multichannel).
    max_len : int, default 200
        Maximum sequence length (truncation cap; positional encoding bound).
    n_classes : int, default 2
        Output class count. IMDB is binary (pos / neg).

    The pooled representation dim is ``n_channels * CL2_DIM`` (= 16
    at the defaults). The plan's ``d=64`` parameter was a misreading
    of the embedding's flat width — the actual per-token vector
    width is fixed by the multivector structure (n_channels * 4).

    Forward shapes
    --------------
    ``token_ids``: (B, L) int64; padding token = 0.
    ``mask``     : (B, L) bool — True where the position is a real token.
                   If None, all positions are pooled (used for the
                   ``shuffled_padding`` invariance test).

    Returns (B, n_classes) class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        enc_depth: int = 3,
        K: int = 4,
        n_channels: int = 4,
        max_len: int = 200,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_channels = int(n_channels)
        self.max_len = int(max_len)
        self.n_classes = int(n_classes)
        # Reuse the encode path of DualPathEncoderDecoder; dec_depth=0
        # builds an empty decoder ModuleList (zero params, zero
        # forward cost — we never call .forward() / .decode_step()).
        self.encoder = DualPathEncoderDecoder(
            vocab_size=vocab_size,
            enc_depth=enc_depth,
            dec_depth=0,
            K=K,
            max_len=max_len,
            tie_embeddings=True,
            n_channels=n_channels,
        )
        d_out = n_channels * CL2_DIM
        self.cls_head = nn.Linear(d_out, n_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be (B, L); got {tuple(token_ids.shape)}"
            )
        if token_ids.shape[1] > self.max_len:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds max_len="
                f"{self.max_len}"
            )
        h, _sigma = self.encoder.encode(token_ids)
        # h is (B, L, 4) at C=1 or (B, L, C, 4) at C>1; flatten to (B, L, C*4).
        if h.dim() == 4:
            B, L, C, D = h.shape
            h_flat = h.reshape(B, L, C * D)
        else:
            h_flat = h
        # Mask-aware mean pool: zero padded positions, divide by mask sum.
        if mask is not None:
            if mask.shape != token_ids.shape:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must equal token_ids "
                    f"{tuple(token_ids.shape)}"
                )
            mask_f = mask.to(h_flat.dtype).unsqueeze(-1)  # (B, L, 1)
            h_flat = h_flat * mask_f
            denom = mask_f.sum(dim=1).clamp_min(1.0)       # (B, 1)
            pooled = h_flat.sum(dim=1) / denom              # (B, C*4)
        else:
            pooled = h_flat.mean(dim=1)
        return self.cls_head(pooled)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
