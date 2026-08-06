"""Token-to-multivector embedding for the text Sequential HSiKAN.

Maps integer token IDs to multivector channel sequences in Cl(2,0)
and adds Clifford-rotor positional encoding.

Plan: docs/plans/2026-05-17-text-encoder-decoder-contest/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .clifford import CL2_DIM
from .clifford_positional import CliffordRotorPositional


class TokenMultivectorEmbedding(nn.Module):
    """Embed token IDs into a multivector channel sequence.

    Token id ∈ {0, …, V-1} → multivector tensor of shape (C, 4) in
    Cl(2,0). After flat embedding (V × 4C) the output is reshaped
    to channels × Clifford-dim. Optionally adds Clifford-rotor
    positional encoding via the ``positional`` kwarg.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size V.
    n_channels : int, default 4
        Number of multivector channels C per token.
    max_len : int, default 1024
        Maximum sequence length (for positional encoding cache).
    positional : bool, default True
        If True, apply Clifford-rotor positional encoding after
        embedding lookup.
    pad_token_id : int | None, default 0
        Token ID to zero-out at the embedding step (so padding
        contributes nothing). If None, no zeroing.

    Input shape:  (B, L) integer tensor of token IDs.
    Output shape: (B, L, C, 4) float multivector tensor.
    """

    def __init__(
        self,
        vocab_size: int,
        n_channels: int = 4,
        max_len: int = 1024,
        positional: bool = True,
        pad_token_id: int | None = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_channels = int(n_channels)
        self.max_len = int(max_len)
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(
            vocab_size, n_channels * CL2_DIM,
            padding_idx=pad_token_id,
        )
        nn.init.normal_(self.embed.weight, std=0.05)
        # nn.init clobbered the padding row; zero it back.
        if pad_token_id is not None:
            with torch.no_grad():
                self.embed.weight[pad_token_id].zero_()
        if positional:
            self.positional = CliffordRotorPositional(
                max_len=max_len, n_channels=n_channels,
            )
        else:
            self.positional = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be (B, L); got {tuple(token_ids.shape)}"
            )
        flat = self.embed(token_ids)              # (B, L, C * 4)
        B, L, _ = flat.shape
        x = flat.view(B, L, self.n_channels, CL2_DIM)
        if self.positional is not None:
            x = self.positional(x)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
