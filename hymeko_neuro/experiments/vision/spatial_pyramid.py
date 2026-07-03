"""SpatialPyramidPool — a compressed, scalable, GPU-friendly flatten readout.

A flattened spatial feature map (``n_items · d`` head) keeps all position but does
not scale and cannot handle a variable item count. This pools a set of positioned
features into a **fixed multi-scale descriptor**: items are binned into a quadtree
pyramid (levels e.g. 1×1, 2×2, 4×4 = 21 cells) by their normalised position and
mean-pooled per cell. ``out_dim = (Σ levelᵢ²)·d`` — independent of the item count.

**It is a linear operator.** Pooling = ``cells = P @ features`` where ``P`` is the
``(n_cells, N)`` row-normalised cell-indicator matrix (each row = one pyramid cell,
summing to 1 = a mean). So all levels collapse into **one matmul**:

* for a *fixed* layout (a patch grid) ``P`` is constant — precompute it once
  (``set_fixed_positions``) and the readout is a single, fused, **batched** matmul
  ``einsum('cn,bnd->bcd', P, features)`` — no per-item kernel launches, no scatter,
  no Python level-loop. GPU-optimal.
* for a *variable* layout (detector anchors, adaptive trees) ``P`` is built per
  call from the positions (still one matmul).

On Cluttered-MNIST this recovers ≈90 % of a flatten readout's accuracy at ≈1/5 of
its parameters. Domain-agnostic — the input is just ``(features, positions∈[0,1]²)``
— so detection heads, RL vision, point/graph readouts can drop it in for flatten.
An ablation (2026-06-30) found the learned per-cell **gate** inert; ``dynamic``
defaults to ``False`` (the pyramid structure is the compressor).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pooling_matrix(positions: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    """Row-normalised cell-indicator matrix ``P (n_cells, N)`` for a quadtree
    pyramid over ``positions`` (N, 2) in ``[0,1]²``. ``P @ features`` mean-pools
    each cell; empty-cell rows are all-zero (→ zero feature, not NaN).
    """
    n = positions.shape[0]
    cols = torch.arange(n, device=positions.device)
    mats = []
    for lvl in levels:
        idx = (positions * lvl).floor().clamp(0, lvl - 1).long()      # (N, 2)
        flat = idx[:, 0] * lvl + idx[:, 1]                            # (N,)
        m = positions.new_zeros(lvl * lvl, n)
        m[flat, cols] = 1.0
        mats.append(m / m.sum(dim=1, keepdim=True).clamp_min(1.0))    # mean per cell
    return torch.cat(mats, dim=0)                                     # (n_cells, N)


class SpatialPyramidPool(nn.Module):
    """Pool ``(features (..., N, d), positions (N, 2) in [0,1])`` → ``(..., out_dim)``.

    Batched: ``features`` may carry leading batch dims; the pooling is one matmul
    over the whole batch. ``set_fixed_positions`` caches ``P`` for a constant
    layout so ``forward(features)`` needs no positions and does a single fused
    matmul (the GPU-optimal path).

    Preconditions: ``positions`` in ``[0,1]²``; ``features.shape[-2] == N``.
    Postconditions: output ``(..., out_dim)``, ``out_dim = (Σ levelᵢ²)·d``, finite.
    """

    _Pbuf: torch.Tensor | None

    def __init__(self, d_hidden: int, levels: tuple[int, ...] = (1, 2, 4),
                 dynamic: bool = False) -> None:
        super().__init__()
        self.d = d_hidden
        self.levels = tuple(levels)
        self.out_dim = sum(lvl * lvl for lvl in self.levels) * d_hidden
        self.gate = nn.Linear(d_hidden, 1) if dynamic else None
        self.register_buffer("_Pbuf", None)

    def set_fixed_positions(self, positions: torch.Tensor) -> None:
        """Precompute and cache ``P`` for a constant layout (e.g. a patch grid)."""
        self._Pbuf = pooling_matrix(positions, self.levels)

    def forward(self, features: torch.Tensor,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is not None:
            p = pooling_matrix(positions, self.levels)
        elif self._Pbuf is not None:
            p = self._Pbuf
        else:
            raise ValueError("pass positions, or call set_fixed_positions first")
        cells = torch.einsum("cn,...nd->...cd", p, features)         # (..., n_cells, d)
        if self.gate is not None:
            cells = cells * torch.sigmoid(self.gate(cells))
        return cells.reshape(*cells.shape[:-2], -1)


def grid_positions(h: int, w: int) -> torch.Tensor:
    """Normalised centres ``(h·w, 2)`` of a row-major ``h×w`` grid — the
    ``positions`` for a dense patch map (so a grid readout reuses the same pool)."""
    rows = (torch.arange(h).float() + 0.5) / h
    cols = (torch.arange(w).float() + 0.5) / w
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=-1)         # (h*w, 2)
