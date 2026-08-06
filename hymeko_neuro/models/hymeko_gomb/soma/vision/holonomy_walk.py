"""Holonomy-group ablation for the Gömb-Soma walk-conv.

The 2026-06-29 vision test used only the smallest holonomy group, **Z₂** (the
brightness *sign*). Holonomy is defined for any structure group; this module
lifts the *same* brightness connection from Z₂ to **U(1)** and compares them on
a fair (position-preserving) readout, so the connection — not the pooling — is
what varies.

Modes (the ablated dimension):

* ``NONE``    — unsigned sum-pool (no connection).
* ``ROUTING`` — Z₂ as a routing switch: dual sign-branched banks (the 2026-06-15
  base-Soma).
* ``Z2``      — Z₂ as a *connection*: single bank, message × σ-product (sign).
* ``U1``      — U(1) magnetic connection: per-edge phase ``θ_e = α·tanh(Δbright)``
  (a continuous lift of the sign; ``α`` learned), walk holonomy ``Σθ_e`` applied
  as a rotation of the message's (x,y) feature pairs. ``α→∞`` recovers Z₂; this is
  the magnetic-Laplacian construction (established) used as the Soma connection.

Reuses ``PatchGraphBuilder`` for topology + brightness (no duplicated graph
logic). Readout is the position-preserving flatten (Phase 1 found mean-pool caps
every connection, so only flatten gives the holonomy a fair test).
"""
from __future__ import annotations

import enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_neuro.models.hymeko_gomb.soma.vision.patch_graph import PatchGraphBuilder


class Holonomy(enum.Enum):
    NONE = "none"
    ROUTING = "routing"
    Z2 = "z2"
    U1 = "u1"


def _rotate_pairs(msg: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive feature pairs of ``msg`` (..., d) by per-row ``theta``
    (...,) — the U(1) action. ``d`` must be even. Batch-safe (any leading dims)."""
    *lead, d = msg.shape
    m = msg.reshape(*lead, d // 2, 2)
    c = torch.cos(theta).unsqueeze(-1)                    # (..., 1)
    s = torch.sin(theta).unsqueeze(-1)
    x0, x1 = m[..., 0], m[..., 1]
    out = torch.stack([c * x0 - s * x1, s * x0 + c * x1], dim=-1)
    return out.reshape(*lead, d)


class HolonomyWalkConv(nn.Module):
    """One walk-conv layer whose connection is selected by ``mode``.

    # Preconditions ``d_in, d_out >= 1``; ``d_out`` even when ``mode is U1``.
    # Postconditions ``forward`` returns ``(n_patches, d_out)``; the U(1) and Z₂
    paths are sum-pooled (single bank), ROUTING uses two sign-branched banks.
    """

    def __init__(self, d_in: int, d_out: int, mode: Holonomy, k_arity: int = 3) -> None:
        super().__init__()
        if mode is Holonomy.U1 and d_out % 2 != 0:
            raise ValueError(f"U1 needs even d_out for pair rotation; got {d_out}")
        self.mode = mode
        self.k_arity = k_arity
        n_banks = 2 if mode is Holonomy.ROUTING else 1
        self.W = nn.Parameter(torch.empty(n_banks, k_arity, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_banks, d_out))
        self.alpha = nn.Parameter(torch.tensor(1.0)) if mode is Holonomy.U1 else None
        bound = (6.0 / (d_in * k_arity)) ** 0.5
        with torch.no_grad():
            self.W.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor, walks: torch.Tensor, walk_signs: torch.Tensor,
                walk_edge_diffs: torch.Tensor, M_v: torch.Tensor) -> torch.Tensor:
        """``x`` (n_patches, d_in) or batched (B, n_patches, d_in); ``walks``
        (n_walks, k) shared; ``walk_signs`` (..., n_walks) σ-product;
        ``walk_edge_diffs`` (..., n_walks, k-1) per-edge brightness deltas;
        ``M_v`` (n_patches, n_walks) shared. The grid topology is batch-shared."""
        batched = x.ndim == 3
        if self.mode is Holonomy.ROUTING:
            branch = (walk_signs < 0).to(torch.long)         # (..., n_walks)
        else:
            branch = torch.zeros(walk_signs.shape, dtype=torch.long,
                                 device=walks.device)
        W = self.W[branch]                                   # (..., n_walks, k, in, out)
        if batched:
            gathered = x[:, walks]                           # (B, n_walks, k, in)
            msg = torch.einsum("bnki,bnkij->bnkj", gathered, W).sum(dim=-2)
        else:
            gathered = x[walks]                              # (n_walks, k, in)
            msg = torch.einsum("nki,nkij->nkj", gathered, W).sum(dim=-2)
        msg = F.gelu(msg + self.bias[branch])
        if self.mode is Holonomy.Z2:
            msg = msg * walk_signs.to(msg.dtype).unsqueeze(-1)
        elif self.mode is Holonomy.U1:
            theta = self.alpha * torch.tanh(walk_edge_diffs).sum(dim=-1)  # (..., n_walks)
            msg = _rotate_pairs(msg, theta)
        if msg.ndim == 2:
            return torch.sparse.mm(M_v, msg)
        b, p, o = msg.shape                                  # batched aggregate
        packed = msg.transpose(0, 1).reshape(p, b * o)
        return torch.sparse.mm(M_v, packed).reshape(-1, b, o).transpose(0, 1)


class HolonomyClassifier(nn.Module):
    """Patch-graph walk-conv classifier with a selectable holonomy and a
    position-preserving (flatten) readout — the ablation testbed."""

    def __init__(self, image_h: int, image_w: int, patch_size: int, in_channels: int,
                 d_hidden: int, n_classes: int, mode: Holonomy) -> None:
        super().__init__()
        self.builder = PatchGraphBuilder(image_h, image_w, patch_size)
        self.mode = mode
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, d_hidden)
        self.conv = HolonomyWalkConv(d_hidden, d_hidden, mode)
        self.head = nn.Linear(self.builder.n_patches * d_hidden, n_classes)

    def _edge_diffs(self, patches: torch.Tensor) -> torch.Tensor:
        """Per-walk (k-1) brightness deltas Δ = bright[src]-bright[dst] for the
        walk's constituent edges — the U(1) connection's raw signal. ``[..., idx]``
        works for (N,) per-image and (B, N) batched alike."""
        bright = patches.mean(dim=-1)                        # (..., n_patches)
        edges = self.builder.edges.to(patches.device)
        ediff = bright[..., edges[:, 0]] - bright[..., edges[:, 1]]   # (..., n_edges)
        return ediff[..., self.builder.walk_edge_idx.to(patches.device)]  # (...,n_walks,k-1)

    def _forward_single(self, image: torch.Tensor) -> torch.Tensor:
        patches, walks, walk_signs, M_v = self.builder.encode(image)
        diffs = self._edge_diffs(patches)
        x = self.patch_embed(patches)
        x = self.conv(x, walks, walk_signs, diffs, M_v)
        return self.head(x.reshape(-1))

    def _forward_batched(self, images: torch.Tensor) -> torch.Tensor:
        """Whole batch in one pass (grid topology shared across the batch)."""
        device = images.device
        patches = self.builder.patchify_batch(images)            # (B, N, patch_dim)
        e_signs = self.builder.edge_signs(patches)               # (B, n_edges)
        w_signs = self.builder.walk_signs(e_signs)               # (B, n_walks)
        diffs = self._edge_diffs(patches)                        # (B, n_walks, k-1)
        x = self.patch_embed(patches)                            # (B, N, d)
        x = self.conv(x, self.builder.walks.to(device), w_signs, diffs,
                      self.builder.M_v.to(device))               # (B, N, d)
        return self.head(x.reshape(x.shape[0], -1))              # (B, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            return self._forward_single(images)
        return self._forward_batched(images)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
