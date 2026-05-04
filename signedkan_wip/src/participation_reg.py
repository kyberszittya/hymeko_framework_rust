"""Vertex-participation regulariser (R2) and hyperedge-density
regulariser (HD).

R2 — vertex-side:

    L_R2 = lam * (1 / |V|) * sum_v deg_H(v)^2 * ||h_v||_2^2

where deg_H(v) = number of triads incident to vertex v. Discourages
hub vertices that dominate aggregation.

HD — triad-side:

    L_HD = lam * (1 / |T|) * sum_t density(t) * ||h_t||_2^2

where density(t) is the number of other triads sharing at least one
vertex with t (normalised). Discourages reliance on triads in dense
clusters whose information is partly captured by their neighbours.

Both fit in the entropy-feedback / structural-prior family — they
operate on the geometric magnitude of embeddings (vertex- or
triad-level) weighted by a structural quantity (vertex-degree or
triad-density), rather than on the spectral distribution.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn


def triad_degree(triads_pyobj, n_nodes: int) -> np.ndarray:
    """Per-vertex count of incident triads."""
    deg = np.zeros(n_nodes, dtype=np.float32)
    for t in triads_pyobj:
        for v in t.v:
            deg[int(v)] += 1.0
    return deg


class ParticipationRegulariser(nn.Module):
    """L = lam * mean_v ( w(deg_H(v)) * ||h_v||^2 ).

    `deg_mode` selects the per-vertex weighting w(d):
      - "sq_max" (default, original): w(d) = d² / max(d)²
      - "log"                     :  w(d) = log(1+d) / log(1+max(d))

    On power-law graphs the squared-then-max-normalised form
    concentrates pressure on the top one or two hubs (their weight is
    ~1, most of the tail ~1e-3). The log form compresses the
    heavy tail so most vertices get a comparable share of the
    regularisation budget — a Tier 4 / F intervention in the
    gap-closing plan.
    """

    def __init__(self, lam: float = 1e-4, eps: float = 1e-8,
                 deg_mode: str = "sq_max"):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.deg_mode = deg_mode
        # The per-vertex weights are a frozen buffer.
        self.register_buffer("deg_sq", torch.zeros(0))
        self.last_value: float = float("nan")

    def set_degrees(self, deg: np.ndarray) -> None:
        deg_t = torch.from_numpy(deg.astype(np.float32))
        if self.deg_mode == "sq_max":
            deg_w = deg_t.pow(2)
        elif self.deg_mode == "log":
            deg_w = torch.log1p(deg_t)
        else:
            raise ValueError(f"unknown deg_mode: {self.deg_mode}")
        # Normalise into [0, 1] so the loss magnitude is dataset-
        # scale-independent at fixed lam.
        denom = deg_w.max().clamp_min(self.eps)
        self.deg_sq = deg_w / denom

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        if self.deg_sq.numel() == 0:
            raise RuntimeError("set_degrees() must be called once before forward()")
        # h_v: (n_nodes, d). Compute per-vertex squared norm.
        norms_sq = h_v.pow(2).sum(dim=-1)             # (n_nodes,)
        weighted = self.deg_sq.to(h_v.device) * norms_sq
        out = weighted.mean()
        self.last_value = float(out.detach().item())
        return self.lam * out


def triad_density(triads_pyobj, n_nodes: int) -> np.ndarray:
    """Per-triad density: number of OTHER triads sharing at least one
    vertex with this triad, normalised to $[0, 1]$.

    Computed via a vertex-to-triad inverted index, so it is
    $O(\\sum_v |T_v|^2)$ rather than $O(|T|^2)$ — tractable on
    Bitcoin and Slashdot.
    """
    triads_v = [tuple(t.v) for t in triads_pyobj]
    inv: list[list[int]] = [[] for _ in range(n_nodes)]
    for ti, vs in enumerate(triads_v):
        for v in vs:
            inv[int(v)].append(ti)
    n_triads = len(triads_v)
    deg = np.zeros(n_triads, dtype=np.float32)
    for ti, vs in enumerate(triads_v):
        neighbours = set()
        for v in vs:
            for tj in inv[int(v)]:
                if tj != ti:
                    neighbours.add(tj)
        deg[ti] = float(len(neighbours))
    if deg.max() > 0:
        deg = deg / deg.max()
    return deg


class HyperedgeDensityRegulariser(nn.Module):
    """L = lam * mean_t ( density(t) * ||h_t||^2 )."""

    def __init__(self, lam: float = 1e-3, eps: float = 1e-8):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.register_buffer("density", torch.zeros(0))
        self.last_value: float = float("nan")

    def set_density(self, density: np.ndarray) -> None:
        self.density = torch.from_numpy(density.astype(np.float32))

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        if self.density.numel() == 0:
            raise RuntimeError("set_density() must be called before forward()")
        # h_t: (n_triads, d) — per-triad embedding from SignedKANLayer.
        norms_sq = h_t.pow(2).sum(dim=-1)             # (n_triads,)
        weighted = self.density.to(h_t.device) * norms_sq
        out = weighted.mean()
        self.last_value = float(out.detach().item())
        return self.lam * out
