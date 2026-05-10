"""HyMeYOLO extension: circle queries + geometric Ricci curvature.

Extends ``HyMeYOLOMulti`` (DETR-style 4-corner box queries) with two
additions per the 2026-05-12 plan:

  1. **Circle queries** (k > 4 corners arranged on a ring) for round /
     elliptical objects.  Different cardinality, same HSiKAN aggregator
     framework.
  2. **Geometric Forman-Ricci** curvature signature per query: a
     scalar derived from the cycle's corner geometry that modulates
     the class head, giving the model an interpretable shape-prior.

Architectural distinction from chordful k-cycles:
  - **k-cycle query**: corners are 4 points, freely placed → represents
    a polygon (potentially with internal chord-like correlations in
    feature space).  Default in `HyMeYOLOMulti`.
  - **circle query**: corners are k > 4 points constrained to lie on a
    common ring → represents a closed curve / silhouette boundary.
    Initialised as a regular k-gon inscribed in a circle.

The Ricci signature is geometric (corner-position-derived), not graph-
topological (which on a chordless cycle is constant and uninformative).

See:
  - `docs/plans_kcycle_vision_2026_05_07.md` (kCVD original)
  - `docs/plans_kcvd_vs_yolo_2026_05_09.md` (convergence with HyMeYOLO)
  - `docs/plans/2026-05-11-hymeko-yolo/SUPERSEDED.md` (auxiliary-feature
    interpretation, rejected)
"""
from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hymeyolo_hungarian import HyMeYOLOMulti
from .hymeyolo_q_smoke import (
    HSiKANAggregator, HighwayHSiKANAggregator,
    MixedArityHSiKANAggregator, TinyBackbone, bilinear_sample,
)


# ─── Geometric Forman-Ricci-style curvature ─────────────────────────


def geometric_ricci_signature(corners: torch.Tensor) -> torch.Tensor:
    """Compute a geometric Ricci-curvature-like signature for each
    k-cycle of corners.

    For each corner i of the k-cycle, exterior angle θ_i = π - interior
    angle.  The curvature signature is the vector of (cos θ_i, sin θ_i)
    plus the scalar mean θ, packed as (B, ..., k * 2 + 1).

    For a regular k-gon: all θ_i = 2π / k → constant; mean = 2π / k.
    For a square: θ_i = π / 2 (90°) for all i → mean = π / 2.
    For a thin rectangle: alternating π/2 and π/2 → still constant
      (because rectangle has all right angles); discriminated by EDGE
      LENGTH ratios instead — see ``edge_length_signature`` below.
    For a degenerate (collinear) polygon: θ_i → 0 → mean → 0.

    Parameters
    ----------
    corners : (..., k, 2)  — corner positions in [0, 1]² (any device).

    Returns
    -------
    (..., 2k + 1)  — concatenation of cos/sin per-corner exterior
      angles + scalar mean.
    """
    k = corners.shape[-2]
    assert k >= 3, f"k must be >= 3 for a cycle, got {k}"
    # Edge vectors v_i = corner_{i+1} - corner_i.
    next_idx = torch.arange(k, device=corners.device)
    next_idx = (next_idx + 1) % k
    v = corners[..., next_idx, :] - corners                # (..., k, 2)
    # Normalise.
    v = F.normalize(v, dim=-1, eps=1e-8)
    # Previous edge vector u_i = v_{i-1}.
    prev_idx = (torch.arange(k, device=corners.device) - 1) % k
    u = v[..., prev_idx, :]                                # (..., k, 2)
    # Interior angle's cosine and sine.  We want the exterior turn:
    #   cos θ_i = -u · v  (sign of dot product gives convex/reflex)
    #   sin θ_i = u × v  (cross product, signed in 2D)
    cos_theta = -(u * v).sum(dim=-1)                       # (..., k)
    sin_theta = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]   # (..., k)
    # Pack: per-corner (cos, sin) + scalar mean angle (atan2 of mean).
    theta = torch.atan2(sin_theta, cos_theta)               # (..., k)
    mean_theta = theta.mean(dim=-1, keepdim=True)           # (..., 1)
    return torch.cat([cos_theta, sin_theta, mean_theta], dim=-1)


def edge_length_signature(corners: torch.Tensor) -> torch.Tensor:
    """Per-edge length signature, normalised to mean=1 per cycle.

    For a regular k-gon: all-ones → variance 0.
    For a thin rectangle: alternating small/large → high variance.

    Returns
    -------
    (..., k)  — normalised edge lengths.
    """
    k = corners.shape[-2]
    next_idx = (torch.arange(k, device=corners.device) + 1) % k
    v = corners[..., next_idx, :] - corners
    lengths = v.norm(dim=-1)                                # (..., k)
    return lengths / (lengths.mean(dim=-1, keepdim=True) + 1e-8)


def ricci_scalar(corners: torch.Tensor) -> torch.Tensor:
    """Scalar curvature: a single number per cycle summarising 'how
    round vs how polygonal'.

    Defined as:
      κ = -log(angle_variance + ε)  if all angles are equal,  κ → +∞
                                    otherwise smaller.

    Output shape: (...) — one scalar per cycle.
    """
    sig = geometric_ricci_signature(corners)
    k = corners.shape[-2]
    cos_part = sig[..., :k]
    sin_part = sig[..., k:2 * k]
    theta = torch.atan2(sin_part, cos_part)
    # Variance of angles.
    var = theta.var(dim=-1, unbiased=False)
    return -torch.log(var + 1e-6)


# ─── Circle query (k corners on a ring) ─────────────────────────────


def _circle_init(k: int, cx: float = 0.5, cy: float = 0.5,
                  r: float = 0.30) -> torch.Tensor:
    """Initialise k corners on a circle of centre (cx, cy) radius r."""
    angles = torch.linspace(0.0, 2 * math.pi, k + 1)[:-1]
    xs = cx + r * torch.cos(angles)
    ys = cy + r * torch.sin(angles)
    return torch.stack([xs, ys], dim=-1)


# ─── Multi-query model with circles + Ricci modulation ───────────────


class RicciHyMeYOLOMulti(nn.Module):
    """HyMeYOLOMulti extended with circle queries + Ricci-modulated
    class head.

    Parameters
    ----------
    n_box_queries : int
        Number of cardinality-4 'box' queries (same as HyMeYOLOMulti's
        `n_queries`).
    n_circle_queries : int
        Number of cardinality-`circle_k` 'circle' queries initialised
        on a ring.
    circle_k : int
        Corner count for circle queries.  Default 8.
    n_classes : int
    d_hidden : int
    ricci_modulation : bool
        If True, the per-query class head gets the ricci signature as
        an extra input.  If False, class prediction is corner-feature
        only (parity with HyMeYOLOMulti).
    """

    def __init__(
        self,
        n_box_queries: int = 4,
        n_circle_queries: int = 2,
        circle_k: int = 8,
        n_classes: int = 10,
        d_hidden: int = 32,
        ricci_modulation: bool = True,
    ):
        super().__init__()
        assert circle_k >= 5, "circle_k should be >= 5 for ring queries"
        self.n_box_queries = n_box_queries
        self.n_circle_queries = n_circle_queries
        self.circle_k = circle_k
        self.n_classes = n_classes
        self.ricci_modulation = ricci_modulation
        self.backbone = TinyBackbone(c_in=3, c_out=d_hidden)

        # Box queries (k=4).  Init as in HyMeYOLOMulti.
        base_box = torch.tensor(
            [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
            dtype=torch.float32,
        )
        gen = torch.Generator().manual_seed(0)
        init_box = base_box.unsqueeze(0).expand(n_box_queries, -1, -1).clone()
        init_box = init_box + torch.randn(
            init_box.shape, generator=gen,
        ) * 0.08
        init_box = init_box.clamp(0.05, 0.95)
        self.box_corners = nn.Parameter(init_box.clone())

        # Circle queries (k=circle_k).
        init_circ = _circle_init(circle_k).unsqueeze(0)            # (1, k, 2)
        init_circ = init_circ.expand(n_circle_queries, -1, -1).clone()
        gen2 = torch.Generator().manual_seed(1)
        init_circ = init_circ + torch.randn(
            init_circ.shape, generator=gen2,
        ) * 0.04
        init_circ = init_circ.clamp(0.05, 0.95)
        self.circle_corners = nn.Parameter(init_circ.clone())

        # Aggregators: one per cardinality (box=4, circle=circle_k).
        self.box_aggregator = HSiKANAggregator(
            d_in=d_hidden, d_hidden=d_hidden, k=4,
        )
        self.circle_aggregator = HSiKANAggregator(
            d_in=d_hidden, d_hidden=d_hidden, k=circle_k,
        )

        # Heads.
        self.head_box_offset = nn.Linear(d_hidden, 4 * 2)
        self.head_circle_offset = nn.Linear(d_hidden, circle_k * 2)
        cls_in = d_hidden + (3 if ricci_modulation else 0)
        self.head_cls = nn.Linear(cls_in, n_classes + 1)
        # 3 extra inputs when ricci_modulation: (κ_scalar, mean_cos_θ, edge_var)

    def _query_features(
        self, corners: torch.Tensor, F_map: torch.Tensor,
        aggregator: nn.Module,
    ) -> torch.Tensor:
        """Bilinear-sample corner features, aggregate → (B, N, d)."""
        B = F_map.shape[0]
        N, k, _ = corners.shape
        # Expand to batch.
        c = corners.unsqueeze(0).expand(B, N, k, 2)
        flat = c.reshape(B, N * k, 2)
        h_flat = bilinear_sample(F_map, flat)                  # (B, N*k, d)
        h_per_q = h_flat.view(B, N, k, -1)                     # (B, N, k, d)
        h_BN = h_per_q.reshape(B * N, k, -1)
        h_aux = aggregator(h_BN)                                # (B*N, d)
        return h_aux.view(B, N, -1)                            # (B, N, d)

    def _ricci_features(self, corners: torch.Tensor) -> torch.Tensor:
        """Compute Ricci shape signature per query: 3 scalars.

        Returns (N, 3): (κ_scalar, mean_cos_θ, edge_length_var).
        """
        k = corners.shape[-2]
        sig = geometric_ricci_signature(corners)               # (N, 2k+1)
        cos_part = sig[..., :k]
        scalar_kappa = ricci_scalar(corners)                   # (N,)
        mean_cos = cos_part.mean(dim=-1)                       # (N,)
        edge_var = edge_length_signature(corners).var(dim=-1)  # (N,)
        return torch.stack([scalar_kappa, mean_cos, edge_var], dim=-1)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) → returns dict with:
            box_corners    : (B, n_box, 4, 2)
            box_cls        : (B, n_box, n_classes+1)
            circle_corners : (B, n_circle, circle_k, 2)
            circle_cls     : (B, n_circle, n_classes+1)
            ricci_box      : (n_box, 3)   shape signatures
            ricci_circle   : (n_circle, 3)
        """
        B = x.shape[0]
        F_map = self.backbone(x)
        d_hidden = F_map.shape[1]
        # Class-head input width depends on ricci_modulation.
        cls_in_extra = 3 if self.ricci_modulation else 0

        # Box branch (skipped if no box queries).
        if self.n_box_queries > 0:
            h_box = self._query_features(
                self.box_corners, F_map, self.box_aggregator,
            )                                                          # (B, Nb, d)
            box_off = self.head_box_offset(h_box).view(B, -1, 4, 2)
            box_off = 0.3 * torch.tanh(box_off)
            box_refined = (self.box_corners.unsqueeze(0).expand(B, -1, -1, -1)
                           + box_off)
            ricci_box = self._ricci_features(
                box_refined.reshape(-1, 4, 2),
            ).view(B, -1, 3)
            cls_in_box = (torch.cat([h_box, ricci_box], dim=-1)
                          if self.ricci_modulation else h_box)
            box_cls = self.head_cls(cls_in_box)
        else:
            box_refined = x.new_zeros(B, 0, 4, 2)
            box_cls = x.new_zeros(B, 0, self.n_classes + 1)
            ricci_box = x.new_zeros(B, 0, 3)

        # Circle branch (skipped if no circle queries).
        if self.n_circle_queries > 0:
            h_circ = self._query_features(
                self.circle_corners, F_map, self.circle_aggregator,
            )
            circ_off = self.head_circle_offset(h_circ).view(
                B, -1, self.circle_k, 2,
            )
            circ_off = 0.20 * torch.tanh(circ_off)
            circ_refined = (self.circle_corners.unsqueeze(0).expand(B, -1, -1, -1)
                            + circ_off)
            ricci_circle = self._ricci_features(
                circ_refined.reshape(-1, self.circle_k, 2),
            ).view(B, -1, 3)
            cls_in_circ = (torch.cat([h_circ, ricci_circle], dim=-1)
                           if self.ricci_modulation else h_circ)
            circle_cls = self.head_cls(cls_in_circ)
        else:
            circ_refined = x.new_zeros(B, 0, self.circle_k, 2)
            circle_cls = x.new_zeros(B, 0, self.n_classes + 1)
            ricci_circle = x.new_zeros(B, 0, 3)

        return {
            "box_corners":    box_refined,
            "box_cls":        box_cls,
            "circle_corners": circ_refined,
            "circle_cls":     circle_cls,
            "ricci_box":      ricci_box,
            "ricci_circle":   ricci_circle,
        }
