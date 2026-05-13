"""HyMeYOLO Ricci × k-cycles curvature mixed variant.

Unifies two HyMeYOLO threads into a single model:

  - **Structural curvature** (signed k-cycle micro-graph σ-product) from
    `hymeyolo_kcycle.KCycleSignedAggregator`. Captures sign-balance over
    the K corners of each query.
  - **Geometric curvature** (Forman-Ricci-style scalars from corner
    positions) from `hymeyolo_circles_ricci`. Captures angle / edge
    geometry — how round vs how polygonal the query is.

The two are complementary: σ-products are sign-blind to geometry,
Ricci scalars are sign-blind to topology. Mixing them gives the
classification head both readouts; mixing them in the **offset head**
fixes the +kcycle localization bug documented in
``reports/2026-05-13-hymeyolo-kcycle-localization-bug.md`` where the
signed-cycle aggregator was wired into cls but not into corner
refinement.

Architectural choices:

  - `KCycleSignedAggregator` runs on BASE corners → produces a
    `(B, N, d)` cycle descriptor. This drives offset prediction
    (alongside the Ricci scalars from BASE corners). Closes the
    +kcycle bug: localization now uses the structural signal.
  - After offset, corners are refined. `KCycleSignedAggregator` runs
    AGAIN on REFINED corners → cls-stage descriptor. Ricci recomputed
    at refined corners too. Classification head reads
    `[mean_pool, cycle_desc, ricci]`.
  - All three signals (mean-pool corner features, signed-cycle
    descriptor, geometric Ricci scalars) feed cls; the latter two feed
    offset. No σ-leakage concerns: cycle σ here is over **feature
    signs of corners**, not over edge signs of a held-out edge —
    different domain than signed graph link prediction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .hymeyolo_circles_ricci import (
    geometric_ricci_signature,
    ricci_scalar,
    edge_length_signature,
)
from .hymeyolo_kcycle import (
    KCycleSignedAggregator,
    corner_signs_from_corners,
    edge_signs_from_corner_signs,
)
from .hymeyolo_q_smoke import TinyBackbone, bilinear_sample


def _regular_kgon(k: int, cx: float, cy: float, r: float) -> torch.Tensor:
    thetas = torch.tensor(
        [2 * math.pi * i / k for i in range(k)], dtype=torch.float32,
    )
    return torch.stack(
        [cx + r * torch.cos(thetas), cy + r * torch.sin(thetas)],
        dim=-1,
    )


def _ricci_scalars(corners: torch.Tensor) -> torch.Tensor:
    """Three geometric scalars per cycle: (κ_scalar, mean_cos_θ, edge_length_var).

    Parameters
    ----------
    corners : tensor of shape ``(..., k, 2)``.

    Returns
    -------
    tensor of shape ``(..., 3)``.
    """
    k = corners.shape[-2]
    sig = geometric_ricci_signature(corners)      # (..., 2k+1)
    cos_part = sig[..., :k]                       # (..., k)
    scalar_kappa = ricci_scalar(corners)          # (...)
    mean_cos = cos_part.mean(dim=-1)              # (...)
    edge_var = edge_length_signature(corners).var(dim=-1)  # (...)
    return torch.stack([scalar_kappa, mean_cos, edge_var], dim=-1)


class RicciKCycleHyMeYOLOMulti(nn.Module):
    """HyMeYOLO with signed-cycle structural curvature AND geometric Ricci.

    Both signals route into BOTH offset prediction and classification.
    Fixes the +kcycle localization bug (aggregator was cls-only) and
    adds the geometric Ricci shape prior on the same forward pass.

    Returns a dict with the same shape contract as
    `KCycleHyMeYOLOMulti` / `RicciHyMeYOLOMulti` for drop-in use in
    `train_circles_ricci.py`'s Hungarian matcher.
    """

    def __init__(
        self,
        n_box_queries: int = 4,
        n_circle_queries: int = 2,
        box_k: int = 4,
        circle_k: int = 8,
        n_classes: int = 10,
        d_hidden: int = 32,
        ricci_modulation: bool = True,
    ):
        super().__init__()
        assert box_k >= 3, "box_k must be >= 3 (signed-cycle aggregator needs k>=3)"
        assert circle_k >= 5, "circle_k must be >= 5 for ring queries"
        self.n_box_queries = n_box_queries
        self.n_circle_queries = n_circle_queries
        self.box_k = box_k
        self.circle_k = circle_k
        self.n_classes = n_classes
        self.d_hidden = d_hidden
        self.ricci_modulation = ricci_modulation
        ricci_dim = 3 if ricci_modulation else 0

        self.backbone = TinyBackbone(c_in=3, c_out=d_hidden)

        if n_box_queries > 0:
            self.box_corners = nn.Parameter(torch.stack([
                _regular_kgon(box_k, cx=0.5, cy=0.5, r=0.30)
                for _ in range(n_box_queries)
            ], dim=0))
            self.box_aggregator = KCycleSignedAggregator(
                d_in=d_hidden, d_hidden=d_hidden, K=box_k,
            )

        if n_circle_queries > 0:
            self.circle_corners = nn.Parameter(torch.stack([
                _regular_kgon(circle_k, cx=0.5, cy=0.5, r=0.25)
                for _ in range(n_circle_queries)
            ], dim=0))
            self.circle_aggregator = KCycleSignedAggregator(
                d_in=d_hidden, d_hidden=d_hidden, K=circle_k,
            )

        # Offset heads: take [cycle_desc, ricci] → k*2 corner offsets.
        self.head_box_offset = nn.Linear(d_hidden + ricci_dim, box_k * 2)
        self.head_circle_offset = nn.Linear(d_hidden + ricci_dim, circle_k * 2)

        # Cls head: takes [mean_pool, cycle_desc, ricci] → n_classes+1.
        self.head_cls = nn.Linear(2 * d_hidden + ricci_dim, n_classes + 1)

    def _query_pipeline(
        self,
        base_corners: torch.Tensor,        # (N, K, 2)
        F_map: torch.Tensor,                # (B, d, H, W)
        aggregator: KCycleSignedAggregator,
        head_offset: nn.Linear,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the offset + cls feature build for one query type.

        Returns
        -------
        refined : (B, N, K, 2)         refined corner positions
        cls_in  : (B, N, cls_in_dim)   classification head input
        """
        B = F_map.shape[0]
        N = base_corners.shape[0]

        # Stage 1 — base corners: drive offset prediction.
        c_base = base_corners.unsqueeze(0).expand(B, N, K, 2)
        h_base = bilinear_sample(F_map, c_base.reshape(B, N * K, 2))
        h_base = h_base.view(B, N, K, -1)                              # (B, N, K, d)

        corner_signs_base = corner_signs_from_corners(c_base)          # (B, N, K)
        edge_signs_base = edge_signs_from_corner_signs(corner_signs_base)  # (B, N, K)

        # Aggregator expects (B, K, d) + (B, K) — flatten over (B, N).
        h_b_flat = h_base.reshape(B * N, K, -1)
        sgn_b_flat = edge_signs_base.reshape(B * N, K)
        h_cycle_base = aggregator(h_b_flat, sgn_b_flat).view(B, N, -1)  # (B, N, d)

        if self.ricci_modulation:
            ricci_base = _ricci_scalars(
                base_corners.reshape(-1, K, 2),
            ).view(N, 3).unsqueeze(0).expand(B, N, 3)                  # (B, N, 3)
            offset_in = torch.cat([h_cycle_base, ricci_base], dim=-1)
        else:
            offset_in = h_cycle_base

        offsets = head_offset(offset_in).view(B, N, K, 2)
        refined = c_base + offsets                                     # (B, N, K, 2)

        # Stage 2 — refined corners: drive classification.
        h_ref_flat = bilinear_sample(F_map, refined.reshape(B, N * K, 2))
        h_ref = h_ref_flat.view(B, N, K, -1)                           # (B, N, K, d)
        h_mean = h_ref.mean(dim=2)                                     # (B, N, d)

        corner_signs_ref = corner_signs_from_corners(refined)
        edge_signs_ref = edge_signs_from_corner_signs(corner_signs_ref)
        h_r_flat = h_ref.reshape(B * N, K, -1)
        sgn_r_flat = edge_signs_ref.reshape(B * N, K)
        h_cycle_ref = aggregator(h_r_flat, sgn_r_flat).view(B, N, -1)

        if self.ricci_modulation:
            ricci_ref = _ricci_scalars(
                refined.reshape(-1, K, 2),
            ).view(B, N, 3)
            cls_in = torch.cat([h_mean, h_cycle_ref, ricci_ref], dim=-1)
        else:
            cls_in = torch.cat([h_mean, h_cycle_ref], dim=-1)

        return refined, cls_in

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        F_map = self.backbone(x)
        out: dict[str, torch.Tensor] = {}

        if self.n_box_queries > 0:
            box_refined, box_cls_in = self._query_pipeline(
                self.box_corners, F_map, self.box_aggregator,
                self.head_box_offset, self.box_k,
            )
            out["box_corners"] = box_refined
            out["box_cls"] = self.head_cls(box_cls_in)
        else:
            out["box_corners"] = x.new_zeros(B, 0, self.box_k, 2)
            out["box_cls"] = x.new_zeros(B, 0, self.n_classes + 1)

        if self.n_circle_queries > 0:
            circ_refined, circ_cls_in = self._query_pipeline(
                self.circle_corners, F_map, self.circle_aggregator,
                self.head_circle_offset, self.circle_k,
            )
            out["circle_corners"] = circ_refined
            out["circle_cls"] = self.head_cls(circ_cls_in)
        else:
            out["circle_corners"] = x.new_zeros(B, 0, self.circle_k, 2)
            out["circle_cls"] = x.new_zeros(B, 0, self.n_classes + 1)

        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def alpha_box(self) -> torch.Tensor:
        return self.box_aggregator.alpha()

    def alpha_circle(self) -> torch.Tensor:
        return self.circle_aggregator.alpha()


__all__ = ["RicciKCycleHyMeYOLOMulti"]
