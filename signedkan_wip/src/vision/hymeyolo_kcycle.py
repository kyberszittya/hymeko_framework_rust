"""K-cycle micro-graph HyMeYOLO (2026-05-11 follow-up to RicciHyMeYOLOMulti).

Unifies the vision branch with the signed-graph branch by treating
each query's K corners as vertices of a signed micro-graph, with
edge signs derived from convex/concave curvature at each corner.
The MixedArityHSiKANAggregator's existing sub-cycle enumeration
(k=2 pairs, k=3 triangles, k=4 quads, generalised here to k≤K) is
applied with signs attached, producing the same kind of structural
readout that broke Slashdot SOTA --- at the per-detection micro
scale.

Concretely for each query:

    1. K corners c_0..c_{K-1} are predicted (same as RicciHyMeYOLO).
    2. Per-corner curvature sign:
           α_i = exterior angle at c_i
           s_i = +1 if α_i < π (convex) else -1 (concave).
    3. Per-edge sign on the polygon's cyclic edges:
           σ(c_{i}, c_{i+1 mod K}) = s_i · s_{i+1 mod K}
       (signed-graph balance product convention).
    4. For k ∈ {2, 3, ..., K}, every C(K, k) subset of corners
       induces a (signed) sub-cycle; per-subset Ricci-like score =
       product of signs along its boundary edges.
    5. MixedArity-style α-routed aggregation pools sub-cycle outputs
       into a single per-query micro-graph feature.
    6. Feature is concatenated to bilinear-sampled corner features
       and fed to the classification head.

This is the discrete-differential-geometry-via-signed-cycles head:
the model reads a query's shape through the same primitive
(signed-cycle Catmull-Rom KAN) that drives our Slashdot SOTA.
"""
from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hymeyolo_q_smoke import (
    HSiKANAggregator,
    TinyBackbone,
    bilinear_sample,
)
from .hymeyolo_circles_ricci import (
    geometric_ricci_signature,
    _circle_init,
)


# ─── Per-corner curvature sign ──────────────────────────────────────


@torch.no_grad()
def corner_signs_from_corners(corners: torch.Tensor) -> torch.Tensor:
    """Exterior-angle-based convex/concave sign at each corner.

    Args:
        corners: (B, N, K, 2)

    Returns:
        signs:   (B, N, K) in {-1, +1}; +1 = convex (α < π),
                 -1 = concave.

    Edge case: degenerate (collinear) corners return +1 by convention.
    """
    K = corners.shape[-2]
    # Cyclic prev / next.
    prev = corners.roll(shifts=+1, dims=-2)
    next_ = corners.roll(shifts=-1, dims=-2)
    # Edge vectors out of and into corner i.
    v_in = corners - prev           # (B, N, K, 2)
    v_out = next_ - corners         # (B, N, K, 2)
    # 2D "cross product" z-component.  Sign indicates turn direction.
    cross = v_in[..., 0] * v_out[..., 1] - v_in[..., 1] * v_out[..., 0]
    # Convention: counter-clockwise polygon → convex corners have
    # cross > 0.  Map to {-1, +1}, treating 0 as +1.
    return torch.where(cross >= 0, torch.ones_like(cross), -torch.ones_like(cross))


def edge_signs_from_corner_signs(corner_signs: torch.Tensor) -> torch.Tensor:
    """Cyclic-edge signs from corner signs.

    σ(edge_i) = s_i · s_{i+1 mod K}; (B, N, K) → (B, N, K).
    """
    next_ = corner_signs.roll(shifts=-1, dims=-1)
    return corner_signs * next_


# ─── Signed sub-cycle enumeration on K vertices ─────────────────────


def _enumerate_signed_subcycles(K: int) -> list[tuple[int, ...]]:
    """All C(K, k) subsets for k in {2..K}, as a flat list of tuples
    (sorted corner indices).  Caller is responsible for attaching
    signs to the subset's boundary edges."""
    out: list[tuple[int, ...]] = []
    for k in range(2, K + 1):
        out.extend(combinations(range(K), k))
    return out


# ─── KCycle aggregator (generalised MixedArity with edge signs) ─────


class KCycleSignedAggregator(nn.Module):
    """α-routed signed-cycle aggregator over a K-corner query.

    Per-arity HSiKAN aggregators run independently for k ∈ {2..K}.
    Each subset's contribution is *signed* by the product of its
    boundary-edge signs (the cycle's balance), so unbalanced cycles
    contribute with opposite sign in the per-arity mean.

    α_κ logits are learnable, init at 0 → uniform 1/(K-1).
    """

    def __init__(self, d_in: int, d_hidden: int = 32, K: int = 4):
        super().__init__()
        assert K >= 3, "K must be >= 3 (need at least one cycle)"
        self.K = K
        self.d_hidden = d_hidden
        # Per-arity aggregators.  k=2 is degenerate ("cycle" of length
        # 2 = a single edge) but we keep it for symmetry with the
        # signed-graph mixed-arity recipe.
        self.aggs = nn.ModuleList([
            HSiKANAggregator(d_in, d_hidden, k=k) for k in range(2, K + 1)
        ])
        # α over arities k=2..K.  Init = uniform after softmax.
        self.alpha_logits = nn.Parameter(torch.zeros(K - 1))
        # Pre-compute subset index tensors per arity, registered as
        # buffers so they move with .to(device) and don't get
        # rematerialised on every forward (the Python-list path was
        # the bottleneck — 247 small-tensor allocations per query
        # forward at K=8).  Buffer name: `subs_k_<ai>` where ai is
        # the arity index (0 ⇒ k=2, 1 ⇒ k=3, …).
        self._n_arities = K - 1
        for ai, k in enumerate(range(2, K + 1)):
            subs = list(combinations(range(K), k))
            subs_t = torch.tensor(
                [list(s) for s in subs], dtype=torch.long,
            )                                              # (n_subsets, k)
            self.register_buffer(f"subs_k_{ai}", subs_t, persistent=False)

    def alpha(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)

    def forward(
        self,
        h_corners: torch.Tensor,     # (B, K, d_in)
        edge_signs: torch.Tensor,    # (B, K) — cyclic-edge signs
    ) -> torch.Tensor:
        """Returns (B, d_hidden) per-query micro-graph descriptor.

        Vectorised — gathers all C(K, k) subsets at once via fancy
        indexing and calls the per-arity aggregator a SINGLE time
        per arity instead of once per subset.  At K=8 this cuts
        247 Python iterations → 7 (one per arity), removing the
        Python-dispatch + small-tensor-alloc bottleneck that was
        causing the +kcycle config to time out at the 1h queue
        wall-clock budget.
        """
        B, K_dim, d_in = h_corners.shape
        per_arity_outs: list[torch.Tensor] = []
        for ai in range(self._n_arities):
            agg = self.aggs[ai]
            subs_t = getattr(self, f"subs_k_{ai}")          # (n_subsets, k_ai)
            n_subsets, k_ai = subs_t.shape

            # (B, n_subsets, k_ai) — sign at each subset's positions.
            sign_per_pos = edge_signs[:, subs_t]            # (B, n_subsets, k_ai)
            sign_product = sign_per_pos.prod(dim=-1)        # (B, n_subsets)

            # (B, n_subsets, k_ai, d_in) — gathered corner features.
            h_per_sub = h_corners[:, subs_t]                # (B, n_subsets, k_ai, d_in)

            # Batched aggregator call: flatten (B, n_subsets) → one
            # leading batch dim, call agg once, reshape back.
            h_flat = h_per_sub.reshape(B * n_subsets, k_ai, d_in)
            out_flat = agg(h_flat)                          # (B*n_subsets, d_hidden)
            d_hidden = out_flat.shape[-1]
            out_per_sub = out_flat.view(B, n_subsets, d_hidden)

            # Weight each subset by its boundary sign-product and
            # mean across subsets of this arity.
            weighted = out_per_sub * sign_product.unsqueeze(-1)
            per_arity_outs.append(weighted.mean(dim=1))     # (B, d_hidden)

        # α-route across arities.
        stacked = torch.stack(per_arity_outs, dim=0)        # (K-1, B, d_hidden)
        a = self.alpha().view(-1, 1, 1)
        return (a * stacked).sum(dim=0)                     # (B, d_hidden)


# ─── KCycle HyMeYOLO model ──────────────────────────────────────────


class KCycleHyMeYOLOMulti(nn.Module):
    """HyMeYOLO variant whose classification head reads each query's
    **signed K-cycle micro-graph descriptor**, fed alongside the
    bilinear-sampled corner features.

    Compatible with the existing combined_set_loss / Hungarian
    matcher in `train_circles_ricci.py` — returns the same dict
    shape (box_corners, box_cls, circle_corners, circle_cls).

    Differences from RicciHyMeYOLOMulti:
        * uses KCycleSignedAggregator instead of MixedArity for both
          box and circle queries
        * concatenates the signed-cycle descriptor + bilinear corner
          features into the classification head (no separate Ricci
          scalar — the cycle product is the curvature readout)
    """

    def __init__(
        self,
        n_box_queries: int = 4,
        n_circle_queries: int = 2,
        box_k: int = 4,
        circle_k: int = 8,
        n_classes: int = 10,
        d_hidden: int = 32,
    ):
        super().__init__()
        self.n_box_queries = n_box_queries
        self.n_circle_queries = n_circle_queries
        self.box_k = box_k
        self.circle_k = circle_k
        self.n_classes = n_classes
        self.d_hidden = d_hidden

        self.backbone = TinyBackbone(c_in=3, c_out=d_hidden)

        # Learnable corner queries.
        if n_box_queries > 0:
            self.box_corners = nn.Parameter(
                torch.stack([
                    self._regular_kgon(box_k, cx=0.5, cy=0.5, r=0.3)
                    for _ in range(n_box_queries)
                ], dim=0)
            )
            self.box_aggregator = KCycleSignedAggregator(
                d_in=d_hidden, d_hidden=d_hidden, K=box_k,
            )

        if n_circle_queries > 0:
            self.circle_corners = nn.Parameter(
                torch.stack([
                    self._regular_kgon(circle_k, cx=0.5, cy=0.5, r=0.25)
                    for _ in range(n_circle_queries)
                ], dim=0)
            )
            self.circle_aggregator = KCycleSignedAggregator(
                d_in=d_hidden, d_hidden=d_hidden, K=circle_k,
            )

        self.head_box_offset = nn.Linear(d_hidden, box_k * 2)
        self.head_circle_offset = nn.Linear(d_hidden, circle_k * 2)
        # Classification head: takes [bilinear-pooled + signed-cycle].
        cls_in = 2 * d_hidden
        self.head_cls = nn.Linear(cls_in, n_classes + 1)

    @staticmethod
    def _regular_kgon(k: int, cx: float, cy: float, r: float) -> torch.Tensor:
        """k points evenly spaced on a circle, centred at (cx, cy)
        with radius r.  For k=4 this gives a square rotated 45°; pass
        through a learnable offset head to specialise per-image."""
        thetas = torch.tensor(
            [2 * math.pi * i / k for i in range(k)], dtype=torch.float32,
        )
        return torch.stack([
            cx + r * torch.cos(thetas),
            cy + r * torch.sin(thetas),
        ], dim=-1)

    def _refine_corners(
        self, base_corners: torch.Tensor, F_map: torch.Tensor,
        aggregator: KCycleSignedAggregator, head_offset: nn.Linear,
    ) -> torch.Tensor:
        """Sample features at base_corners, run aggregator, predict
        per-corner offsets, return refined corners."""
        B = F_map.shape[0]
        N, K, _ = base_corners.shape
        flat = base_corners.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 2)
        h_flat = bilinear_sample(F_map, flat)                  # (B, N*K, d)
        h_corners = h_flat.view(B, N, K, -1)                   # (B, N, K, d)
        # Compute signed-cycle descriptor for offset prediction.
        h_query = h_corners.mean(dim=2)                        # (B, N, d) — coarse
        offsets = head_offset(h_query).view(B, N, K, 2)
        return base_corners.unsqueeze(0).expand(B, -1, -1, -1) + offsets

    def forward(self, x: torch.Tensor) -> dict:
        F_map = self.backbone(x)
        B = F_map.shape[0]
        out: dict[str, torch.Tensor] = {}

        if self.n_box_queries > 0:
            box_refined = self._refine_corners(
                self.box_corners, F_map, self.box_aggregator, self.head_box_offset,
            )
            box_corner_feats = bilinear_sample(
                F_map,
                box_refined.reshape(B, -1, 2),
            ).view(B, self.n_box_queries, self.box_k, -1)
            # Signed-cycle micro-graph descriptor per box query.
            corner_signs_b = corner_signs_from_corners(box_refined)
            edge_signs_b = edge_signs_from_corner_signs(corner_signs_b)
            box_cycle_feats: list[torch.Tensor] = []
            for qi in range(self.n_box_queries):
                box_cycle_feats.append(self.box_aggregator(
                    box_corner_feats[:, qi],     # (B, K, d)
                    edge_signs_b[:, qi],         # (B, K)
                ))
            box_cycle = torch.stack(box_cycle_feats, dim=1)  # (B, n_box, d)
            # Pooled corner features (mean) for the classification head.
            box_pooled = box_corner_feats.mean(dim=2)         # (B, n_box, d)
            box_cls_in = torch.cat([box_pooled, box_cycle], dim=-1)
            out["box_corners"] = box_refined
            out["box_cls"] = self.head_cls(box_cls_in)
        else:
            out["box_corners"] = x.new_zeros(B, 0, self.box_k, 2)
            out["box_cls"] = x.new_zeros(B, 0, self.n_classes + 1)

        if self.n_circle_queries > 0:
            circle_refined = self._refine_corners(
                self.circle_corners, F_map, self.circle_aggregator,
                self.head_circle_offset,
            )
            circle_corner_feats = bilinear_sample(
                F_map,
                circle_refined.reshape(B, -1, 2),
            ).view(B, self.n_circle_queries, self.circle_k, -1)
            corner_signs_c = corner_signs_from_corners(circle_refined)
            edge_signs_c = edge_signs_from_corner_signs(corner_signs_c)
            circle_cycle_feats: list[torch.Tensor] = []
            for qi in range(self.n_circle_queries):
                circle_cycle_feats.append(self.circle_aggregator(
                    circle_corner_feats[:, qi],
                    edge_signs_c[:, qi],
                ))
            circle_cycle = torch.stack(circle_cycle_feats, dim=1)
            circle_pooled = circle_corner_feats.mean(dim=2)
            circle_cls_in = torch.cat([circle_pooled, circle_cycle], dim=-1)
            out["circle_corners"] = circle_refined
            out["circle_cls"] = self.head_cls(circle_cls_in)
        else:
            out["circle_corners"] = x.new_zeros(B, 0, self.circle_k, 2)
            out["circle_cls"] = x.new_zeros(B, 0, self.n_classes + 1)

        return out

    def alpha_box(self) -> torch.Tensor:
        """Inspect learned α_κ over box-query arities (k=2..K)."""
        return self.box_aggregator.alpha()

    def alpha_circle(self) -> torch.Tensor:
        return self.circle_aggregator.alpha()


def _smoke() -> None:
    """Forward + backward smoke at toy scale."""
    B, K = 2, 4
    model = KCycleHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2,
        box_k=4, circle_k=8, n_classes=10, d_hidden=16,
    )
    x = torch.randn(B, 3, 64, 64, requires_grad=False)
    out = model(x)
    print("box_corners:", tuple(out["box_corners"].shape))
    print("box_cls:    ", tuple(out["box_cls"].shape))
    print("circle_corners:", tuple(out["circle_corners"].shape))
    print("circle_cls:    ", tuple(out["circle_cls"].shape))
    print("α_box:    ", model.alpha_box().detach().tolist())
    print("α_circle: ", model.alpha_circle().detach().tolist())
    # Backward
    loss = out["box_cls"].mean() + out["circle_cls"].mean()
    loss.backward()
    print(f"loss = {loss.item():.4f}  (backward OK)")


if __name__ == "__main__":
    _smoke()
