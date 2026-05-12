"""HyMeYOLO-Q smoke with HSiKAN-style σ-masked aggregation.

Re-implementation of the HyMeYOLO-Q toy (`examples/hymeyolo_toy.py` in
the sibling HyMeKoConv repo) using HSiKAN's signed-cycle aggregation
in place of HyMeKoConv's per-Θ_τ head.  Demonstrates that the two
operators are interchangeable in the object-as-hyperedge framing,
which is the architectural convergence claim of the kCVD-vs-YOLO plan
(`docs/plans_kcvd_vs_yolo_2026_05_09.md`).

Architecture
------------
- Backbone:   3 conv layers (3 → 16 → 32 → 64), output 8×8×64 feature map
- Query:      1 learnable cardinality-4 query hyperedge.  Each corner
              has a learnable (x, y) location in [0, 1]² (refined by
              the head).
- Sampling:   Bilinear sample backbone features at each predicted
              corner location → 4 × 64 corner features.
- Aggregation (the HSiKAN bit):
                σ-masked Catmull-Rom-style branch aggregation over the
                4-corner cycle.  Sign per corner edge from the
                gradient-direction of the local feature window;
                σ_v ∈ {±1} from cycle-edge parity.  Aggregator returns
                a per-query descriptor h_aux ∈ R^d.
- Heads:      corner offset MLP (h_aux → 4×2 offset),
              class MLP (h_aux → n_classes), no objectness in this
              smoke (single object per image).

The point of this smoke is *not* to beat YOLO --- it is to show the
HSiKAN aggregator drives end-to-end detection training and the loss
descends.  Full PASCAL VOC training is Phase 4 of
`IMPLEMENTATION_PLAN.md`; this is the gate for that phase.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..splines import _catmull_rom_eval


# ─── Synthetic dataset: one coloured rectangle per image ────────────


def make_synthetic_rectangles(n: int, H: int = 32, W: int = 32,
                                n_classes: int = 3,
                                seed: int = 0):
    """Generate `n` 32×32 RGB images with one solid rectangle each.

    Returns
    -------
    X     : (n, 3, H, W) float32 in [0, 1]
    boxes : (n, 4) — (x0, y0, x1, y1) in [0, 1]² normalised coords
    cls   : (n,)   — int class label in {0, 1, 2}
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 3, H, W), dtype=np.float32)
    # Slight noise on the background.
    X += rng.normal(0, 0.02, size=X.shape).astype(np.float32)
    boxes = np.zeros((n, 4), dtype=np.float32)
    cls = np.zeros(n, dtype=np.int64)
    palette = np.eye(3, dtype=np.float32)  # red, green, blue
    for i in range(n):
        c = int(rng.integers(0, n_classes))
        cls[i] = c
        x0 = rng.uniform(0.05, 0.45)
        y0 = rng.uniform(0.05, 0.45)
        w = rng.uniform(0.20, 0.45)
        h = rng.uniform(0.20, 0.45)
        x1 = min(0.95, x0 + w)
        y1 = min(0.95, y0 + h)
        boxes[i] = [x0, y0, x1, y1]
        # Paint.
        ix0, iy0 = int(x0 * W), int(y0 * H)
        ix1, iy1 = int(x1 * W), int(y1 * H)
        X[i, c, iy0:iy1, ix0:ix1] = 1.0
    return X, boxes, cls


# ─── Backbone ───────────────────────────────────────────────────────


class TinyBackbone(nn.Module):
    def __init__(self, c_in: int = 3, c_out: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32→16
        self.conv3 = nn.Conv2d(32, c_out, 3, stride=2, padding=1)  # 16→8

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x  # (B, c_out, 8, 8)


# ─── Bilinear sampling at predicted corner locations ────────────────


def bilinear_sample(F_map: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Sample a (B, C, H, W) feature map at (B, N, 2) normalised pts ∈
    [0, 1]² → (B, N, C).

    Wrapper around F.grid_sample with align_corners=True.
    """
    # grid_sample expects pts in [-1, 1]; convert.
    grid = pts * 2 - 1                          # (B, N, 2)
    grid = grid.unsqueeze(1)                    # (B, 1, N, 2)
    sampled = F.grid_sample(F_map, grid,
                              mode="bilinear",
                              padding_mode="zeros",
                              align_corners=True)
    # sampled: (B, C, 1, N) → (B, N, C)
    return sampled.squeeze(2).transpose(1, 2)


# ─── HSiKAN-style σ-masked aggregation over a query's 4 corners ─────


class HSiKANAggregator(nn.Module):
    """Aggregates `(B, k, d_in)` corner features into `(B, d_out)` aux
    descriptors via the SignedKAN factorisation:

        u_{b, i, s} = φ_in^s(h_corner_{b, i})              (per-sign)
        a_{b, s}    = mean_{i : σ_{b, i} = s} u_{b, i, s}
        h_aux_b     = sum_s φ_out^s(a_{b, s})

    For this smoke σ_{b, i} is derived deterministically from the
    cycle-edge sign parity (computed from local feature-direction
    sign), but for simplicity we use σ_{b, i} = +1 if the corner's
    feature mean is above the cycle's centroid, else −1.  This matches
    HSiKAN's σ-as-cycle-parity semantics in the simplest possible way.
    """
    def __init__(self, d_in: int, d_hidden: int = 32, k: int = 4):
        super().__init__()
        self.k = k
        self.embed = nn.Linear(d_in, d_hidden)
        self.phi_in_pos = nn.Linear(d_hidden, d_hidden)
        self.phi_in_neg = nn.Linear(d_hidden, d_hidden)
        self.phi_out_pos = nn.Linear(d_hidden, d_hidden)
        self.phi_out_neg = nn.Linear(d_hidden, d_hidden)

    def forward(self, h_corners: torch.Tensor) -> torch.Tensor:
        """h_corners: (B, k, d_in) → (B, d_hidden)"""
        h = self.embed(h_corners)                    # (B, k, d)
        # σ from feature-mean polarity.
        per_corner = h.mean(dim=-1)                  # (B, k)
        centroid = per_corner.mean(dim=-1, keepdim=True)  # (B, 1)
        sigma = (per_corner > centroid).float() * 2 - 1   # ±1, (B, k)
        # Per-sign branches.
        u_pos = torch.tanh(self.phi_in_pos(h))       # (B, k, d)
        u_neg = torch.tanh(self.phi_in_neg(h))
        # Mask + mean.
        m_pos = (sigma == 1).float().unsqueeze(-1)   # (B, k, 1)
        m_neg = (sigma == -1).float().unsqueeze(-1)
        cnt_pos = m_pos.sum(dim=1).clamp(min=1)
        cnt_neg = m_neg.sum(dim=1).clamp(min=1)
        a_pos = (u_pos * m_pos).sum(dim=1) / cnt_pos     # (B, d)
        a_neg = (u_neg * m_neg).sum(dim=1) / cnt_neg
        h_aux = (torch.tanh(self.phi_out_pos(a_pos))
                 + torch.tanh(self.phi_out_neg(a_neg)))
        return h_aux


# ─── Highway-gated quaternion-attention path ───────────────────────


class HighwayHSiKANAggregator(nn.Module):
    """HSiKAN σ-masked aggregator + parallel quaternion-attention pool,
    mixed by a learnable Highway gate.  Replicates the Slashdot
    SOTA-beating recipe at the per-query level: at init the Highway
    gate is closed (g ≈ 0.05) so the model behaves like the uniform
    HSiKAN aggregator; gradient pushes the gate open only where the
    sparse-attention pool is more informative than the σ-masked
    aggregation.

    The attention is single-query per corner-subset (the query is the
    mean of corner features) with Hamilton-product real-part scoring
    over the corner keys.  Same kernel structure as
    `_QuaternionAttentionM_e` from the Slashdot recipe, scaled to
    cardinality-k corner sets.
    """

    def __init__(self, d_in: int, d_hidden: int, k: int,
                 init_gate_logit: float = -3.0, gate_max: float = 1.0,
                 gate_kind: str = "scalar", n_grid: int = 8):
        super().__init__()
        self.uniform = HSiKANAggregator(d_in=d_in, d_hidden=d_hidden, k=k)
        # d_attn must be divisible by 4 for quaternion blocks.
        d_attn = d_hidden if d_hidden % 4 == 0 else ((d_hidden // 4) * 4)
        self.d_attn = d_attn
        self.n_quat = d_attn // 4
        self.W_q = nn.Linear(d_in, d_attn, bias=False)
        self.W_k = nn.Linear(d_in, d_attn, bias=False)
        self.phi_out = nn.Linear(d_attn, d_hidden)
        with torch.no_grad():
            self.W_q.weight.mul_(0.01)
            self.W_k.weight.mul_(0.01)
        self.gate_kind = gate_kind
        if gate_kind == "scalar":
            self.gate_logit = nn.Parameter(
                torch.tensor(init_gate_logit, dtype=torch.float32),
            )
            self.gate_proj = None
            self.gate_coef = None
        elif gate_kind == "edge_cr":
            # Per-edge KAN-style gate: query feature → 2 logits via a
            # learnable Catmull-Rom spline, softmax-normalised to
            # (uniform_weight, attn_weight).  No sigmoid anywhere.
            self.gate_logit = None
            self.gate_proj = nn.Linear(d_in, 2)
            with torch.no_grad():
                self.gate_proj.weight.mul_(0.01)
                # Bias so that initial softmax leans uniform (low attn).
                self.gate_proj.bias.copy_(torch.tensor([3.0, 0.0]))
            self.n_grid = n_grid
            # Init CR control points as identity curve over [-1, 1] →
            # range [-3, 3] (gives σ-like behaviour at start).
            init_coefs = (torch.linspace(-3.0, 3.0, n_grid)
                          .unsqueeze(0).expand(2, -1).contiguous())
            self.gate_coef = nn.Parameter(init_coefs)
        else:
            raise ValueError(f"unknown gate_kind: {gate_kind!r}")
        self.gate_max = gate_max
        self.scale = self.n_quat ** -0.5

    def gate(self, h_query: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the attention-vs-uniform mix weight.

        Returns a scalar (when gate_kind="scalar") or a per-edge
        weight tensor of shape (B,) (when gate_kind="edge_cr").
        """
        if self.gate_kind == "scalar":
            pair = torch.stack(
                [torch.zeros_like(self.gate_logit), self.gate_logit]
            )
            return self.gate_max * F.softmax(pair, dim=0)[1]
        # edge_cr: per-edge KAN-style gate.  h_query is required.
        assert h_query is not None
        x = torch.tanh(self.gate_proj(h_query))      # (B, 2) ∈ [-1, 1]
        cr_out = _catmull_rom_eval(self.gate_coef, x, self.n_grid)
        # softmax over the two channels → (B, 2) convex weights.
        weights = F.softmax(cr_out, dim=-1)           # (B, 2)
        # Return the attention weight (uniform weight = 1 - this).
        return weights[:, 1]

    def forward(self, h_corners: torch.Tensor) -> torch.Tensor:
        """h_corners: (B, n, d_in) → (B, d_hidden)."""
        # Uniform σ-masked branch (the structural prior).
        uniform_out = self.uniform(h_corners)               # (B, d_hidden)

        # Quaternion attention path.  Query = mean of corners.
        q_in = h_corners.mean(dim=1)                        # (B, d_in)
        q = self.W_q(q_in).view(-1, self.n_quat, 4)         # (B, n_quat, 4)
        k = self.W_k(h_corners).view(
            *h_corners.shape[:-1], self.n_quat, 4
        )                                                    # (B, n, n_quat, 4)
        q_b = q.unsqueeze(1)                                 # (B, 1, n_quat, 4)
        scores = (
            q_b[..., 0] * k[..., 0]
            - q_b[..., 1] * k[..., 1]
            - q_b[..., 2] * k[..., 2]
            - q_b[..., 3] * k[..., 3]
        ).sum(dim=-1) * self.scale                          # (B, n)
        attn = F.softmax(scores, dim=-1)                     # (B, n)
        # Value = key projection (parameter-efficient).
        v = k.flatten(-2, -1)                                # (B, n, d_attn)
        attn_pool = (attn.unsqueeze(-1) * v).sum(dim=1)      # (B, d_attn)
        attn_out = self.phi_out(attn_pool)                   # (B, d_hidden)

        # Convex-combination mix.  Scalar gate → broadcast; per-edge
        # gate → per-batch-element weight.
        if self.gate_kind == "scalar":
            g = self.gate()
            return (1.0 - g) * uniform_out + g * attn_out
        # edge_cr: g(B,) per-batch convex weight on attention.
        g = self.gate(h_query=q_in).unsqueeze(-1)            # (B, 1)
        return (1.0 - g) * uniform_out + g * attn_out


# ─── Mixed-arity α-routed aggregator ────────────────────────────────


class MixedArityHSiKANAggregator(nn.Module):
    """α-routed mix of per-arity HSiKAN aggregators over the same
    4-corner query.

    For a cardinality-4 query, three primitive types coexist:

        k=2 — 6 corner-pair edges
        k=3 — 4 triangles formed by corner triples (C(4,3)=4)
        k=4 — the box-corner cycle itself

    Each primitive runs an independent HSiKAN σ-masked aggregator and
    is averaged over all subsets of that arity present in the query.
    A softmax-normalised α_κ vector then mixes the per-arity
    descriptors.  This is the same α-routing primitive that gave the
    SOTA-beating Slashdot result on 2026-05-08; here it routes signal
    across primitive geometries instead of cycle / walk slots.

    The trained α_κ at convergence is the per-primitive structural
    readout for detection: which polytope-arity does the model
    predict from --- the box itself (k=4), one of its triangulations
    (k=3), or the corner-pair edges (k=2).
    """

    def __init__(self, d_in: int, d_hidden: int = 32,
                 highway_attention: bool = False,
                 gate_kind: str = "scalar"):
        super().__init__()
        self.highway_attention = highway_attention
        self.gate_kind = gate_kind
        if highway_attention:
            self.agg_k2 = HighwayHSiKANAggregator(
                d_in, d_hidden, k=2, gate_kind=gate_kind,
            )
            self.agg_k3 = HighwayHSiKANAggregator(
                d_in, d_hidden, k=3, gate_kind=gate_kind,
            )
            self.agg_k4 = HighwayHSiKANAggregator(
                d_in, d_hidden, k=4, gate_kind=gate_kind,
            )
        else:
            self.agg_k2 = HSiKANAggregator(d_in, d_hidden, k=2)
            self.agg_k3 = HSiKANAggregator(d_in, d_hidden, k=3)
            self.agg_k4 = HSiKANAggregator(d_in, d_hidden, k=4)
        # α over (k=2, k=3, k=4).  Init at 0 → softmax = uniform 1/3.
        self.alpha_logits = nn.Parameter(torch.zeros(3))
        # Pre-compute corner-subset index lists.
        from itertools import combinations
        self._pairs = list(combinations(range(4), 2))      # 6
        self._triangles = list(combinations(range(4), 3))   # 4

    def alpha(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)

    def gates(self) -> list[float] | None:
        """Per-slot Highway gate values (scalar mode only).  In
        edge_cr mode the gate is per-edge so a single readout
        scalar is not well-defined; returns None."""
        if not self.highway_attention or self.gate_kind != "scalar":
            return None
        return [
            float(self.agg_k2.gate().detach().cpu()),
            float(self.agg_k3.gate().detach().cpu()),
            float(self.agg_k4.gate().detach().cpu()),
        ]

    def forward(self, h_corners: torch.Tensor) -> torch.Tensor:
        """h_corners: (B, 4, d_in) → (B, d_hidden)."""
        # k=4: full box.
        out_k4 = self.agg_k4(h_corners)
        # k=3: mean over 4 triangles.
        tri_outs = [self.agg_k3(h_corners[:, list(tri)])
                    for tri in self._triangles]
        out_k3 = torch.stack(tri_outs, dim=0).mean(dim=0)
        # k=2: mean over 6 pairs.
        pair_outs = [self.agg_k2(h_corners[:, list(p)])
                     for p in self._pairs]
        out_k2 = torch.stack(pair_outs, dim=0).mean(dim=0)
        # α-mix.
        a = self.alpha()                            # (3,)
        stacked = torch.stack([out_k2, out_k3, out_k4], dim=1)  # (B, 3, d)
        h_aux = (a.unsqueeze(0).unsqueeze(-1) * stacked).sum(dim=1)
        return h_aux


# ─── HyMeYOLO-Q model ───────────────────────────────────────────────


class HyMeYOLOQ(nn.Module):
    def __init__(self, n_classes: int = 3, d_hidden: int = 32,
                 mixed_arity: bool = False,
                 highway_attention: bool = False,
                 gate_kind: str = "scalar"):
        super().__init__()
        self.backbone = TinyBackbone(c_in=3, c_out=d_hidden)
        init_corners = torch.tensor(
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            dtype=torch.float32,
        )
        self.query_corners = nn.Parameter(init_corners.clone())
        self.mixed_arity = mixed_arity
        self.highway_attention = highway_attention
        self.gate_kind = gate_kind
        if mixed_arity:
            self.aggregator = MixedArityHSiKANAggregator(
                d_in=d_hidden, d_hidden=d_hidden,
                highway_attention=highway_attention,
                gate_kind=gate_kind,
            )
        elif highway_attention:
            self.aggregator = HighwayHSiKANAggregator(
                d_in=d_hidden, d_hidden=d_hidden, k=4,
                gate_kind=gate_kind,
            )
        else:
            self.aggregator = HSiKANAggregator(
                d_in=d_hidden, d_hidden=d_hidden, k=4,
            )
        self.head_offset = nn.Linear(d_hidden, 4 * 2)
        self.head_cls = nn.Linear(d_hidden, n_classes)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) → corners (B, 4, 2), class logits (B, n_cls)."""
        B = x.shape[0]
        F_map = self.backbone(x)                          # (B, d, 8, 8)
        # Broadcast the query corners across the batch.
        corners = self.query_corners.unsqueeze(0).expand(B, -1, -1)
        # Bilinear sample at corner locations.
        h_corners = bilinear_sample(F_map, corners)       # (B, 4, d)
        # HSiKAN aggregation.
        h_aux = self.aggregator(h_corners)                # (B, d)
        # Predict corner offsets and class.
        offset = self.head_offset(h_aux).view(B, 4, 2)
        offset = 0.3 * torch.tanh(offset)                  # bound
        refined = corners + offset
        cls_logits = self.head_cls(h_aux)
        return refined, cls_logits


# ─── Loss + IoU ─────────────────────────────────────────────────────


def corner_to_box(corners: torch.Tensor) -> torch.Tensor:
    """4-corner polygon → (x0, y0, x1, y1) axis-aligned bbox.

    corners: (B, 4, 2) → (B, 4)."""
    x0 = corners[..., 0].min(dim=-1).values
    y0 = corners[..., 1].min(dim=-1).values
    x1 = corners[..., 0].max(dim=-1).values
    y1 = corners[..., 1].max(dim=-1).values
    return torch.stack([x0, y0, x1, y1], dim=-1)


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two (B, 4) bboxes."""
    ix0 = torch.max(a[..., 0], b[..., 0])
    iy0 = torch.max(a[..., 1], b[..., 1])
    ix1 = torch.min(a[..., 2], b[..., 2])
    iy1 = torch.min(a[..., 3], b[..., 3])
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-9)


def gt_corners_from_box(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """(B, 4) → (B, 4, 2) — corners in TL, TR, BR, BL order to match
    the query's init."""
    x0, y0, x1, y1 = boxes_xyxy.unbind(-1)
    return torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x1, y1], dim=-1),
            torch.stack([x0, y1], dim=-1),
        ],
        dim=1,
    )


# ─── Smoke training loop ────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed-arity", action="store_true",
                    help="Use the α-routed mixed-arity aggregator "
                          "(k=2,3,4 over the 4 corners) instead of "
                          "the single-arity k=4 aggregator.")
    ap.add_argument("--highway-attention", action="store_true",
                    help="Enable Highway-gated quaternion attention "
                          "as a parallel pool to the σ-masked HSiKAN "
                          "aggregation (per-slot Highway gate "
                          "init=−3 → ≈0.05).")
    ap.add_argument("--gate-kind", default="scalar",
                    choices=["scalar", "edge_cr"],
                    help="Highway gate parameterisation. "
                          "'scalar' = sigmoid-free 2-element softmax "
                          "(equivalent to per-slot scalar gate). "
                          "'edge_cr' = per-edge Catmull-Rom spline "
                          "over query embedding, softmax-normalised "
                          "(KAN-aligned, no fixed nonlinearities).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data.
    X_tr, B_tr, C_tr = make_synthetic_rectangles(args.n_train, seed=args.seed)
    X_te, B_te, C_te = make_synthetic_rectangles(args.n_test, seed=args.seed + 1)
    X_tr_t = torch.tensor(X_tr, device=device)
    B_tr_t = torch.tensor(B_tr, device=device)
    C_tr_t = torch.tensor(C_tr, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, device=device)
    B_te_t = torch.tensor(B_te, device=device)
    C_te_t = torch.tensor(C_te, dtype=torch.long, device=device)
    print(f"[data] train={X_tr.shape}, test={X_te.shape}")

    model = HyMeYOLOQ(n_classes=3, d_hidden=args.d_hidden,
                       mixed_arity=args.mixed_arity,
                       highway_attention=args.highway_attention,
                       gate_kind=args.gate_kind,
                       ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    print(f"[model] params={n_params}")

    t0 = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        ep_loss = 0.0
        n_batches = 0
        for bs in range(0, args.n_train, args.batch):
            be = min(bs + args.batch, args.n_train)
            idx = perm[bs:be]
            x_b = X_tr_t[idx]
            box_b = B_tr_t[idx]
            cls_b = C_tr_t[idx]
            corners_pred, cls_logits = model(x_b)
            corners_gt = gt_corners_from_box(box_b)
            loss_corner = F.l1_loss(corners_pred, corners_gt)
            loss_cls = F.cross_entropy(cls_logits, cls_b)
            loss = loss_corner + 0.5 * loss_cls
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        # Test eval.
        model.eval()
        with torch.no_grad():
            corners_pred, cls_logits = model(X_te_t)
            box_pred = corner_to_box(corners_pred)
            box_gt = B_te_t
            iou = iou_xyxy(box_pred, box_gt)
            cls_acc = (cls_logits.argmax(-1) == C_te_t).float().mean().item()
        print(f"  epoch {epoch+1:2d}  loss={ep_loss/n_batches:.4f}  "
              f"mIoU={iou.mean().item():.3f}  cls_acc={cls_acc:.3f}")
    train_s = time.time() - t0

    if args.mixed_arity:
        alpha_vec = [float(a) for a in
                      model.aggregator.alpha().detach().cpu().tolist()]
        gates_vec = model.aggregator.gates()
    elif args.highway_attention:
        alpha_vec = None
        gates_vec = [float(model.aggregator.gate().detach().cpu())]
    else:
        alpha_vec = None
        gates_vec = None
    out = dict(
        task="hymeyolo_q_smoke_with_hsikan",
        mixed_arity=args.mixed_arity,
        highway_attention=args.highway_attention,
        alpha=alpha_vec,
        alpha_labels=["k=2", "k=3", "k=4"] if args.mixed_arity else None,
        gates=gates_vec,
        gate_labels=(["k=2", "k=3", "k=4"]
                     if args.mixed_arity and args.highway_attention
                     else (["k=4"] if args.highway_attention else None)),
        n_train=args.n_train, n_test=args.n_test,
        n_epochs=args.n_epochs, d_hidden=args.d_hidden,
        n_params=n_params,
        miou=float(iou.mean().item()),
        cls_acc=float(cls_acc),
        train_s=train_s, seed=args.seed,
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
