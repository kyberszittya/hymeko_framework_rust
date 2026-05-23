"""HyMeYOLO multi-object detection with Hungarian matching.

Extension of `hymeyolo_q_smoke.py` to N learnable queries + DETR-style
set prediction loss.  Synthetic 32×32 RGB with 1-3 coloured
rectangles per image; per-image bipartite Hungarian matcher pairs
predicted-corner sets to ground-truth corner sets via
`scipy.optimize.linear_sum_assignment`.  Set loss = matched corner
L1 + matched class CE + unmatched no-object CE.

Reference: `examples/hymeyolo_hungarian.py` in the sibling HyMeKoConv
repo (mIoU 59.2%, recall 100%, cls_acc 89.5% at 21,372 params, 12
epochs).  This re-implementation uses HSiKAN's mixed-arity α-routing
+ Highway-quat sparse attention (when enabled) in place of
HyMeKoConv's per-Θ_τ aggregation.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .hymeyolo_q_smoke import (
    HSiKANAggregator, HighwayHSiKANAggregator,
    MixedArityHSiKANAggregator, TinyBackbone,
    bilinear_sample, corner_to_box, gt_corners_from_box, iou_xyxy,
)


# ─── Synthetic multi-rectangle dataset ──────────────────────────────


def make_synthetic_multi_rectangles(
    n: int, H: int = 32, W: int = 32,
    n_classes: int = 3, max_objects: int = 3,
    seed: int = 0,
):
    """Generate `n` 32×32 RGB images with 1-`max_objects` solid
    coloured rectangles each.

    Returns
    -------
    X       : (n, 3, H, W) float32 in [0, 1]
    boxes   : (n, max_objects, 4) — (x0, y0, x1, y1) normalised; padded
              with zeros for missing objects
    classes : (n, max_objects) — class id ∈ {0..n_classes-1};
              `n_classes` (== "no-object") for padding
    counts  : (n,) — actual object count per image
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 3, H, W), dtype=np.float32)
    X += rng.normal(0, 0.02, size=X.shape).astype(np.float32)
    boxes = np.zeros((n, max_objects, 4), dtype=np.float32)
    classes = np.full((n, max_objects), n_classes, dtype=np.int64)  # pad = no-object
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        n_obj = int(rng.integers(1, max_objects + 1))
        counts[i] = n_obj
        for j in range(n_obj):
            c = int(rng.integers(0, n_classes))
            classes[i, j] = c
            x0 = rng.uniform(0.05, 0.6)
            y0 = rng.uniform(0.05, 0.6)
            w = rng.uniform(0.15, 0.30)
            h = rng.uniform(0.15, 0.30)
            x1 = min(0.95, x0 + w)
            y1 = min(0.95, y0 + h)
            boxes[i, j] = [x0, y0, x1, y1]
            ix0, iy0 = int(x0 * W), int(y0 * H)
            ix1, iy1 = int(x1 * W), int(y1 * H)
            X[i, c, iy0:iy1, ix0:ix1] = 1.0
    return X, boxes, classes, counts


# ─── HyMeYOLO multi-query model ─────────────────────────────────────


class HyMeYOLOMulti(nn.Module):
    def __init__(self, n_queries: int = 6, n_classes: int = 3,
                 d_hidden: int = 32,
                 mixed_arity: bool = True,
                 highway_attention: bool = False,
                 gate_kind: str = "scalar"):
        super().__init__()
        self.n_queries = n_queries
        self.n_classes = n_classes  # excluding the "no-object" class
        self.backbone = TinyBackbone(c_in=3, c_out=d_hidden)
        # N learnable queries, each a cardinality-4 hyperedge.
        # Initialise on a centred square then add per-query random
        # offset to break symmetry across queries.
        base = torch.tensor(
            [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
            dtype=torch.float32,
        )
        init_corners = base.unsqueeze(0).expand(n_queries, -1, -1).clone()
        gen = torch.Generator().manual_seed(0)
        init_corners = init_corners + torch.randn(
            init_corners.shape, generator=gen,
        ) * 0.08
        init_corners = init_corners.clamp(0.05, 0.95)
        self.query_corners = nn.Parameter(init_corners.clone())

        # Aggregator (one shared across queries).
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
        # Output heads (one shared across queries).
        # Class head produces n_classes + 1 logits (last = no-object).
        self.head_offset = nn.Linear(d_hidden, 4 * 2)
        self.head_cls = nn.Linear(d_hidden, n_classes + 1)
        self.mixed_arity = mixed_arity

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) → (B, N, 4, 2) corners,
                              (B, N, n_classes+1) class logits."""
        B = x.shape[0]
        N = self.n_queries
        F_map = self.backbone(x)                              # (B, d, h, w)
        # (B, N, 4, 2)
        corners = self.query_corners.unsqueeze(0).expand(B, -1, -1, -1)
        # Flatten queries × corners to bilinear-sample once.
        flat = corners.reshape(B, N * 4, 2)                   # (B, N*4, 2)
        h_flat = bilinear_sample(F_map, flat)                 # (B, N*4, d)
        h_per_q = h_flat.view(B, N, 4, -1)                    # (B, N, 4, d)
        # Aggregator over (B*N, 4, d). Reshape, run, reshape back.
        h_BN = h_per_q.reshape(B * N, 4, -1)
        h_aux = self.aggregator(h_BN)                         # (B*N, d)
        h_aux = h_aux.view(B, N, -1)                          # (B, N, d)
        # Heads.
        offset = self.head_offset(h_aux).view(B, N, 4, 2)
        offset = 0.3 * torch.tanh(offset)
        refined = corners + offset                             # (B, N, 4, 2)
        cls_logits = self.head_cls(h_aux)                     # (B, N, n_classes+1)
        return refined, cls_logits


# ─── Hungarian-matched set loss ─────────────────────────────────────


def hungarian_set_loss(
    pred_corners: torch.Tensor,    # (B, N, 4, 2)
    pred_cls: torch.Tensor,        # (B, N, n_classes+1)
    gt_corners: torch.Tensor,      # (B, M, 4, 2)
    gt_classes: torch.Tensor,      # (B, M)
    gt_counts: torch.Tensor,       # (B,)
    n_classes: int,
    lam_corner: float = 5.0,
    lam_cls: float = 1.0,
    lam_no_obj: float = 0.5,
    *,
    cls_loss_kind: str = "ce",    # "ce" | "focal" (Stage A-3)
    box_loss_kind: str = "l1",    # "l1" | "giou" (Stage A-3)
):
    """DETR-style set-prediction loss with per-image Hungarian
    matching.

    Stage A-3 (2026-05-16 evening):
        cls_loss_kind="focal" → use multi-class focal loss instead of
            cross-entropy (better for the heavy class-imbalance from
            many no-object queries).
        box_loss_kind="giou" → use 1-GIoU on AABBs derived from the
            matched corners instead of L1 on the corner positions
            (better for box regression — aligned with the eval mAP@0.5
            metric, which uses IoU).
    Defaults preserve pre-2026-05-16 behaviour (ce + L1).
    """
    # Lazy import: avoid a circular dependency at module load.
    from .train_circles_ricci import focal_loss_ce, giou_loss_xyxy

    def _cls_loss(logits, targets, *, reduction="mean"):
        if cls_loss_kind == "focal":
            return focal_loss_ce(
                logits, targets, gamma=2.0, reduction=reduction,
            )
        return F.cross_entropy(logits, targets, reduction=reduction)

    def _box_loss_on_matched(matched_pred_corners, matched_gt_corners):
        if box_loss_kind == "giou":
            # Box-loss term computes GIoU on the AABB derived from
            # each (4-corner) prediction vs the AABB of each GT
            # 4-corner set. Same input dimensionality as the L1
            # branch; just different reduction.
            pred_aabb = torch.stack([
                matched_pred_corners[..., 0].min(dim=-1).values,
                matched_pred_corners[..., 1].min(dim=-1).values,
                matched_pred_corners[..., 0].max(dim=-1).values,
                matched_pred_corners[..., 1].max(dim=-1).values,
            ], dim=-1)
            gt_aabb = torch.stack([
                matched_gt_corners[..., 0].min(dim=-1).values,
                matched_gt_corners[..., 1].min(dim=-1).values,
                matched_gt_corners[..., 0].max(dim=-1).values,
                matched_gt_corners[..., 1].max(dim=-1).values,
            ], dim=-1)
            return giou_loss_xyxy(pred_aabb, gt_aabb, reduction="mean")
        return (matched_pred_corners - matched_gt_corners).abs().mean()
    B, N = pred_corners.shape[:2]
    device = pred_corners.device
    pred_probs = F.softmax(pred_cls, dim=-1)        # (B, N, n_classes+1)

    total_corner_loss = pred_corners.new_zeros(())
    total_cls_loss = pred_corners.new_zeros(())
    total_no_obj_loss = pred_corners.new_zeros(())
    matched_count_total = 0
    n_matched_correct = 0

    for b in range(B):
        n_gt = int(gt_counts[b].item())
        if n_gt == 0:
            # All queries → no-object.
            no_obj_targets = torch.full(
                (N,), n_classes, dtype=torch.long, device=device,
            )
            total_no_obj_loss = total_no_obj_loss + _cls_loss(
                pred_cls[b], no_obj_targets, reduction="sum",
            )
            continue
        # Cost matrix: (N, n_gt).
        # Corner L1: each pred against each gt. Matching cost stays
        # on L1 even when box_loss_kind=="giou" — see note in the
        # circle variant: per-pair GIoU as a matching cost is more
        # expensive without measurable matching difference.
        pc = pred_corners[b].unsqueeze(1)            # (N, 1, 4, 2)
        gc = gt_corners[b, :n_gt].unsqueeze(0)       # (1, n_gt, 4, 2)
        corner_cost = (pc - gc).abs().mean(dim=(-1, -2))  # (N, n_gt)
        # Class cost: −prob(gt class).
        gt_cls_b = gt_classes[b, :n_gt]               # (n_gt,)
        cls_cost = -pred_probs[b][:, gt_cls_b]        # (N, n_gt)
        cost = (lam_corner * corner_cost + lam_cls * cls_cost
                ).detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        # rows: query indices that match. cols: gt indices.
        # Build no-object target for unmatched queries.
        matched_pred = torch.tensor(rows, device=device, dtype=torch.long)
        matched_gt = torch.tensor(cols, device=device, dtype=torch.long)
        # Matched corner/box loss (dispatched by box_loss_kind).
        total_corner_loss = total_corner_loss + _box_loss_on_matched(
            pred_corners[b][matched_pred],
            gt_corners[b][matched_gt],
        )
        # Matched class loss (dispatched by cls_loss_kind).
        matched_cls_targets = gt_classes[b][matched_gt]
        total_cls_loss = total_cls_loss + _cls_loss(
            pred_cls[b][matched_pred], matched_cls_targets,
        )
        n_matched_correct += int((
            pred_cls[b][matched_pred].argmax(-1) == matched_cls_targets
        ).sum().item())
        matched_count_total += int(matched_pred.numel())
        # Unmatched → no-object.
        all_idx = torch.arange(N, device=device)
        unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
        unmatched_mask[matched_pred] = False
        unmatched = all_idx[unmatched_mask]
        if unmatched.numel() > 0:
            no_obj_targets = torch.full(
                (unmatched.numel(),), n_classes,
                dtype=torch.long, device=device,
            )
            total_no_obj_loss = total_no_obj_loss + _cls_loss(
                pred_cls[b][unmatched], no_obj_targets,
            )

    # Average over batch.
    loss = (lam_corner * total_corner_loss
            + lam_cls * total_cls_loss
            + lam_no_obj * total_no_obj_loss) / max(1, B)
    cls_acc = (n_matched_correct / max(1, matched_count_total)
               if matched_count_total > 0 else 0.0)
    return loss, cls_acc, matched_count_total


def evaluate_multi(model: HyMeYOLOMulti,
                   X: torch.Tensor, boxes: torch.Tensor,
                   classes: torch.Tensor, counts: torch.Tensor,
                   n_classes: int):
    """Compute mIoU and recall on the matched-prediction set."""
    model.eval()
    with torch.no_grad():
        pred_corners, pred_cls = model(X)
        pred_boxes = corner_to_box(pred_corners.view(-1, 4, 2)).view(
            pred_corners.shape[0], pred_corners.shape[1], 4,
        )
        pred_probs = F.softmax(pred_cls, dim=-1)
        # Take argmax class per query, drop "no-object".
        pred_class_id = pred_probs.argmax(dim=-1)            # (B, N)
        pred_score = pred_probs.max(dim=-1).values            # (B, N)

        ious_matched = []
        n_recalled = 0
        n_total_gt = 0
        for b in range(X.shape[0]):
            n_gt = int(counts[b].item())
            n_total_gt += n_gt
            if n_gt == 0:
                continue
            # Match each gt to its best-IoU query that predicts a
            # non-no-object class.
            gt_boxes_b = boxes[b, :n_gt]
            valid_mask = pred_class_id[b] != n_classes
            if valid_mask.sum() == 0:
                continue
            valid_pred_boxes = pred_boxes[b][valid_mask]
            for j in range(n_gt):
                ious = iou_xyxy(
                    valid_pred_boxes,
                    gt_boxes_b[j].unsqueeze(0).expand_as(valid_pred_boxes),
                )
                if ious.numel() == 0:
                    continue
                best_iou = ious.max().item()
                ious_matched.append(best_iou)
                if best_iou > 0.5:
                    n_recalled += 1
        miou = float(np.mean(ious_matched)) if ious_matched else 0.0
        recall = n_recalled / max(1, n_total_gt)
    return miou, recall


# ─── Smoke training ─────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-queries", type=int, default=6)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--n-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed-arity", action="store_true")
    ap.add_argument("--highway-attention", action="store_true")
    ap.add_argument("--gate-kind", default="scalar",
                    choices=["scalar", "edge_cr"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 3
    X_tr, B_tr, C_tr, K_tr = make_synthetic_multi_rectangles(
        args.n_train, max_objects=args.max_objects, seed=args.seed,
    )
    X_te, B_te, C_te, K_te = make_synthetic_multi_rectangles(
        args.n_test, max_objects=args.max_objects, seed=args.seed + 1,
    )
    X_tr_t = torch.tensor(X_tr, device=device)
    B_tr_t = torch.tensor(B_tr, device=device)
    C_tr_t = torch.tensor(C_tr, device=device)
    K_tr_t = torch.tensor(K_tr, device=device)
    X_te_t = torch.tensor(X_te, device=device)
    B_te_t = torch.tensor(B_te, device=device)
    C_te_t = torch.tensor(C_te, device=device)
    K_te_t = torch.tensor(K_te, device=device)
    print(f"[data] train={X_tr.shape}, test={X_te.shape}, "
          f"max_objects={args.max_objects}, mean_objs="
          f"{K_tr.mean():.2f}")

    # Convert gt boxes → gt corners for the loss.
    gt_corners_tr = gt_corners_from_box(B_tr_t.view(-1, 4)).view(
        args.n_train, args.max_objects, 4, 2,
    )
    gt_corners_te = gt_corners_from_box(B_te_t.view(-1, 4)).view(
        args.n_test, args.max_objects, 4, 2,
    )

    model = HyMeYOLOMulti(
        n_queries=args.n_queries, n_classes=n_classes,
        d_hidden=args.d_hidden,
        mixed_arity=args.mixed_arity,
        highway_attention=args.highway_attention,
        gate_kind=args.gate_kind,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    print(f"[model] params={n_params}, n_queries={args.n_queries}, "
          f"mixed_arity={args.mixed_arity}, "
          f"highway_attention={args.highway_attention}")

    t0 = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        ep_loss = 0.0; n_batches = 0; ep_cls_acc = 0.0
        for bs in range(0, args.n_train, args.batch):
            be = min(bs + args.batch, args.n_train)
            idx = perm[bs:be]
            x_b = X_tr_t[idx]
            gtc_b = gt_corners_tr[idx]
            gtcls_b = C_tr_t[idx]
            cnt_b = K_tr_t[idx]
            pred_corners, pred_cls = model(x_b)
            loss, cls_acc, _ = hungarian_set_loss(
                pred_corners, pred_cls, gtc_b, gtcls_b, cnt_b,
                n_classes=n_classes,
            )
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
            ep_cls_acc += cls_acc
        miou_te, recall_te = evaluate_multi(
            model, X_te_t, B_te_t, C_te_t, K_te_t, n_classes=n_classes,
        )
        print(f"  epoch {epoch+1:2d}  loss={ep_loss/n_batches:.4f}  "
              f"cls_acc={ep_cls_acc/n_batches:.3f}  "
              f"mIoU={miou_te:.3f}  recall={recall_te:.3f}")
    train_s = time.time() - t0

    out = dict(
        task="hymeyolo_hungarian_with_hsikan",
        n_train=args.n_train, n_test=args.n_test,
        n_queries=args.n_queries, max_objects=args.max_objects,
        n_epochs=args.n_epochs, d_hidden=args.d_hidden,
        n_params=n_params,
        miou=miou_te, recall=recall_te,
        train_s=train_s, seed=args.seed,
        mixed_arity=args.mixed_arity,
        highway_attention=args.highway_attention,
    )
    if args.mixed_arity:
        out["alpha"] = [
            float(a) for a in
            model.aggregator.alpha().detach().cpu().tolist()
        ]
        out["alpha_labels"] = ["k=2", "k=3", "k=4"]
        out["gates"] = model.aggregator.gates()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
