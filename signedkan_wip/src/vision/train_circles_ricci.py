"""Phase 1 smoke training for RicciHyMeYOLOMulti on Cluttered MNIST.

Integrates circles + Ricci shape-prior with the existing Hungarian
set-prediction framework.  Each batch performs TWO Hungarian solves:
one for box queries (k=4), one for circle queries (k=8).  Both match
to the same ground-truth box set (with no-object padding for unmatched
queries).  This lets box queries specialise on rectangular objects and
circle queries on round objects --- in Cluttered MNIST (digit shapes
are rectangular-ish), we expect box queries to dominate matching and
circle queries to mostly converge on no-object; the test of whether
the model learns is whether the loss curve still descends.

Usage:
    python -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 100 --epochs 10 --seed 0

Configs:
    - "baseline"      : HyMeYOLOMulti (existing, no circles, no ricci)
    - "boxes-only"    : RicciHyMeYOLOMulti(n_circle_queries=0)
    - "circles-only"  : RicciHyMeYOLOMulti(n_box_queries=0)
    - "boxes+circles" : RicciHyMeYOLOMulti(both)
    - "+ricci"        : boxes+circles with Ricci-modulated class head
"""
from __future__ import annotations
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .cluttered_mnist import make_cluttered_mnist_hungarian_format
from .hymeyolo_circles_ricci import RicciHyMeYOLOMulti
from .hymeyolo_hungarian import HyMeYOLOMulti, hungarian_set_loss
from .hymeyolo_kcycle import KCycleHyMeYOLOMulti
from .hymeyolo_ricci_kcycle import RicciKCycleHyMeYOLOMulti
from .hymeyolo_q_smoke import gt_corners_from_box


# ─── Hungarian for circle queries matched to box GTs ───────────────


def circles_aabb(corners: torch.Tensor) -> torch.Tensor:
    """k-corner circle → axis-aligned bounding box (x0, y0, x1, y1)."""
    x0 = corners[..., 0].min(dim=-1).values
    y0 = corners[..., 1].min(dim=-1).values
    x1 = corners[..., 0].max(dim=-1).values
    y1 = corners[..., 1].max(dim=-1).values
    return torch.stack([x0, y0, x1, y1], dim=-1)


def hungarian_set_loss_circle_vs_box(
    pred_corners: torch.Tensor,    # (B, N, k_circle, 2)
    pred_cls: torch.Tensor,        # (B, N, n_classes+1)
    gt_boxes_xyxy: torch.Tensor,   # (B, M, 4)
    gt_classes: torch.Tensor,      # (B, M)
    gt_counts: torch.Tensor,       # (B,)
    n_classes: int,
    lam_aabb: float = 5.0,
    lam_cls: float = 1.0,
    lam_no_obj: float = 0.5,
    *,
    cls_loss_kind: str = "ce",     # "ce" | "focal" (Stage A-3)
    box_loss_kind: str = "l1",     # "l1" | "giou" (Stage A-3)
):
    """Circle queries are matched via their AABB (the bounding box
    enclosing the k circle corners) against GT xyxy boxes.  Loss has
    the same shape as ``hungarian_set_loss`` but with an AABB-L1 term
    in place of corner-L1.
    """
    B, N, _, _ = pred_corners.shape
    device = pred_corners.device
    pred_aabb = circles_aabb(pred_corners)              # (B, N, 4)
    pred_probs = F.softmax(pred_cls, dim=-1)

    # Stage A-3 dispatch: which cls / box loss kernel to use.
    def _cls_loss(logits, targets, *, reduction="mean"):
        if cls_loss_kind == "focal":
            return focal_loss_ce(
                logits, targets, gamma=2.0, reduction=reduction,
            )
        return F.cross_entropy(logits, targets, reduction=reduction)

    def _box_loss(pred_box, target_box):
        if box_loss_kind == "giou":
            return giou_loss_xyxy(pred_box, target_box, reduction="mean")
        return (pred_box - target_box).abs().mean()

    total_box_loss = pred_corners.new_zeros(())
    total_cls_loss = pred_corners.new_zeros(())
    total_no_obj_loss = pred_corners.new_zeros(())
    matched_count_total = 0
    n_matched_correct = 0

    for b in range(B):
        n_gt = int(gt_counts[b].item())
        if n_gt == 0:
            no_obj_t = torch.full(
                (N,), n_classes, dtype=torch.long, device=device,
            )
            total_no_obj_loss = total_no_obj_loss + _cls_loss(
                pred_cls[b], no_obj_t, reduction="sum",
            )
            continue
        pc = pred_aabb[b].unsqueeze(1)                  # (N, 1, 4)
        gc = gt_boxes_xyxy[b, :n_gt].unsqueeze(0)       # (1, n_gt, 4)
        # Hungarian cost stays on L1-style for matching (GIoU as a cost
        # matrix would require a per-pair compute that's slower without
        # measurable matching difference; keep matching cheap, swap
        # only the *loss* on matched pairs).
        box_cost = (pc - gc).abs().mean(dim=-1)         # (N, n_gt)
        gt_cls_b = gt_classes[b, :n_gt]
        cls_cost = -pred_probs[b][:, gt_cls_b]
        cost = (lam_aabb * box_cost + lam_cls * cls_cost
                ).detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        m_pred = torch.tensor(rows, device=device, dtype=torch.long)
        m_gt = torch.tensor(cols, device=device, dtype=torch.long)
        total_box_loss = total_box_loss + _box_loss(
            pred_aabb[b][m_pred], gt_boxes_xyxy[b][m_gt],
        )
        m_cls_t = gt_classes[b][m_gt]
        total_cls_loss = total_cls_loss + _cls_loss(
            pred_cls[b][m_pred], m_cls_t,
        )
        n_matched_correct += int(
            (pred_cls[b][m_pred].argmax(-1) == m_cls_t).sum().item()
        )
        matched_count_total += int(m_pred.numel())
        all_idx = torch.arange(N, device=device)
        unmatched = torch.ones(N, dtype=torch.bool, device=device)
        unmatched[m_pred] = False
        if unmatched.any():
            uidx = all_idx[unmatched]
            no_obj_t = torch.full(
                (uidx.numel(),), n_classes,
                dtype=torch.long, device=device,
            )
            total_no_obj_loss = total_no_obj_loss + _cls_loss(
                pred_cls[b][uidx], no_obj_t,
            )

    loss = (
        lam_aabb * total_box_loss
        + lam_cls * total_cls_loss
        + lam_no_obj * total_no_obj_loss
    ) / max(1, B)
    cls_acc = (n_matched_correct / max(1, matched_count_total)
                if matched_count_total > 0 else 0.0)
    return loss, cls_acc


# ─── Stage A-3 loss components ────────────────────────────────────────


def focal_loss_ce(
    logits: torch.Tensor, targets: torch.Tensor,
    *, gamma: float = 2.0, alpha: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class focal loss (Lin et al. 2017, Eq. 5) on softmax logits.

    Drop-in for ``F.cross_entropy``; same input shape ``(N, C)`` +
    ``(N,)`` target. For ``gamma=0`` and ``alpha=None`` reduces to
    standard cross-entropy. Default ``gamma=2.0`` is the original
    Focal-loss recommendation. ``alpha`` is an optional per-class
    weight vector ``(C,)`` for class imbalance; ``None`` = uniform.

    Implementation: compute log-softmax once, gather the target log-
    prob, multiply by ``(1 − p_t) ** gamma``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_p = log_probs.gather(
        -1, targets.unsqueeze(-1),
    ).squeeze(-1)
    pt = target_log_p.exp().clamp(min=1e-9, max=1.0)
    focal_weight = (1.0 - pt) ** gamma
    loss = -focal_weight * target_log_p
    if alpha is not None:
        loss = loss * alpha.to(loss.device)[targets]
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def giou_loss_xyxy(
    pred: torch.Tensor, target: torch.Tensor,
    *, reduction: str = "mean",
) -> torch.Tensor:
    """Generalised IoU loss (Rezatofighi et al. 2019) on
    axis-aligned (x0, y0, x1, y1) boxes.

    Args:
        pred:   (N, 4) predicted boxes.
        target: (N, 4) ground-truth boxes.

    Returns:
        Scalar loss = mean of (1 − GIoU). Higher GIoU = better; the
        loss is non-negative and zero only when pred == target.

    Robust to degenerate (zero-area) boxes via standard area clamps.
    """
    if pred.shape != target.shape or pred.shape[-1] != 4:
        raise ValueError(
            f"giou_loss expects matching (N, 4) shapes; got "
            f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    # Per-box areas.
    pa = (pred[..., 2] - pred[..., 0]).clamp_min(0) \
       * (pred[..., 3] - pred[..., 1]).clamp_min(0)
    ta = (target[..., 2] - target[..., 0]).clamp_min(0) \
       * (target[..., 3] - target[..., 1]).clamp_min(0)
    # Intersection.
    lt = torch.maximum(pred[..., :2], target[..., :2])
    rb = torch.minimum(pred[..., 2:], target[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = pa + ta - inter
    iou = inter / union.clamp_min(1e-9)
    # Enclosing box.
    elt = torch.minimum(pred[..., :2], target[..., :2])
    erb = torch.maximum(pred[..., 2:], target[..., 2:])
    ewh = (erb - elt).clamp_min(0)
    enc_area = ewh[..., 0] * ewh[..., 1]
    giou = iou - (enc_area - union) / enc_area.clamp_min(1e-9)
    loss = 1.0 - giou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ─── Joint Hungarian for the combined model ──────────────────────────


def combined_set_loss(
    out: dict, gt_boxes_xyxy: torch.Tensor,
    gt_classes: torch.Tensor, gt_counts: torch.Tensor,
    n_classes: int,
    *,
    cls_loss_kind: str = "ce",
    box_loss_kind: str = "l1",
    lam_no_obj: float = 0.5,
    gate_curriculum: bool = False,
    lam_gate_neg_override: float | None = None,
    lam_gate_match_cost_override: float | None = None,
    gate_loss_kind: str = "bce",
    gate_focal_gamma: float = 2.0,
):
    """Box loss (corners vs corners) + Circle loss (AABB vs box).  Both
    branches see the SAME GT box set; both can match independently.

    ``lam_no_obj`` (default 0.5, pre-2026-05-18 byte-identical) weights
    the unmatched-query "predict no-object" CE term. Stage D-2b
    diagnostic showed that at the default 0.5, queries fail to learn
    to suppress on VOC; raising to 2.0–5.0 puts more gradient pressure
    on the no-object slot.

    2026-05-19 Stage D-3: when ``out`` contains ``box_gates`` (the model
    was built with ``query_head_kind="nodelet"``), dispatch to
    :func:`hungarian_set_loss_gated` instead of the legacy path. The
    legacy and gated paths return the same ``(loss, accs)`` interface so
    the trainer is unchanged.

    Stage D-3-bis (2026-05-18): ``lam_gate_neg_override`` (None default)
    passes through to ``hungarian_set_loss_gated.lam_gate_neg``. None
    keeps the auto-balance recipe; an explicit float (e.g. 1.0) makes
    the suppression-side gradient ~5× larger in aggregate (D-3
    diagnostic showed auto-balance under-suppresses on VOC).
    """
    B, M = gt_boxes_xyxy.shape[:2]
    gt_corners = gt_corners_from_box(gt_boxes_xyxy.view(-1, 4)).view(
        B, M, 4, 2,
    )

    use_gated = "box_gates" in out or "circle_gates" in out
    if use_gated:
        from .nodelet_head import hungarian_set_loss_gated

    bx_loss, box_cls_acc = (None, 0.0)
    if out["box_corners"].shape[1] > 0:
        if use_gated:
            gated_kwargs: dict = dict(
                n_classes=n_classes,
                cls_loss_kind=cls_loss_kind,
                box_loss_kind=box_loss_kind,
                gate_curriculum=gate_curriculum,
                lam_gate_neg=lam_gate_neg_override,
                gate_loss_kind=gate_loss_kind,
                gate_focal_gamma=gate_focal_gamma,
            )
            if lam_gate_match_cost_override is not None:
                gated_kwargs["lam_gate_match_cost"] = lam_gate_match_cost_override
            bx_loss, box_cls_acc, _ = hungarian_set_loss_gated(
                out["box_corners"], out["box_cls"], out["box_gates"],
                gt_corners, gt_classes, gt_counts,
                **gated_kwargs,
            )
        else:
            bx_loss, box_cls_acc, _ = hungarian_set_loss(
                out["box_corners"], out["box_cls"],
                gt_corners, gt_classes, gt_counts,
                n_classes=n_classes,
                cls_loss_kind=cls_loss_kind,
                box_loss_kind=box_loss_kind,
                lam_no_obj=lam_no_obj,
            )

    circ_loss, circ_cls_acc = (None, 0.0)
    if out["circle_corners"].shape[1] > 0:
        if use_gated:
            # Stage D-3 v1: nodelet head + circle queries combination
            # is not yet wired. The fix is a gated circle-vs-box loss
            # mirroring hungarian_set_loss_gated; deferred until a
            # use case needs it (VOC uses box queries only).
            raise NotImplementedError(
                "Stage D-3 nodelet head + circle queries: not implemented. "
                "Use n_circle_queries=0 with query_head_kind='nodelet'."
            )
        circ_loss, circ_cls_acc = hungarian_set_loss_circle_vs_box(
            out["circle_corners"], out["circle_cls"],
            gt_boxes_xyxy, gt_classes, gt_counts,
            n_classes=n_classes,
            cls_loss_kind=cls_loss_kind,
            box_loss_kind=box_loss_kind,
            lam_no_obj=lam_no_obj,
        )

    if bx_loss is None and circ_loss is None:
        raise ValueError("no queries in model")
    if bx_loss is None:
        return circ_loss, dict(box_cls_acc=0.0, circ_cls_acc=circ_cls_acc)
    if circ_loss is None:
        return bx_loss, dict(box_cls_acc=box_cls_acc, circ_cls_acc=0.0)
    return bx_loss + circ_loss, dict(
        box_cls_acc=box_cls_acc, circ_cls_acc=circ_cls_acc,
    )


# ─── Detection metrics (IoU, AP@0.5, mAP@0.5:0.95) ──────────────────


@torch.no_grad()
def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for (N,4) vs (M,4) axis-aligned boxes (x0,y0,x1,y1).

    Returns (N, M) IoU matrix.  Handles zero-area degenerates by
    treating them as IoU=0.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0) \
           * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0) \
           * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0)
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp_min(1e-9)


@torch.no_grad()
def _aabb_from_corners(corners: torch.Tensor) -> torch.Tensor:
    """(B, M, k, 2) corners → (B, M, 4) AABB (x0, y0, x1, y1)."""
    return torch.stack([
        corners[..., 0].min(dim=-1).values,
        corners[..., 1].min(dim=-1).values,
        corners[..., 0].max(dim=-1).values,
        corners[..., 1].max(dim=-1).values,
    ], dim=-1)


@torch.no_grad()
def compute_detection_metrics(
    model, X: torch.Tensor, boxes: torch.Tensor,
    classes: torch.Tensor, counts: torch.Tensor,
    n_classes: int = 10, batch_size: int = 32,
) -> dict:
    """Per-image score-then-IoU mAP, computed at IoU@0.5 and the COCO
    mAP@0.5:0.95 ladder.

    For each image: take the model's predicted queries (boxes from box
    queries, AABBs derived from circle queries), score them by softmax
    max-class prob, run NMS-free greedy IoU matching (each prediction
    consumes at most one GT), and accumulate per-class TP/FP at the
    listed IoU thresholds. Then per-class AP via the standard
    precision-recall integration, and mAP as the arithmetic mean of
    per-class APs.

    Returns:
        {
          'mAP_50': float,
          'mAP_50_95': float,        # COCO ladder (10 thresholds)
          'mean_iou_matched': float, # mean IoU over score-thresholded matches
          'n_preds_used': int,
          'n_gts_total': int,
        }
    """
    model.eval()
    iou_levels = [0.5 + 0.05 * i for i in range(10)]  # 0.50..0.95
    # 2026-05-22 Phase 7: X may now be a CPU tensor (lazy-loader path
    # for VOC at 320+ px).  Take the device from the model so the
    # per-batch slices below land on the right device.
    device = next(model.parameters()).device
    # Accumulate (score, is_tp_at_iou_threshold) per class, per IoU level.
    per_class_records: dict[int, list[tuple[float, list[bool]]]] = {
        c: [] for c in range(n_classes)
    }
    n_gt_per_class = [0] * n_classes
    iou_sum_matched = 0.0
    n_matched = 0
    for s in range(0, X.shape[0], batch_size):
        e = min(s + batch_size, X.shape[0])
        # Lazy transfer — no-op if X is already on `device`.
        xb = X[s:e].to(device, non_blocking=True)
        pred = model(xb)
        if isinstance(pred, dict):
            # RicciHyMeYOLOMulti returns dict with box_corners, box_cls,
            # circle_corners, circle_cls (when both query types present).
            # Stage D-3 (2026-05-19): when the model has the nodelet
            # head, box_gates / circle_gates are emitted; multiply
            # per-query score by gate so unsuppressed queries rank
            # low at eval time (mAP is computed over ALL queries by
            # default — gates are the eval-time suppression signal).
            pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
            if "box_corners" in pred and pred["box_corners"].numel() > 0:
                pieces.append((
                    _aabb_from_corners(pred["box_corners"]),
                    pred["box_cls"],
                    pred.get("box_gates"),
                ))
            if "circle_corners" in pred and pred["circle_corners"].numel() > 0:
                pieces.append((
                    _aabb_from_corners(pred["circle_corners"]),
                    pred["circle_cls"],
                    pred.get("circle_gates"),
                ))
            if not pieces:
                continue
            pred_boxes = torch.cat([p[0] for p in pieces], dim=1)
            pred_cls = torch.cat([p[1] for p in pieces], dim=1)
            # Concatenate gates (if any). If a piece has no gates
            # (legacy head), substitute ones so the multiplicative
            # combination leaves its score unchanged.
            gate_parts = []
            for box_part, cls_part, gate_part in pieces:
                if gate_part is not None:
                    gate_parts.append(gate_part)
                else:
                    gate_parts.append(box_part.new_ones(box_part.shape[:2]))
            pred_gates = torch.cat(gate_parts, dim=1)  # (B, Q)
        else:
            corners, cls_logits = pred
            pred_boxes = _aabb_from_corners(corners)
            pred_cls = cls_logits
            pred_gates = pred_boxes.new_ones(pred_boxes.shape[:2])
        # pred_boxes: (B, Q, 4); pred_cls: (B, Q, n_classes [+1]).
        # For the legacy Hungarian head, drop the no-object slot
        # before scoring (it's the (n_classes)-th index). For the
        # nodelet head, n_cls_logits == n_classes already.
        probs = F.softmax(pred_cls, dim=-1)
        n_cls_logits = pred_cls.shape[-1]
        if n_cls_logits == n_classes + 1:
            obj_probs = probs[..., :n_classes]
        else:
            obj_probs = probs                   # nodelet: already n_classes
        # Per-prediction best-class score
        best_score, best_class = obj_probs.max(dim=-1)  # (B, Q)
        # Multiply by gate (gates default to 1 for legacy heads).
        best_score = best_score * pred_gates
        for bi in range(xb.shape[0]):
            gt_n = int(counts[s + bi].item())
            if gt_n == 0:
                continue
            gt_boxes_i = boxes[s + bi, :gt_n]                # (gt_n, 4)
            gt_classes_i = classes[s + bi, :gt_n].tolist()   # list[int]
            for c in gt_classes_i:
                if 0 <= c < n_classes:
                    n_gt_per_class[c] += 1
            pred_boxes_i = pred_boxes[bi]                    # (Q, 4)
            pred_scores_i = best_score[bi]                   # (Q,)
            pred_class_i = best_class[bi].tolist()           # list[int]
            # Score-descending order
            order = pred_scores_i.argsort(descending=True).tolist()
            iou_mat = _box_iou(pred_boxes_i, gt_boxes_i)     # (Q, gt_n)
            # COCO/PASCAL-VOC greedy matching: each GT may be consumed
            # AT MOST ONCE per IoU threshold, by the highest-scoring
            # prediction that hits it above that threshold. Prior code
            # never marked GTs consumed, so multiple predictions all
            # got credited as TP against the same GT — recall would
            # then exceed 1 and AP > 1 (see seed-3 +ricci-mod = 1.017
            # in the 2026-05-13 backfill). Bookkeeping is per IoU
            # level because a GT may be hit by one prediction at
            # IoU=0.6 and a different one at IoU=0.8.
            gt_consumed_per_level: list[list[bool]] = [
                [False] * gt_n for _ in iou_levels
            ]
            for q in order:
                cls = pred_class_i[q]
                if cls < 0 or cls >= n_classes:
                    continue
                score = float(pred_scores_i[q].item())
                is_tp_per_level = [False] * len(iou_levels)
                gt_mask = [gc == cls for gc in gt_classes_i]
                iou_row = iou_mat[q].tolist() if any(gt_mask) else None
                # For the diagnostic mean-IoU-of-matched line we
                # keep the un-consumed-best-IoU semantics so this
                # number stays comparable to prior reports.
                if iou_row is not None:
                    best_iou_unconsumed = max(
                        (iou_row[gi] for gi, m in enumerate(gt_mask) if m),
                        default=0.0,
                    )
                    if best_iou_unconsumed > 0:
                        iou_sum_matched += best_iou_unconsumed
                        n_matched += 1
                    # Per IoU level: pick the best unconsumed GT of
                    # matching class above threshold; mark it consumed.
                    for li, thr in enumerate(iou_levels):
                        best_gi = -1
                        best_iou = thr  # must strictly exceed threshold
                        for gi, m in enumerate(gt_mask):
                            if not m or gt_consumed_per_level[li][gi]:
                                continue
                            if iou_row[gi] >= best_iou:
                                best_iou = iou_row[gi]
                                best_gi = gi
                        if best_gi >= 0:
                            is_tp_per_level[li] = True
                            gt_consumed_per_level[li][best_gi] = True
                per_class_records[cls].append((score, is_tp_per_level))

    # Per-class AP at each IoU level via the standard PASCAL VOC
    # all-points integration.
    def _ap_at_level(records: list[tuple[float, list[bool]]],
                     n_gt: int, level_idx: int) -> float:
        if n_gt == 0:
            return float("nan")
        if not records:
            return 0.0
        recs = sorted(records, key=lambda r: -r[0])
        tp = [int(r[1][level_idx]) for r in recs]
        fp = [1 - t for t in tp]
        tp_cum = 0
        fp_cum = 0
        precisions: list[float] = []
        recalls: list[float] = []
        for t, f in zip(tp, fp):
            tp_cum += t
            fp_cum += f
            precisions.append(tp_cum / max(1, tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)
        # All-points integration: at each unique recall, take max
        # precision to the right of that recall.
        if not precisions:
            return 0.0
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        ap = 0.0
        prev_r = 0.0
        for p, r in zip(precisions, recalls):
            ap += p * (r - prev_r)
            prev_r = r
        return ap

    aps_50 = []
    aps_5095 = []
    for c in range(n_classes):
        if n_gt_per_class[c] == 0:
            continue
        ap_50 = _ap_at_level(per_class_records[c], n_gt_per_class[c], 0)
        ap_levels = [
            _ap_at_level(per_class_records[c], n_gt_per_class[c], li)
            for li in range(len(iou_levels))
        ]
        aps_50.append(ap_50)
        aps_5095.append(sum(ap_levels) / len(ap_levels))

    mAP_50 = sum(aps_50) / max(1, len(aps_50)) if aps_50 else 0.0
    mAP_5095 = sum(aps_5095) / max(1, len(aps_5095)) if aps_5095 else 0.0
    mean_iou = iou_sum_matched / max(1, n_matched)
    return dict(
        mAP_50=mAP_50,
        mAP_50_95=mAP_5095,
        mean_iou_matched=mean_iou,
        n_preds_used=sum(len(v) for v in per_class_records.values()),
        n_gts_total=sum(n_gt_per_class),
    )


# ─── Phase 1 smoke trainer ──────────────────────────────────────────


def train_one_config(
    label: str, model, X, boxes, classes, counts,
    epochs: int, lr: float, device: torch.device,
    batch_size: int = 32,
    *,
    schedule: str = "constant",
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
    weight_decay: float = 0.0,
    cls_loss_kind: str = "ce",
    box_loss_kind: str = "l1",
    lam_no_obj: float = 0.5,
    lam_gate_neg_override: float | None = None,
    lam_gate_match_cost_override: float | None = None,
    gate_loss_kind: str = "bce",
    gate_focal_gamma: float = 2.0,
) -> dict:
    """Minibatched trainer; epoch = one full pass over the dataset.

    Optional LR schedule (Stage A-2 from the 2026-05-16 YOLO-parity
    ladder):

    * ``schedule = "constant"`` (default) — `lr` constant across all
      epochs. Pre-2026-05-16 byte-identical behaviour.
    * ``schedule = "cosine"`` — linear warmup from 0 → `lr` over
      ``warmup_epochs`` epochs, then cosine anneal from `lr` to
      `min_lr_ratio * lr` over the remaining epochs.

    The schedule is applied per epoch via ``param_group['lr']``
    rewrite at epoch start. Existing callers get the constant
    behaviour without changes.
    """
    model = model.to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    losses_per_epoch = []
    last_accs = dict(box_cls_acc=0.0, circ_cls_acc=0.0)
    n = X.shape[0]
    t0 = time.perf_counter()

    # Pre-compute the per-epoch LR schedule.
    if schedule == "constant":
        lr_per_epoch = [lr] * epochs
    elif schedule == "cosine":
        if warmup_epochs >= epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be < epochs "
                f"({epochs})"
            )
        lr_per_epoch = []
        for ep in range(epochs):
            if ep < warmup_epochs:
                # Linear warmup; ep=0 gives lr * 1/warmup, not 0,
                # to keep the very first step from being a zero-LR
                # waste (Adam needs at least one step to populate
                # state before the cosine phase).
                lr_per_epoch.append(lr * (ep + 1) / warmup_epochs)
            else:
                t = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
                # cos(0)=1 → lr*1; cos(π)=-1 → lr*min_lr_ratio
                cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
                lr_per_epoch.append(
                    lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos_val)
                )
    else:
        raise ValueError(
            f"unknown schedule {schedule!r}; expected "
            f"'constant' or 'cosine'"
        )

    for ep in range(epochs):
        # Apply schedule for this epoch.
        for pg in opt.param_groups:
            pg["lr"] = lr_per_epoch[ep]
        model.train()
        # Permute indices for the epoch.  Permutation lives on CPU so
        # it indexes X (which may be on CPU under the Phase 7 lazy-
        # loader path).  For Cluttered MNIST X is on GPU and the
        # `.to(device)` below is a no-op.
        perm = torch.randperm(n)
        ep_losses = []
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            # Lazy GPU transfer for the image batch only.  boxes /
            # classes / counts are small enough to stay on GPU; index
            # them with idx moved to device.
            xb = X[idx].to(device, non_blocking=True)
            idx_d = idx.to(device, non_blocking=True)
            bb = boxes[idx_d]
            cb = classes[idx_d]
            kb = counts[idx_d]
            pred = model(xb)
            # Infer n_classes from the model's class-head width
            # (out has shape (B, N, n_classes + 1)). Replaces the
            # pre-2026-05-18 hardcoded n_classes=10, which routed the
            # no-object CE target to logit index 10 — for VOC's 20
            # classes that meant the "diningtable" slot was being
            # trained as the no-object catcher, and the real
            # no-object slot at index 20 was never supervised.
            if isinstance(pred, dict):
                cls_dim = pred["box_cls"].shape[-1] \
                    if pred["box_cls"].shape[1] > 0 \
                    else pred["circle_cls"].shape[-1]
                n_cls_dyn = cls_dim - 1
                loss, accs = combined_set_loss(
                    pred, bb, cb, kb, n_classes=n_cls_dyn,
                    cls_loss_kind=cls_loss_kind,
                    box_loss_kind=box_loss_kind,
                    lam_no_obj=lam_no_obj,
                    lam_gate_neg_override=lam_gate_neg_override,
                    lam_gate_match_cost_override=lam_gate_match_cost_override,
                    gate_loss_kind=gate_loss_kind,
                    gate_focal_gamma=gate_focal_gamma,
                )
            else:
                B_, M_ = bb.shape[:2]
                gt_corners = gt_corners_from_box(bb.view(-1, 4)).view(
                    B_, M_, 4, 2,
                )
                corners, cls_logits = pred
                n_cls_dyn = cls_logits.shape[-1] - 1
                loss, cls_acc, _ = hungarian_set_loss(
                    corners, cls_logits, gt_corners, cb, kb,
                    n_classes=n_cls_dyn,
                    cls_loss_kind=cls_loss_kind,
                    box_loss_kind=box_loss_kind,
                    lam_no_obj=lam_no_obj,
                )
                accs = dict(box_cls_acc=cls_acc, circ_cls_acc=0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_losses.append(float(loss.detach()))
            last_accs = accs
        losses_per_epoch.append(sum(ep_losses) / max(1, len(ep_losses)))
    wall = time.perf_counter() - t0
    drop_pct = (losses_per_epoch[0] - losses_per_epoch[-1]) / max(
        1e-9, losses_per_epoch[0],
    ) * 100.0
    # Compute detection metrics (mAP@0.5, mAP@0.5:0.95, mean matched IoU)
    # on the same set used for training. This is "training mAP" — a
    # step up from class-accuracy but not held-out generalisation.
    # Infer n_classes from the model's class-head width (same fix as
    # the training loop above).
    with torch.no_grad():
        _probe = model(X[:1].to(device, non_blocking=True))
        if isinstance(_probe, dict):
            _cls_dim = _probe["box_cls"].shape[-1] \
                if _probe["box_cls"].shape[1] > 0 \
                else _probe["circle_cls"].shape[-1]
        else:
            _cls_dim = _probe[1].shape[-1]
    n_cls_dyn = _cls_dim - 1
    metrics = compute_detection_metrics(
        model, X, boxes, classes, counts,
        n_classes=n_cls_dyn, batch_size=batch_size,
    )
    print(f"  {label:<18s}  start={losses_per_epoch[0]:7.3f}  "
          f"end={losses_per_epoch[-1]:7.3f}  drop={drop_pct:5.1f}%  "
          f"wall={wall:5.1f}s  "
          f"box_acc={last_accs['box_cls_acc']:.2f}  "
          f"circ_acc={last_accs['circ_cls_acc']:.2f}  "
          f"mAP50={metrics['mAP_50']:.3f}  "
          f"mAP50:95={metrics['mAP_50_95']:.3f}  "
          f"mIoU={metrics['mean_iou_matched']:.3f}")
    return dict(label=label, losses=losses_per_epoch, wall=wall,
                accs=last_accs, det_metrics=metrics)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jsonl-out", default=None,
                    help="Optional path to write per-config results as jsonl.")
    p.add_argument(
        "--configs", default=None,
        help="Comma-separated subset of {baseline,boxes-only,circles-only,"
             "boxes+circles,+ricci-mod,+kcycle,+ricci+kcycle}; default = all seven.",
    )
    p.add_argument(
        "--ricci-scale", type=float, default=1.0,
        help="Scalar multiplier on the 3 Ricci features before the class/offset "
             "head. Only affects +ricci-mod and +ricci+kcycle configs. Default 1.0 "
             "= byte-identical to pre-2026-05-16 behaviour. Used by the "
             "2026-05-16 ricci-scale sweep.",
    )
    # 2026-05-16 evening: defaults flipped after Stage A-1 and Stage A-2
    # delivered overwhelming paired wins (Δ +0.124 and +0.118 respectively,
    # both z > 4, both 5/5 win-rate on 5 seeds). The previous "opt-in via
    # flag" stance is now obsolete; `--warm-start` and `--schedule cosine
    # --warmup-epochs 10` are the canonical protocol for any future
    # HyMeYOLO experiment at this scale. Pass `--no-warm-start` /
    # `--schedule constant` / `--warmup-epochs 0` to explicitly opt out
    # (used by the ricci-scale sweep harness for protocol parity with
    # the pre-2026-05-16 baseline measurement).
    p.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction, default=True,
        help="Saliency-driven warm-start of query corners (Stage A-1 of the "
             "2026-05-16 YOLO-parity ladder). ON BY DEFAULT as of "
             "2026-05-16 evening; pass `--no-warm-start` to disable. See "
             "docs/plans/2026-05-16-hymeyolo-warmstart-query-init/ and "
             "reports/2026-05-16-hymeyolo-warmstart-5seed.md (paired Δ "
             "+0.124, z=+4.68, 5/5 wins).",
    )
    p.add_argument(
        "--warmstart-bootstrap-n", type=int, default=128,
        help="Bootstrap sample size used by --warm-start. Smaller = noisier "
             "saliency map, larger = more bootstrap memory at init time "
             "(but still cheap). Default 128.",
    )
    p.add_argument(
        "--schedule", choices=["constant", "cosine"], default="cosine",
        help="LR schedule (Stage A-2 of the 2026-05-16 YOLO-parity ladder). "
             "DEFAULT 'cosine' as of 2026-05-16 evening; pass "
             "'--schedule constant' to opt out. Cosine does linear warmup "
             "for --warmup-epochs then anneals to lr * --min-lr-ratio. "
             "Stage A-2 evidence: paired Δ +0.118, z=+14.01, 5/5 wins.",
    )
    p.add_argument(
        "--warmup-epochs", type=int, default=10,
        help="Linear-warmup epochs at the start of training when "
             "--schedule=cosine. DEFAULT 10 (was 0 pre-2026-05-16 evening), "
             "which is ~10%% of typical --epochs=100. No-op when "
             "--schedule=constant.",
    )
    p.add_argument(
        "--min-lr-ratio", type=float, default=0.01,
        help="Final LR as a fraction of --lr at the end of cosine annealing. "
             "Default 0.01 (anneal to 1%% of the peak LR).",
    )
    # ─── Stage A-3 levers (2026-05-16 evening). All four default off
    # so existing experiments are unaffected. The ladder orchestrator's
    # `a3` stage turns them all on simultaneously.
    p.add_argument(
        "--use-layernorm", action="store_true",
        help="Add a LayerNorm before the +ricci-mod class-head linear. "
             "Stage A-3 lever 1/4. Off by default; on for ladder stage a3.",
    )
    p.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="Adam weight_decay. Stage A-3 lever 2/4. Recommended 1e-4.",
    )
    p.add_argument(
        "--cls-loss", choices=["ce", "focal"], default="ce",
        help="Classification loss kernel. Stage A-3 lever 3/4. 'focal' uses "
             "γ=2 multi-class focal loss (Lin et al. 2017) to down-weight the "
             "many easy no-object queries.",
    )
    p.add_argument(
        "--box-loss", choices=["l1", "giou"], default="l1",
        help="Box-regression loss kernel. Stage A-3 lever 4/4. 'giou' uses "
             "1 − GIoU on AABBs derived from matched corners (Rezatofighi "
             "et al. 2019) — IoU-aligned with the eval mAP@0.5 metric.",
    )
    p.add_argument(
        "--lam-no-obj", type=float, default=0.5,
        help="Weight on the no-object CE term for unmatched queries. "
             "Default 0.5 is byte-identical to pre-2026-05-18 behaviour. "
             "Stage D-2b probe: raise to 2.0 or 5.0 to apply more "
             "gradient pressure on queries to learn to suppress.",
    )
    p.add_argument(
        "--backbone", choices=["tiny", "resnet", "hsikan", "resnet18_imagenet"], default="tiny",
        help="Backbone architecture (Stage B of the YOLO-parity ladder). "
             "'tiny' = 3-conv TinyBackbone (pre-Stage-B default; byte-"
             "identical for backward compat). 'resnet' = residual-block "
             "backbone (~107k params at d_hidden=32). 'hsikan' = ResNet "
             "topology with Catmull-Rom basis-function activations in "
             "place of ReLU (KAN primitive from the HSiKAN family). "
             "Affects +ricci-mod and +ricci+kcycle; not the bare baseline.",
    )
    p.add_argument(
        "--fpn", choices=["none", "2level"], default="none",
        help="Stage C FPN: multi-scale feature heads. 'none' (default) "
             "= single-scale forward, byte-identical to Stage B. "
             "'2level' = 2-level FPN at /4 + /8 with lateral 1x1 + "
             "top-down upsample + 3x3 smooth; multi-scale bilinear "
             "sampling at query corners with concat→project. Requires "
             "--backbone resnet or hsikan (TinyBackbone has no /4 tap).",
    )
    p.add_argument(
        "--save-checkpoint", default=None,
        help="If set, save trained model state_dicts (one per config) "
             "to this directory after training. Filename pattern: "
             "<config>_<seed>.pt. Used by the demo GUI "
             "(demo_hymeyolo_tk.py) to load a pre-trained model "
             "without re-training.",
    )
    args = p.parse_args()
    device = torch.device(args.device)
    print(f"\nPhase 1 smoke: n_images={args.n_images}  epochs={args.epochs}  "
          f"lr={args.lr}  device={device}")

    # Cluttered MNIST in Hungarian format.  canvas=64 to fit up to 3 digits.
    Xn, boxes_n, classes_n, counts_n = make_cluttered_mnist_hungarian_format(
        n=args.n_images, canvas=64, max_objects=3, seed=args.seed, rgb=True,
    )
    X = torch.from_numpy(Xn).to(device)
    boxes = torch.from_numpy(boxes_n).to(device)
    classes = torch.from_numpy(classes_n).to(device)
    counts = torch.from_numpy(counts_n).to(device)
    print(f"  dataset: X={tuple(X.shape)}  counts.mean={counts.float().mean():.2f}")

    configs = [
        ("baseline",      HyMeYOLOMulti(n_queries=6, n_classes=10, d_hidden=32)),
        ("boxes-only",    RicciHyMeYOLOMulti(n_box_queries=6, n_circle_queries=0,
                                              ricci_modulation=False)),
        ("circles-only",  RicciHyMeYOLOMulti(n_box_queries=0, n_circle_queries=4,
                                              ricci_modulation=False)),
        ("boxes+circles", RicciHyMeYOLOMulti(n_box_queries=4, n_circle_queries=2,
                                              ricci_modulation=False)),
        ("+ricci-mod",    RicciHyMeYOLOMulti(n_box_queries=4, n_circle_queries=2,
                                              ricci_modulation=True,
                                              ricci_scale=args.ricci_scale,
                                              use_layernorm=args.use_layernorm,
                                              backbone=args.backbone,
                                              fpn=args.fpn)),
        # 2026-05-11 new: K-cycle signed micro-graph head.  Each
        # query's K corners form a signed sub-graph; α-routed
        # aggregation over all C(K,k) sub-cycles (k=2..K).  Composes
        # with both box (K=4, 3 arities) and circle (K=8, 7 arities)
        # query types.
        ("+kcycle",       KCycleHyMeYOLOMulti(n_box_queries=4, n_circle_queries=2,
                                                box_k=4, circle_k=8, d_hidden=32)),
        # 2026-05-13 new: Ricci × k-cycles curvature mixed.
        # Unifies the signed k-cycle σ-product (structural curvature)
        # with the geometric Ricci scalar signature (κ, mean_cos_θ,
        # edge_var). Both signals feed offset prediction AND
        # classification — closes the +kcycle localization bug where
        # the signed-cycle aggregator was cls-only. See
        # reports/2026-05-13-hymeyolo-kcycle-localization-bug.md and
        # signedkan_wip/tests/test_ricci_kcycle.py for the regression
        # gate.
        ("+ricci+kcycle", RicciKCycleHyMeYOLOMulti(n_box_queries=4, n_circle_queries=2,
                                                    box_k=4, circle_k=8, d_hidden=32,
                                                    ricci_modulation=True,
                                                    ricci_scale=args.ricci_scale)),
    ]
    if args.configs:
        keep = set(args.configs.split(","))
        configs = [(name, m) for (name, m) in configs if name in keep]
        if not configs:
            raise SystemExit(
                f"--configs={args.configs!r} matches no known config; "
                f"valid: baseline, boxes-only, circles-only, "
                f"boxes+circles, +ricci-mod, +kcycle, +ricci+kcycle",
            )

    print(f"\n  {'config':<18s}  {'start':>7s}  {'end':>7s}  "
          f"{'drop':>5s}  {'wall':>5s}  box_acc circ_acc")
    print(f"  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}  -------  --------")

    # 2026-05-16: Stage-A-1 — saliency-driven warm-start of query
    # corners. Replaces the seed-dependent fixed-base + Gaussian-noise
    # init of box_corners / circle_corners with a deterministic
    # spatial-coverage init derived from a small bootstrap of the
    # training set. Only applied to configs that have query-corner
    # parameters (i.e., not the bare HyMeYOLOMulti baseline, which uses
    # `query_corners` differently). Off by default; see
    # docs/plans/2026-05-16-hymeyolo-warmstart-query-init/.
    if args.warm_start:
        from .hymeyolo_warmstart import warmstart_query_corners
        boot_n = min(args.warmstart_bootstrap_n, X.shape[0])
        X_boot = X[:boot_n]
        for label, model in configs:
            if not (hasattr(model, "box_corners")
                    or hasattr(model, "circle_corners")):
                continue
            warmstart_query_corners(
                model, X_boot, seed=args.seed,
            )
        print(f"  [warm-start] applied to "
              f"{sum(1 for (_, m) in configs if hasattr(m, 'box_corners') or hasattr(m, 'circle_corners'))} "
              f"configs (bootstrap n={boot_n}, seed={args.seed})")

    results = []
    for label, model in configs:
        torch.manual_seed(args.seed)
        results.append(
            train_one_config(
                label, model, X, boxes, classes, counts,
                args.epochs, args.lr, device,
                schedule=args.schedule,
                warmup_epochs=args.warmup_epochs,
                min_lr_ratio=args.min_lr_ratio,
                weight_decay=args.weight_decay,
                cls_loss_kind=args.cls_loss,
                box_loss_kind=args.box_loss,
                lam_no_obj=args.lam_no_obj,
            )
        )
        # Optional checkpoint persistence (Stage A-3 follow-up: lets
        # the Tk demo GUI load a pre-trained model without re-training).
        if args.save_checkpoint is not None:
            import os as _os
            _os.makedirs(args.save_checkpoint, exist_ok=True)
            # Strip "+" from labels (filesystem safety) — e.g. "+ricci-mod"
            # → "ricci-mod" in the filename.
            safe_label = label.lstrip("+").replace("/", "_")
            ckpt_path = _os.path.join(
                args.save_checkpoint,
                f"{safe_label}_seed{args.seed}.pt",
            )
            torch.save({
                "label": label,
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "ricci_scale": float(args.ricci_scale),
                "warm_start": bool(args.warm_start),
                "schedule": args.schedule,
                "warmup_epochs": int(args.warmup_epochs),
                "use_layernorm": bool(args.use_layernorm),
                "weight_decay": float(args.weight_decay),
                "cls_loss": args.cls_loss,
                "box_loss": args.box_loss,
                "backbone": args.backbone,
                "fpn": args.fpn,
                "state_dict": model.state_dict(),
                "model_class": type(model).__name__,
            }, ckpt_path)
            print(f"  [checkpoint] saved {label} → {ckpt_path}")

    print("\n=== Phase 1 acceptance ===")
    # Acceptance: every config's loss decreases by ≥ 30%.
    all_pass = True
    for r in results:
        decreased = r["losses"][0] - r["losses"][-1] > 0.30 * r["losses"][0]
        flag = "PASS" if decreased else "FAIL"
        r["acceptance_pass"] = bool(decreased)
        if not decreased:
            all_pass = False
        print(f"  {r['label']:<18s} : loss drop ≥ 30%? {flag}")
    print(f"\n  PHASE-1 SMOKE: {'PASS' if all_pass else 'FAIL'}")

    # Optional jsonl persistence — one line per config.
    if args.jsonl_out:
        import json
        with open(args.jsonl_out, "w") as fh:
            for r in results:
                det = r.get("det_metrics", {})
                rec = {
                    "label":       r["label"],
                    "n_images":    args.n_images,
                    "epochs":      args.epochs,
                    "lr":          args.lr,
                    "seed":        args.seed,
                    # 2026-05-16: persist the Ricci-feature scale so
                    # the sweep aggregator can group runs by it. Always
                    # written (defaults to 1.0); only meaningful for
                    # +ricci-mod and +ricci+kcycle rows.
                    "ricci_scale": args.ricci_scale,
                    # 2026-05-16: persist the warm-start flag + bootstrap
                    # size so the Stage-A-1 vs no-warm-start paired
                    # comparison can be aggregated cleanly.
                    "warm_start": bool(args.warm_start),
                    "warmstart_bootstrap_n":
                        (args.warmstart_bootstrap_n if args.warm_start else 0),
                    # 2026-05-16: persist the LR schedule for Stage A-2
                    # cosine-vs-constant comparison.
                    "schedule": args.schedule,
                    "warmup_epochs": args.warmup_epochs,
                    "min_lr_ratio": args.min_lr_ratio,
                    # 2026-05-16 evening: persist the four Stage A-3 levers
                    # so paired comparisons can be aggregated cleanly across
                    # ladder stages.
                    "use_layernorm": bool(args.use_layernorm),
                    "weight_decay": args.weight_decay,
                    "cls_loss": args.cls_loss,
                    "box_loss": args.box_loss,
                    # 2026-05-16 evening: persist the backbone kind for
                    # Stage B's paired comparison (tiny / resnet / hsikan).
                    "backbone": args.backbone,
                    # 2026-05-17: persist the FPN kind for Stage C's
                    # paired comparison (none / 2level).
                    "fpn": args.fpn,
                    "wall_s":      r["wall"],
                    "loss_start":  r["losses"][0],
                    "loss_end":    r["losses"][-1],
                    "loss_drop_pct": (r["losses"][0] - r["losses"][-1])
                                      / max(1e-9, r["losses"][0]) * 100.0,
                    "box_cls_acc": r["accs"]["box_cls_acc"],
                    "circ_cls_acc": r["accs"]["circ_cls_acc"],
                    # Detection metrics on the training set (no held-out
                    # split in current pipeline; comparable across configs).
                    "mAP_50":      det.get("mAP_50"),
                    "mAP_50_95":   det.get("mAP_50_95"),
                    "mean_iou_matched": det.get("mean_iou_matched"),
                    "n_preds_used":     det.get("n_preds_used"),
                    "n_gts_total":      det.get("n_gts_total"),
                    "acceptance_pass": r["acceptance_pass"],
                    "losses_per_epoch": r["losses"],
                }
                fh.write(json.dumps(rec) + "\n")
        print(f"\n  Results written to {args.jsonl_out}")


if __name__ == "__main__":
    main()
