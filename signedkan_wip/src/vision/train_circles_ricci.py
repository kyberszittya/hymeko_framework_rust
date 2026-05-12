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
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .cluttered_mnist import make_cluttered_mnist_hungarian_format
from .hymeyolo_circles_ricci import RicciHyMeYOLOMulti
from .hymeyolo_hungarian import HyMeYOLOMulti, hungarian_set_loss
from .hymeyolo_kcycle import KCycleHyMeYOLOMulti
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
            total_no_obj_loss = total_no_obj_loss + F.cross_entropy(
                pred_cls[b], no_obj_t, reduction="sum",
            )
            continue
        pc = pred_aabb[b].unsqueeze(1)                  # (N, 1, 4)
        gc = gt_boxes_xyxy[b, :n_gt].unsqueeze(0)       # (1, n_gt, 4)
        box_cost = (pc - gc).abs().mean(dim=-1)         # (N, n_gt)
        gt_cls_b = gt_classes[b, :n_gt]
        cls_cost = -pred_probs[b][:, gt_cls_b]
        cost = (lam_aabb * box_cost + lam_cls * cls_cost
                ).detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        m_pred = torch.tensor(rows, device=device, dtype=torch.long)
        m_gt = torch.tensor(cols, device=device, dtype=torch.long)
        total_box_loss = total_box_loss + (
            pred_aabb[b][m_pred] - gt_boxes_xyxy[b][m_gt]
        ).abs().mean()
        m_cls_t = gt_classes[b][m_gt]
        total_cls_loss = total_cls_loss + F.cross_entropy(
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
            total_no_obj_loss = total_no_obj_loss + F.cross_entropy(
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


# ─── Joint Hungarian for the combined model ──────────────────────────


def combined_set_loss(
    out: dict, gt_boxes_xyxy: torch.Tensor,
    gt_classes: torch.Tensor, gt_counts: torch.Tensor,
    n_classes: int,
):
    """Box loss (corners vs corners) + Circle loss (AABB vs box).  Both
    branches see the SAME GT box set; both can match independently.
    """
    B, M = gt_boxes_xyxy.shape[:2]
    gt_corners = gt_corners_from_box(gt_boxes_xyxy.view(-1, 4)).view(
        B, M, 4, 2,
    )

    box_loss, box_cls_acc = (None, 0.0)
    if out["box_corners"].shape[1] > 0:
        box_loss, box_cls_acc, _ = hungarian_set_loss(
            out["box_corners"], out["box_cls"],
            gt_corners, gt_classes, gt_counts,
            n_classes=n_classes,
        )

    circ_loss, circ_cls_acc = (None, 0.0)
    if out["circle_corners"].shape[1] > 0:
        circ_loss, circ_cls_acc = hungarian_set_loss_circle_vs_box(
            out["circle_corners"], out["circle_cls"],
            gt_boxes_xyxy, gt_classes, gt_counts,
            n_classes=n_classes,
        )

    if box_loss is None and circ_loss is None:
        raise ValueError("no queries in model")
    if box_loss is None:
        return circ_loss, dict(box_cls_acc=0.0, circ_cls_acc=circ_cls_acc)
    if circ_loss is None:
        return box_loss, dict(box_cls_acc=box_cls_acc, circ_cls_acc=0.0)
    return box_loss + circ_loss, dict(
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
    device = X.device
    # Accumulate (score, is_tp_at_iou_threshold) per class, per IoU level.
    per_class_records: dict[int, list[tuple[float, list[bool]]]] = {
        c: [] for c in range(n_classes)
    }
    n_gt_per_class = [0] * n_classes
    iou_sum_matched = 0.0
    n_matched = 0
    for s in range(0, X.shape[0], batch_size):
        e = min(s + batch_size, X.shape[0])
        xb = X[s:e]
        pred = model(xb)
        if isinstance(pred, dict):
            # RicciHyMeYOLOMulti returns dict with box_corners, box_cls,
            # circle_corners, circle_cls (when both query types present).
            pieces: list[tuple[torch.Tensor, torch.Tensor]] = []
            if "box_corners" in pred and pred["box_corners"].numel() > 0:
                pieces.append((_aabb_from_corners(pred["box_corners"]),
                                pred["box_cls"]))
            if "circle_corners" in pred and pred["circle_corners"].numel() > 0:
                pieces.append((_aabb_from_corners(pred["circle_corners"]),
                                pred["circle_cls"]))
            if not pieces:
                continue
            pred_boxes = torch.cat([p[0] for p in pieces], dim=1)
            pred_cls = torch.cat([p[1] for p in pieces], dim=1)
        else:
            corners, cls_logits = pred
            pred_boxes = _aabb_from_corners(corners)
            pred_cls = cls_logits
        # pred_boxes: (B, Q, 4); pred_cls: (B, Q, n_classes+1)
        # Slot "n_classes" is the no-object class; drop it for scoring.
        probs = F.softmax(pred_cls, dim=-1)
        obj_probs = probs[..., :n_classes]  # (B, Q, C)
        # Per-prediction best-class score
        best_score, best_class = obj_probs.max(dim=-1)  # (B, Q)
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
            # Greedy matching, one GT per prediction, per IoU level
            for q in order:
                cls = pred_class_i[q]
                if cls < 0 or cls >= n_classes:
                    continue
                score = float(pred_scores_i[q].item())
                is_tp_per_level = [False] * len(iou_levels)
                # Only GTs matching this class can be TP
                gt_mask = [gc == cls for gc in gt_classes_i]
                if any(gt_mask):
                    iou_row = iou_mat[q].tolist()
                    best_iou = 0.0
                    for gi, m in enumerate(gt_mask):
                        if m and iou_row[gi] > best_iou:
                            best_iou = iou_row[gi]
                    for li, thr in enumerate(iou_levels):
                        if best_iou >= thr:
                            is_tp_per_level[li] = True
                    if best_iou > 0:
                        iou_sum_matched += best_iou
                        n_matched += 1
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
) -> dict:
    """Minibatched trainer; epoch = one full pass over the dataset."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses_per_epoch = []
    last_accs = dict(box_cls_acc=0.0, circ_cls_acc=0.0)
    n = X.shape[0]
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        # Permute indices for the epoch.
        perm = torch.randperm(n, device=device)
        ep_losses = []
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = X[idx]
            bb = boxes[idx]
            cb = classes[idx]
            kb = counts[idx]
            pred = model(xb)
            if isinstance(pred, dict):
                loss, accs = combined_set_loss(
                    pred, bb, cb, kb, n_classes=10,
                )
            else:
                B_, M_ = bb.shape[:2]
                gt_corners = gt_corners_from_box(bb.view(-1, 4)).view(
                    B_, M_, 4, 2,
                )
                corners, cls_logits = pred
                loss, cls_acc, _ = hungarian_set_loss(
                    corners, cls_logits, gt_corners, cb, kb,
                    n_classes=10,
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
    metrics = compute_detection_metrics(
        model, X, boxes, classes, counts,
        n_classes=10, batch_size=batch_size,
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
             "boxes+circles,+ricci-mod,+kcycle}; default = all six.",
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
                                              ricci_modulation=True)),
        # 2026-05-11 new: K-cycle signed micro-graph head.  Each
        # query's K corners form a signed sub-graph; α-routed
        # aggregation over all C(K,k) sub-cycles (k=2..K).  Composes
        # with both box (K=4, 3 arities) and circle (K=8, 7 arities)
        # query types.
        ("+kcycle",       KCycleHyMeYOLOMulti(n_box_queries=4, n_circle_queries=2,
                                                box_k=4, circle_k=8, d_hidden=32)),
    ]
    if args.configs:
        keep = set(args.configs.split(","))
        configs = [(name, m) for (name, m) in configs if name in keep]
        if not configs:
            raise SystemExit(
                f"--configs={args.configs!r} matches no known config; "
                f"valid: baseline, boxes-only, circles-only, "
                f"boxes+circles, +ricci-mod, +kcycle",
            )

    print(f"\n  {'config':<18s}  {'start':>7s}  {'end':>7s}  "
          f"{'drop':>5s}  {'wall':>5s}  box_acc circ_acc")
    print(f"  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}  -------  --------")
    results = []
    for label, model in configs:
        torch.manual_seed(args.seed)
        results.append(
            train_one_config(label, model, X, boxes, classes, counts,
                             args.epochs, args.lr, device)
        )

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
