"""Training infrastructure for `RicciStimDetector` on Cluttered MNIST.

GömbSoma-Ricci-Stim Phase 8-bench. Provides:

  * `assign_anchors_to_gt` — IoU-based anchor-target assignment.
  * `detection_loss` — combined cross-entropy (cls) + smooth-L1 (bbox).
  * `train_one_seed` — full training loop with periodic eval.
  * `evaluate_map50` — simplified per-class mAP at IoU 0.5.

Decoding convention matches `RicciStimDetector.decode_boxes`:

    cx = anchor_cx + dx * anchor_size
    cy = anchor_cy + dy * anchor_size
    w  = anchor_size * exp(dw)
    h  = anchor_size * exp(dh)

GT boxes in the Cluttered MNIST dataset are `[x_min, y_min, x_max, y_max]`
in pixel coordinates; we convert to (cx, cy, w, h) at assignment time.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim-bench/.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------


def _iou_boxes_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU between rows of (n, 4) and (m, 4) in [x1,y1,x2,y2] format."""
    n = box_a.shape[0]
    m = box_b.shape[0]
    if n == 0 or m == 0:
        return torch.zeros(n, m)
    x1 = torch.maximum(box_a[:, 0:1], box_b[:, 0:1].T)
    y1 = torch.maximum(box_a[:, 1:2], box_b[:, 1:2].T)
    x2 = torch.minimum(box_a[:, 2:3], box_b[:, 2:3].T)
    y2 = torch.minimum(box_a[:, 3:4], box_b[:, 3:4].T)
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a.unsqueeze(-1) + area_b.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-6)


def _anchors_as_xyxy(
    anchor_positions: torch.Tensor, anchor_sizes: torch.Tensor,
) -> torch.Tensor:
    """Convert (n, 2) top-left positions + (n,) sizes to (n, 4) x1y1x2y2."""
    r = anchor_positions[:, 0].float()
    c = anchor_positions[:, 1].float()
    s = anchor_sizes.float()
    return torch.stack([c, r, c + s, r + s], dim=-1)


# ---------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------


@dataclass
class AnchorAssignment:
    """Per-anchor target assignment.

    Attributes
    ----------
    cls_targets : LongTensor[n_anchors]
        0 = background; 1..n_classes = digit class + 1.
    bbox_targets : FloatTensor[n_anchors, 4]
        (dx, dy, dw, dh) regression targets. Zeros for background anchors.
    positive_mask : BoolTensor[n_anchors]
        True for foreground-assigned anchors.
    """
    cls_targets: torch.Tensor
    bbox_targets: torch.Tensor
    positive_mask: torch.Tensor


def assign_anchors_to_gt(
    anchor_positions: torch.Tensor,
    anchor_sizes: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_pos: float = 0.3,
) -> AnchorAssignment:
    """For each anchor, find the highest-IoU ground-truth box. If IoU
    exceeds ``iou_pos``, mark anchor as foreground for that GT and
    compute regression offsets.

    Background anchors get cls_target = 0, bbox_target = 0.
    """
    n = anchor_positions.shape[0]
    device = anchor_positions.device
    if gt_boxes_xyxy.shape[0] == 0:
        # No objects → all anchors are background.
        return AnchorAssignment(
            cls_targets=torch.zeros(n, dtype=torch.long, device=device),
            bbox_targets=torch.zeros(n, 4, device=device),
            positive_mask=torch.zeros(n, dtype=torch.bool, device=device),
        )
    anchors_xyxy = _anchors_as_xyxy(anchor_positions, anchor_sizes)
    iou = _iou_boxes_xyxy(anchors_xyxy, gt_boxes_xyxy.to(device))  # (n, n_gt)
    best_iou, best_gt = iou.max(dim=1)
    positive = best_iou > iou_pos
    # cls_target: 0 for background; gt_labels[best_gt] + 1 for foreground.
    cls_targets = torch.zeros(n, dtype=torch.long, device=device)
    cls_targets[positive] = gt_labels[best_gt[positive]].long().to(device) + 1
    # bbox regression target: encode the assigned GT relative to the anchor.
    bbox_targets = torch.zeros(n, 4, device=device)
    if positive.any():
        # GT box → (cx, cy, w, h)
        gt = gt_boxes_xyxy[best_gt[positive]].float().to(device)
        gt_cx = (gt[:, 0] + gt[:, 2]) / 2
        gt_cy = (gt[:, 1] + gt[:, 3]) / 2
        gt_w = gt[:, 2] - gt[:, 0]
        gt_h = gt[:, 3] - gt[:, 1]
        # Anchor → (cx, cy, s)
        anchor_size = anchor_sizes[positive].float()
        anchor_r = anchor_positions[positive, 0].float()
        anchor_c = anchor_positions[positive, 1].float()
        anchor_cx = anchor_c + 0.5 * anchor_size
        anchor_cy = anchor_r + 0.5 * anchor_size
        # Offsets (matching decode_boxes formula).
        dx = (gt_cx - anchor_cx) / anchor_size.clamp(min=1e-6)
        dy = (gt_cy - anchor_cy) / anchor_size.clamp(min=1e-6)
        dw = torch.log((gt_w / anchor_size).clamp(min=1e-6))
        dh = torch.log((gt_h / anchor_size).clamp(min=1e-6))
        bbox_targets[positive] = torch.stack([dx, dy, dw, dh], dim=-1)
    return AnchorAssignment(
        cls_targets=cls_targets,
        bbox_targets=bbox_targets,
        positive_mask=positive,
    )


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


def detection_loss(
    output,          # DetectionOutput
    assignment: AnchorAssignment,
    bbox_weight: float = 1.0,
) -> dict:
    """Combined cls (cross-entropy) + bbox (smooth-L1) loss.

    Returns
    -------
    {"loss": Tensor, "cls": float, "bbox": float, "n_pos": int}
    """
    cls_loss = F.cross_entropy(output.cls_logits, assignment.cls_targets)
    if assignment.positive_mask.any():
        bbox_loss = F.smooth_l1_loss(
            output.bbox_offsets[assignment.positive_mask],
            assignment.bbox_targets[assignment.positive_mask],
        )
    else:
        bbox_loss = torch.tensor(0.0, device=output.cls_logits.device)
    loss = cls_loss + bbox_weight * bbox_loss
    return {
        "loss": loss,
        "cls": float(cls_loss.item()),
        "bbox": float(bbox_loss.item()),
        "n_pos": int(assignment.positive_mask.sum().item()),
    }


# ---------------------------------------------------------------------
# Greedy NMS
# ---------------------------------------------------------------------


def _greedy_nms(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.5,
    max_dets: int = 100,
) -> torch.Tensor:
    """Return LongTensor of kept indices."""
    n = boxes_xyxy.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    order = torch.argsort(scores, descending=True)
    kept = []
    while order.numel() > 0 and len(kept) < max_dets:
        i = order[0].item()
        kept.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _iou_boxes_xyxy(
            boxes_xyxy[i:i + 1], boxes_xyxy[rest],
        ).squeeze(0)
        order = rest[ious < iou_thresh]
    return torch.tensor(kept, dtype=torch.long)


# ---------------------------------------------------------------------
# mAP50 (per-image, then averaged)
# ---------------------------------------------------------------------


def evaluate_map50(
    model: nn.Module,
    eval_loader,
    device: torch.device,
    score_thresh: float = 0.05,
) -> float:
    """Simplified mAP50: per-image, find positive detections (cls != 0,
    score > thresh, then NMS); match against GT at IoU 0.5; compute
    precision and recall; report mean precision averaged over images.

    This is NOT the full COCO mAP50 (which averages per-class AP at
    IoU 0.5); it is the per-image precision-at-IoU-0.5 averaged
    over images. Good enough for ablation comparison; the report
    notes the difference explicitly.
    """
    model.eval()
    image_aps = []
    with torch.no_grad():
        for images, bboxes_list, labels_list in eval_loader:
            images = images.to(device)
            outs = model(images)
            for b_idx in range(len(outs)):
                out = outs[b_idx] if isinstance(outs, list) else outs
                gt_boxes = bboxes_list[b_idx]
                gt_labels = labels_list[b_idx]
                if gt_boxes.shape[0] == 0:
                    continue
                # Per-anchor: softmax → max non-bg class.
                probs = F.softmax(out.cls_logits, dim=-1)
                bg_score = probs[:, 0]
                # Best non-bg class per anchor.
                cls_probs = probs[:, 1:]                  # (n, n_classes)
                best_cls = cls_probs.argmax(dim=-1)
                best_score = cls_probs.max(dim=-1).values
                keep = best_score > score_thresh
                if keep.sum().item() == 0:
                    image_aps.append(0.0)
                    continue
                pred_boxes = type(model).decode_boxes(out)[keep]
                # Convert (cx, cy, w, h) → (x1, y1, x2, y2).
                pb = torch.zeros_like(pred_boxes)
                pb[:, 0] = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
                pb[:, 1] = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
                pb[:, 2] = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
                pb[:, 3] = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
                pred_classes = best_cls[keep]
                pred_scores = best_score[keep]
                # NMS.
                keep_nms = _greedy_nms(pb, pred_scores)
                pb = pb[keep_nms]
                pred_classes = pred_classes[keep_nms]
                # Match against GT.
                if pb.shape[0] == 0:
                    image_aps.append(0.0)
                    continue
                ious = _iou_boxes_xyxy(pb.cpu(), gt_boxes.float())
                # A prediction is correct if it has IoU>0.5 with a GT
                # of matching class.
                gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
                tp = 0
                for p_idx in range(pb.shape[0]):
                    best_gt_iou = -1.0
                    best_gt_idx = -1
                    for g_idx in range(gt_boxes.shape[0]):
                        if gt_used[g_idx]:
                            continue
                        if pred_classes[p_idx].item() != gt_labels[g_idx].item():
                            continue
                        if ious[p_idx, g_idx].item() > best_gt_iou:
                            best_gt_iou = ious[p_idx, g_idx].item()
                            best_gt_idx = g_idx
                    if best_gt_idx >= 0 and best_gt_iou > 0.5:
                        gt_used[best_gt_idx] = True
                        tp += 1
                precision = tp / max(1, pb.shape[0])
                recall = tp / max(1, gt_boxes.shape[0])
                # mAP50 proxy: harmonic mean (F1 at IoU 0.5).
                if precision + recall == 0:
                    image_aps.append(0.0)
                else:
                    image_aps.append(
                        2 * precision * recall / (precision + recall)
                    )
    return float(sum(image_aps) / max(1, len(image_aps)))


# ---------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------


def train_one_seed(
    detector_factory,    # callable() -> RicciStimDetector
    train_loader,
    eval_loader,
    *,
    n_epochs: int = 20,
    lr: float = 3e-3,
    bbox_weight: float = 1.0,
    iou_pos: float = 0.3,
    device: torch.device = torch.device("cpu"),
    log_every: int = 200,
    eval_every_epoch: bool = True,
) -> dict:
    model = detector_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    history = []
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for batch_idx, (images, bboxes_list, labels_list) in enumerate(train_loader):
            images = images.to(device)
            opt.zero_grad()
            outs = model(images)
            if not isinstance(outs, list):
                outs = [outs]
            total_loss = 0.0
            for b_idx, out in enumerate(outs):
                assignment = assign_anchors_to_gt(
                    out.anchor_positions, out.anchor_sizes,
                    bboxes_list[b_idx].to(device),
                    labels_list[b_idx].to(device),
                    iou_pos=iou_pos,
                )
                loss_dict = detection_loss(out, assignment, bbox_weight)
                total_loss = total_loss + loss_dict["loss"]
            total_loss = total_loss / max(1, len(outs))
            total_loss.backward()
            opt.step()
            ep_loss += float(total_loss.item())
            n_batches += 1
        epoch_loss = ep_loss / max(1, n_batches)
        result = {"epoch": epoch + 1, "train_loss": epoch_loss}
        if eval_every_epoch:
            result["mAP50_proxy"] = evaluate_map50(model, eval_loader, device)
        history.append(result)
    wall = time.perf_counter() - t0
    return {
        "n_params": n_params,
        "wall_s": wall,
        "history": history,
        "final_mAP50_proxy": (
            history[-1].get("mAP50_proxy", None) if history else None
        ),
    }
