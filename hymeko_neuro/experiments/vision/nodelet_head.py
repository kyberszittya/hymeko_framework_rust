"""Nodelet query head: explicit per-query objectness gates + gate-aware Hungarian loss.

Stage D-3 architectural fix for the HyMeYOLO Hungarian-head
over-provisioning bottleneck. Per the Stage D-2 diagnostic (2026-05-18),
the flat $K+1$-class softmax cannot produce both high matched-cls
accuracy and clean no-object suppression with any setting of
``lam_no_obj``. This module replaces it with:

* A class head that emits exactly $K$ logits (no no-object slot).
* A separate **gate head** that emits one sigmoid scalar per query.
  $g_q \\in [0, 1]$ answers "am I a real object?" independently of
  "which class am I?".
* A **Hungarian matcher** that uses $(1 - g_q)$ as part of the cost
  (low-gate queries are penalised in matching).
* A **loss** with four terms: GIoU on matched boxes, CE on matched
  classes, BCE on gates pushed toward 1 for matched and 0 for
  unmatched.

Plan: docs/plans/2026-05-19-stage-d3-nodelet-head/.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ─── Heads ──────────────────────────────────────────────────────────


def make_nodelet_class_head(d_in: int, n_classes: int,
                             use_layernorm: bool = False) -> nn.Module:
    """Class head for the nodelet regime: ``d_in -> n_classes``.

    No no-object slot — the gate head handles objectness separately.
    """
    layers: list[nn.Module] = []
    if use_layernorm:
        layers.append(nn.LayerNorm(d_in))
    layers.append(nn.Linear(d_in, n_classes))
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


def make_nodelet_gate_head(d_in: int) -> nn.Module:
    """Gate head: ``d_in -> 1`` logit, sigmoid applied at forward time
    by the model (so the same module can be queried for both pre- and
    post-sigmoid values during loss/eval)."""
    return nn.Linear(d_in, 1)


# ─── Hungarian matcher with gate cost ──────────────────────────────


def _box_giou(pred_aabb: torch.Tensor, gt_aabb: torch.Tensor) -> torch.Tensor:
    """Lazy import to avoid the circular dep with train_circles_ricci."""
    from .train_circles_ricci import giou_loss_xyxy
    # giou_loss_xyxy returns 1 - GIoU; we want GIoU for the cost matrix
    # (lower cost is better, so we use 1 - GIoU directly = the loss itself).
    return giou_loss_xyxy(pred_aabb, gt_aabb, reduction="none")


def _corners_to_aabb(corners: torch.Tensor) -> torch.Tensor:
    """(..., k, 2) → (..., 4) xyxy."""
    x = corners[..., 0]
    y = corners[..., 1]
    return torch.stack([
        x.min(dim=-1).values, y.min(dim=-1).values,
        x.max(dim=-1).values, y.max(dim=-1).values,
    ], dim=-1)


def hungarian_set_loss_gated(
    pred_corners: torch.Tensor,    # (B, N, k, 2)
    pred_cls: torch.Tensor,        # (B, N, n_classes)
    pred_gates: torch.Tensor,      # (B, N) in [0, 1]
    gt_corners: torch.Tensor,      # (B, M, k, 2)
    gt_classes: torch.Tensor,      # (B, M)  long
    gt_counts: torch.Tensor,       # (B,)    long
    n_classes: int,
    *,
    lam_corner: float = 5.0,
    lam_cls: float = 1.0,
    lam_gate_match_cost: float = 1.0,
    lam_gate_pos: float = 1.0,
    lam_gate_neg: float | None = None,  # None → auto-balance per-image
    cls_loss_kind: str = "ce",
    box_loss_kind: str = "l1",
    gate_curriculum: bool = False,
    gate_loss_kind: str = "bce",        # "bce" | "focal"
    gate_focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, float, dict]:
    """Hungarian set loss with explicit objectness gates.

    Cost matrix entry $(q, g)$:
        $C_{qg} = \\lambda_{\\text{corner}} \\cdot L_1(c_q, c_g)
                + \\lambda_{\\text{cls}} \\cdot (-p_q[\\text{cls}(g)])
                + \\lambda_{\\text{gate\\_match}} \\cdot (1 - g_q)$

    Loss:
        $\\mathcal{L} =
            \\lambda_{\\text{corner}} \\cdot \\mathcal{L}_{\\text{box}}
          + \\lambda_{\\text{cls}}    \\cdot \\mathcal{L}_{\\text{cls}}
          + \\lambda_{\\text{gpos}}   \\cdot \\mathcal{L}_{\\text{gate}}^{+}
          + \\lambda_{\\text{gneg}}   \\cdot \\mathcal{L}_{\\text{gate}}^{-}$

    ``gate_curriculum=True`` (used for the first few epochs) sets
    ``lam_gate_match_cost = 0.0`` in the matcher, letting the model
    learn matched-cls + matched-box first; the gate BCE still
    supervises so the gate head trains. Switch off after epoch 5.

    Stage D-3-tris (2026-05-18): ``gate_loss_kind="focal"`` replaces
    the unmatched-side BCE :math:`-\\log(1-g)` with focal
    :math:`g^\\gamma \\cdot -\\log(1-g)`. Already-suppressed
    queries (gate ≈ 0) contribute almost nothing; borderline ones
    (gate ≈ 0.5) get the modulated weight. Matched-side stays BCE
    (focal would *reduce* matched gradient, opposite of intent).
    Default ``"bce"`` preserves D-3 / D-3-bis behaviour
    byte-identical.

    Returns: (total_loss, matched_cls_accuracy, diagnostics_dict).
    """
    from .train_circles_ricci import focal_loss_ce, giou_loss_xyxy

    if gate_loss_kind not in ("bce", "focal"):
        raise ValueError(
            f"gate_loss_kind must be 'bce' or 'focal'; got {gate_loss_kind!r}"
        )

    def _gate_neg_loss(gates_unmatched: torch.Tensor) -> torch.Tensor:
        """Sum of unmatched-gate loss values (NOT mean)."""
        zeros = torch.zeros_like(gates_unmatched)
        if gate_loss_kind == "bce":
            return F.binary_cross_entropy(gates_unmatched, zeros,
                                          reduction="sum")
        # Focal: gate^gamma * -log(1 - gate), per-sample then summed.
        # Clamp gates away from 1.0 to avoid log(0).
        g = gates_unmatched.clamp(min=1e-7, max=1.0 - 1e-7)
        loss = (g ** gate_focal_gamma) * (-torch.log(1.0 - g))
        return loss.sum()

    def _cls_loss(logits: torch.Tensor, targets: torch.Tensor,
                  reduction: str = "mean") -> torch.Tensor:
        if cls_loss_kind == "focal":
            return focal_loss_ce(logits, targets, gamma=2.0, reduction=reduction)
        return F.cross_entropy(logits, targets, reduction=reduction)

    def _box_loss(matched_pred: torch.Tensor,
                  matched_gt: torch.Tensor) -> torch.Tensor:
        if box_loss_kind == "giou":
            pred_aabb = _corners_to_aabb(matched_pred)
            gt_aabb = _corners_to_aabb(matched_gt)
            return giou_loss_xyxy(pred_aabb, gt_aabb, reduction="mean")
        return (matched_pred - matched_gt).abs().mean()

    B, N = pred_corners.shape[:2]
    device = pred_corners.device
    pred_probs = F.softmax(pred_cls, dim=-1)             # (B, N, n_classes)

    matcher_gate_lambda = 0.0 if gate_curriculum else lam_gate_match_cost

    total_box_loss = pred_corners.new_zeros(())
    total_cls_loss = pred_corners.new_zeros(())
    total_gate_pos = pred_corners.new_zeros(())
    total_gate_neg = pred_corners.new_zeros(())
    n_matched_correct = 0
    matched_count_total = 0
    n_gate_pos_total = 0
    n_gate_neg_total = 0

    for b in range(B):
        n_gt = int(gt_counts[b].item())
        gates_b = pred_gates[b]                          # (N,)
        if n_gt == 0:
            # No GT in this image — all queries must learn gate=0.
            total_gate_neg = total_gate_neg + _gate_neg_loss(gates_b)
            n_gate_neg_total += N
            continue
        # Cost matrix.
        pc = pred_corners[b].unsqueeze(1)                # (N, 1, k, 2)
        gc = gt_corners[b, :n_gt].unsqueeze(0)           # (1, n_gt, k, 2)
        corner_cost = (pc - gc).abs().mean(dim=(-1, -2)) # (N, n_gt)
        gt_cls_b = gt_classes[b, :n_gt]                  # (n_gt,)
        cls_cost = -pred_probs[b][:, gt_cls_b]           # (N, n_gt)
        # Gate cost: low-gate queries cost more to match.
        gate_cost = (1.0 - gates_b.unsqueeze(1)).expand(N, n_gt)
        cost = (lam_corner * corner_cost
                + lam_cls * cls_cost
                + matcher_gate_lambda * gate_cost
               ).detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        m_pred = torch.tensor(rows, device=device, dtype=torch.long)
        m_gt = torch.tensor(cols, device=device, dtype=torch.long)

        # Box + cls loss on matched pairs.
        total_box_loss = total_box_loss + _box_loss(
            pred_corners[b][m_pred], gt_corners[b][m_gt],
        )
        m_cls_t = gt_classes[b][m_gt]
        total_cls_loss = total_cls_loss + _cls_loss(
            pred_cls[b][m_pred], m_cls_t,
        )
        n_matched_correct += int(
            (pred_cls[b][m_pred].argmax(-1) == m_cls_t).sum().item()
        )
        matched_count_total += int(m_pred.numel())

        # Gate supervision: matched queries → 1, unmatched → 0.
        gpos_targets = torch.ones(m_pred.numel(), device=device)
        total_gate_pos = total_gate_pos + F.binary_cross_entropy(
            gates_b[m_pred], gpos_targets, reduction="sum",
        )
        n_gate_pos_total += int(m_pred.numel())

        # Unmatched mask.
        unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
        unmatched_mask[m_pred] = False
        n_unmatched = int(unmatched_mask.sum())
        if n_unmatched > 0:
            total_gate_neg = total_gate_neg + _gate_neg_loss(
                gates_b[unmatched_mask],
            )
            n_gate_neg_total += n_unmatched

    # Auto-balance gate-neg weight so the matched/unmatched gradient
    # signal is comparable.
    if lam_gate_neg is None:
        if n_gate_neg_total > 0 and n_gate_pos_total > 0:
            lam_gate_neg = float(n_gate_pos_total) / float(n_gate_neg_total)
        else:
            lam_gate_neg = 1.0

    total_box_loss = total_box_loss / max(1, B)
    total_cls_loss = total_cls_loss / max(1, B)
    # Mean per supervised element for the gate terms.
    total_gate_pos_mean = (total_gate_pos / max(1, n_gate_pos_total)
                            if n_gate_pos_total else total_gate_pos)
    total_gate_neg_mean = (total_gate_neg / max(1, n_gate_neg_total)
                            if n_gate_neg_total else total_gate_neg)

    total_loss = (lam_corner * total_box_loss
                  + lam_cls * total_cls_loss
                  + lam_gate_pos * total_gate_pos_mean
                  + lam_gate_neg * total_gate_neg_mean)

    cls_acc = (n_matched_correct / max(1, matched_count_total)
                if matched_count_total > 0 else 0.0)
    diagnostics = {
        "matched_count": matched_count_total,
        "n_gate_pos":    n_gate_pos_total,
        "n_gate_neg":    n_gate_neg_total,
        "gate_neg_lambda": float(lam_gate_neg),
        "mean_gate_pos_loss": float(total_gate_pos_mean.detach().item()),
        "mean_gate_neg_loss": float(total_gate_neg_mean.detach().item()),
        "gate_loss_kind": gate_loss_kind,
        "gate_focal_gamma": float(gate_focal_gamma),
        "lam_gate_match_cost": float(lam_gate_match_cost),
    }
    return total_loss, cls_acc, diagnostics


def filter_predictions_by_gate(
    pred_corners: torch.Tensor,
    pred_cls: torch.Tensor,
    pred_gates: torch.Tensor,
    threshold: float = 0.5,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Inference helper: per image in the batch, return only the
    queries whose gate exceeds the threshold.

    Returns a list of length B; each entry is (kept_corners, kept_cls).
    The class indices are integers (argmax over the n_classes logits).
    """
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    B = pred_corners.shape[0]
    for b in range(B):
        keep = pred_gates[b] > threshold
        kept_corners = pred_corners[b][keep]
        kept_cls = pred_cls[b][keep].argmax(dim=-1)
        out.append((kept_corners, kept_cls))
    return out
