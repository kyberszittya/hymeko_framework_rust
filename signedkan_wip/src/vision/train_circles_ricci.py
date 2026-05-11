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
    print(f"  {label:<18s}  start={losses_per_epoch[0]:7.3f}  "
          f"end={losses_per_epoch[-1]:7.3f}  drop={drop_pct:5.1f}%  "
          f"wall={wall:5.1f}s  "
          f"box_acc={last_accs['box_cls_acc']:.2f}  "
          f"circ_acc={last_accs['circ_cls_acc']:.2f}")
    return dict(label=label, losses=losses_per_epoch, wall=wall, accs=last_accs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jsonl-out", default=None,
                    help="Optional path to write per-config results as jsonl.")
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
    ]

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
                    "acceptance_pass": r["acceptance_pass"],
                    "losses_per_epoch": r["losses"],
                }
                fh.write(json.dumps(rec) + "\n")
        print(f"\n  Results written to {args.jsonl_out}")


if __name__ == "__main__":
    main()
