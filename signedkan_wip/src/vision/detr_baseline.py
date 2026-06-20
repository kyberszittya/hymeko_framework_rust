"""MiniDETR — a budget-matched transformer detector baseline for Cluttered MNIST.

Purpose: turn "RicciStim (5 896 params, 0.228 mAP50_proxy) is at parity with DETR"
from a hypothesis into a measured, FAIR head-to-head. Two size presets:

  * ``standard`` — a genuine small DETR (~10^5–10^6 params);
  * ``tiny``     — parameter-matched to RicciStim (~10^4 params): the
                   apples-to-apples efficiency comparison.

Fairness is the whole point (plan: docs/plans/2026-06-16-detr-baseline/):
  * the set-prediction loss is the EXISTING ``hungarian_set_loss``
    (vision/hymeyolo_hungarian.py) — not a re-implementation;
  * evaluation uses the EXISTING ``match_f1_at_iou50`` shared with the RicciStim
    evaluator, so both models are scored by identical code;
  * an overfit guard (test_detr_baseline.py) proves the model is competent — a
    DETR that cannot overfit a handful of images would be a strawman.

Only the DETR *body* (conv stem → transformer enc/dec → object queries → heads)
is new here.

Run:
    python -m signedkan_wip.src.vision.detr_baseline --size tiny \\
        --n-train 5000 --n-eval 1000 --n-epochs 40 --canvas 64 --max-digits 3 \\
        --seed 0 --device cuda --out-jsonl reports/detr_tiny_cluttered.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_train import (
    match_f1_at_iou50,
)
from signedkan_wip.src.vision.cluttered_mnist import (
    ClutteredMNIST,
    collate_cluttered,
)
from signedkan_wip.src.vision.hymeyolo_hungarian import hungarian_set_loss


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DETRConfig:
    """MiniDETR hyper-parameters. ``PRESETS`` holds the two reported sizes."""

    d_model: int
    n_enc_layers: int
    n_dec_layers: int
    n_heads: int
    dim_ff: int
    n_queries: int


PRESETS: dict[str, DETRConfig] = {
    # A genuine small transformer detector.
    "standard": DETRConfig(d_model=128, n_enc_layers=3, n_dec_layers=3,
                           n_heads=4, dim_ff=256, n_queries=10),
    # Parameter-matched to RicciStim (~10^4 params) — the efficiency comparison.
    "tiny": DETRConfig(d_model=16, n_enc_layers=1, n_dec_layers=1,
                       n_heads=2, dim_ff=32, n_queries=8),
}


class MiniDETR(nn.Module):
    """Conv stem → flattened tokens + learned positional embedding →
    Transformer encoder/decoder over ``n_queries`` object queries → per-query
    class + box heads.

    Forward returns the contract expected by ``hungarian_set_loss``:
        corners : Tensor[B, N, 4, 2]  (normalised, in [0, 1])
        cls     : Tensor[B, N, n_classes + 1]  (last logit = no-object)
    """

    def __init__(
        self,
        cfg: DETRConfig,
        image_h: int = 64,
        image_w: int = 64,
        in_channels: int = 1,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        d = cfg.d_model

        # Conv stem: three stride-2 convs → /8 spatial (64 → 8), `d` channels.
        c1 = max(8, d // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, d, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.grid_h = image_h // 8
        self.grid_w = image_w // 8
        n_tokens = self.grid_h * self.grid_w
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
        self.query_embed = nn.Parameter(torch.randn(cfg.n_queries, d) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.n_enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=0.0, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.n_dec_layers)

        self.class_head = nn.Linear(d, n_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 4),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"expected (B, C, H, W); got {tuple(images.shape)}")
        b = images.shape[0]
        feat = self.stem(images)                       # (B, d, gh, gw)
        tokens = feat.flatten(2).transpose(1, 2)       # (B, n_tokens, d)
        tokens = tokens + self.pos_embed
        memory = self.encoder(tokens)                  # (B, n_tokens, d)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)  # (B, N, d)
        hs = self.decoder(queries, memory)             # (B, N, d)

        cls_logits = self.class_head(hs)               # (B, N, n_classes+1)
        box = torch.sigmoid(self.box_head(hs))         # (B, N, 4) = (cx,cy,w,h)
        corners = _box_cxcywh_to_corners(box)          # (B, N, 4, 2)
        return corners, cls_logits

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _box_cxcywh_to_corners(box: torch.Tensor) -> torch.Tensor:
    """(…, 4)=(cx,cy,w,h) → (…, 4, 2) corners TL, TR, BR, BL (normalised)."""
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    tl = torch.stack([x0, y0], dim=-1)
    tr = torch.stack([x1, y0], dim=-1)
    br = torch.stack([x1, y1], dim=-1)
    bl = torch.stack([x0, y1], dim=-1)
    return torch.stack([tl, tr, br, bl], dim=-2)


def _corners_to_xyxy(corners: torch.Tensor) -> torch.Tensor:
    """(…, 4, 2) corners → (…, 4)=(x0,y0,x1,y1) axis-aligned bbox."""
    xs, ys = corners[..., 0], corners[..., 1]
    return torch.stack([xs.min(-1).values, ys.min(-1).values,
                        xs.max(-1).values, ys.max(-1).values], dim=-1)


def build_detr(size: str, canvas: int = 64, n_classes: int = 10) -> MiniDETR:
    if size not in PRESETS:
        raise SystemExit(f"unknown size {size!r}; choose from {list(PRESETS)}")
    return MiniDETR(PRESETS[size], image_h=canvas, image_w=canvas,
                    in_channels=1, n_classes=n_classes)


# ---------------------------------------------------------------------
# GT padding for the (reused) set loss
# ---------------------------------------------------------------------


def _pad_targets(
    bboxes_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    canvas: int,
    n_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (gt_corners[B,M,4,2] normalised, gt_classes[B,M] pad=n_classes,
    gt_counts[B]) from a batch's variable-length pixel-xyxy GT."""
    b = len(bboxes_list)
    counts = [int(bx.shape[0]) for bx in bboxes_list]
    m = max(1, max(counts))
    gt_corners = torch.zeros(b, m, 4, 2, device=device)
    gt_classes = torch.full((b, m), n_classes, dtype=torch.long, device=device)
    for i, (bx, lb) in enumerate(zip(bboxes_list, labels_list)):
        n = counts[i]
        if n == 0:
            continue
        box = bx.float().to(device) / canvas            # normalise to [0,1]
        x0, y0, x1, y1 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        corners = torch.stack([
            torch.stack([x0, y0], -1), torch.stack([x1, y0], -1),
            torch.stack([x1, y1], -1), torch.stack([x0, y1], -1),
        ], dim=-2)                                       # (n, 4, 2)
        gt_corners[i, :n] = corners
        gt_classes[i, :n] = lb.long().to(device)
    return gt_corners, gt_classes, torch.tensor(counts, device=device)


# ---------------------------------------------------------------------
# Eval — identical metric to RicciStim (match_f1_at_iou50)
# ---------------------------------------------------------------------


def evaluate_detr_map50(
    model: MiniDETR,
    eval_loader: DataLoader,
    device: torch.device,
    canvas: int,
    score_thresh: float = 0.05,
) -> float:
    """Per-image F1@IoU0.5 averaged over images — the SAME metric as
    `ricci_stim_train.evaluate_map50`, via the shared `match_f1_at_iou50`."""
    model.eval()
    n_classes = model.n_classes
    image_f1 = []
    with torch.no_grad():
        for images, bboxes_list, labels_list in eval_loader:
            images = images.to(device)
            corners, cls_logits = model(images)          # (B,N,4,2),(B,N,nc+1)
            probs = F.softmax(cls_logits, dim=-1)
            cls_probs = probs[..., :n_classes]           # drop no-object
            best_cls = cls_probs.argmax(dim=-1)          # (B, N)
            best_score = cls_probs.max(dim=-1).values    # (B, N)
            xyxy = _corners_to_xyxy(corners) * canvas    # back to pixels
            for b in range(images.shape[0]):
                gt_boxes = bboxes_list[b]
                gt_labels = labels_list[b]
                if gt_boxes.shape[0] == 0:
                    continue
                keep = best_score[b] > score_thresh
                if keep.sum().item() == 0:
                    image_f1.append(0.0)
                    continue
                image_f1.append(match_f1_at_iou50(
                    xyxy[b][keep], best_cls[b][keep], gt_boxes, gt_labels,
                ))
    return float(sum(image_f1) / max(1, len(image_f1)))


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_seed(
    size: str,
    seed: int,
    n_train: int,
    n_eval: int,
    n_epochs: int,
    canvas: int,
    max_digits: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    set_seed(seed)
    n_classes = 10
    train_ds = ClutteredMNIST(n_samples=n_train, canvas=canvas,
                              max_digits=max_digits, seed=seed, train=True)
    eval_ds = ClutteredMNIST(n_samples=n_eval, canvas=canvas,
                             max_digits=max_digits, seed=seed + 50_000,
                             train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_cluttered, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_cluttered, num_workers=0)
    model = build_detr(size, canvas=canvas, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n_params = model.n_parameters()

    history = []
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        model.train()
        ep_loss, n_b = 0.0, 0
        for images, bboxes_list, labels_list in train_loader:
            images = images.to(device)
            gt_corners, gt_classes, gt_counts = _pad_targets(
                bboxes_list, labels_list, canvas, n_classes, device)
            corners, cls_logits = model(images)
            # l1giou (= corner-L1 + AABB-GIoU): pure GIoU saturates at IoU≈0.46,
            # just under the 0.5 the metric needs; the L1 term drives it past.
            # Diagnosed by the overfit guard — reports/2026-06-16-detr-overfit-fix.md.
            loss, _cls_acc, _n = hungarian_set_loss(
                corners, cls_logits, gt_corners, gt_classes, gt_counts,
                n_classes=n_classes, box_loss_kind="l1giou",
            )
            opt.zero_grad()
            loss.backward()
            # Grad-norm clip. NOTE the regime split (reports/2026-06-16-detr-
            # overfit-fix.md + …-clip-sweep): the 16-image overfit GUARD needs a
            # tight clip 0.1 (lr 1e-3 diverges otherwise), but on the full 5000-
            # image run clip 0.1 strangles learning (clips every mini-batch grad
            # to ~0 → flat loss). The full-run default is 1.0 (stable AND learns;
            # 0 learns marginally faster but is less safe over 40 ep unattended).
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        m = evaluate_detr_map50(model, eval_loader, device, canvas)
        history.append({"epoch": epoch + 1, "train_loss": ep_loss / max(1, n_b),
                        "mAP50_proxy": m})
        print(f"  [size={size} seed={seed}] epoch {epoch + 1}/{n_epochs} "
              f"loss={ep_loss / max(1, n_b):.4f} mAP50_proxy={m:.4f}", flush=True)
    wall = time.perf_counter() - t0
    return {"model": f"detr_{size}", "seed": seed, "n_train": n_train,
            "n_eval": n_eval, "n_epochs": n_epochs, "n_params": n_params,
            "wall_s": wall, "history": history,
            "final_mAP50_proxy": history[-1]["mAP50_proxy"] if history else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(PRESETS), default="tiny")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--max-digits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="grad-norm clip (0 disables); 1.0 for the full run "
                         "(0.1 strangles it — see detr-overfit-fix report)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[detr] size={args.size} device={device} n_train={args.n_train} "
          f"n_eval={args.n_eval} n_epochs={args.n_epochs}", flush=True)
    rec = train_one_seed(args.size, args.seed, args.n_train, args.n_eval,
                         args.n_epochs, args.canvas, args.max_digits,
                         args.batch_size, args.lr, device,
                         grad_clip=args.grad_clip)
    print(f"[detr] DONE size={args.size} n_params={rec['n_params']} "
          f"final_mAP50_proxy={rec['final_mAP50_proxy']} "
          f"wall={rec['wall_s']:.1f}s", flush=True)
    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
