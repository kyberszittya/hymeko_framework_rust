"""Stage H — single-class person detection on PASCAL VOC2007.

Plan: docs/plans/2026-05-19-stage-h-voc-eyes-for-rapport/.

Design choices motivated by Stage D-2's diagnostic:

* ``n_classes = 1`` (just "person"; CE head has 2 output logits
  — person + no-object). The 20-class no-object signal-to-noise
  problem that killed Stage D / D-1 / D-2 doesn't apply at 1 class.
* ``n_box_queries = 2`` to roughly match VOC2007's mean
  person-count-per-image (1.85). The 12.5× over-provisioning ratio
  from Stage D becomes ~1:1.
* ``--backbone resnet18_imagenet`` is the Stage D-1 contribution
  (it was correct; the bottleneck was at the head).

Usage:
    # CPU smoke (200 images, 1 epoch):
    python -m signedkan_wip.src.vision.train_voc_person \\
        --n-images 200 --epochs 1 --input-size 96 --batch-size 4 \\
        --seed 0 --device cpu

    # Production smoke (full VOC person trainval, 30 epochs, GPU):
    python -m signedkan_wip.src.vision.train_voc_person \\
        --image-set trainval --epochs 30 --input-size 224 --seed 0 \\
        --save-checkpoint signedkan_wip/experiments/results/stage_h/checkpoints \\
        --jsonl-out signedkan_wip/experiments/results/stage_h/smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .hymeyolo_circles_ricci import RicciHyMeYOLOMulti
from .train_circles_ricci import train_one_config
from .voc_person_dataset import (
    PERSON_CLASSES, load_voc_person_hungarian,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="2007", choices=["2007", "2012"])
    p.add_argument("--image-set", default="trainval",
                   choices=["train", "val", "trainval", "test"])
    p.add_argument("--n-images", type=int, default=None,
                   help="Subset size; None = full person-filtered split.")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--max-objects", type=int, default=6,
                   help="Pad slot count. Person count per VOC image rarely "
                        "exceeds 5; 6 is the safe default.")
    p.add_argument("--n-box-queries", type=int, default=2,
                   help="Hungarian query count. VOC person mean=1.85; "
                        "n=2 keeps the over-provisioning ratio near 1.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="data/torchvision")
    p.add_argument("--jsonl-out", default=None)
    p.add_argument("--save-checkpoint", default=None)
    p.add_argument("--ricci-scale", type=float, default=1.0)
    p.add_argument("--schedule", default="cosine",
                   choices=["constant", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--backbone",
                   choices=["resnet", "resnet18_imagenet"],
                   default="resnet18_imagenet")
    p.add_argument("--lam-no-obj", type=float, default=0.5,
                   help="No-object CE weight. Stage D-2d tested 0.1/0.5/1.0/2.0; "
                        "default 0.5 is reasonable for 1 class + 2 queries.")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    n_classes = len(PERSON_CLASSES)  # = 1

    print(f"\n=== Stage H — VOC{args.year} person-only detection ===")
    print(f"  image_set={args.image_set}  input={args.input_size}  "
          f"epochs={args.epochs}  batch={args.batch_size}  "
          f"n_box_queries={args.n_box_queries}  "
          f"n_classes={n_classes}  device={device}  seed={args.seed}")
    print(f"  arch: backbone={args.backbone!r}  fpn='2level'  "
          f"ricci_modulation=True  lam_no_obj={args.lam_no_obj}")

    t0 = time.perf_counter()
    Xn, boxes_n, classes_n, counts_n, names = load_voc_person_hungarian(
        year=args.year, image_set=args.image_set,
        input_size=args.input_size, max_objects=args.max_objects,
        root=args.data_root, subset_n=args.n_images, download=True,
    )
    load_wall = time.perf_counter() - t0
    print(f"  loaded {Xn.shape[0]} person-images in {load_wall:.1f}s  "
          f"persons.mean={counts_n.mean():.2f}  persons.max={counts_n.max()}")

    X = torch.from_numpy(Xn).to(device)
    boxes = torch.from_numpy(boxes_n).to(device)
    classes = torch.from_numpy(classes_n).to(device)
    classes_safe = torch.where(classes < 0, torch.zeros_like(classes),
                                classes)
    counts = torch.from_numpy(counts_n).to(device)

    label = "stage_h_voc_person"
    model = RicciHyMeYOLOMulti(
        n_box_queries=args.n_box_queries,
        n_circle_queries=0,
        n_classes=n_classes,
        d_hidden=32,
        ricci_modulation=True,
        ricci_scale=args.ricci_scale,
        use_layernorm=False,
        backbone=args.backbone,
        fpn="2level",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    torch.manual_seed(args.seed)
    result = train_one_config(
        label, model, X, boxes, classes_safe, counts,
        epochs=args.epochs, lr=args.lr, device=device,
        batch_size=args.batch_size,
        lam_no_obj=args.lam_no_obj,
    )

    det = result.get("det_metrics", {})
    loss_drop_pct = (result["losses"][0] - result["losses"][-1]) \
                    / max(1e-9, result["losses"][0]) * 100.0

    print(f"\n=== Stage H result ===")
    print(f"  loss_start={result['losses'][0]:.4f}  "
          f"loss_end={result['losses'][-1]:.4f}  "
          f"drop={loss_drop_pct:.1f}%")
    print(f"  box_cls_acc={result['accs']['box_cls_acc']:.4f}")
    print(f"  mAP_50={det.get('mAP_50')}  mAP_50_95={det.get('mAP_50_95')}")
    print(f"  wall_s={result['wall']:.1f}")

    if args.save_checkpoint:
        ckpt_dir = Path(args.save_checkpoint)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{label}_seed{args.seed}.pt"
        torch.save({
            "label": label,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "ricci_scale": float(args.ricci_scale),
            "schedule": args.schedule,
            "warmup_epochs": int(args.warmup_epochs),
            "use_layernorm": False,
            "weight_decay": 0.0,
            "cls_loss": "ce",
            "box_loss": "giou",
            "backbone": args.backbone,
            "fpn": "2level",
            "n_box_queries": int(args.n_box_queries),
            "n_classes": n_classes,
            "input_size": int(args.input_size),
            "lam_no_obj": float(args.lam_no_obj),
            "state_dict": model.state_dict(),
            "model_class": "RicciHyMeYOLOMulti",
            "dataset": f"voc{args.year}_person_{args.image_set}",
        }, ckpt_path)
        print(f"  [checkpoint] saved → {ckpt_path}")

    if args.jsonl_out:
        Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl_out, "w") as fh:
            rec = {
                "dataset":      f"voc{args.year}_person_{args.image_set}",
                "label":        label,
                "n_images":     Xn.shape[0],
                "input_size":   args.input_size,
                "max_objects":  args.max_objects,
                "n_box_queries": args.n_box_queries,
                "n_classes":    n_classes,
                "epochs":       args.epochs,
                "lr":           args.lr,
                "seed":         args.seed,
                "n_params":     n_params,
                "backbone":     args.backbone,
                "fpn":          "2level",
                "lam_no_obj":   args.lam_no_obj,
                "wall_s":       result["wall"],
                "loss_start":   result["losses"][0],
                "loss_end":     result["losses"][-1],
                "loss_drop_pct": loss_drop_pct,
                "box_cls_acc":  result["accs"]["box_cls_acc"],
                "mAP_50":       det.get("mAP_50"),
                "mAP_50_95":    det.get("mAP_50_95"),
                "mean_iou_matched": det.get("mean_iou_matched"),
                "n_preds_used":  det.get("n_preds_used"),
                "n_gts_total":   det.get("n_gts_total"),
                "losses_per_epoch": result["losses"],
            }
            fh.write(json.dumps(rec) + "\n")
        print(f"  Wrote {args.jsonl_out}")

    print(f"\n  Total wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
