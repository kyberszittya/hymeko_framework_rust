"""HyMeYOLO training on Pascal VOC.

Apples-to-apples-ish comparison against standard YOLO/DETR reports.
Same 5 configs as `train_circles_ricci.py` (baseline / boxes-only /
circles-only / boxes+circles / +ricci-mod) but on real VOC images.

Usage:
    python -m signedkan_wip.src.vision.train_voc \
        --year 2007 --image-set train \
        --n-images 1000 --epochs 30 --input-size 128 --seed 0 \
        --jsonl-out results/voc2007_train_n1k_e30_s0.jsonl

VOC2007 trainval = 5011 images. With 128x128, 5011 × 3 × 128 × 128 ×
4 B ≈ 940 MB — fits comfortably in 16 GB cap.

VOC has 20 classes vs Cluttered MNIST's 10. The same RicciHyMeYOLOMulti
architecture handles it via the `n_classes` constructor arg.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .hymeyolo_circles_ricci import RicciHyMeYOLOMulti
from .hymeyolo_hungarian import HyMeYOLOMulti
from .train_circles_ricci import train_one_config
from .voc_dataset import VOC_CLASSES, load_voc_hungarian


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="2007", choices=["2007", "2012"])
    p.add_argument("--image-set", default="train",
                    choices=["train", "val", "trainval", "test"])
    p.add_argument("--n-images", type=int, default=None,
                    help="Subset size; None = full split.")
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--max-objects", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="data/torchvision")
    p.add_argument("--jsonl-out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    n_classes = len(VOC_CLASSES)

    print(f"\nVOC{args.year} {args.image_set}: "
          f"input={args.input_size}  max_objects={args.max_objects}  "
          f"epochs={args.epochs}  lr={args.lr}  device={device}")

    t0 = time.perf_counter()
    Xn, boxes_n, classes_n, counts_n, names = load_voc_hungarian(
        year=args.year, image_set=args.image_set,
        input_size=args.input_size, max_objects=args.max_objects,
        root=args.data_root, subset_n=args.n_images, download=True,
    )
    load_wall = time.perf_counter() - t0
    print(f"  loaded in {load_wall:.1f}s — X={Xn.shape}  "
          f"counts.mean={counts_n.mean():.2f}  "
          f"counts.max={counts_n.max()}  n_classes={n_classes}")

    X = torch.from_numpy(Xn).to(device)
    boxes = torch.from_numpy(boxes_n).to(device)
    # classes -1 (padding) becomes max class index for the no-object
    # slot: the model uses (n_classes + 1) outputs, slot n_classes
    # being "background".  The Hungarian loss treats the GT padding
    # rows as no-object via the counts mask, so the -1 sentinel never
    # leaks into the loss.  Cast as-is.
    classes = torch.from_numpy(classes_n).to(device)
    # Hungarian set loss expects long-tensor class ids; cluttered
    # MNIST pipeline used non-negative ids. -1 indicates padding;
    # combined_set_loss masks by counts so the value here doesn't
    # matter for loss but it MUST be a valid index for any tensor
    # indexing the model might do.  Map -1 → 0 for safety; the
    # counts mask will exclude these from loss.
    classes_safe = torch.where(classes < 0, torch.zeros_like(classes),
                                 classes)
    counts = torch.from_numpy(counts_n).to(device)

    configs = [
        ("baseline",
         HyMeYOLOMulti(n_queries=8, n_classes=n_classes, d_hidden=32)),
        ("boxes-only",
         RicciHyMeYOLOMulti(n_box_queries=8, n_circle_queries=0,
                              ricci_modulation=False, n_classes=n_classes)),
        ("circles-only",
         RicciHyMeYOLOMulti(n_box_queries=0, n_circle_queries=6,
                              ricci_modulation=False, n_classes=n_classes)),
        ("boxes+circles",
         RicciHyMeYOLOMulti(n_box_queries=6, n_circle_queries=2,
                              ricci_modulation=False, n_classes=n_classes)),
        ("+ricci-mod",
         RicciHyMeYOLOMulti(n_box_queries=6, n_circle_queries=2,
                              ricci_modulation=True, n_classes=n_classes)),
    ]

    print(f"\n  {'config':<18s}  {'start':>7s}  {'end':>7s}  "
          f"{'drop':>5s}  {'wall':>6s}  box_acc circ_acc  mAP50 mAP50:95 mIoU")
    print(f"  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*6}  -------  --------  "
          f"----- -------- -----")

    results = []
    for label, model in configs:
        torch.manual_seed(args.seed)
        results.append(train_one_config(
            label, model, X, boxes, classes_safe, counts,
            epochs=args.epochs, lr=args.lr, device=device,
            batch_size=args.batch_size,
        ))

    # Acceptance: every config's loss drops ≥ 30%.
    all_pass = True
    print("\n=== VOC acceptance ===")
    for r in results:
        decreased = r["losses"][0] - r["losses"][-1] > 0.30 * r["losses"][0]
        r["acceptance_pass"] = bool(decreased)
        if not decreased:
            all_pass = False
        print(f"  {r['label']:<18s} : loss drop ≥ 30%? "
              f"{'PASS' if decreased else 'FAIL'}")
    print(f"\n  VOC SMOKE: {'PASS' if all_pass else 'FAIL'}")

    if args.jsonl_out:
        Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl_out, "w") as fh:
            for r in results:
                det = r.get("det_metrics", {})
                rec = {
                    "dataset":     f"voc{args.year}_{args.image_set}",
                    "label":       r["label"],
                    "n_images":    Xn.shape[0],
                    "input_size":  args.input_size,
                    "max_objects": args.max_objects,
                    "n_classes":   n_classes,
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
                    "mAP_50":      det.get("mAP_50"),
                    "mAP_50_95":   det.get("mAP_50_95"),
                    "mean_iou_matched": det.get("mean_iou_matched"),
                    "n_preds_used":     det.get("n_preds_used"),
                    "n_gts_total":      det.get("n_gts_total"),
                    "acceptance_pass": r["acceptance_pass"],
                    "losses_per_epoch": r["losses"],
                }
                fh.write(json.dumps(rec) + "\n")
        print(f"\n  Wrote {args.jsonl_out}")

    print(f"\n  Total wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
