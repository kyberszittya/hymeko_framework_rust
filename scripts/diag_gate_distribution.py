"""Snapshot the per-image gate distribution of a trained nodelet-head
checkpoint on the VOC trainval split. Used to produce the diagnostic
table in Stage D-3 / D-3-bis reports.

Usage:
    python scripts/diag_gate_distribution.py \\
        --ckpt path/to/stage_d3bis_seed0.pt \\
        --n-images 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
    RicciHyMeYOLOMulti,
)
from signedkan_wip.src.vision.voc_dataset import (
    VOC_CLASSES, load_voc_hungarian,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-images", type=int, default=64)
    p.add_argument("--data-root", default="data/torchvision")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                       else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    n_classes = ckpt.get("n_classes", len(VOC_CLASSES))
    n_box_queries = ckpt.get("n_box_queries", 16)
    backbone = ckpt.get("backbone", "resnet18_imagenet")
    input_size = ckpt.get("input_size", 224)

    model = RicciHyMeYOLOMulti(
        n_box_queries=n_box_queries,
        n_circle_queries=0,
        n_classes=n_classes,
        d_hidden=32,
        ricci_modulation=True,
        ricci_scale=ckpt.get("ricci_scale", 1.0),
        use_layernorm=False,
        backbone=backbone,
        fpn="2level",
        query_head_kind="nodelet",
    )
    model.load_state_dict(sd)
    model = model.to(args.device).eval()

    Xn, _, _, _, _ = load_voc_hungarian(
        year="2007", image_set="trainval",
        input_size=input_size, max_objects=12,
        root=args.data_root, subset_n=args.n_images, download=False,
    )
    X = torch.from_numpy(Xn).to(args.device)
    with torch.no_grad():
        out = model(X)
    gates = out["box_gates"].cpu().numpy()  # (B, n_q)

    flat = gates.ravel()
    above = float((flat > args.threshold).mean())
    above_03 = float((flat > 0.3).mean())

    sorted_gates = np.sort(gates[0])
    print("image 0 gate values (sorted):")
    print(f"  {sorted_gates.round(3).tolist()}")

    print()
    print(f"n_images = {gates.shape[0]}")
    print(f"n_queries_per_image = {gates.shape[1]}")
    print(f"min  = {flat.min():.4f}")
    print(f"mean = {flat.mean():.4f}")
    print(f"max  = {flat.max():.4f}")
    print(f"std  = {flat.std():.4f}")
    print(f"fraction > {args.threshold} = {above:.4f}")
    print(f"fraction > 0.3 = {above_03:.4f}")
    # Per-image firing fraction (mean across images, vs across all queries).
    per_image_firing = (gates > args.threshold).mean(axis=1)
    print(f"per-image firing fraction "
          f"(mean over {gates.shape[0]} images, threshold {args.threshold}): "
          f"{per_image_firing.mean():.4f} "
          f"(σ {per_image_firing.std():.4f}, "
          f"min {per_image_firing.min():.4f}, "
          f"max {per_image_firing.max():.4f})")


if __name__ == "__main__":
    main()
