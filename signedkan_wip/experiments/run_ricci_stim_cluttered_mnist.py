"""Ricci-Stim Cluttered MNIST training runner.

Drives `RicciStimDetector` against the Cluttered MNIST dataset, using
the training infrastructure in
`signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_train`. Supports
the 5-config ablation matrix from the Phase 8-bench plan.

Usage::

    python -m signedkan_wip.experiments.run_ricci_stim_cluttered_mnist \\
        --config E --n-train 5000 --n-eval 1000 --n-epochs 20 \\
        --seed 0 --device cuda --out-jsonl results.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from signedkan_wip.src.hymeko_gomb.soma.vision import RicciStimDetector
from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_train import (
    train_one_seed,
)
from signedkan_wip.src.vision.cluttered_mnist import (
    ClutteredMNIST, collate_cluttered,
)


# ---------------------------------------------------------------------
# Ablation configs (from the Phase 8-bench plan)
# ---------------------------------------------------------------------

CONFIGS: dict[str, dict] = {
    "A": dict(bochner_alpha=0.0, bochner_beta=0.0, use_sdrf=False),
    "B": dict(bochner_alpha=0.1, bochner_beta=0.0, use_sdrf=False),
    "C": dict(bochner_alpha=0.0, bochner_beta=0.1, use_sdrf=False),
    "D": dict(bochner_alpha=0.1, bochner_beta=0.1, use_sdrf=False),
    # 2026-05-15: Config E was the planned headline (Bochner full + SDRF).
    # 2026-05-16 measurements showed SDRF rewiring is **net negative** on
    # Cluttered MNIST: Config D landed at mAP50_proxy=0.174 vs Config E at
    # 0.141 — a -0.033 regression. The Hodge-vectorize report
    # (`reports/2026-05-16-gomb-soma-hodge-vectorize.md` §11) recommends
    # dropping SDRF from the canonical operational config. We preserve
    # Config E for backward-compatible measurement reproduction and
    # introduce Config F as the new canonical "Bochner full, no SDRF"
    # recommendation (== Config D under another name, but Config D is
    # part of the orthogonal ablation grid; Config F is the *headline*).
    "E": dict(bochner_alpha=0.1, bochner_beta=0.1, use_sdrf=True),
    # 2026-05-17 canonical: Bochner full, no SDRF. Same as Config D but
    # promoted from "ablation cell" to "operational baseline". Saves
    # ~3 ms vs E (no SDRF rewire iterations) and gains +0.033 mAP50_proxy
    # on the 5000-image Cluttered MNIST grid measured 2026-05-16.
    "F": dict(bochner_alpha=0.1, bochner_beta=0.1, use_sdrf=False),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--max-digits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--iou-pos", type=float, default=0.3)
    ap.add_argument("--bbox-weight", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = CONFIGS[args.config]
    print(
        f"[bench] config={args.config} {cfg} seed={args.seed} "
        f"n_train={args.n_train} n_eval={args.n_eval} "
        f"n_epochs={args.n_epochs} device={device}",
        flush=True,
    )
    set_seed(args.seed)

    train_ds = ClutteredMNIST(
        n_samples=args.n_train, canvas=args.canvas,
        max_digits=args.max_digits, seed=args.seed, train=True,
    )
    eval_ds = ClutteredMNIST(
        n_samples=args.n_eval, canvas=args.canvas,
        max_digits=args.max_digits, seed=args.seed + 50_000, train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_cluttered, num_workers=0,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_cluttered, num_workers=0,
    )

    def detector_factory():
        # Anchor scales sized for 28-px digits on a 64-px canvas.
        return RicciStimDetector(
            image_h=args.canvas, image_w=args.canvas,
            patch_size_initial=args.canvas // 2,   # 32 for canvas=64
            patch_size_min=4,
            max_depth=3,
            max_anchors=256,
            in_channels=1, d_hidden=16, n_classes=10,
            **cfg,
        )

    t0 = time.perf_counter()
    result = train_one_seed(
        detector_factory, train_loader, eval_loader,
        n_epochs=args.n_epochs, lr=args.lr,
        bbox_weight=args.bbox_weight, iou_pos=args.iou_pos,
        device=device, eval_every_epoch=True,
    )
    wall = time.perf_counter() - t0

    record = {
        "config": args.config,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "n_epochs": args.n_epochs,
        "n_params": result["n_params"],
        "wall_s": wall,
        "history": result["history"],
        "final_mAP50_proxy": result["final_mAP50_proxy"],
        "ablation_settings": cfg,
    }
    print(f"[bench] DONE config={args.config} seed={args.seed} "
          f"final_mAP50_proxy={result['final_mAP50_proxy']} "
          f"wall={wall:.1f}s", flush=True)
    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
