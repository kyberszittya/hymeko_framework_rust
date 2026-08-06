"""Unified single-cell runner for the hypergraph-vision vs CNN re-benchmark.

Trains ONE (model, dataset, seed) cell at a fixed budget and prints a JSON
result row. Dispatches over the five image classifiers — all share the
``(B, 1, 28, 28) -> (B, n_classes)`` interface:

    cnn         TinyCNN                 (translation-equivariant baseline)
    mlp         MLP                     (structure-free baseline)
    hgnn        NeocogHGNN              (Feng-2019 receptive-field hyperconv)
    hsikan      HSiKANVisionClassifier  (signed-branch Catmull-Rom hyperconv)
    ricci_stim  RicciStimClassifier     (quadtree -> signed patch-graph ->
                                         Bochner walks/polygons/triangles)

All models train on the SAME ``--train-subset N`` and ``--n-epochs`` so the
comparison is apples-to-apples and the per-image RicciStim path stays
tractable (it sets the budget ceiling). See
``docs/plans/2026-05-28-vision-hypergraph-vs-cnn-rebench/``.

Usage
-----
    python -m hymeko_neuro.experiments.vision.vision_bench_cell \\
        --model hsikan --dataset mnist --n-epochs 15 \\
        --train-subset 8000 --batch-size 128 --hidden 32 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

MODEL_NAMES = ("cnn", "mlp", "hgnn", "hsikan", "ricci_stim", "fuzzy_sig")
DATASETS = ("mnist", "fashion")
DEFAULT_DATA_ROOT = "/tmp/torchvision_cache"


def subset_indices(n_total: int, n_subset: int, seed: int) -> list[int]:
    """Deterministic training subset: a seed-shuffled prefix of ``n_total``.

    ``n_subset <= 0`` or ``>= n_total`` selects everything (full dataset).
    Pure (no torch) so it is unit-testable in isolation.
    """
    if n_subset <= 0 or n_subset >= n_total:
        return list(range(n_total))
    import random

    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    return idx[:n_subset]


def build_model(name: str, *, h: int, w: int, n_classes: int, hidden: int,
                tie_we: bool = False, spatial_filter: str = "none",
                n_layers: int = 2, pooling: str = "sum",
                t_norm_kind: str = "product",
                t_conorm_kind: str = "probsum"):
    """Construct one classifier by name. Model classes imported lazily so
    pure helpers (and the orchestrator) need not pull torch/the wheel.

    ``tie_we`` (bool) / ``spatial_filter`` / ``n_layers`` / ``pooling``
    ({"sum","min","product","lukasiewicz"}): only consumed by ``hsikan``;
    silently ignored by other models.

    ``t_norm_kind`` / ``t_conorm_kind``: only consumed by ``fuzzy_sig``;
    silently ignored by other models.
    """
    if name == "cnn":
        from hymeko_neuro.experiments.vision.neocog_hgnn import TinyCNN

        return TinyCNN(n_classes, hidden=hidden)
    if name == "mlp":
        from hymeko_neuro.experiments.vision.neocog_hgnn import MLP

        return MLP(h * w, n_classes, hidden=hidden)
    if name == "hgnn":
        from hymeko_neuro.experiments.vision.neocog_hgnn import NeocogHGNN

        return NeocogHGNN(h, w, n_classes, hidden=hidden)
    if name == "hsikan":
        from hymeko_neuro.experiments.vision.hsikan_vision import HSiKANVisionClassifier

        return HSiKANVisionClassifier(h, w, n_classes, hidden=hidden,
                                      n_layers=n_layers,
                                      tie_we=tie_we,
                                      spatial_filter=spatial_filter,
                                      pooling=pooling)
    if name == "ricci_stim":
        from hymeko_neuro.models.hymeko_gomb.soma.vision import RicciStimClassifier

        return RicciStimClassifier(
            image_h=h, image_w=w, d_hidden=hidden, n_classes=n_classes,
        )
    if name == "fuzzy_sig":
        from hymeko_neuro.experiments.vision.fuzzy_signature import (
            FuzzySignatureClassifier,
        )

        return FuzzySignatureClassifier(
            H=h, W=w, n_classes=n_classes, d=hidden, n_layers=n_layers,
            t_norm_kind=t_norm_kind, t_conorm_kind=t_conorm_kind,
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def load_split(dataset: str, root: str, train_subset: int, seed: int):
    """Return (train_dataset, test_dataset) with the train subset applied."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import Subset

    tf = T.Compose([T.ToTensor()])
    cls = {
        "mnist": torchvision.datasets.MNIST,
        "fashion": torchvision.datasets.FashionMNIST,
    }[dataset]
    train = cls(root=root, train=True, download=True, transform=tf)
    test = cls(root=root, train=False, download=True, transform=tf)
    idx = subset_indices(len(train), train_subset, seed)
    if len(idx) < len(train):
        train = Subset(train, idx)
    return train, test


def _accuracy(model, loader, device) -> float:
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
    return correct / total if total else float("nan")


def train_and_eval(
    *, model_name: str, dataset: str, n_epochs: int, train_subset: int,
    batch_size: int, hidden: int, lr: float, seed: int, data_root: str,
    tie_we: bool = False, compile_model: bool = False, amp: bool = False,
    spatial_filter: str = "none", n_layers: int = 2, pooling: str = "sum",
    t_norm_kind: str = "product", t_conorm_kind: str = "probsum",
) -> dict:
    """Train one cell and return a result dict (accuracy, params, wall)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = load_split(dataset, data_root, train_subset, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_model(model_name, h=28, w=28, n_classes=10, hidden=hidden,
                        tie_we=tie_we, spatial_filter=spatial_filter,
                        n_layers=n_layers, pooling=pooling,
                        t_norm_kind=t_norm_kind,
                        t_conorm_kind=t_conorm_kind).to(device)
    if compile_model:
        # `reduce-overhead` mode is right for many small calls; trace
        # cost amortises after ~50 batches. Profiled wins on HSiKAN at
        # n_epochs ≥ 10 (2026-05-29 Tier-1 probe: 67.7 s → 54.7 s at 10 ep).
        model = torch.compile(model, mode="reduce-overhead")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if amp else None

    t0 = time.monotonic()
    for _ in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if amp:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
    train_time_s = time.monotonic() - t0

    acc = _accuracy(model, test_loader, device)
    return {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "n_epochs": n_epochs,
        "train_subset": train_subset,
        "hidden": hidden,
        "lr": lr,
        "tie_we": bool(tie_we),
        "spatial_filter": str(spatial_filter),
        "n_layers": int(n_layers),
        "pooling": str(pooling),
        "t_norm_kind": str(t_norm_kind),
        "t_conorm_kind": str(t_conorm_kind),
        "compile": bool(compile_model),
        "amp": bool(amp),
        "test_accuracy": acc,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_time_s": round(train_time_s, 2),
        "device": device.type,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--dataset", default="mnist", choices=DATASETS)
    ap.add_argument("--n-epochs", type=int, default=15)
    ap.add_argument("--train-subset", type=int, default=8000,
                    help="0 = full training set; else N seed-shuffled samples.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2,
                    help="HSiKAN depth axis (number of HSiKANVisionLayer "
                         "blocks); silently ignored by non-hsikan models.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--tie-we", action="store_true",
                    help="HSiKAN translation-equivariance variant: tie "
                         "the per-edge weight W_e to a single scalar "
                         "shared across all RF positions. Silently "
                         "ignored by other models.")
    ap.add_argument("--spatial-filter", default="none",
                    choices=["none", "scalar", "per_channel"],
                    help="HSiKAN within-RF spatial filter. "
                         "'scalar': W_pos[K] (one scalar per "
                         "position-within-RF, channel-invariant). "
                         "'per_channel': W_pos[K, d_out] (per-output-channel). "
                         "W_pos init = ones matches uniform-mean baseline "
                         "bit-for-bit either way.")
    ap.add_argument("--pooling", default="sum",
                    choices=["sum", "min", "product", "lukasiewicz"],
                    help="HSiKAN aggregation t-norm (2026-05-30 fuzzy-"
                         "signature work). 'sum' = current behavior "
                         "(CNN/HGNN view, NOT a t-norm). 'min' = Gödel; "
                         "'product' = algebraic; 'lukasiewicz' = bounded. "
                         "T-norm modes apply sigmoid to inputs before "
                         "pooling so values lie in [0,1].")
    ap.add_argument("--t-norm", default="product",
                    choices=["min", "product", "lukasiewicz"],
                    help="FuzzySignatureLayer t-norm (fuzzy AND).")
    ap.add_argument("--t-conorm", default="probsum",
                    choices=["max", "probsum", "lukasiewicz"],
                    help="FuzzySignatureLayer t-conorm (fuzzy OR).")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(model, mode='reduce-overhead'). "
                         "Profiled win on HSiKAN at n_epochs >= 10.")
    ap.add_argument("--amp", action="store_true",
                    help="autocast + GradScaler. Profiled NULL on HSiKAN "
                         "MNIST (model too small for FP16 matmul to "
                         "amortise autocast overhead); included for "
                         "completeness.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the resolved cell config and exit (no torch).")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(json.dumps({
            "model": args.model, "dataset": args.dataset, "seed": args.seed,
            "n_epochs": args.n_epochs, "train_subset": args.train_subset,
            "batch_size": args.batch_size, "hidden": args.hidden, "lr": args.lr,
            "tie_we": args.tie_we, "spatial_filter": args.spatial_filter,
            "n_layers": args.n_layers, "pooling": args.pooling,
            "compile": args.compile, "amp": args.amp,
        }))
        return 0

    row = train_and_eval(
        model_name=args.model, dataset=args.dataset, n_epochs=args.n_epochs,
        train_subset=args.train_subset, batch_size=args.batch_size,
        hidden=args.hidden, lr=args.lr, seed=args.seed, data_root=args.data_root,
        tie_we=args.tie_we, compile_model=args.compile, amp=args.amp,
        spatial_filter=args.spatial_filter, n_layers=args.n_layers,
        pooling=args.pooling,
        t_norm_kind=args.t_norm, t_conorm_kind=args.t_conorm,
    )
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
