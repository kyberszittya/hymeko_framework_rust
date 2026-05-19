"""GömbSoma MNIST benchmark — Phase 3-V-bench.

First hard number for the GömbSoma vision stack: train
WalkConvImageClassifier on MNIST for n_epochs with n_seeds different
random initialisations; report mean ± pstd test accuracy, parameter
count, and wall time.

A parameter-light Linear-baseline (Linear(784, 10) = 7 850 params)
trains alongside as a control: if GömbSoma's 2 010 params can't
beat a 7 850-param linear classifier on MNIST, the walks-only
sensorimotor hypothesis is in trouble. If it ties or wins at fewer
params, walks are doing real structural work.

Plan: docs/plans/2026-05-14-gomb-soma/.

Run via the orchestrator script:
    signedkan_wip/experiments/run_gomb_soma_mnist_bench_2026_05_14.sh
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    WalkConvImageClassifier,
)

DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "mnist"


class LinearBaseline(nn.Module):
    """Trivial reference: flatten + Linear → 10. ~7 850 params."""

    def __init__(self, image_h: int = 28, image_w: int = 28,
                 in_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels * image_h * image_w, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            return self.fc(images.reshape(-1))
        return self.fc(images.reshape(images.shape[0], -1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(n_train: int, n_test: int, batch_size: int, seed: int):
    """MNIST with sub-sampling for speed."""
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(
        str(DATA_ROOT), train=True, download=True, transform=tfm,
    )
    test = datasets.MNIST(
        str(DATA_ROOT), train=False, download=True, transform=tfm,
    )
    rng = np.random.default_rng(seed)
    if n_train < len(train):
        idx = rng.choice(len(train), size=n_train, replace=False)
        train = Subset(train, idx.tolist())
    if n_test < len(test):
        idx = rng.choice(len(test), size=n_test, replace=False)
        test = Subset(test, idx.tolist())
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader


def build_model(model_type: str, device: torch.device) -> nn.Module:
    if model_type == "gomb_soma":
        m = WalkConvImageClassifier(
            image_h=28, image_w=28, patch_size=4,
            in_channels=1, d_hidden=16, n_classes=10,
        )
    elif model_type == "linear":
        m = LinearBaseline()
    else:
        raise SystemExit(f"unknown model_type {model_type!r}")
    return m.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            n_correct += (pred == y).sum().item()
            n_total += y.shape[0]
    return n_correct / max(1, n_total)


def train_one_seed(
    model_type: str,
    seed: int,
    n_train: int,
    n_test: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    set_seed(seed)
    train_loader, test_loader = load_mnist(n_train, n_test, batch_size, seed)
    model = build_model(model_type, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        model.train()
        ep_loss, n_steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_steps += 1
        train_loss = ep_loss / max(1, n_steps)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"  [seed={seed}] epoch {epoch + 1}/{n_epochs} "
            f"loss={train_loss:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )
    wall = time.perf_counter() - t0
    final_acc = evaluate(model, test_loader, device)
    return {
        "model": model_type,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_params": n_params,
        "test_acc": final_acc,
        "wall_s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("gomb_soma", "linear"),
                     default="gomb_soma")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default=None,
                     help="cuda / cpu; auto-detect if omitted")
    ap.add_argument("--out-jsonl", default=None,
                     help="append per-seed records to this JSONL")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[bench] model={args.model} device={device} "
          f"n_train={args.n_train} n_test={args.n_test} "
          f"n_epochs={args.n_epochs}", flush=True)

    rows = []
    for seed in args.seeds:
        print(f"[bench] seed={seed}", flush=True)
        rec = train_one_seed(
            args.model, seed, args.n_train, args.n_test,
            args.n_epochs, args.batch_size, args.lr, device,
        )
        rows.append(rec)
        if args.out_jsonl:
            with open(args.out_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")

    accs = [r["test_acc"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    print(f"[bench] === summary ({args.model}, n={len(accs)} seeds) ===",
          flush=True)
    print(f"[bench]   n_params = {rows[0]['n_params']}", flush=True)
    print(f"[bench]   test_acc per seed: {[round(a, 4) for a in accs]}",
          flush=True)
    if len(accs) > 1:
        m = statistics.mean(accs)
        s = statistics.pstdev(accs)
        print(f"[bench]   mean = {m:.4f}  pstd = {s:.4f}", flush=True)
    print(f"[bench]   wall per seed: "
          f"{[round(w, 1) for w in walls]} s", flush=True)


if __name__ == "__main__":
    main()
