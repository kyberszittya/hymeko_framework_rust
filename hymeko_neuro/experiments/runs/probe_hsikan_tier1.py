"""Tier-1 wall-speedup smoke for HSiKAN: try torch.compile and AMP on
top of the existing CR + cat fixes, on a quick 5ep / 2000-subset cell.
Report each variant's wall and AUC.

Usage:
    PYTHONPATH=$PWD python -m hymeko_neuro.experiments.runs.probe_hsikan_tier1 [--mode {baseline,compile,amp,compile_amp}]
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from hymeko_neuro.experiments.vision.vision_bench_cell import build_model, load_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline",
                    choices=["baseline", "compile", "amp", "compile_amp"])
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--train-subset", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="/tmp/torchvision_cache")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    device = torch.device("cuda")
    train_ds, test_ds = load_split("mnist", args.data_root, args.train_subset, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model("hsikan", h=28, w=28, n_classes=10, hidden=args.hidden).to(device)
    use_compile = args.mode in ("compile", "compile_amp")
    use_amp = args.mode in ("amp", "compile_amp")

    if use_compile:
        compile_mode = "reduce-overhead"
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    t0 = time.monotonic()
    for _ in range(args.n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if use_amp:
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

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(x).argmax(dim=1)
            else:
                pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
    acc = correct / total

    print(json.dumps({
        "mode": args.mode, "test_accuracy": acc, "train_time_s": round(train_time_s, 2),
        "n_epochs": args.n_epochs, "train_subset": args.train_subset,
        "batch_size": args.batch_size, "hidden": args.hidden, "seed": args.seed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
