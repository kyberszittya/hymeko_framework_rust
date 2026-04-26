"""Architecture-scale sweep for scalar spectral-entropy regularization.

Characterizes how the regularizer's accuracy boost (and its effect on
the spectrum) varies with network size. Keeps input (784) and output
(10) fixed, sweeps hidden widths (h0, h1) through a geometric
progression from the thesis-scale (16, 8) up to (512, 256).

For each size, runs N seeds × {baseline, scalar_entropy} at the best
single λ found in the λ sweep (λ=0.1). Reports per-size Δ-accuracy,
paired t-stat, win count, and final spectral entropy. A clean monotone
dilution curve (effect strong at small N, vanishing at large N) is
the hypothesis — the λ=0.1 mass might be right-sized only for a
narrow band of neuron counts.

Output:
- Console: per-size summary.
- CSV: `data/benchmarks/thesis_iv_scale_<timestamp>.csv` for plotting.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse infrastructure from the main benchmark.
sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark import (  # type: ignore
    MNISTScaled, mnist_loaders,
    normalized_laplacian_eigvals, spectral_entropy_bits,
)


@dataclass
class ScaleRunResult:
    h0: int
    h1: int
    n_spectral_neurons: int
    arm: str
    seed: int
    final_val_acc: float
    final_entropy: float
    wall_seconds: float


def train_once(
    h0: int, h1: int, arm: str, seed: int,
    *, epochs: int, lr: float, lam: float, batch_size: int,
    reg_every_n: int, device: torch.device,
) -> ScaleRunResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tr_loader, va_loader = mnist_loaders(seed, batch_size=batch_size)
    model = MNISTScaled(h0, h1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if arm == "scalar_entropy" and batch_idx % reg_every_n == 0:
                eigs = normalized_laplacian_eigvals(model.spectral_weights())
                if eigs is not None:
                    loss = loss + lam * spectral_entropy_bits(eigs)

            loss.backward()
            optim.step()

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in va_loader:
            x, y = x.to(device), y.to(device)
            correct += int((model(x).argmax(dim=1) == y).sum().item())
            total += y.size(0)

    with torch.no_grad():
        eigs = normalized_laplacian_eigvals(model.spectral_weights())
        entropy = float(spectral_entropy_bits(eigs).item()) if eigs is not None else 0.0

    return ScaleRunResult(
        h0=h0, h1=h1,
        n_spectral_neurons=784 + h0 + h1 + 10,
        arm=arm, seed=seed,
        final_val_acc=correct / total,
        final_entropy=entropy,
        wall_seconds=time.time() - t0,
    )


def run_sweep(
    sizes: list[tuple[int, int]], seeds: int, epochs: int, lr: float,
    lam: float, batch_size: int, reg_every_n: int,
    device: torch.device,
) -> list[ScaleRunResult]:
    results: list[ScaleRunResult] = []
    total = len(sizes) * 2 * seeds
    done = 0
    for h0, h1 in sizes:
        for arm in ("baseline", "scalar_entropy"):
            for seed in range(seeds):
                r = train_once(
                    h0, h1, arm, seed,
                    epochs=epochs, lr=lr, lam=lam,
                    batch_size=batch_size, reg_every_n=reg_every_n,
                    device=device,
                )
                results.append(r)
                done += 1
                print(f"  [{done}/{total}] ({h0},{h1}) {arm}/seed={seed} "
                      f"acc={r.final_val_acc:.4f} H={r.final_entropy:.3f} "
                      f"{r.wall_seconds:.1f}s", flush=True)
    return results


def summarise(results: list[ScaleRunResult]) -> None:
    # Group by (h0, h1).
    from collections import defaultdict
    by_size: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    entropy_by_size: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))

    for r in results:
        by_size[(r.h0, r.h1)][r.arm].append(r.final_val_acc)
        entropy_by_size[(r.h0, r.h1)][r.arm].append(r.final_entropy)

    print()
    print(f"{'size':>14}  {'N':>5}  {'base avg':>9}  {'ent avg':>9}  "
          f"{'Δ':>9}  {'t-stat':>7}  {'W/L/T':>9}  {'H base':>7}  {'H ent':>7}")
    for size in sorted(by_size.keys()):
        h0, h1 = size
        base = by_size[size]["baseline"]
        ent = by_size[size]["scalar_entropy"]
        if len(base) != len(ent):
            continue
        # Paired by seed index.
        deltas = [e - b for b, e in zip(base, ent)]
        mean_d = statistics.mean(deltas)
        sd_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        se = sd_d / (len(deltas) ** 0.5) if sd_d > 0 else 1.0
        t = mean_d / se if se > 0 else 0.0
        w = sum(1 for d in deltas if d > 0)
        l = sum(1 for d in deltas if d < 0)
        tie = sum(1 for d in deltas if d == 0)
        n_neurons = 784 + h0 + h1 + 10
        base_avg = statistics.mean(base)
        ent_avg = statistics.mean(ent)
        h_base = statistics.mean(entropy_by_size[size]["baseline"])
        h_ent = statistics.mean(entropy_by_size[size]["scalar_entropy"])
        print(f"  {h0:>4} → {h1:<5}  {n_neurons:>5}  "
              f"{base_avg:>9.4f}  {ent_avg:>9.4f}  {mean_d:>+9.5f}  "
              f"{t:>+7.2f}  {w:>3}/{l:>2}/{tie:>2}  {h_base:>7.3f}  {h_ent:>7.3f}")


def write_csv(results: list[ScaleRunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h0", "h1", "n_spectral_neurons", "arm", "seed",
                    "final_val_acc", "final_entropy", "wall_seconds"])
        for r in results:
            w.writerow([r.h0, r.h1, r.n_spectral_neurons, r.arm, r.seed,
                        f"{r.final_val_acc:.6f}",
                        f"{r.final_entropy:.6f}",
                        f"{r.wall_seconds:.2f}"])


DEFAULT_SIZES = [
    (8, 4),
    (16, 8),     # thesis scale
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--reg-every-n", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Sizes (h0, h1): {DEFAULT_SIZES}")
    print(f"Seeds: {args.seeds}  Epochs: {args.epochs}  λ: {args.lam}")
    print()

    t0 = time.time()
    results = run_sweep(
        DEFAULT_SIZES, args.seeds, args.epochs, args.lr,
        args.lam, args.batch_size, args.reg_every_n, device,
    )
    elapsed = time.time() - t0

    if not args.no_csv:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = args.out_dir / f"thesis_iv_scale_{stamp}.csv"
        write_csv(results, csv_path)
        print(f"\nWrote {len(results)} records to {csv_path}")

    summarise(results)
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
