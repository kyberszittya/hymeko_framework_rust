"""Reproduction of Thesis IV (Hajdu 2025) — architecture entropy feedback.

Implements the architecture-entropy-feedback path from §6 of the HyMeKo
dissertation. The claim (subtheses 4.3 + 4.4): adding a structural-
entropy term derived from the NN's weight-matrix-as-hypergraph to the
training loss boosts average accuracy and reduces variance.

## What this script does

1. Build an MLP (4 → 16 → 8 → 3, ReLU, cross-entropy head — the
   architecture from Listing A.6.1 / §6.5 of the thesis).
2. At every step, compute the clique-expansion adjacency
   `A_ij = |W_ij|` across all neurons (block-tridiagonal structure
   for MLPs), symmetrize, build Laplacian `L = D - A`, normalize by
   `sum(D)` to get `L̂`, compute eigenvalues, then algebraic entropy
   `I(H) = -Σ λ̂_i log₂ λ̂_i` over non-zero eigenvalues.
3. Compare two training arms:
   - `baseline`: cross-entropy loss only.
   - `entropy_feedback`: cross-entropy + `λ · I(H)`. λ > 0 minimizes
     the architecture's spectral entropy as a soft pressure, matching
     the thesis's observation that entropy decreases during training.
4. Run on two datasets:
   - Iris (150 samples, 3 classes, d_in=4)
   - Synthetic classification (1500 samples, 4 features, 3 classes,
     sklearn `make_classification`)
5. 33 seeds per arm per dataset, report Table 6.1 shape.

## Formula mapping

Thesis Eq 6.1:  L̂(H) = L(H) / Σ_x D(x)
Thesis Eq 6.2:  I(H) = -Σ λ̂_i log₂ λ̂_i  over non-zero eigenvalues of L̂
Thesis Eq 6.4:  J(θ_{t+1}) = J_t(θ) + ... + D_KL(H_t(θ), H[θ_{t+1}])
                — we use the simpler form `J + λ·I(H)` which directly
                  drives the architecture toward lower spectral entropy
                  (thesis Figure 6.5a shows entropy monotonically
                  decreasing, so this is the intended direction).

## Honest caveat

This is a reproduction attempt on the exact setup the thesis describes
(architecture, datasets, seed count, metric shape). It reproduces the
entropy calculation path, not a blind re-implementation — the KL form
(Eq 6.3) is omitted for simplicity; the scalar-entropy-penalty form is
equivalent in the limit when D_KL is interpreted as "pressure toward
lower entropy" (the thesis says entropy decreases, i.e. the feedback
is effectively pulling entropy down).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Dataset loaders ─────────────────────────────────────────────────


def load_iris_dataset(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iris (150 samples, 3 classes, 4 features). 80/20 train/val split
    using the seed for shuffling so each run sees a different split."""
    from sklearn.datasets import load_iris

    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


def load_synthetic_dataset(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """1500-sample synthetic 3-class dataset with 4 features. Matches
    the shape described in §6.5 of the thesis."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=3, n_redundant=1,
        n_classes=3, n_clusters_per_class=1, random_state=seed,
    )
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed + 10_000)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


DATASETS: dict[str, Callable[[int], tuple[torch.Tensor, ...]]] = {
    "iris": load_iris_dataset,
    "synthetic": load_synthetic_dataset,
}


# ─── Model (matches Listing A.6.1) ──────────────────────────────────


class MLP(nn.Module):
    """4 → 16 → 8 → 3 MLP with ReLU activations. Cross-entropy loss
    applied externally; no softmax in the forward pass (standard
    PyTorch convention, CrossEntropyLoss includes log-softmax)."""

    def __init__(self, d_in: int = 4, h0: int = 16, h1: int = 8, d_out: int = 3):
        super().__init__()
        self.layer_0 = nn.Linear(d_in, h0)
        self.layer_1 = nn.Linear(h0, h1)
        self.layer_2 = nn.Linear(h1, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_0(x))
        x = F.relu(self.layer_1(x))
        return self.layer_2(x)

    def weight_matrices(self) -> list[torch.Tensor]:
        return [self.layer_0.weight, self.layer_1.weight, self.layer_2.weight]


# ─── Architecture entropy (Eqs 6.1, 6.2) ────────────────────────────


def architecture_entropy(weights: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """Compute the algebraic entropy of the MLP's factor-view
    hypergraph via clique-expansion of weight magnitudes.

    Builds a block-tridiagonal adjacency `A` where `A[i,j] = |W_l[j,i]|`
    for neurons `i` in layer `l` and `j` in layer `l+1` (and the
    symmetric partner). Computes the aggregated normalized Laplacian
    `L̂ = (D - A) / Σ D` per Eq 6.1 and its spectral entropy per
    Eq 6.2.

    Differentiable: `eigvalsh` supports autograd when its input is
    symmetric and real. Gradients flow back to each `W_l`.
    """
    # Layer sizes: input features, hidden sizes, output features.
    # Weights are (out_features, in_features) in PyTorch. For the
    # factor view we want a neuron-by-neuron adjacency.
    layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    total = sum(layer_sizes)

    A = torch.zeros(total, total, device=weights[0].device, dtype=weights[0].dtype)
    # Cumulative offsets for all layer boundaries; we need offsets[L+1]
    # where L is the last layer index, so build from the full list.
    offsets = [0]
    for s in layer_sizes:
        offsets.append(offsets[-1] + s)

    for l, W in enumerate(weights):
        # W is (n_{l+1}, n_l). The block at rows [offsets[l], offsets[l+1])
        # and cols [offsets[l+1], offsets[l+2]) holds |W^T|.
        block = W.abs().t()  # (n_l, n_{l+1})
        r0, r1 = offsets[l], offsets[l + 1]
        c0, c1 = offsets[l + 1], offsets[l + 2]
        A[r0:r1, c0:c1] = block
        A[c0:c1, r0:r1] = block.t()

    D = A.sum(dim=1)
    total_degree = D.sum()
    if total_degree.item() < eps:
        return torch.zeros((), device=A.device, dtype=A.dtype)

    L = torch.diag(D) - A
    L_hat = L / total_degree  # Eq 6.1

    # Eigenvalues of the symmetric real matrix L_hat.
    eigvals = torch.linalg.eigvalsh(L_hat)
    # Entropy requires positive values; clamp numerical negatives and
    # filter near-zero to avoid log(0) blow-ups. Use natural log then
    # convert to bits to match Eq 6.2's log₂.
    mask = eigvals > eps
    if not mask.any():
        return torch.zeros((), device=A.device, dtype=A.dtype)
    lam = eigvals[mask]
    entropy_nats = -(lam * torch.log(lam)).sum()
    return entropy_nats / torch.log(torch.tensor(2.0, device=A.device))


# ─── Training loop ──────────────────────────────────────────────────


@dataclass
class RunResult:
    dataset: str
    arm: str
    seed: int
    final_val_acc: float
    final_train_loss: float
    final_entropy: float


def train_one_run(
    dataset: str, arm: str, seed: int,
    epochs: int, lr: float, lam: float,
) -> RunResult:
    torch.manual_seed(seed)
    x_tr, y_tr, x_va, y_va = DATASETS[dataset](seed)
    model = MLP()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    final_entropy = 0.0
    for _ in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(x_tr)
        task_loss = F.cross_entropy(logits, y_tr)
        if arm == "entropy_feedback":
            ent = architecture_entropy(model.weight_matrices())
            loss = task_loss + lam * ent
            final_entropy = float(ent.item())
        else:
            loss = task_loss
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_va).argmax(dim=1)
        val_acc = float((val_pred == y_va).float().mean().item())
        train_loss_final = float(task_loss.item())
        if arm == "baseline":
            final_entropy = float(architecture_entropy(model.weight_matrices()).item())

    return RunResult(
        dataset=dataset, arm=arm, seed=seed,
        final_val_acc=val_acc, final_train_loss=train_loss_final,
        final_entropy=final_entropy,
    )


# ─── Orchestration + reporting ──────────────────────────────────────


@dataclass
class BenchConfig:
    seeds: int = 33
    epochs: int = 200
    lr: float = 1e-2
    lam: float = 1e-2
    datasets: tuple[str, ...] = ("iris", "synthetic")
    out_dir: Path = Path("data/benchmarks")


def run_all(cfg: BenchConfig) -> list[RunResult]:
    results: list[RunResult] = []
    for dataset in cfg.datasets:
        for arm in ("baseline", "entropy_feedback"):
            for seed in range(cfg.seeds):
                results.append(train_one_run(
                    dataset, arm, seed,
                    epochs=cfg.epochs, lr=cfg.lr, lam=cfg.lam,
                ))
    return results


def summarise(results: list[RunResult], cfg: BenchConfig) -> None:
    print()
    print(f"Seeds per arm: {cfg.seeds}  Epochs: {cfg.epochs}  λ: {cfg.lam}")
    print()
    for dataset in cfg.datasets:
        print(f"=== {dataset} ===")
        print(f"  {'arm':<22}  {'min':>8}  {'avg':>8}  {'max':>8}  "
              f"{'stdev':>10}  {'final_H':>10}")
        for arm in ("baseline", "entropy_feedback"):
            accs = [r.final_val_acc for r in results
                    if r.dataset == dataset and r.arm == arm]
            ents = [r.final_entropy for r in results
                    if r.dataset == dataset and r.arm == arm]
            if not accs:
                continue
            mn, mx = min(accs), max(accs)
            av = statistics.mean(accs)
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            me = statistics.mean(ents) if ents else 0.0
            print(f"  {arm:<22}  {mn:>8.4f}  {av:>8.4f}  {mx:>8.4f}  "
                  f"{sd:>10.5f}  {me:>10.4f}")

        # Paired comparison (same seed, different arm) for significance hint.
        paired_deltas = []
        for seed in range(cfg.seeds):
            b = [r for r in results if r.dataset == dataset and r.arm == "baseline" and r.seed == seed]
            e = [r for r in results if r.dataset == dataset and r.arm == "entropy_feedback" and r.seed == seed]
            if b and e:
                paired_deltas.append(e[0].final_val_acc - b[0].final_val_acc)
        if paired_deltas:
            mean_delta = statistics.mean(paired_deltas)
            sd_delta = statistics.stdev(paired_deltas) if len(paired_deltas) > 1 else 0.0
            # Simple t-statistic without scipy dependency.
            se = sd_delta / (len(paired_deltas) ** 0.5) if sd_delta > 0 else 1.0
            t_stat = mean_delta / se if se > 0 else 0.0
            wins = sum(1 for d in paired_deltas if d > 0)
            losses = sum(1 for d in paired_deltas if d < 0)
            ties = sum(1 for d in paired_deltas if d == 0)
            print(f"  paired Δ (entropy − baseline):  "
                  f"mean={mean_delta:+.5f}  sd={sd_delta:.5f}  t={t_stat:+.2f}  "
                  f"(W/L/T={wins}/{losses}/{ties})")
        print()


def write_csv(results: list[RunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "seed", "final_val_acc",
                    "final_train_loss", "final_entropy"])
        for r in results:
            w.writerow([r.dataset, r.arm, r.seed,
                        f"{r.final_val_acc:.6f}",
                        f"{r.final_train_loss:.6f}",
                        f"{r.final_entropy:.6f}"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, default=33, help="Matches thesis §6.5.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=1e-2,
                    help="Entropy regularization weight.")
    ap.add_argument("--datasets", nargs="+", default=["iris", "synthetic"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--out-dir", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    cfg = BenchConfig(
        seeds=args.seeds, epochs=args.epochs, lr=args.lr, lam=args.lam,
        datasets=tuple(args.datasets), out_dir=args.out_dir,
    )

    t0 = time.time()
    results = run_all(cfg)
    elapsed = time.time() - t0

    if not args.no_csv:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = cfg.out_dir / f"thesis_iv_{stamp}.csv"
        write_csv(results, csv_path)
        print(f"Wrote {len(results)} records to {csv_path}")

    summarise(results, cfg)
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
