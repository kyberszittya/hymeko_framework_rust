"""Scaffold benchmark for the entropy hot-swap weight-transfer path.

**Scope.** This measures a narrower question than the full entropy-
feedback claim: does mid-training architectural rewriting + compatible-
subset weight transfer produce a usable model, versus (a) training the
smaller architecture from scratch and (b) training the post-rewrite
architecture from scratch? The underlying layers are plain `nn.Linear`s
— the real signed-incidence + GGK kernel math from `ehk_torch` isn't
wired up yet, so this benchmark does **not** measure the theoretical
claim that structural entropy drives better representations. It
measures the plumbing: did the hot-swap preserve enough information
that continued training catches up?

**Task.** Deterministic synthetic regression y ∈ ℝ² from x ∈ ℝ³:
    y₀ = sin(x₀) + cos(x₁)
    y₁ = tanh(x₂² − x₀·x₁)
Nonlinear enough that hidden capacity matters, cheap enough to train
200 epochs in seconds.

**Conditions** (each run with `--seeds` independent seeds):
  - `baseline_small`  — train hidden=5 MLP for full budget
  - `baseline_large`  — train hidden=8 MLP for full budget (target arch)
  - `hotswap_widen`   — train hidden=5 for first half, rebuild as
                        hidden=8 with compatible weights transferred,
                        continue training for second half
  - `hotswap_same`    — train hidden=5 for first half, rebuild as
                        hidden=5 (same shape) — control that proves
                        the plumbing preserves the full state_dict
                        when no shape changes; should match
                        baseline_small trajectory exactly after swap

Writes `data/benchmarks/hotswap_<timestamp>.csv` (raw per-epoch) and
prints a summary comparison. CSV columns:
  `condition, seed, epoch, train_loss, val_loss`
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

# Ensure the in-repo ehk_torch_stub is importable without pip-install.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "python" / "ehk_torch_stub" / "src"))

from ehk_torch_stub import transfer_compatible_weights  # noqa: E402


# ─── Task + model ────────────────────────────────────────────────────


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (x, y) with the fixed synthetic relation described
    in the module docstring. Same `seed` → same data."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    y0 = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    y1 = torch.tanh(x[:, 2] ** 2 - x[:, 0] * x[:, 1])
    y = torch.stack([y0, y1], dim=1)
    return x, y


class MLP(nn.Module):
    """3 → hidden → 2 ReLU MLP. The hidden dim is the single knob that
    the hot-swap rewires mid-training."""

    def __init__(self, hidden: int):
        super().__init__()
        self.layer_0 = nn.Linear(3, hidden)
        self.act = nn.ReLU()
        self.layer_1 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_1(self.act(self.layer_0(x)))


def seeded_mlp(hidden: int, seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP(hidden)


# ─── Training loop ───────────────────────────────────────────────────


@dataclass
class TrainRecord:
    condition: str
    seed: int
    epoch: int
    train_loss: float
    val_loss: float


def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optim: torch.optim.Optimizer,
) -> float:
    model.train()
    optim.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    optim.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(nn.functional.mse_loss(model(x), y).item())


def train(
    model: nn.Module,
    x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_va: torch.Tensor, y_va: torch.Tensor,
    epochs: int,
    lr: float,
    condition: str,
    seed: int,
    start_epoch: int = 0,
) -> list[TrainRecord]:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    records: list[TrainRecord] = []
    for e in range(start_epoch, start_epoch + epochs):
        tl = train_one_epoch(model, x_tr, y_tr, optim)
        vl = eval_loss(model, x_va, y_va)
        records.append(TrainRecord(condition, seed, e, tl, vl))
    return records


# ─── Conditions ──────────────────────────────────────────────────────


def run_baseline(
    hidden: int, epochs: int, lr: float,
    x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_va: torch.Tensor, y_va: torch.Tensor,
    seed: int, condition: str,
) -> list[TrainRecord]:
    model = seeded_mlp(hidden, seed)
    return train(model, x_tr, y_tr, x_va, y_va, epochs, lr, condition, seed)


def run_hotswap(
    hidden_start: int, hidden_end: int, epochs: int, lr: float,
    swap_at: int,
    x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_va: torch.Tensor, y_va: torch.Tensor,
    seed: int,
    condition: str,
) -> tuple[list[TrainRecord], int]:
    """Train hidden=hidden_start for `swap_at` epochs, then rebuild as
    hidden=hidden_end, transfer compatible weights, continue to
    `epochs` total. Returns (records, n_transferred_keys)."""
    records: list[TrainRecord] = []

    old = seeded_mlp(hidden_start, seed)
    records.extend(train(old, x_tr, y_tr, x_va, y_va,
                         swap_at, lr, condition, seed))

    # Rebuild with the new hidden width. Use a different seed offset
    # so fresh params for the extra neurons aren't identical to the
    # old run's trajectory.
    new = seeded_mlp(hidden_end, seed + 10_000)
    report = transfer_compatible_weights(old, new)

    # Continue training.
    records.extend(train(new, x_tr, y_tr, x_va, y_va,
                         epochs - swap_at, lr, condition, seed,
                         start_epoch=swap_at))

    return records, len(report.transferred)


# ─── Orchestration ──────────────────────────────────────────────────


@dataclass
class BenchConfig:
    seeds: int = 5
    epochs: int = 200
    swap_at: int = 100
    lr: float = 1e-2
    n_train: int = 512
    n_val: int = 128
    hidden_small: int = 5
    hidden_large: int = 8
    out_dir: Path = Path("data/benchmarks")


def run_all(cfg: BenchConfig) -> tuple[list[TrainRecord], dict]:
    all_records: list[TrainRecord] = []
    widen_transfer: list[int] = []
    same_transfer: list[int] = []

    for seed in range(cfg.seeds):
        # Data is seed-coupled so each run sees the same task.
        x_tr, y_tr = make_dataset(cfg.n_train, seed)
        x_va, y_va = make_dataset(cfg.n_val, seed + 100_000)

        all_records.extend(run_baseline(
            cfg.hidden_small, cfg.epochs, cfg.lr,
            x_tr, y_tr, x_va, y_va, seed, "baseline_small",
        ))
        all_records.extend(run_baseline(
            cfg.hidden_large, cfg.epochs, cfg.lr,
            x_tr, y_tr, x_va, y_va, seed, "baseline_large",
        ))
        widen_records, n_widen = run_hotswap(
            cfg.hidden_small, cfg.hidden_large,
            cfg.epochs, cfg.lr, cfg.swap_at,
            x_tr, y_tr, x_va, y_va, seed, "hotswap_widen",
        )
        all_records.extend(widen_records)
        widen_transfer.append(n_widen)

        same_records, n_same = run_hotswap(
            cfg.hidden_small, cfg.hidden_small,
            cfg.epochs, cfg.lr, cfg.swap_at,
            x_tr, y_tr, x_va, y_va, seed, "hotswap_same",
        )
        all_records.extend(same_records)
        same_transfer.append(n_same)

    stats = summarise(all_records, widen_transfer, same_transfer, cfg)
    return all_records, stats


def summarise(
    records: list[TrainRecord],
    widen_transfer: list[int],
    same_transfer: list[int],
    cfg: BenchConfig,
) -> dict:
    by_cond: dict[str, list[float]] = {}
    for r in records:
        if r.epoch == cfg.epochs - 1:
            by_cond.setdefault(r.condition, []).append(r.val_loss)
    summary = {}
    for cond, losses in by_cond.items():
        summary[cond] = {
            "mean_val_loss": statistics.mean(losses),
            "std_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "n_seeds": len(losses),
            "final_losses": losses,
        }
    summary["transfer_keys_widen_per_run"] = widen_transfer
    summary["transfer_keys_same_per_run"] = same_transfer
    return summary


def write_csv(records: list[TrainRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "epoch", "train_loss", "val_loss"])
        for r in records:
            w.writerow([r.condition, r.seed, r.epoch,
                        f"{r.train_loss:.6f}", f"{r.val_loss:.6f}"])


def print_summary(stats: dict) -> None:
    print()
    print("Final val loss (epoch N-1), averaged across seeds:")
    print(f"  {'condition':<20}  {'mean':>10}  {'± std':>10}  {'n':>3}")
    for cond in ("baseline_small", "hotswap_same", "hotswap_widen", "baseline_large"):
        s = stats.get(cond)
        if s is None:
            continue
        print(f"  {cond:<20}  {s['mean_val_loss']:>10.5f}  "
              f"{s['std_val_loss']:>10.5f}  {s['n_seeds']:>3}")
    print()
    widen_tc = stats.get("transfer_keys_widen_per_run", [])
    same_tc = stats.get("transfer_keys_same_per_run", [])
    if widen_tc:
        print(f"hotswap_widen transfer keys per run: {widen_tc} "
              f"(mean {statistics.mean(widen_tc):.1f})")
    if same_tc:
        print(f"hotswap_same  transfer keys per run: {same_tc} "
              f"(mean {statistics.mean(same_tc):.1f})")

    small = stats.get("baseline_small", {}).get("mean_val_loss")
    large = stats.get("baseline_large", {}).get("mean_val_loss")
    widen = stats.get("hotswap_widen",  {}).get("mean_val_loss")
    same  = stats.get("hotswap_same",   {}).get("mean_val_loss")
    if None in (small, large, widen, same):
        return

    print()
    print("Interpretation:")

    # Plumbing check: key count, not loss equality. Loss trajectory
    # legitimately differs because we rebuild the Adam optimizer at
    # the swap — its momentum state resets, which acts like a mild
    # learning-rate restart (SGDR-style). Loss after the restart can
    # go either way on a noisy landscape; what we verify here is that
    # every tensor carries over when the shape doesn't change.
    same_keys = statistics.mean(stats.get("transfer_keys_same_per_run", [0]))
    widen_keys = statistics.mean(stats.get("transfer_keys_widen_per_run", [0]))
    if same_keys == 4:
        print(f"  ✓ hotswap_same transfers 4/4 keys — plumbing preserves "
              f"full state_dict on no-op reshape")
    else:
        print(f"  ! hotswap_same transfers {same_keys}/4 keys — "
              f"plumbing dropped something unexpectedly")
    print(f"  (Note: hotswap_same loss ≠ baseline_small because the "
          f"Adam optimizer state is rebuilt at the swap, producing a "
          f"mild LR-restart effect — loss can go either way, that's "
          f"expected, not a transfer bug.)")

    # Gap analysis: shape-changing swap
    gap_total = small - large
    if gap_total <= 0:
        print("  - baseline_small already matches baseline_large on this "
              "task; shape-change hotswap has no signal to measure")
        return
    closed = (small - widen) / gap_total * 100
    print(f"  gap baseline_small → baseline_large = {gap_total:+.5f}")
    print(f"  hotswap_widen closes {closed:+.1f}% of it "
          f"(transferred {widen_keys:.0f}/4 keys)")
    print(f"  (0% = no better than small, 100% = matches large)")
    print("  Caveat: compatible-subset-only transfer wipes most weights "
          "when shapes change — only tensors whose shapes coincidentally")
    print("  match (output biases here) survive. Partial-row/col transfer "
          "is future work; the low gap-close % reflects that correctly.")


# ─── CLI ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--swap-at", type=int, default=100,
                    help="Epoch at which to perform the hot-swap.")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--hidden-small", type=int, default=5)
    ap.add_argument("--hidden-large", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip CSV output; just print the summary.")
    args = ap.parse_args()

    cfg = BenchConfig(
        seeds=args.seeds, epochs=args.epochs, swap_at=args.swap_at,
        lr=args.lr, n_train=args.n_train, n_val=args.n_val,
        hidden_small=args.hidden_small, hidden_large=args.hidden_large,
        out_dir=args.out_dir,
    )

    t0 = time.time()
    records, stats = run_all(cfg)
    elapsed = time.time() - t0

    if not args.no_csv:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = cfg.out_dir / f"hotswap_{stamp}.csv"
        write_csv(records, out_path)
        print(f"Wrote {len(records)} records to {out_path}")

    print_summary(stats)
    print(f"\nElapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
