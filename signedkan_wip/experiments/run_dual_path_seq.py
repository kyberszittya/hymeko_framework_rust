"""Train the Sequential HSiKAN + CliffordFIR dual-path model on the
Tier-1 synthetic benchmark.

Ablations (selectable via --variant):
  full          : DualPathSequenceModel with learned routing
  signal_only   : gate pinned to g=1 (pure CliffordFIR path)
  info_only     : gate pinned to g=0 (pure HSiKAN-seq path)
  fixed_mix     : gate pinned to g=0.5 (no learned routing)

Each variant trains for ``--epochs`` epochs with Adam, evaluates on a
held-out test split, reports macro-F1 + AUROC + accuracy.

Usage:
    python -m signedkan_wip.experiments.run_dual_path_seq \\
        --variant full --seeds 5 --epochs 100 --supervised-sign

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/ §6.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from signedkan_wip.src.sequence.dual_path_model import DualPathSequenceModel
from signedkan_wip.src.sequence.synthetic_seq import SynthConfig, make_dataset


VARIANTS = ("full", "signal_only", "info_only", "fixed_mix")


def _pin_routers(model: DualPathSequenceModel, variant: str) -> None:
    """For the ablation variants, freeze each block's PositionRouter to
    produce a constant gate value. We do this by zeroing the router's
    linear-projection weight and setting the bias to drive sigmoid to
    the target."""
    if variant == "full":
        return
    targets = {
        "signal_only": 10.0,    # sigmoid(10) ≈ 1
        "info_only":   -10.0,   # sigmoid(-10) ≈ 0
        "fixed_mix":   0.0,     # sigmoid(0) = 0.5
    }
    bias = targets[variant]
    with torch.no_grad():
        for block in model.blocks:
            block.router.proj.weight.zero_()
            block.router.proj.bias.fill_(bias)
    # Freeze routers so they don't drift during training.
    for block in model.blocks:
        for p in block.router.parameters():
            p.requires_grad_(False)


def _split(
    raw: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor,
    test_frac: float = 0.2,
):
    n = raw.shape[0]
    n_test = int(n * test_frac)
    n_train = n - n_test
    return (
        (raw[:n_train], sigma[:n_train], labels[:n_train]),
        (raw[n_train:], sigma[n_train:], labels[n_train:]),
    )


def _eval(
    model: DualPathSequenceModel,
    raw: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor,
    supervised_sign: bool,
) -> dict:
    model.eval()
    with torch.no_grad():
        if supervised_sign:
            logits = model(raw, sigma_override=sigma)
        else:
            logits = model(raw)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    y = labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "auroc": float(roc_auc_score(y, probs)),
    }


def train_one_seed(
    variant: str, seed: int, epochs: int,
    n_train: int, L: int, supervised_sign: bool,
    lb_weight: float, verbose: bool,
) -> dict:
    """Train one seed of one variant. Returns metrics dict."""
    cfg = SynthConfig(n_samples=n_train, L=L, seed=seed)
    raw, sigma, labels = make_dataset(cfg)
    train, test = _split(raw, sigma, labels, test_frac=0.2)
    raw_tr, sig_tr, lab_tr = train
    raw_te, sig_te, lab_te = test

    torch.manual_seed(seed)
    model = DualPathSequenceModel(
        in_features=2, n_classes=2, depth=3, K=4,
        supervised_sign=supervised_sign,
    )
    _pin_routers(model, variant)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                             lr=3e-3)
    losses = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        if supervised_sign:
            logits = model(raw_tr, sigma_override=sig_tr)
        else:
            logits = model(raw_tr)
        task_loss = F.cross_entropy(logits, lab_tr)
        if variant == "full" and lb_weight > 0:
            if supervised_sign:
                lb_loss = model.gate_load_balance_loss(raw_tr, sigma_override=sig_tr)
            else:
                lb_loss = model.gate_load_balance_loss(raw_tr)
            loss = task_loss + lb_weight * lb_loss
        else:
            loss = task_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(task_loss.item()))
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  variant={variant} seed={seed} ep={ep:3d}  loss={task_loss.item():.4f}")
    wall = time.perf_counter() - t0
    metrics = _eval(model, raw_te, sig_te, lab_te,
                      supervised_sign=supervised_sign)
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "variant": variant, "seed": seed, "epochs": epochs,
        "n_samples": n_train, "L": L, "supervised_sign": supervised_sign,
        "wall_s": wall, "n_params": n_params,
        "n_train_params": n_train_params,
        "loss_start": losses[0], "loss_end": losses[-1],
        **metrics,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=VARIANTS + ("all",), default="full",
                   help="Ablation variant to train. 'all' runs every variant in sequence.")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds per variant.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=1024,
                   help="Total dataset size (80/20 train/test split).")
    p.add_argument("--L", type=int, default=256,
                   help="Sequence length.")
    p.add_argument("--supervised-sign", action="store_true",
                   help="Use ground-truth σ from the generator (cleaner ablation).")
    p.add_argument("--lb-weight", type=float, default=0.01,
                   help="Load-balance auxiliary loss weight (full variant only).")
    p.add_argument("--out-jsonl", type=Path, default=None,
                   help="Append per-run rows to this jsonl file.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    all_rows = []
    for v in variants:
        per_seed = []
        for seed in range(args.seeds):
            row = train_one_seed(
                variant=v, seed=seed, epochs=args.epochs,
                n_train=args.n_samples, L=args.L,
                supervised_sign=args.supervised_sign,
                lb_weight=args.lb_weight, verbose=args.verbose,
            )
            per_seed.append(row)
            all_rows.append(row)
            if args.out_jsonl is not None:
                with args.out_jsonl.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        # Per-variant summary.
        for metric in ("accuracy", "macro_f1", "auroc"):
            vals = [r[metric] for r in per_seed]
            print(
                f"variant={v:12s} {metric:9s} mean={statistics.mean(vals):.4f} "
                f"pstdev={statistics.pstdev(vals):.4f} "
                f"min={min(vals):.4f} max={max(vals):.4f} (n={len(vals)})"
            )
        wall_mean = statistics.mean(r["wall_s"] for r in per_seed)
        print(
            f"variant={v:12s} wall_mean={wall_mean:.1f}s "
            f"params={per_seed[0]['n_params']}"
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
