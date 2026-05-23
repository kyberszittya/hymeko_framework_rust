"""Train the Sequential HSiKAN IMDBClassifier on IMDB binary sentiment.

Per plan ``docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/``.

Usage::

    # Unit-test-scale smoke (500-doc subset, 1 epoch, CPU OK):
    python -m signedkan_wip.src.sequence.train_imdb_classifier \\
        --n-train 500 --n-test 500 --epochs 1 --batch-size 16

    # Production-scale smoke (full IMDB, 5 epochs):
    python -m signedkan_wip.src.sequence.train_imdb_classifier \\
        --epochs 5 --seed 0 --jsonl-out smoke.jsonl

    # Single-seed headline (full IMDB, 20 epochs):
    python -m signedkan_wip.src.sequence.train_imdb_classifier \\
        --epochs 20 --seed 0 --jsonl-out seed0.jsonl

Falsifier per plan: 5-seed mean test accuracy < 0.70 → architecture
does not transfer to natural language at this scale.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .imdb_dataset import IMDBSplit, load_imdb, split_to_tensors
from .text_classifier import IMDBClassifier


def _val_split(train: IMDBSplit, val_frac: float, seed: int
               ) -> tuple[IMDBSplit, IMDBSplit]:
    rng = np.random.default_rng(seed)
    n = len(train)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return (
        IMDBSplit(ids=train.ids[tr_idx], mask=train.mask[tr_idx],
                   labels=train.labels[tr_idx]),
        IMDBSplit(ids=train.ids[val_idx], mask=train.mask[val_idx],
                   labels=train.labels[val_idx]),
    )


def _epoch_accuracy(
    model: IMDBClassifier,
    ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    n_correct = 0
    n = ids.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = slice(s, s + batch_size)
            logits = model(ids[idx], mask=mask[idx])
            pred = logits.argmax(dim=-1)
            n_correct += int((pred == labels[idx]).sum())
    return n_correct / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/imdb")
    p.add_argument("--vocab-size", type=int, default=20_000)
    p.add_argument("--L-max", type=int, default=200)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--n-train", type=int, default=None,
                    help="Train subset size (None = full 25k).")
    p.add_argument("--n-test", type=int, default=None,
                    help="Test subset size (None = full 25k).")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--enc-depth", type=int, default=3)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-channels", type=int, default=4)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jsonl-out", default=None)
    p.add_argument("--no-download", action="store_true",
                    help="Don't curl-fetch IMDB; require data/imdb/aclImdb/ "
                         "to already exist.")
    p.add_argument("--pretrained-state-dict", default=None,
                    help="Path to a state_dict to load into the model "
                         "before fine-tuning (MLM-pretrain → fine-tune path). "
                         "The cls_head's random init is preserved.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    t0 = time.perf_counter()

    print(f"\n=== Sequential HSiKAN IMDB ===")
    print(f"  vocab_size={args.vocab_size}  L_max={args.L_max}  "
          f"enc_depth={args.enc_depth}  K={args.K}  C={args.n_channels}  "
          f"epochs={args.epochs}  lr={args.lr}  device={device}  seed={args.seed}")

    train, test, vocab = load_imdb(
        root=args.data_root,
        vocab_size=args.vocab_size, L_max=args.L_max, min_freq=args.min_freq,
        download=not args.no_download,
    )
    print(f"  IMDB loaded: |train|={len(train)} |test|={len(test)} "
          f"|V|={len(vocab)} in {time.perf_counter() - t0:.1f}s")

    # Shuffle + optional subset
    train = train.shuffle(args.seed)
    if args.n_train is not None:
        train = IMDBSplit(ids=train.ids[:args.n_train],
                           mask=train.mask[:args.n_train],
                           labels=train.labels[:args.n_train])
    if args.n_test is not None:
        test = IMDBSplit(ids=test.ids[:args.n_test],
                          mask=test.mask[:args.n_test],
                          labels=test.labels[:args.n_test])

    tr, val = _val_split(train, val_frac=args.val_frac, seed=args.seed + 1)
    print(f"  splits: train={len(tr)}  val={len(val)}  test={len(test)}")

    tr_ids, tr_mask, tr_labels = split_to_tensors(tr, device)
    val_ids, val_mask, val_labels = split_to_tensors(val, device)
    te_ids, te_mask, te_labels = split_to_tensors(test, device)

    model = IMDBClassifier(
        vocab_size=len(vocab), enc_depth=args.enc_depth,
        K=args.K, n_channels=args.n_channels,
        max_len=args.L_max, n_classes=args.n_classes,
    ).to(device)
    n_params = model.num_params()
    print(f"  model params: {n_params:,}")

    if args.pretrained_state_dict is not None:
        sd = torch.load(args.pretrained_state_dict, map_location=device)
        # Drop the cls_head keys — they were initialised under the pretrain
        # config (n_classes=2 too, but the linear weights are random); keep
        # our fresh init.
        sd = {k: v for k, v in sd.items()
              if k in model.state_dict() and not k.startswith("cls_head.")}
        model.load_state_dict(sd, strict=False)
        print(f"  loaded pretrained weights from "
              f"{args.pretrained_state_dict}: {len(sd)} tensors")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    losses_per_epoch: list[float] = []
    val_acc_per_epoch: list[float] = []
    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(tr_ids.shape[0], device=device)
        ep_losses = []
        for s in range(0, tr_ids.shape[0], args.batch_size):
            idx = perm[s:s + args.batch_size]
            logits = model(tr_ids[idx], mask=tr_mask[idx])
            loss = F.cross_entropy(logits, tr_labels[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_losses.append(float(loss.detach()))
        ep_mean_loss = sum(ep_losses) / max(1, len(ep_losses))
        val_acc = _epoch_accuracy(
            model, val_ids, val_mask, val_labels, args.batch_size,
        )
        losses_per_epoch.append(ep_mean_loss)
        val_acc_per_epoch.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
        print(f"  [ep {ep:3d}] loss={ep_mean_loss:.4f}  "
              f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = _epoch_accuracy(
        model, te_ids, te_mask, te_labels, args.batch_size,
    )
    wall = time.perf_counter() - t0
    print(f"\n  test_acc={test_acc:.4f}  best_val_acc={best_val_acc:.4f}  "
          f"wall={wall:.1f}s")

    if args.jsonl_out:
        Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset":        "imdb",
            "model":          "SeqHSiKAN-IMDBClassifier",
            "vocab_size":     len(vocab),
            "L_max":          args.L_max,
            "enc_depth":      args.enc_depth,
            "K":              args.K,
            "n_channels":     args.n_channels,
            "n_classes":      args.n_classes,
            "n_params":       n_params,
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "seed":           args.seed,
            "n_train":        len(tr),
            "n_val":          len(val),
            "n_test":         len(test),
            "test_accuracy":  test_acc,
            "best_val_accuracy": best_val_acc,
            "val_acc_per_epoch":  val_acc_per_epoch,
            "losses_per_epoch":   losses_per_epoch,
            "wall_s":         wall,
            "pretrained_state_dict": args.pretrained_state_dict,
        }
        with open(args.jsonl_out, "w") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Wrote {args.jsonl_out}")


if __name__ == "__main__":
    main()
