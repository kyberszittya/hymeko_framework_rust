"""Train the iso-param Transformer baseline on IMDB binary sentiment.

Mirrors ``train_imdb_classifier.py`` but instantiates
``IMDBTransformerBaseline`` instead of ``IMDBClassifier``. Used as
the architectural-fairness Phase 2 baseline: same corpus, same
training budget, matched parameter count → which architecture wins?

Plan: docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/
(2026-05-18 architectural-fairness probe).
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
from .iso_param_transformer import IMDBTransformerBaseline


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


def _epoch_accuracy(model, ids, mask, labels, batch_size):
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
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--dim-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jsonl-out", default=None)
    p.add_argument("--pretrained-state-dict", default=None,
                    help="Path to a state_dict to load into the model "
                         "before fine-tuning. Used for the MLM-pretrain "
                         "→ fine-tune path.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    t0 = time.perf_counter()

    print(f"\n=== Iso-param Transformer (IMDB) ===")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  "
          f"dim_ff={args.dim_ff}  n_layers={args.n_layers}  "
          f"epochs={args.epochs}  lr={args.lr}  device={device}  seed={args.seed}")

    train, test, vocab = load_imdb(
        root=args.data_root,
        vocab_size=args.vocab_size, L_max=args.L_max,
    )
    print(f"  IMDB loaded: |train|={len(train)} |test|={len(test)} "
          f"|V|={len(vocab)}", flush=True)

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

    model = IMDBTransformerBaseline(
        vocab_size=len(vocab),
        d_model=args.d_model, n_heads=args.n_heads,
        dim_ff=args.dim_ff, n_layers=args.n_layers,
        max_len=args.L_max, n_classes=args.n_classes,
        dropout=args.dropout,
    ).to(device)
    n_params = model.num_params()
    print(f"  model params: {n_params:,}", flush=True)

    if args.pretrained_state_dict is not None:
        sd = torch.load(args.pretrained_state_dict, map_location=device)
        # The cls_head was reinitialised; keep its random init.
        missing = [k for k in model.state_dict() if k not in sd]
        ignored = [k for k in sd if k not in model.state_dict()]
        sd = {k: v for k, v in sd.items() if k in model.state_dict()
              and not k.startswith("cls_head.")}
        model.load_state_dict(sd, strict=False)
        print(f"  loaded pretrained weights from {args.pretrained_state_dict}: "
              f"{len(sd)} tensors  ({len(missing)} missing, "
              f"{len(ignored)} ignored)", flush=True)

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
              f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}", flush=True)

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
            "model":          "IsoParamTransformer",
            "vocab_size":     len(vocab),
            "L_max":          args.L_max,
            "d_model":        args.d_model,
            "n_heads":        args.n_heads,
            "dim_ff":         args.dim_ff,
            "n_layers":       args.n_layers,
            "dropout":        args.dropout,
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
