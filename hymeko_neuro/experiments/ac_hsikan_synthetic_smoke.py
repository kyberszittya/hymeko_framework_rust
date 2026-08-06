"""Synthetic signed-parity benchmark: AC-HSiKAN vs Transformer.

Task: length-L sequences over {-1, +1}; label = product of all
symbols (signed-parity). The mapping is fundamentally
multiplicative-sign-aware; AC-HSiKAN's signed-cycle pool should
win, transformer should struggle without an explicit XOR-style
inductive bias.

Tokens: 0 -> -1, 1 -> +1 (vocab_size=2).
3-seed paired; CPU-runnable (small).
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hymeko_neuro.models.ac_hsikan import AcHsikanClassifier, AcHsikanConfig
from hymeko_neuro.experiments.sequence.iso_param_transformer import (
    IMDBTransformerBaseline,
)


# ── Synthetic data ───────────────────────────────────────────────────

def make_parity_dataset(n: int, L: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 2, (n, L), generator=g)
    # symbols in {-1, +1}
    sym = tokens * 2 - 1
    labels = (sym.prod(dim=1) > 0).long()    # 1 if product +1 else 0
    return TensorDataset(tokens, labels)


# ── Training loop (one model x one seed) ─────────────────────────────

def train_one(
    model: nn.Module, train: DataLoader, val: DataLoader,
    n_epochs: int, lr: float, device: str,
) -> dict:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_acc = 0.0
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for tokens, labels in train:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for tokens, labels in val:
                tokens, labels = tokens.to(device), labels.to(device)
                pred = model(tokens).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total   += labels.numel()
        acc = correct / total
        if acc > best_val_acc:
            best_val_acc = acc
    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"val_acc": best_val_acc, "wall_s": round(wall, 1), "n_params": n_params}


# ── Run ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="/tmp/ac_hsikan_parity_smoke.json")
    args = p.parse_args(argv)

    L = args.seq_len
    results: dict[str, list[dict]] = {"ac_hsikan": [], "transformer": []}

    print(f"=== synthetic signed-parity, L={L}, "
          f"n_train={args.n_train}, n_val={args.n_val}, "
          f"epochs={args.n_epochs}, seeds={args.seeds}, "
          f"device={args.device} ===")

    for seed in args.seeds:
        torch.manual_seed(seed)
        train_ds = make_parity_dataset(args.n_train, L, seed)
        val_ds   = make_parity_dataset(args.n_val,   L, seed + 10_000)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # AC-HSiKAN
        cfg = AcHsikanConfig(
            d_model=16, n_positions=L, arities=(2, 3, 4, 5),
            top_k_per_position=min(8, L - 1),
            n_layers=2, n_classes=2, dropout=0.0,
        )
        torch.manual_seed(seed)
        ac = AcHsikanClassifier(vocab_size=2, cfg=cfg)
        r_ac = train_one(ac, train_dl, val_dl, args.n_epochs, args.lr,
                         args.device)
        r_ac["seed"] = seed
        r_ac["alpha"] = [a.tolist() for a in ac.alpha()]
        results["ac_hsikan"].append(r_ac)
        print(f"  seed={seed} AC-HSiKAN  val_acc={r_ac['val_acc']:.4f} "
              f"wall={r_ac['wall_s']}s n_params={r_ac['n_params']}")

        # Transformer baseline (use IMDBTransformerBaseline with vocab=2)
        torch.manual_seed(seed)
        tr = IMDBTransformerBaseline(vocab_size=2, d_model=16, n_heads=2,
                                      dim_ff=64, n_layers=2)
        r_tr = train_one(tr, train_dl, val_dl, args.n_epochs, args.lr,
                         args.device)
        r_tr["seed"] = seed
        results["transformer"].append(r_tr)
        print(f"  seed={seed} Transform  val_acc={r_tr['val_acc']:.4f} "
              f"wall={r_tr['wall_s']}s n_params={r_tr['n_params']}")

    # Aggregate
    import statistics
    print()
    print("=== paired result (3-seed) ===")
    for name in ("ac_hsikan", "transformer"):
        accs = [r["val_acc"] for r in results[name]]
        params = results[name][0]["n_params"]
        m = statistics.mean(accs)
        s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {name:<12} acc={m:.4f} ± {s:.4f}   n_params={params:,}")
    # delta
    ac_m = statistics.mean([r["val_acc"] for r in results["ac_hsikan"]])
    tr_m = statistics.mean([r["val_acc"] for r in results["transformer"]])
    print(f"  delta (AC - TR): {ac_m - tr_m:+.4f}")
    # Always-positive parity check: if both ~0.5 the task is too easy
    # for the baseline; if both ~1.0 too easy for the model.

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
