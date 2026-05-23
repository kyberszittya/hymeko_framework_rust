"""HyMeKo "query as classifier" — rule-based, zero parameters.

Implements §3.2 / §4 of the GNN experiment plan: the HyMeKo branch is
a deterministic query over the incidence matrix that returns a 0/1
label. No learning. The point is that a query directly expressing the
property's structural definition will trivially solve WL-hard
benchmarks that GNNs conflate.

This Python implementation is a temporary stand-in for the Rust binary
called out in §10 step 4. The Rust binary will run the same predicates
through `hymeko_query`'s pattern-matching engine; behaviour must
match. We test parity between this and the Rust binary as soon as the
Rust side ships.

Run:
    python3 -m src.hymeko_classifier \
        --in data/synth_n32_k5.npz --property has_triangle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .synthetic import (
    has_triangle_subhypergraph,
    is_k_regular,
    n_components_after_random_removal,
)


# ─── Property predicates (rule-based, no learning) ────────────────────


def predict_is_3_regular(B: np.ndarray) -> int:
    return int(is_k_regular(B, 3))


def predict_is_5_regular(B: np.ndarray) -> int:
    return int(is_k_regular(B, 5))


def predict_has_triangle(B: np.ndarray) -> int:
    return int(has_triangle_subhypergraph(B))


def predict_n_components_ge2(B: np.ndarray) -> int:
    return int(n_components_after_random_removal(B) >= 2)


PREDICATES = {
    "is_3_regular": predict_is_3_regular,
    "is_5_regular": predict_is_5_regular,
    "has_triangle": predict_has_triangle,
    "n_components_ge2": predict_n_components_ge2,
}


# ─── Evaluation ──────────────────────────────────────────────────────


def evaluate(npz_path: Path, prop: str) -> dict:
    """Run the rule-based classifier over every sample in `npz_path`
    and report per-property accuracy / F1."""
    data = np.load(npz_path, allow_pickle=True)
    Bs = data["B"]
    labels = data["labels"]
    keys = list(data["labels_keys"])
    if prop not in PREDICATES:
        raise ValueError(f"unknown property {prop}; available: {sorted(PREDICATES)}")
    if prop not in keys:
        raise ValueError(f"property {prop} not in dataset labels {keys}")
    j = keys.index(prop)
    pred_fn = PREDICATES[prop]
    n = Bs.shape[0]
    preds = np.zeros(n, dtype=np.int8)
    for i in range(n):
        preds[i] = pred_fn(Bs[i])
    truth = labels[:, j]
    acc = float((preds == truth).mean())
    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(
        property=prop,
        n_samples=n,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion=dict(tp=tp, fp=fp, fn=fn,
                        tn=int(((preds == 0) & (truth == 0)).sum())),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", type=Path, required=True)
    ap.add_argument("--property", default="has_triangle",
                    choices=sorted(PREDICATES))
    args = ap.parse_args()
    result = evaluate(args.input_path, args.property)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
