"""Cross-graph inductive transfer test for the rotor signed-link line.

Plan: ``docs/plans/2026-06-18-inductive-transfer-test/``. The rotor line's
distinctive (untested) claim is **inductive**: it embeds nodes from train-only
*structural features*, with no per-node identity table, so a model trained on one
graph should transfer to another. The transductive baselines
(``sgcn``/``sigat``/``dadsgnn``) use ``nn.Embedding(n_nodes)`` — a per-graph table
that cannot embed a different graph's nodes — so transfer is a **discriminating**
test, not an incremental tweak.

Protocol: train all weights on graph ``A`` (strict train edges, optional
sign-shuffle for the gate), then evaluate the **frozen** model on graph ``B``'s
strict deduped held-out edges, with ``B``'s node embeddings computed purely from
``B``'s train-only context via the frozen weights. No training/metric logic is
re-implemented — training reuses ``run_baseline_audit._train`` and scoring reuses
``_evaluate`` (§6.5 #3).

Transfer leakage gate: a model trained on **shuffled-A** must not transfer above
chance — otherwise the "transfer" is reading B's structure trivially, not carrying
learned signed structure.

One file, mode arg (§6.5 #13):
    python -m signedkan_wip.experiments.runs.run_inductive_transfer --smoke
    python -m signedkan_wip.experiments.runs.run_inductive_transfer --full
"""
from __future__ import annotations

import argparse
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signedkan_wip.src.baselines.registry import GraphMeta, HParams, get_baseline
from signedkan_wip.src.datasets import (
    drop_train_pairs,
    load,
    split,
    undirected_pair,
)
from signedkan_wip.experiments.runs.run_baseline_audit import (
    SHUFFLE_SEED_OFFSET,
    _evaluate,
    _train,
)

class Arm(Enum):
    """A transfer-decomposition arm: how graph A is (or is not) used to train M.

    Each arm fixes the ``(shuffle_train_signs, train)`` pair handed to
    ``transfer_cell``. Together the three arms isolate *transported A-knowledge* from
    the eval graph's own structural prior (signed message-passing exposes B's signs
    regardless of training): ``learned = real − max(shuffle, randinit)``.

    Preconditions: none. Invariant: ``Arm.of(shuffle=a.shuffle_signs,
    trained=a.trains) is a`` for every arm ``a`` (round-trips the resumption key).
    """

    REAL = "real"          # train on A's real signs
    SHUFFLE = "shuffle"    # train on A's permuted signs (leakage gate → chance)
    RANDINIT = "randinit"  # no A-training at all (B's structural-prior floor)

    @property
    def shuffle_signs(self) -> bool:
        return self is Arm.SHUFFLE

    @property
    def trains(self) -> bool:
        return self is not Arm.RANDINIT

    @classmethod
    def of(cls, *, shuffle: bool, trained: bool) -> Arm:
        """Recover the arm from a recorded ``(shuffle, trained)`` row (inverse map)."""
        if not trained:
            return cls.RANDINIT
        return cls.SHUFFLE if shuffle else cls.REAL


ARMS_GATE = (Arm.REAL, Arm.SHUFFLE)                   # 2-arm: real + leakage gate
ARMS_DECOMP = (Arm.REAL, Arm.SHUFFLE, Arm.RANDINIT)   # 3-arm: + random-init floor

# Default comparison: the inductive rotor (+ its walk-enriched variant) vs a
# transductive table baseline (the degeneration control — cannot index B's nodes).
TRANSFER_MODELS = ("cayley_rotor", "cayley_rotor_walk", "dadsgnn")
PAIRS = (("bitcoin_otc", "bitcoin_alpha"), ("bitcoin_alpha", "bitcoin_otc"))

# Harder, more-distinct pairs: train-small (bitcoin) → eval-large (slashdot/epinions).
# These push the learned-transfer increment past the bitcoin-pair σ (BACKLOG NEXT STEP).
HARD_PAIRS = (
    ("bitcoin_otc", "slashdot"), ("bitcoin_alpha", "slashdot"),
    ("bitcoin_otc", "epinions"), ("bitcoin_alpha", "epinions"),
)
DECOMP_MODELS = ("cayley_rotor", "cayley_rotor_walk")


def transfer_cell(
    model_name: str, train_ds: str, eval_ds: str, seed: int,
    *, dedup: bool = True, shuffle_train_signs: bool = False, train: bool = True,
    n_epochs: int = 120, overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on ``train_ds``, evaluate the frozen model on ``eval_ds``.

    Preconditions: both datasets load; ``model_name`` is a registered baseline.
    Postconditions: dict with ``auc`` (transfer AUROC on ``eval_ds`` held-out,
    ``nan`` if the model cannot transfer), ``transferred`` (bool), ``n_test``. The
    eval context is built from ``eval_ds``'s train edges only (strict; no eval-test
    edge enters it).

    Two controls isolate *learned* transfer from the eval graph's own structural
    prior (B's real signed adjacency lets signed message-passing encode B's signs
    into embeddings regardless of training): ``shuffle_train_signs`` permutes the
    *training* (A) signs, and ``train=False`` evaluates the **random-init** model on
    B (no A-training at all). Learned transfer = real − max(shuffled, random-init).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    strat = get_baseline(model_name)
    hp: HParams = strat.default_hparams().merged(
        n_epochs=n_epochs, **(overrides or {}))

    # --- train on A (strict) ---
    g_a = load(train_ds)
    tr_a, va_a, _ = split(g_a, seed=seed)
    e_tr, s_tr = g_a.edges[tr_a], g_a.signs[tr_a].copy()
    e_va, s_va = g_a.edges[va_a], g_a.signs[va_a]
    if shuffle_train_signs:
        perm = np.random.default_rng(seed + SHUFFLE_SEED_OFFSET).permutation(len(s_tr))
        s_tr = s_tr[perm]
    ctx_a = strat.build_context(e_tr, s_tr, g_a.n_nodes, device)
    model = strat.build_model(GraphMeta(n_nodes=g_a.n_nodes), hp).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.time()
    auc = f1 = float("nan")
    n_test = 0
    transferred = False
    note = ""
    try:
        if train:
            model, _, _, _ = _train(model, hp, ctx_a, e_tr, s_tr, e_va, s_va, device)
        # else: random-init control — no A-training, isolates B's structural prior.
        # --- evaluate the frozen model on B (strict deduped held-out) ---
        g_b = load(eval_ds)
        tr_b, _, te_b = split(g_b, seed=seed)
        e_tr_b, s_tr_b = g_b.edges[tr_b], g_b.signs[tr_b]
        e_te_b, s_te_b = g_b.edges[te_b], g_b.signs[te_b]
        if dedup:
            pairs = {undirected_pair(e) for e in e_tr_b}
            e_te_b, s_te_b = drop_train_pairs(e_te_b, s_te_b, pairs)
        ctx_b = strat.build_context(e_tr_b, s_tr_b, g_b.n_nodes, device)
        auc, f1 = _evaluate(model, ctx_b, e_te_b, s_te_b, device)
        n_test = len(e_te_b)
        transferred = True
    except (IndexError, RuntimeError) as exc:
        # The intended discrimination: a transductive table (nn.Embedding sized for
        # A's nodes) cannot index B's nodes ⇒ it *cannot* transfer. Recorded, not
        # masked. Other model/data errors are not swallowed (specific catch).
        note = f"cannot transfer: {type(exc).__name__}: {str(exc)[:120]}"

    return dict(
        model=model_name, train_ds=train_ds, eval_ds=eval_ds, seed=seed,
        shuffle=shuffle_train_signs, trained=train, dedup=dedup,
        auc=(None if np.isnan(auc) else round(float(auc), 4)),
        test_f1_macro=(None if np.isnan(f1) else round(float(f1), 4)),
        n_test=int(n_test), transferred=transferred, note=note,
        n_params=int(n_params), elapsed_s=round(time.time() - t0, 2),
        device=str(device),
    )


def _row_key(r: dict[str, Any]) -> tuple[str, str, str, int, str]:
    """Resumption key, arm-aware.

    The arm is *derived* from ``(shuffle, trained)`` rather than read from a stored
    field, so the random-init arm (``shuffle=False, trained=False``) no longer
    collides with the real arm (``shuffle=False, trained=True``) — the latent bug that
    silently dropped the random-init arm from the 2-arm grid's resume set. Backward-
    compatible with pre-``arm`` rows (no ``trained`` field ⇒ defaults ``True`` ⇒
    real/shuffle).
    """
    arm = Arm.of(shuffle=bool(r["shuffle"]), trained=bool(r.get("trained", True)))
    return (r["model"], r["train_ds"], r["eval_ds"], int(r["seed"]), arm.value)


def _print_cell(m: str, a: str, b: str, seed: int,
                cells: dict[Arm, dict[str, Any]]) -> None:
    def fmt(r: dict[str, Any] | None) -> str:
        if r is None:
            return "  -   "
        return "cannot" if r["auc"] is None else f"{r['auc']:.4f}"
    parts = " ".join(f"{arm.value}={fmt(cells.get(arm))}" for arm in cells)
    n_test = next((c["n_test"] for c in cells.values() if c.get("n_test")), 0)
    print(f"  {m:<20} {a}->{b:<16} seed={seed} {parts} n_test={n_test}")


def run_grid(models: tuple[str, ...], pairs: tuple[tuple[str, str], ...],
             seeds: tuple[int, ...], arms: tuple[Arm, ...],
             out_path: Any, *, label: str = "") -> list[dict[str, Any]]:
    """Run the model × pair × seed × arm transfer grid; resumable per cell.

    One loop serves both the 2-arm gate grid (``ARMS_GATE``) and the 3-arm
    decomposition (``ARMS_DECOMP``) — no loop duplication (§6.5 #3). Each ``arm``
    fixes ``transfer_cell``'s ``(shuffle_train_signs, train)`` pair; rows append to
    ``out_path`` and are skipped on resume via the arm-aware ``_row_key`` (so a
    re-run never recomputes a done arm, nor drops the random-init arm).

    Preconditions: every dataset in ``pairs`` loads; every name in ``models`` is a
    registered baseline. Postcondition: ``out_path`` holds exactly one row per
    (model, pair, seed, arm) cell.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, str, str, int, str]] = set()
    if out_path.exists():
        rows = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
        done = {_row_key(r) for r in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inductive transfer [{label}]: {len(models)} models × {len(pairs)} pairs × "
          f"{len(seeds)} seeds × {len(arms)} arms on {device}")

    def cell(m: str, a: str, b: str, seed: int, arm: Arm) -> dict[str, Any]:
        """Resume the (m, a, b, seed, arm) row or compute and append it."""
        key = (m, a, b, seed, arm.value)
        hit = next((r for r in rows if _row_key(r) == key), None)
        if hit is None:
            hit = transfer_cell(m, a, b, seed,
                                shuffle_train_signs=arm.shuffle_signs, train=arm.trains)
            hit["arm"] = arm.value
            _append(out_path, hit, rows, done)
        return hit

    for a, b in pairs:
        for m in models:
            for seed in seeds:
                cells = {arm: cell(m, a, b, seed, arm) for arm in arms}
                _print_cell(m, a, b, seed, cells)

    print(f"\nWrote {out_path} ({len(rows)} rows)")
    return rows


def main(smoke: bool = True, out_path: Any = None,
         seeds: tuple[int, ...] | None = None,
         models: tuple[str, ...] | None = None,
         pairs: tuple[tuple[str, str], ...] | None = None) -> list[dict[str, Any]]:
    """2-arm gate grid (real + shuffle) on the bitcoin pairs."""
    if smoke:
        models = models or ("cayley_rotor", "dadsgnn")
        pairs = pairs or (("bitcoin_otc", "bitcoin_alpha"),)
        seeds = seeds or (0,)
    else:
        models = models or TRANSFER_MODELS
        pairs = pairs or PAIRS
        seeds = seeds or (0, 1, 2, 3, 4)
    out_path = out_path or (
        f"signedkan_wip/experiments/results/inductive_transfer_"
        f"{'smoke' if smoke else 'full'}.jsonl")
    return run_grid(models, pairs, seeds, ARMS_GATE, out_path,
                    label="smoke" if smoke else "full")


def main_decomp(out_path: Any = None, seeds: tuple[int, ...] | None = None,
                models: tuple[str, ...] | None = None,
                pairs: tuple[tuple[str, str], ...] | None = None,
                ) -> list[dict[str, Any]]:
    """3-arm decomposition (real + shuffle + random-init) on the harder
    train-small → eval-large pairs (bitcoin → slashdot/epinions)."""
    out_path = out_path or (
        "signedkan_wip/experiments/results/inductive_transfer_decomp_hard.jsonl")
    return run_grid(models or DECOMP_MODELS, pairs or HARD_PAIRS,
                    seeds or (0, 1, 2, 3, 4), ARMS_DECOMP, out_path, label="decomp")


def _append(out_path: Path, row: dict[str, Any],
            rows: list[dict[str, Any]],
            done: set[tuple[str, str, str, int, str]]) -> None:
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    rows.append(row)
    done.add(_row_key(row))


def _parse_pairs(spec: str | None) -> tuple[tuple[str, str], ...] | None:
    """``"bitcoin_alpha:epinions,bitcoin_otc:slashdot"`` → tuple of (train, eval)."""
    if not spec:
        return None
    return tuple((p.split(":", 1)[0], p.split(":", 1)[1])
                 for p in spec.split(",") if ":" in p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="1 pair × 2 models × 1 seed (2-arm gate)")
    mode.add_argument("--full", action="store_true",
                      help="both bitcoin pairs × 3 models × 5 seeds (2-arm gate)")
    mode.add_argument("--decomp", action="store_true",
                      help="3-arm decomposition on bitcoin → slashdot/epinions")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--models", default=None, help="comma-separated model names")
    ap.add_argument("--pairs", default=None,
                    help="comma-separated train:eval pairs, e.g. bitcoin_alpha:epinions")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = (tuple(int(s) for s in a.seeds.split(",") if s.strip())
             if a.seeds else None)
    models = (tuple(s for s in a.models.split(",") if s.strip())
              if a.models else None)
    pairs = _parse_pairs(a.pairs)
    if a.decomp:
        main_decomp(out_path=a.out, seeds=seeds, models=models, pairs=pairs)
    else:
        main(smoke=not a.full, out_path=a.out, seeds=seeds, models=models, pairs=pairs)
