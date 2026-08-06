"""Shared CEM optimiser + linear policy — the framework core the humanoid CEM trainers reuse.

The three CEM trainers (`train_footstep_walk`, `train_target_footstep`, `train_balance_walk`) all ran the
*same* cross-entropy-method loop (sample → evaluate in parallel → refit to the elite) over the *same*
linear ``tanh(W·obs + b)`` policy, differing only in the task rollout and the saved metrics. That loop
and policy live here once (CLAUDE.md §6.1 / §6.5 #3: one framework, not one copy per experiment); a
trainer is now a rollout + a config + a metrics dict.

# Preconditions: ``eval_fn((theta, cfg))`` is a top-level (picklable) callable returning ``(return, …)``.
# Postconditions: ``cem_optimize`` returns the maximum-return ``theta`` and its full result tuple.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np


def policy_dim(obs_dim: int, act_dim: int = 2) -> int:
    """Parameter count of the linear policy ``a = tanh(W·obs + b)`` (``W`` is act×obs, ``b`` is act)."""
    return act_dim * obs_dim + act_dim


def linear_policy(theta: np.ndarray, obs: np.ndarray, obs_dim: int, act_dim: int = 2) -> np.ndarray:
    """Evaluate the linear policy ``tanh(W·obs + b)`` from a flat parameter vector ``theta``."""
    w = theta[: act_dim * obs_dim].reshape(act_dim, obs_dim)
    b = theta[act_dim * obs_dim:]
    return np.tanh(w @ obs + b)


def cem_optimize(eval_fn: "Callable[[tuple], tuple]", cfg: Any, dim: int, *, iters: int, pop: int,
                 elite: int, workers: int = 1, seed: int = 0, sigma0: float = 0.5,
                 warm: "np.ndarray | None" = None, out: "Path | None" = None, label: str = "cem",
                 report: "Callable[[tuple], str] | None" = None) -> "tuple[np.ndarray, tuple]":
    """Cross-entropy-method search over ``eval_fn((theta, cfg)) -> (return, *extra)``.

    Samples ``pop`` candidates from ``N(mu, sig)``, evaluates them (in parallel when ``workers > 1``),
    refits ``mu, sig`` to the top-``elite`` by return, and repeats for ``iters``. Returns the best-return
    ``theta`` and its full result tuple. Writes ``journal.jsonl`` to ``out`` and prints
    ``[label] iterN elite_ret=… <report(best)>`` each iteration.

    # Preconditions: ``1 <= elite <= pop``; ``eval_fn`` picklable when ``workers > 1``. # Postconditions:
    #   the returned theta has the maximum return seen; the journal has one row per iteration.
    """
    rng = np.random.default_rng(seed)
    mu = warm.copy() if (warm is not None and warm.shape[0] == dim) else np.zeros(dim)
    sig = np.ones(dim) * (0.25 if warm is not None else sigma0)     # tighter search when warm-started
    best_theta, best = np.zeros(dim), (-1e9,)
    journal = None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        journal = (out / "journal.jsonl").open("w")
    for it in range(iters):
        cand = mu + sig * rng.standard_normal((pop, dim))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(eval_fn, [(c, cfg) for c in cand]))
        else:
            res = [eval_fn((c, cfg)) for c in cand]
        scores = np.array([r[0] for r in res])
        idx = np.argsort(scores)[::-1][:elite]
        mu, sig = cand[idx].mean(0), cand[idx].std(0) + 0.04
        for c, r in zip(cand, res):
            if r[0] > best[0]:
                best_theta, best = c.copy(), r
        row = {"iter": it, "elite_ret": float(scores[idx].mean()), "best_ret": float(best[0])}
        if journal is not None:
            journal.write(json.dumps(row) + "\n")
            journal.flush()
        tail = report(best) if report is not None else f"best_ret={best[0]:.1f}"
        print(f"[{label}] iter{it} elite_ret={row['elite_ret']:.1f} {tail}", flush=True)
    if journal is not None:
        journal.close()
    return best_theta, best
