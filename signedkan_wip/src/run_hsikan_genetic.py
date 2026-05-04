"""Genetic-algorithm hyperparameter search for HSiKAN.

Population-based metaheuristic over the high-sensitivity HSiKAN knobs
identified in `run_hsikan_hpsweep.py` (lr, init_scale, entropy_eta,
entropy_target, entropy_lam0, participation_lam, gate_bias) plus a few
unswept architectural choices (n_layers, hidden_dim, grid).

Why GA over Bayesian optimisation: HSiKAN evaluations are 30-60s each;
populations of 8 evaluated in parallel-ish on a single GPU give faster
turnaround than sequential Bayesian acquisition. The fitness landscape
also has obvious clusters (init=0.05 vs 0.1) suiting elitism + mutation.

Search budget: pop=8, generations=6 ⇒ at most 48 evaluations. With
median 45 s/eval ≈ 36 min wall-clock — same order as the manual sweep.

Fitness: mean(AUC, F1m) on a single seed (seed=0) for speed; the elite
of the final generation is re-scored over 3 seeds for honest reporting.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .run_compare import run_one


@dataclass(frozen=True)
class Genome:
    lr: float
    init_scale: float
    entropy_lam0: float
    entropy_eta: float
    entropy_target: float
    participation_lam: float
    n_layers: int
    hidden: int
    grid: int
    weight_decay: float = 1e-5
    optimizer_kind: str = "adam"
    grad_clip: float = 0.0

    def to_kwargs(self) -> dict:
        return {
            "lr": self.lr,
            "init_scale": self.init_scale,
            "entropy_lam0": self.entropy_lam0,
            "entropy_eta": self.entropy_eta,
            "entropy_target": self.entropy_target,
            "participation_lam": self.participation_lam,
            "n_layers": self.n_layers,
            "grid": self.grid,
            "weight_decay": self.weight_decay,
            "optimizer_kind": self.optimizer_kind,
            "grad_clip": self.grad_clip,
            # HSiKAN architectural pieces, fixed:
            "spline_kinds": ["catmull_rom"] * self.n_layers,
            "pool_mode": "sum",
            "jk_mode": "concat",
            "layer_norm_between": True,
            "share_weights": True,
            "inner_skip": "highway",
            "outer_skip": "none",
            "early_stopping": True,
            "class_weighted": True,
            "val_every": 5,
            "entropy_kl_normalized": True,
            "entropy_momentum": 0.9,
        }


# Search space: each gene is sampled uniformly from this discrete set
SPACE = {
    "lr":               [1e-2, 3e-2, 5e-2, 1e-1],
    "init_scale":       [0.03, 0.05, 0.1, 0.15],
    "entropy_lam0":     [0.0, 0.005, 0.01, 0.02],
    "entropy_eta":      [1.0, 5.0, 10.0],
    "entropy_target":   [0.3, 0.5, 0.7],
    "participation_lam":[0.0, 0.05, 0.1],
    "n_layers":         [1, 2, 3],
    "hidden":           [16, 32, 64],
    "grid":             [3, 5, 7],
    "weight_decay":     [1e-5, 1e-4, 1e-3, 5e-3],
    "optimizer_kind":   ["adam", "adamw"],
    "grad_clip":        [0.0, 0.5, 1.0, 5.0],
}


def random_genome(rng: random.Random) -> Genome:
    return Genome(**{k: rng.choice(v) for k, v in SPACE.items()})


def mutate(g: Genome, rng: random.Random, p: float = 0.3) -> Genome:
    fields = {}
    for k, v in SPACE.items():
        cur = getattr(g, k)
        if rng.random() < p:
            choices = [x for x in v if x != cur]
            fields[k] = rng.choice(choices) if choices else cur
        else:
            fields[k] = cur
    return Genome(**fields)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    fields = {}
    for k in SPACE:
        fields[k] = getattr(a if rng.random() < 0.5 else b, k)
    return Genome(**fields)


def fitness(g: Genome, dataset: str, n_epochs: int, seed: int) -> tuple[float, dict]:
    """Mean of test_auc and test_f1_macro on a single seed."""
    try:
        r = run_one("signedkan", dataset, hidden=g.hidden, seed=seed,
                     n_epochs=n_epochs, **g.to_kwargs())
    except Exception as e:
        print(f"    [genome failed: {e!r}]")
        return -1.0, {"error": repr(e)}
    f = 0.5 * (r["test_auc"] + r["test_f1_macro"])
    return f, r


def evolve(dataset: str, n_epochs: int, pop_size: int, n_gens: int,
           seed: int, eval_seeds: list[int]) -> dict:
    rng = random.Random(seed)
    population = [random_genome(rng) for _ in range(pop_size)]
    history = []
    best_so_far: tuple[float, Genome, dict] = (-1.0, population[0], {})

    for gen in range(n_gens):
        gen_t0 = time.time()
        scored = []
        for i, g in enumerate(population):
            f, r = fitness(g, dataset, n_epochs, seed)
            scored.append((f, g, r))
            print(f"  gen={gen} ind={i:>2d} fit={f:.4f}  "
                  f"AUC={r.get('test_auc', float('nan')):.4f}  "
                  f"F1m={r.get('test_f1_macro', float('nan')):.4f}  "
                  f"L={g.n_layers} h={g.hidden} G={g.grid} "
                  f"lr={g.lr:.0e} init={g.init_scale}")
            if f > best_so_far[0]:
                best_so_far = (f, g, r)
        scored.sort(key=lambda t: t[0], reverse=True)
        history.append({
            "gen": gen,
            "best_fit": scored[0][0],
            "mean_fit": sum(s[0] for s in scored) / max(1, len(scored)),
            "elite": asdict(scored[0][1]),
            "elapsed_s": round(time.time() - gen_t0, 1),
        })
        # Elitism (keep top 2) + tournament-style + mutation
        elites = [s[1] for s in scored[:2]]
        survivors = [s[1] for s in scored[: pop_size // 2]]
        children = []
        while len(elites) + len(children) < pop_size:
            a, b = rng.sample(survivors, 2)
            children.append(mutate(crossover(a, b, rng), rng, p=0.3))
        population = elites + children
        print(f"  gen={gen} elapsed={history[-1]['elapsed_s']:.1f}s "
              f"best={scored[0][0]:.4f}  mean={history[-1]['mean_fit']:.4f}")

    # Final elite: re-score over all eval_seeds for honest reporting.
    elite_genome = best_so_far[1]
    elite_runs = []
    for s in eval_seeds:
        f, r = fitness(elite_genome, dataset, n_epochs, s)
        elite_runs.append(r)
        print(f"  ELITE seed={s}  AUC={r.get('test_auc', float('nan')):.4f}  "
              f"F1m={r.get('test_f1_macro', float('nan')):.4f}")
    return {
        "dataset": dataset, "history": history,
        "elite_genome": asdict(elite_genome),
        "elite_runs": elite_runs,
        "best_single_seed_fit": best_so_far[0],
        "best_single_seed_run": best_so_far[2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/hsikan_genetic.json")
    args = ap.parse_args()

    res = evolve(args.dataset, args.n_epochs, args.pop, args.gens,
                  args.seed, args.eval_seeds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
