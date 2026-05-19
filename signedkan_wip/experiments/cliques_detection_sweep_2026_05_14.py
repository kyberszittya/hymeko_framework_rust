"""Cliques detection foundation — performance + recall sweep.

Emits one JSONL row per (n_robots, planted_profile, detector, seed)
cell, with wall-time + recall metrics. The output feeds the
foundation report and gates Stage 1 of the NP-hard pivot plan.

==============================================================
Mathematical objects measured
==============================================================

Let `H = (V, E, σ)` be a signed graph, with `V = {1..n}` the robot
indices, `E ⊆ V × V` the undirected edges, and `σ: E → {±1}` the
sign function. A planted-clique generator produces a bundle
``B = (H, {P_1, ..., P_K})`` where each ``P_i`` is a balanced
clique inserted into `H`. **Balance** for a clique is defined via
the triangle-product check (Heider 1946 / Cartwright-Harary 1956):

    P is balanced  ⟺  ∀ {a, b, c} ⊂ P : σ(a,b)·σ(b,c)·σ(a,c) = +1

i.e., every triangle has σ-product = +1. Equivalently, the clique
admits a 2-coloring s.t. within-color edges are + and across-color
are −. (Note: the all-edges-product is NOT a valid balance check
for k ≥ 4 — see ``cliques._clique_balance_indicator`` docstring.)

A detector D consumes `H` and returns a list of balanced cliques
``D(H) = (Q_1, ..., Q_M)``. We evaluate D against the planted
ground truth via Jaccard overlap with threshold `τ`:

    overlap(P, Q) = |P ∩ Q| / |P ∪ Q|

    recall(D, B, τ) = #{ P_i : ∃ Q_j with overlap(P_i, Q_j) ≥ τ } / K

    precision(D, B, τ) = #{ Q_j : ∃ P_i with overlap(P_i, Q_j) ≥ τ } / M

For τ = 0.5 a planted clique is "recovered" iff a detected clique
shares at least half its members (and inherits ≤ |Q|/|P ∪ Q|
spurious members at the same threshold).

==============================================================
Detectors compared
==============================================================

  D₁  Bron-Kerbosch + balance check (exact, NetworkX-backed)
       — exponential worst case, treated as ground truth on
         networks where it completes inside the timeout.

  D₂  Triangle-density greedy
       — rank vertices by triangle-balance score, greedy expansion
         along balance-preserving extensions.

  D₃  Greedy balanced (degree-seeded)
       — high-degree seed + balance-preserving growth.

  D₄  Spectral balanced
       — signed-Laplacian eigenvectors → k-means clusters → per-
         cluster balance check + greedy expansion.

==============================================================
Sweep grid
==============================================================

n_robots          ∈ {30, 50, 100, 200, 500}
planted profiles  ∈ {[6,5,4,3],
                     [8,5,4],
                     [10]}
detectors         ∈ {D₁, D₂, D₃, D₄}
seeds             ∈ {0, 1, 2, 3, 4}
comm_range        = 4.0 (constant)
noise_prob        = 0.05 (constant — on non-planted edges only)
n_factions        = 2 (ambient signal)
timeout per call  = 60 s

= 5 × 3 × 4 × 5 = 300 cells.

==============================================================
Output
==============================================================

JSONL with fields per row:
  ts                     — ISO-8601 launch timestamp
  n_robots               — number of vertices
  comm_range             — radius
  planted_sizes          — list[int]
  seed                   — int
  detector               — str name
  wall_time_s            — float
  timed_out              — bool
  n_detected             — int
  largest_size           — int
  recall_at_0_5          — float
  precision_at_0_5       — float
  largest_planted_size   — int (max over planted_sizes)
  n_edges                — int
  n_negative_edges       — int

Streams to disk as it runs; safe to read tail of file during the run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from signedkan_wip.src.demo.cliques_bench import (  # noqa: E402
    benchmark_detector, default_detectors, recall_against_planted,
)
from signedkan_wip.src.demo.cliques_planted import (  # noqa: E402
    make_planted_balanced_cliques,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--output",
        default=str(REPO_ROOT
                     / "signedkan_wip/experiments/results"
                     / f"cliques_detection_sweep_"
                       f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"),
    )
    ap.add_argument("--n-robots", nargs="+", type=int,
                       default=[30, 50, 100, 200, 500])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--noise-prob", type=float, default=0.05)
    ap.add_argument("--n-factions", type=int, default=2)
    ap.add_argument("--comm-range", type=float, default=4.0)
    args = ap.parse_args()

    profiles = [[6, 5, 4, 3], [8, 5, 4], [10]]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output)

    print(f"=== cliques detection sweep ===")
    print(f"  output: {log_path}")
    print(f"  grid: n={args.n_robots} profiles={len(profiles)} "
          f"detectors={len(default_detectors())} seeds={len(args.seeds)}")
    print(f"  total cells: "
          f"{len(args.n_robots) * len(profiles) * len(default_detectors()) * len(args.seeds)}")

    cell = 0
    t_start = time.perf_counter()
    with log_path.open("w") as fh:
        for n_robots in args.n_robots:
            for profile in profiles:
                if sum(profile) > n_robots:
                    # planted cliques would overlap — skip cell.
                    continue
                for seed in args.seeds:
                    bundle = make_planted_balanced_cliques(
                        n_robots=n_robots,
                        clique_sizes=profile,
                        area_size=10.0,
                        comm_range=args.comm_range,
                        noise_prob=args.noise_prob,
                        n_factions=args.n_factions,
                        seed=seed,
                    )
                    largest_planted = max(c.size for c in bundle.planted_cliques)
                    for detector in default_detectors():
                        cell += 1
                        r = benchmark_detector(
                            detector, bundle,
                            min_size=3, max_size=max(8, largest_planted + 2),
                            limit=30, timeout_s=args.timeout_s,
                        )
                        if r.error:
                            row = {
                                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "n_robots": n_robots,
                                "comm_range": args.comm_range,
                                "planted_sizes": profile,
                                "seed": seed,
                                "detector": detector.name,
                                "error": r.error,
                                "wall_time_s": r.wall_time_s,
                            }
                        else:
                            metrics = recall_against_planted(
                                r.cliques, bundle.planted_cliques,
                                overlap_threshold=0.5,
                            )
                            row = {
                                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "n_robots": n_robots,
                                "comm_range": args.comm_range,
                                "planted_sizes": profile,
                                "seed": seed,
                                "detector": detector.name,
                                "wall_time_s": r.wall_time_s,
                                "timed_out": r.timed_out,
                                "n_detected": metrics["n_detected"],
                                "largest_size": r.largest_size,
                                "recall_at_0_5": metrics["recall"],
                                "precision_at_0_5": metrics["precision"],
                                "largest_planted_size": largest_planted,
                                "n_edges": bundle.n_edges,
                                "n_negative_edges": bundle.n_negative_edges,
                            }
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        # Live progress.
                        elapsed = time.perf_counter() - t_start
                        status = "T/O" if row.get("timed_out") else ""
                        recall = row.get("recall_at_0_5", float("nan"))
                        print(
                            f"  [{cell:>3d} | {elapsed:>6.1f}s]  "
                            f"n={n_robots:>3} sz={profile} seed={seed} "
                            f"{detector.name:<26} "
                            f"wall={row.get('wall_time_s', float('nan')):>7.3f}s  "
                            f"recall={recall:.3f}  {status}",
                            flush=True,
                        )
    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. Output: {log_path}")


if __name__ == "__main__":
    main()
