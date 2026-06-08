"""Regime A/B/C 5-seed comparison for the HSiKAN-mixed protocol sweep.

Answers the quality question the Composite-regime pipeline deferred: does
the P-graph regime change which HSiKAN-mixed architecture you should
pick? The three *distinct* regimes on
``sweep_msg_mixed_protocols.hymeko`` are

    A  Canonical               — admits 12 clean archs (incl. quaternion)
    B  No-Excess (≡ Composite) — admits 8  (quaternion pruned as waste)
    C  Cost-dominance          — admits 2  (cost-Pareto)

Composite ≡ No-Excess here because No-Excess ⊆ Canonical ⟹
Composite = Canonical ∩ No-Excess = No-Excess (reported, not hidden).

Pipeline: derive each regime's admissible CLEAN architectures (one unit
per axis) from the authoritative Rust SSG → train the Canonical superset
(12 archs) × N seeds, checkpointing every (arch, seed) row to jsonl →
aggregate per regime + paired Δ between the per-regime best architectures.

Usage
-----
    PYTHONPATH=$PWD systemd-run --user --scope -p MemoryMax=16G \\
        /home/kyberszittya/miniconda3/bin/python -m \\
        signedkan_wip.experiments.runs.run_regime_abc_5seed \\
            --seeds 0,1,2,3,4 --n-epochs 20 \\
            --results-file /tmp/regime_abc/results.jsonl

``--dry-run`` lists the admissible sets + the (arch, seed) job list with
no torch. ``--analyze-only`` re-aggregates an existing results jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from signedkan_wip.experiments.runs.run_hsikan_mixed_composite_smoke import (
    ATTENTION_TOPK_K,
    _git_sha,
    _repo_root,
    run_cell,
    solve_regime,
    structure_to_env,
)
from signedkan_wip.src.hsikan_pgraph_mapping import merge_structure_knobs

DEFAULT_SWEEP = Path("data/hsikan/sweep_msg_mixed_protocols.hymeko")
MIXED_TUPLES = "c3,c4,w2,w3"
# The three regimes that produce distinct admissible sets on this sweep.
REGIMES = ("canonical", "no-excess", "cost-dominance")
_AXIS_PREFIXES = ("struct_", "attn_", "gate_", "dm_", "model_h", "train_")

# An architecture signature is the (attention_kind, per_edge_gate,
# direct_messaging) triple — the protocol axes that distinguish regimes.
# hidden / n_epochs are held fixed across the comparison.
Arch = tuple[str, bool, bool]


def _clean_signature(units: list[str]) -> Arch | None:
    """Return the (attn, gate, dm) signature of a clean structure.

    A *clean* structure selects exactly one unit per mandatory axis.
    Structures with redundant or missing axis units return ``None`` (they
    do not denote a single trainable architecture).
    """
    axis_unit: dict[str, str] = {}
    for u in units:
        for p in _AXIS_PREFIXES:
            if u.startswith(p):
                if p in axis_unit:
                    return None  # redundant unit on this axis
                axis_unit[p] = u
    if len(axis_unit) != len(_AXIS_PREFIXES):
        return None  # an axis is unselected
    knobs = merge_structure_knobs(units)
    return (
        knobs["attention_kind"],
        bool(knobs["per_edge_gate"]),
        bool(knobs["direct_messaging"]),
    )


def regime_admissible_archs(repo: Path, sweep: Path, regime: str) -> set[Arch]:
    """Derive the set of clean trainable architectures a regime admits.

    Authoritative: reads the Rust SSG solution-structure enumeration under
    ``regime`` rather than hardcoding the pruning rule.
    """
    data = solve_regime(repo, sweep, regime, algorithm="ssg")
    structures = data.get("ssg_structures") or []
    archs: set[Arch] = set()
    for s in structures:
        sig = _clean_signature(s)
        if sig is not None:
            archs.add(sig)
    return archs


def arch_env(arch: Arch, attention_topk_k: int = ATTENTION_TOPK_K) -> dict[str, str]:
    """Build the HSIKAN_* env patch for one architecture (mixed family fixed).

    ``attention_topk_k`` sets the per-vertex cycle-pool cap for attention
    archs (default 8); raise it to give attention a richer pool.
    """
    attn, gate, dm = arch
    env, _ = structure_to_env(
        {
            "mixed_tuples": MIXED_TUPLES,
            "attention_kind": attn,
            "per_edge_gate": gate,
            "direct_messaging": dm,
        },
        attention_topk_k=attention_topk_k,
    )
    return env


def _arch_key(arch: Arch) -> str:
    attn, gate, dm = arch
    return f"attn={attn}|gate={'edge_cr' if gate else 'scalar'}|dm={'on' if dm else 'off'}"


def _load_done(results_file: Path) -> set[tuple[str, int]]:
    """Completed (arch_key, seed) pairs in an existing jsonl (resume support)."""
    done: set[tuple[str, int]] = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["arch_key"], int(row["seed"])))
    return done


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(xs) / n
    if n == 1:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (m, math.sqrt(var))


def _paired_delta(a: dict[int, float], b: dict[int, float]) -> dict[str, Any]:
    """Paired Δ (a − b) over shared seeds; returns mean, sd, sigma, n, wins."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    m, sd = _mean_sd(diffs)
    sigma = (m / (sd / math.sqrt(len(diffs)))) if (sd > 0 and len(diffs) > 1) else 0.0
    return {
        "delta_mean": m,
        "delta_sd": sd,
        "sigma": sigma,
        "n_paired": len(diffs),
        "wins_a": sum(1 for d in diffs if d > 0),
    }


def aggregate(rows: list[dict[str, Any]], regime_sets: dict[str, set[Arch]]) -> dict[str, Any]:
    """Per-arch mean±sd, per-regime best architecture, and the paired Δ
    between the Canonical-best and No-Excess-best architectures."""
    # arch_key → {seed → auc}
    by_arch: dict[str, dict[int, float]] = {}
    key_to_arch: dict[str, Arch] = {}
    for r in rows:
        if r.get("auc") is None:
            continue
        by_arch.setdefault(r["arch_key"], {})[int(r["seed"])] = float(r["auc"])
        key_to_arch[r["arch_key"]] = tuple(r["arch"])  # type: ignore[assignment]

    per_arch = {
        k: {"mean": _mean_sd(list(v.values()))[0],
            "sd": _mean_sd(list(v.values()))[1],
            "n": len(v),
            "seeds": v}
        for k, v in by_arch.items()
    }

    def best_in(regime: str) -> str | None:
        admissible_keys = [
            k for k, a in key_to_arch.items() if a in regime_sets[regime]
        ]
        scored = [(k, per_arch[k]["mean"]) for k in admissible_keys if per_arch[k]["n"] > 0]
        return max(scored, key=lambda kv: kv[1])[0] if scored else None

    regime_best = {reg: best_in(reg) for reg in regime_sets}

    out: dict[str, Any] = {
        "per_arch": {k: {kk: vv for kk, vv in v.items() if kk != "seeds"}
                     for k, v in per_arch.items()},
        "regime_best": regime_best,
        "regime_admissible_counts": {r: len(s) for r, s in regime_sets.items()},
    }
    # The headline: does Canonical's extra (quaternion) set beat No-Excess?
    cb, nb = regime_best.get("canonical"), regime_best.get("no-excess")
    if cb and nb:
        out["canonical_vs_noexcess"] = {
            "canonical_best": cb,
            "noexcess_best": nb,
            "same_architecture": cb == nb,
            "paired": _paired_delta(by_arch[cb], by_arch[nb]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--results-file", default="/tmp/regime_abc/results.jsonl")
    ap.add_argument("--log-dir", default="/tmp/regime_abc")
    ap.add_argument("--mem-cap-gib", type=float, default=7.0)
    ap.add_argument("--topk-k", type=int, default=8,
                    help="Uniform per-vertex cycle-pool cap for ALL archs "
                         "(controls the GPU-memory-binding budget; attention "
                         "archs cannot exceed it without OOM).")
    ap.add_argument("--sparse-attn-k", type=int, default=0,
                    help="HSIKAN_SPARSE_ATTN_K: top-K row-wise attention "
                         "focusing (0 = dense). Lets attention pick a small "
                         "subset of a larger cycle pool.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args(argv)

    # Base env applied to EVERY arch so the cycle budget is uniform across
    # the comparison (no attention-vs-non-attention enumeration confound).
    # expandable_segments reduces CUDA fragmentation at higher caps.
    base_env = {
        "HSIKAN_TOPK_MODE": "per_vertex",
        "HSIKAN_TOPK_K": str(args.topk_k),
        "HSIKAN_SPARSE_ATTN_K": str(args.sparse_attn_k),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    repo = _repo_root()
    sweep = repo / args.sweep
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results_file = Path(args.results_file)

    regime_sets = {reg: regime_admissible_archs(repo, sweep, reg) for reg in REGIMES}
    union = sorted(regime_sets["canonical"])  # canonical is the superset

    if args.analyze_only:
        rows = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()]
        print(json.dumps(aggregate(rows, regime_sets), indent=2, default=list))
        return 0

    print("[abc] admissible architecture counts: "
          + ", ".join(f"{r}={len(regime_sets[r])}" for r in REGIMES), file=sys.stderr)

    if args.dry_run:
        print(json.dumps({
            "regime_admissible": {r: sorted(map(list, regime_sets[r])) for r in REGIMES},
            "union_archs": [list(a) for a in union],
            "jobs": [{"arch_key": _arch_key(a), "seed": s} for a in union for s in seeds],
            "n_jobs": len(union) * len(seeds),
        }, indent=2))
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(results_file)
    sha = _git_sha(repo)

    n_total = len(union) * len(seeds)
    n_done = 0
    for arch in union:
        key = _arch_key(arch)
        # base_env first (uniform budget), arch-specific knobs win on top.
        env_patch = {**base_env, **arch_env(arch, attention_topk_k=args.topk_k)}
        for seed in seeds:
            n_done += 1
            if (key, seed) in done:
                print(f"[abc] skip {key} seed={seed} (done) [{n_done}/{n_total}]", file=sys.stderr)
                continue
            log_path = log_dir / f"{key.replace('|', '_')}_seed{seed}.log"
            try:
                row, wall_s, rss = run_cell(
                    repo, dataset=args.dataset, seed=seed, env_patch=env_patch,
                    hidden=args.hidden, n_epochs=args.n_epochs, log_path=log_path,
                )
            except RuntimeError as err:
                print(f"[abc] FAIL {key} seed={seed}: {err}", file=sys.stderr)
                rec = {"arch": list(arch), "arch_key": key, "seed": seed,
                       "auc": None, "error": str(err), "git_sha": sha}
                with open(results_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                continue
            rec = {
                "arch": list(arch), "arch_key": key, "seed": seed,
                "auc": row.get("auc"), "params": row.get("n_params", row.get("params")),
                "wall_s": round(wall_s, 2), "peak_rss_gib": round(rss, 3),
                "n_epochs": args.n_epochs, "hidden": args.hidden,
                "dataset": args.dataset, "git_sha": sha,
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[abc] {key} seed={seed} auc={row.get('auc'):.4f} "
                  f"wall={wall_s:.0f}s rss={rss:.2f}G [{n_done}/{n_total}]", file=sys.stderr)

    rows = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()]
    summary = aggregate(rows, regime_sets)
    print(json.dumps(summary, indent=2, default=list))
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=list))
    print(f"[abc] summary → {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
