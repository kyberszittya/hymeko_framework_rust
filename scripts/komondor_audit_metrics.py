#!/usr/bin/env python3
"""Komondor HSiKAN-edge_cr audit metrics report.

Aggregates per-run wall, AUC, MaxRSS, per-arity setup wall and memory,
cache hit/miss ratio, and error states across the 4-dataset × 5-seed
× 2-mode = 40-run audit. Run after both chain jobs (13885723 +
13885739) have completed; pulls data from:

  - hsikan_edge_cr_audit/{results,results_ba_otc}.jsonl  (per-run JSON)
  - hsikan_edge_cr_audit/{slashdot,epinions,bitcoin_alpha,bitcoin_otc}_*_seed*.log  (MEM prints)
  - slurm sacct                                          (MaxRSS, ExitCode)

Usage:
  ssh komondor "cd /scratch/.../hymeko_framework_rust && \
                python3 scripts/komondor_audit_metrics.py"
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO / "hsikan_edge_cr_audit"
SLURM_DIR = REPO / "slurm_logs"

# (filename, label) pairs for the two chain jsonl outputs.
JSONL_FILES = [
    (AUDIT_DIR / "results.jsonl", "slashdot+epinions"),
    (AUDIT_DIR / "results_ba_otc.jsonl", "BA+OTC"),
]

MEM_RE = re.compile(
    r"\[MEM\]\s+(?P<event>[^:]+):\s+rss=(?P<rss>[\d.]+)G\s+"
    r"cuda_alloc=(?P<calloc>[\d.]+)G\s+cuda_reserved=(?P<cres>[\d.]+)G"
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_mem_log(log_path: Path) -> dict:
    """Extract per-arity memory + wall trajectory from a single-run log.

    Returns ``{"arities": [{"kind", "k", "n_t", "rss_peak", "cuda_peak"}],
                "any_cuda_alloc_above_gb": float, ...}``.
    """
    if not log_path.exists():
        return {}
    arities: list[dict] = []
    current: dict | None = None
    rss_peak_overall = 0.0
    cuda_peak_overall = 0.0
    for line in log_path.read_text(errors="replace").splitlines():
        m = MEM_RE.match(line.strip())
        if not m:
            continue
        rss = float(m["rss"])
        cuda_alloc = float(m["calloc"])
        cuda_res = float(m["cres"])
        rss_peak_overall = max(rss_peak_overall, rss)
        cuda_peak_overall = max(cuda_peak_overall, cuda_res)
        event = m["event"].strip()
        if event.startswith("before "):
            # Start of a new arity.
            parts = event.split()
            # event like "before cycle k=2" / "before walk k=4"
            if len(parts) >= 3:
                kind = parts[1]
                k_str = parts[2].lstrip("k=")
                try:
                    k_v = int(k_str)
                except ValueError:
                    k_v = -1
                if current is not None:
                    arities.append(current)
                current = {
                    "kind": kind,
                    "k": k_v,
                    "rss_start": rss,
                    "rss_peak": rss,
                    "cuda_peak": cuda_res,
                    "n_t": None,
                }
        elif "after build_me(e_tr)" in event and current is not None:
            current["rss_peak"] = max(current["rss_peak"], rss)
            current["cuda_peak"] = max(current["cuda_peak"], cuda_res)
            n_t_m = re.search(r"n_t=(\d+)", event)
            if n_t_m:
                current["n_t"] = int(n_t_m.group(1))
        elif "after build_me(e_te)" in event and current is not None:
            current["rss_peak"] = max(current["rss_peak"], rss)
            current["cuda_peak"] = max(current["cuda_peak"], cuda_res)
        elif "after gc+trim+empty_cache" in event and current is not None:
            current["rss_end"] = rss
    if current is not None:
        arities.append(current)
    return {
        "arities": arities,
        "rss_peak_overall": rss_peak_overall,
        "cuda_peak_overall": cuda_peak_overall,
    }


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) >= 2 else 0.0
    return f"{m:.4f} ± {s:.4f}"


def fmt_p50_max(values: list[float], unit: str = "") -> str:
    if not values:
        return "n/a"
    return (
        f"p50 {statistics.median(values):.1f}{unit}  "
        f"max {max(values):.1f}{unit}  "
        f"n={len(values)}"
    )


def sacct_metrics(job_ids: list[str]) -> dict[str, dict]:
    """Per-job SLURM accounting: State, MaxRSS, ExitCode."""
    if not job_ids:
        return {}
    cmd = [
        "sacct", "-j", ",".join(job_ids),
        "--format=JobID,State,Elapsed,MaxRSS,ExitCode",
        "-P", "-n",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=30,
        ).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    res: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, state, elapsed, max_rss, exit_code = parts[:5]
        res[jid] = {
            "state": state,
            "elapsed": elapsed,
            "max_rss": max_rss,
            "exit_code": exit_code,
        }
    return res


def report() -> None:
    rows: list[dict] = []
    chain_origin: list[str] = []
    for jsonl_path, label in JSONL_FILES:
        sub = load_jsonl(jsonl_path)
        for r in sub:
            r["_chain"] = label
        rows.extend(sub)
        chain_origin.append(f"{jsonl_path.name}: {len(sub)} rows")

    print("=" * 72)
    print(f"Komondor HSiKAN-edge_cr audit — metrics report")
    print("=" * 72)
    print(f"Sourced jsonl files:")
    for s in chain_origin:
        print(f"  - {s}")
    print(f"Total rows: {len(rows)}")
    print()

    if not rows:
        print("(no jsonl rows found — chains not yet started or path wrong)")
        return

    # ── Per (dataset, mode) AUC stats ────────────────────────────
    by_dsmode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        ds = r.get("dataset", "?")
        mode = r.get("_audit_mode", "?")
        by_dsmode[(ds, mode)].append(r)

    print("=== AUC and wall time per (dataset, mode) ===")
    print(f"{'dataset':<14} {'mode':<8} {'n':>3}  {'auc mean ± std':<18}  "
          f"{'wall p50/max (s)':<26}")
    print("-" * 72)
    for (ds, mode), runs in sorted(by_dsmode.items()):
        aucs = [r["auc"] for r in runs if isinstance(r.get("auc"), (int, float))]
        walls = [r["_audit_elapsed_s"] for r in runs if "_audit_elapsed_s" in r]
        print(f"{ds:<14} {mode:<8} {len(runs):>3}  {fmt_mean_std(aucs):<18}  "
              f"{fmt_p50_max(walls, 's'):<26}")
    print()

    # ── Cache hit ratio (warm vs cold wall) ───────────────────────
    # Heuristic: walls <60s are warm-cache hits; >60s are cold-cache builds.
    print("=== Cache hit ratio (heuristic: wall ≤ 60 s = warm) ===")
    print(f"{'dataset':<14} {'mode':<8} {'warm/total':<12}  warm-walls (s)         cold-walls (s)")
    print("-" * 72)
    for (ds, mode), runs in sorted(by_dsmode.items()):
        walls = [r["_audit_elapsed_s"] for r in runs if "_audit_elapsed_s" in r]
        warm = [w for w in walls if w <= 60]
        cold = [w for w in walls if w > 60]
        print(
            f"{ds:<14} {mode:<8} "
            f"{len(warm)}/{len(walls):<10}  "
            f"{fmt_p50_max(warm, 's') if warm else 'n/a':<22}  "
            f"{fmt_p50_max(cold, 's') if cold else 'n/a'}"
        )
    print()

    # ── Per-arity memory peak from MEM logs ───────────────────────
    print("=== Per-arity peak resident memory (averaged across runs) ===")
    print(f"{'dataset':<14} {'arity':<14} {'n_t':<8} {'rss peak GB':<14} {'cuda reserved GB':<18}")
    print("-" * 72)
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r.get("dataset", "?")].append(r)
    for ds, runs in sorted(by_ds.items()):
        per_arity_rss: dict[tuple[str, int], list[float]] = defaultdict(list)
        per_arity_cuda: dict[tuple[str, int], list[float]] = defaultdict(list)
        per_arity_nt: dict[tuple[str, int], list[int]] = defaultdict(list)
        for r in runs:
            mode_tag = r.get("_audit_mode", "real")
            seed = r.get("_audit_seed", 0)
            log_name = f"{ds}_{mode_tag}_seed{seed}.log"
            log_path = AUDIT_DIR / log_name
            mem = parse_mem_log(log_path)
            for a in mem.get("arities", []):
                key = (a.get("kind", "?"), a.get("k", -1))
                if a.get("rss_peak") is not None:
                    per_arity_rss[key].append(a["rss_peak"])
                if a.get("cuda_peak") is not None:
                    per_arity_cuda[key].append(a["cuda_peak"])
                if a.get("n_t") is not None:
                    per_arity_nt[key].append(a["n_t"])
        for key in sorted(per_arity_rss):
            kind, k_v = key
            label = f"{kind} k={k_v}"
            rss_med = statistics.median(per_arity_rss[key])
            cuda_med = statistics.median(per_arity_cuda[key])
            n_t_med = statistics.median(per_arity_nt[key]) if per_arity_nt[key] else "?"
            print(f"{ds:<14} {label:<14} {str(n_t_med):<8} "
                  f"{rss_med:<14.2f} {cuda_med:<18.2f}")
        if per_arity_rss:
            print()

    # ── Parallelism inferred from job environment ─────────────────
    print("=== Parallelism / threading inferred from MEM logs ===")
    # The OMP_NUM_THREADS / MALLOC_ARENA_MAX caps are set in the
    # SBATCH scripts; cite them so the metrics report is self-
    # contained. Rust-side enumeration is single-threaded for the
    # default edge_cr config (no HSIKAN_TOPK_MODE=per_vertex), so the
    # parallelism degree is dominated by PyTorch CUDA kernels on the
    # GPU rather than CPU rayon.
    env_caps = {
        "OMP_NUM_THREADS": "4 (capped in SBATCH --env)",
        "MALLOC_ARENA_MAX": "4 (glibc arena per-CPU cap)",
        "cpus_per_task": "4 (SBATCH)",
        "Rust enum threads": (
            "1 (serial DFS — HSIKAN_TOPK_MODE not set; per_vertex / global "
            "top-K variants would activate rayon-parallel paths)"
        ),
        "PyTorch CUDA": "single A100-SXM4-40GB per job (SBATCH --gres=gpu:1)",
    }
    for k, v in env_caps.items():
        print(f"  {k:<20} {v}")
    print()

    # ── Failures + exit codes ────────────────────────────────────
    print("=== Errors and failure analysis ===")
    n_fail = 0
    for r in rows:
        if r.get("auc") is None:
            n_fail += 1
    # Also check the orchestrator log for FAIL lines that didn't get a jsonl row.
    orch_fails: list[str] = []
    for log_name in ("chain.log", "chain_ba_otc.log"):
        log_path = AUDIT_DIR / log_name
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            if "FAIL" in line:
                orch_fails.append(line.strip())
    print(f"  jsonl rows with null AUC: {n_fail}")
    print(f"  orchestrator FAIL lines:  {len(orch_fails)}")
    for line in orch_fails[:10]:
        print(f"    {line}")
    print()

    # ── SLURM sacct (if running on Komondor) ──────────────────────
    print("=== SLURM job-level accounting (sacct) ===")
    job_ids: list[str] = []
    for log_name in (
        "hsikan-edge-cr-audit-13885723.out",
        "hsikan-edge-cr-ba-otc-13885739.out",
    ):
        m = re.search(r"-(\d+)\.out$", log_name)
        if m:
            job_ids.append(m.group(1))
    job_info = sacct_metrics(job_ids)
    if job_info:
        for jid, info in sorted(job_info.items()):
            print(f"  {jid:<16} state={info['state']:<10} "
                  f"elapsed={info['elapsed']:<10} "
                  f"MaxRSS={info['max_rss']:<10} "
                  f"exit={info['exit_code']}")
    else:
        print("  (sacct unavailable — run this script on Komondor)")
    print()

    # ── Comparison vs published / local baselines ────────────────
    print("=== Headline comparisons ===")
    published = {
        ("slashdot", "real"):    (0.9067, 0.0029, "edge_cr 5-seed (2026-05-09 report)"),
        ("epinions", "real"):    (0.8409, None, "edge_cr 5-seed single, project_epinions_edge_cr_null"),
        ("bitcoin_alpha", "real"): (0.9959, 0.0011, "Optuna-best 10-seed (project_bitcoin_optuna_best_10seed)"),
        ("bitcoin_otc", "real"):   (0.9933, 0.0023, "Optuna-best 10-seed"),
    }
    for (ds, mode), (mu, sig, src) in published.items():
        runs = by_dsmode.get((ds, mode), [])
        aucs = [r["auc"] for r in runs if isinstance(r.get("auc"), (int, float))]
        if aucs:
            our_mu = statistics.mean(aucs)
            our_sig = statistics.stdev(aucs) if len(aucs) >= 2 else 0.0
            delta = our_mu - mu
            print(
                f"  {ds:<14} {mode:<8} "
                f"ours={our_mu:.4f}±{our_sig:.4f} (n={len(aucs)})  "
                f"published={mu:.4f}±{(sig or 0):.4f}  Δ={delta:+.4f}  "
                f"({src})"
            )
        else:
            print(f"  {ds:<14} {mode:<8} (no jsonl rows yet)")


if __name__ == "__main__":
    report()
