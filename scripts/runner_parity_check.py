#!/usr/bin/env python3
"""Framework parity check: run the new Experiment runner on a smoke
config, then compare AUC to a prior baseline measurement.

Used to validate that the new runner framework produces the same
numerical AUC as the old shell-script path it replaces.

Usage:
  python3 scripts/runner_parity_check.py \\
      --config hymeko_neuro/experiments/configs/_smoke_ba_real_seed0.yaml \\
      --baseline-auc 0.9946 \\
      --tolerance 0.005

Exits non-zero if measured AUC deviates from baseline by > tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config to run via the framework")
    p.add_argument("--baseline-auc", required=True, type=float,
                   help="prior AUC measurement to compare against")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="acceptable |delta| (default 0.005)")
    p.add_argument("--baseline-label", default="prior",
                   help="label for the baseline in the report")
    args = p.parse_args(argv)

    # Import + run the framework. Subprocess'd so the runner runs in
    # a clean env (same as the user-facing CLI).
    import subprocess
    print(f"[parity] running framework on {args.config.name} ...")
    rc = subprocess.call([
        sys.executable, "-m", "hymeko_neuro.experiments.run",
        "--config", str(args.config),
    ], env={**__import__("os").environ, "PYTHONPATH": "."})
    if rc != 0:
        print(f"[parity] FAILED: framework run rc={rc}", file=sys.stderr)
        return 1

    # The framework emits JSONL to the config-specified path; we read it back.
    import yaml
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    jsonl_path = Path(cfg["output"]["jsonl"].replace("${name}", cfg["name"]))
    if not jsonl_path.exists():
        print(f"[parity] FAILED: no jsonl at {jsonl_path}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        print(f"[parity] FAILED: empty jsonl at {jsonl_path}", file=sys.stderr)
        return 3

    # For SingleCellExperiment the jsonl has 1 row.
    aucs = [r["auc"] for r in rows if isinstance(r.get("auc"), (int, float))]
    if not aucs:
        print(f"[parity] FAILED: no auc field in jsonl rows", file=sys.stderr)
        return 4

    measured = sum(aucs) / len(aucs)
    delta = measured - args.baseline_auc
    within = abs(delta) <= args.tolerance

    print()
    print("=" * 60)
    print(f"  baseline ({args.baseline_label}):  {args.baseline_auc:.4f}")
    print(f"  framework (this run):  {measured:.4f}  (n={len(aucs)})")
    print(f"  delta:    {delta:+.4f}  (tolerance: ±{args.tolerance:.4f})")
    print(f"  verdict:  {'PASS' if within else 'FAIL'}")
    print("=" * 60)

    return 0 if within else 5


if __name__ == "__main__":
    sys.exit(main())
