"""Run signed_spmv on the topology-variant fixtures so the per-call
times can be added to Table 5 alongside Table 4 in the paper.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

import numpy as np

SPMV_BIN = REPO / "target" / "release" / "examples" / "spmv_from_json"


def synth_csr(n_v: int, n_e: int, mean_arity: float, seed: int = 0):
    """Build a signed-incidence CSR: rows = vertices, cols =
    hyperedges, entries in {-1,+1}."""
    rng = np.random.default_rng(seed)
    rows_for = [[] for _ in range(n_v)]
    for j in range(n_e):
        ar = max(2, int(rng.poisson(mean_arity)))
        members = rng.choice(n_v, size=ar, replace=False)
        for k, m in enumerate(members):
            sign = 1.0 if (k == 0) else -1.0
            rows_for[int(m)].append((j, sign))

    row_ptr = [0]
    col_ind = []
    val     = []
    for r in rows_for:
        for j, s in r:
            col_ind.append(int(j))
            val.append(float(s))
        row_ptr.append(len(col_ind))
    x = (rng.random(n_e) * 2.0 - 1.0).tolist()
    return n_v, n_e, row_ptr, col_ind, val, x


def run_spmv(n_v, n_e, row_ptr, col_ind, val, x, n_repeat=50):
    payload = json.dumps({
        "n_rows": n_v, "n_cols": n_e,
        "row_ptr": row_ptr, "col_ind": col_ind,
        "val": val, "x": x, "n_repeat": n_repeat,
    })
    proc = subprocess.run(
        [str(SPMV_BIN)], input=payload,
        capture_output=True, text=True, check=True,
    )
    out = json.loads(proc.stdout)
    return out["per_call_ms"], len(col_ind)


def main():
    if not SPMV_BIN.exists():
        raise SystemExit(f"missing binary: {SPMV_BIN}")
    fixtures = [
        ("sparse-1e4   (arity 2)", 10_000, 25_000, 2.0),
        ("dense-1e4    (arity 8)", 10_000, 25_000, 8.0),
        ("synth-3e4    (arity 4)", 30_000, 75_000, 4.0),
        ("synth-1e5    (arity 4)", 100_000, 250_000, 4.0),
    ]
    print(f"{'fixture':30s} {'|V|':>7s} {'nnz':>9s} "
          f"{'per_call_ms':>13s}")
    for label, n_v, n_e, ar in fixtures:
        n, _, rp, ci, val, x = synth_csr(n_v, n_e, ar)
        ms, nnz = run_spmv(n, n_e, rp, ci, val, x, n_repeat=100)
        print(f"{label:30s} {n_v:>7d} {nnz:>9d} {ms:>13.3f}")


if __name__ == "__main__":
    main()
