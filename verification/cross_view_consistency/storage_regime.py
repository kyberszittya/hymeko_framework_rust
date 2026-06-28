"""Storage-overhead honesty (review point #2): rho is CONTROLLED, asymptotically vanishing for HIGH arity.

Proposition 4 (proved symbolically in propositions/p4_storage_overhead.py) gives
    rho - 1 = c (n + m) / (m * d_bar)  =  O(log n / d_bar)  -> 0  as d_bar -> inf.
The reviewer-fair caveat: robotics is the LOW-arity regime. A revolute/prismatic joint is a *binary* parent-child
hyperedge (arity 2), so d_bar ~ 2 and rho ~ 1.5..2.0 there -- emphatically NOT rho -> 1. The rho -> 1 limit is
the HIGH-arity regime (n-ary constraints, multi-body contact groups, n-ary SysML trace edges). This script
tabulates both regimes from the same symbolic rho, so the article can state "controlled overhead, asymptotically
vanishing for high-arity relations" rather than the attackable "rho -> 1".

Reuses the symbolic derivation; does not re-derive it (CLAUDE.md 6.1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "propositions"))

n, m, d, c = sp.symbols("n m d c", positive=True)
# rho - 1 = c (n + m) / (m d), from p4's size equations (IR = (n+m) records + m d incidences; adj = m d).
overhead = c * (n + m) / (m * d)
rho = 1 + overhead


def regime_table() -> list[tuple[str, int, float]]:
    """rho across arity regimes for a representative robot pool (n = m, c = 1 => rho = 1 + 2/d_bar)."""
    expr = sp.simplify(overhead.subs({c: 1, n: m}))          # = 2/d
    assert sp.simplify(expr - 2 / d) == 0
    rows = []
    regimes = [("robotics (binary joints)", 2),
               ("low-arity (tri/quad)", 3),
               ("mixed", 6),
               ("high-arity relations", 20),
               ("very high arity", 200)]
    for label, dv in regimes:
        rho_v = float((1 + expr).subs(d, dv))
        rows.append((label, dv, rho_v))
    return rows


def run() -> bool:
    print("Storage overhead rho across arity regimes (n=m, c=1 ; rho = 1 + 2/d_bar):\n")
    print(f"  {'regime':<28} {'d_bar':>6} {'rho':>7}")
    rows = regime_table()
    for label, dv, rho_v in rows:
        print(f"  {label:<28} {dv:>6} {rho_v:>7.2f}")
    lim = sp.limit(overhead.subs({c: 1, n: m}), d, sp.oo)
    deriv_neg = sp.simplify(sp.diff(overhead.subs({c: 1, n: m}), d)) < 0
    print(f"\n  lim_(d_bar->inf) rho = {1 + lim}  (vanishing overhead) ; d(rho)/d(d_bar) < 0 : "
          f"{bool(deriv_neg.subs(d, 2))}  (monotone decreasing)")
    print("  HONEST framing for the article: overhead is CONTROLLED, asymptotically vanishing for HIGH-arity")
    print("  relations; in the binary-joint robotics regime (d_bar~2) rho~2.0 -- a small constant, not rho->1.")
    # sanity: robotics regime is the worst (largest rho) and stays a small constant.
    robotics_rho = rows[0][2]
    return abs(robotics_rho - 2.0) < 1e-9 and (1 + lim) == 1


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
