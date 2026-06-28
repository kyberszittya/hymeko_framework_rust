"""Z3 proof: cross-view consistency reduces EXACTLY to per-view faithfulness under the shared-query architecture.

The empirical harness (cross_view.py) checks that each extractor recovers the same invariant. This script proves
the *logical* half: WHY that suffices, and precisely WHAT must be verified empirically. We model the article's
hub-and-spoke dispatcher abstractly over a finite universe of entities (e.g. links / joints):

  Q : Set         -- the ground-truth structural query over the IR (the single source of truth);
  render_f : Set  -- the set of entities the emitter for view f materialises;
  X_f(render_f)   -- the extractor; faithful iff it recovers exactly render_f.

Two theorems, discharged by Z3 (sets as Array(Int,Bool)):

  T1 (positive). Hub-and-spoke axiom: every view renders from the SAME query, render_f == Q. Then for all views
     f, g: X_f(eps_f) == X_g(eps_g). Z3 proves the negation UNSAT -> the commuting square holds by construction.

  T2 (negative, falsifiable). Drop the shared-query axiom for ONE view -- i.e. that view re-implements the
     extraction with an independent predicate (the per-format-converter anti-pattern the article argues against).
     Z3 finds a MODEL where the views disagree -> drift is possible exactly when a view stops sharing Q.

So cross-view consistency is *not* an extra runtime check: it is entailed by the architecture, and the only
empirical obligation is per-view faithfulness (X_f recovers what was rendered) -- which is what cross_view.py
falsification-tests against the real emitters over 16 fixtures x 5 views.
"""
from __future__ import annotations

import z3


def _set(name: str) -> z3.ArrayRef:
    """An abstract finite set of entities as a characteristic function Int -> Bool."""
    return z3.Array(name, z3.IntSort(), z3.BoolSort())


def theorem_positive() -> bool:
    """T1: shared-query architecture => pairwise cross-view agreement. Returns True iff Z3 proves it."""
    e = z3.Int("e")
    Q = _set("Q")
    render_urdf, render_sdf, render_mjcf = _set("render_urdf"), _set("render_sdf"), _set("render_mjcf")

    # Hub-and-spoke axiom: each emitter materialises exactly the shared query Q (it consumes the same H and the
    # same named query). Faithful extraction recovers the rendered set, so X_f(eps_f) = render_f = Q.
    shared = z3.And(
        z3.ForAll([e], render_urdf[e] == Q[e]),
        z3.ForAll([e], render_sdf[e] == Q[e]),
        z3.ForAll([e], render_mjcf[e] == Q[e]),
    )
    # Goal: all three views agree everywhere.
    agree = z3.ForAll([e], z3.And(render_urdf[e] == render_sdf[e], render_sdf[e] == render_mjcf[e]))

    s = z3.Solver()
    s.add(shared)
    s.add(z3.Not(agree))            # try to violate agreement under the architecture
    result = s.check()
    proved = result == z3.unsat     # no counterexample -> theorem holds
    print(f"  T1 positive  (shared query => agreement): negation is {result}  -> "
          f"{'PROVED' if proved else 'NOT proved'}")
    return proved


def theorem_negative() -> bool:
    """T2: a view that abandons the shared query CAN drift. Returns True iff Z3 finds a divergence model."""
    e = z3.Int("e")
    Q = _set("Q")
    render_urdf, render_sdf = _set("render_urdf"), _set("render_sdf")

    # urdf still renders from Q; sdf re-implements extraction with an INDEPENDENT predicate (no shared-query tie).
    partial_share = z3.ForAll([e], render_urdf[e] == Q[e])
    disagree = z3.Exists([e], render_urdf[e] != render_sdf[e])

    s = z3.Solver()
    s.add(partial_share)
    s.add(disagree)                 # is drift satisfiable when sdf is untethered?
    result = s.check()
    found = result == z3.sat        # a concrete drift model exists
    print(f"  T2 negative  (untethered view can drift): {result}  -> "
          f"{'drift witnessed (architecture is load-bearing)' if found else 'unexpectedly impossible'}")
    return found


def run() -> bool:
    print("Z3 cross-view consistency proof (shared-query dispatcher abstraction):\n")
    t1 = theorem_positive()
    t2 = theorem_negative()
    ok = t1 and t2
    print("\n  => cross-view agreement is ENTAILED by the shared-query architecture (T1); it FAILS exactly when "
          "a view stops sharing the query (T2).")
    print(f"  => the sole empirical obligation is per-view faithfulness, checked by cross_view.py. [{ok}]")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
