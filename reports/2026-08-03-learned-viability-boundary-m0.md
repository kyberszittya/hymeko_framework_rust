# M0 — learned viability boundary + curvature geometry

**Date:** 2026-08-03
**Plan:** `docs/plans/2026-08-02-learned-viability-boundary/` (on disk, per the repo's gitignored-plans convention)
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `66471ed9`)

---

## Summary

Implemented milestone **M0** of the learned-viability-boundary plan, with the curvature geometry it depends on:

- **`geometry.py`** — the differential geometry of the mechanical metric `M(q)`: a `RiemannianMetric` class
  (Christoffel → Riemann → Ricci → scalar / Gauss curvature, cached), the connection-corrected Bakry–Émery
  Ricci `Ric(g)+∇²V`, and a **numeric** `scalar_curvature_numeric(metric_fn, q0)` (nested central differences)
  so the full humanoid's curvature is reachable from `mj_fullM` where the closed form is intractable.
- **`viability.py`** — the boundary as a level set of `H_d`: analytic separatrix `c*=½kπ²`, exact `in_roa`,
  a **vectorised** closed-loop rollout labeller (`sample_viability`, struct-of-arrays over the whole grid),
  a **numpy-only** `LearnedBoundary` (logistic on the curvature-aware features `(u²,v²)` that can represent the
  ellipse exactly — no new dependency), and `validate_boundary` (IoU + per-class error vs the analytic ROA on a
  held-out, half-cell-shifted grid).

**Self-validation result (the point of M0):** trained on the analytic ROA the learned boundary recovers the
separatrix to **IoU 0.997** (err_recover 0.0, err_fall 0.021); trained purely on **rollout data** it still tracks
the analytic ROA to **IoU 0.941**. Rollout labels agree with the analytic ROA on **93.5%** of the grid.

**Honest physics (not a bug):** the rollout (dynamic) ROA is a *mild super-set* of the conservative energy
sublevel `{H_d<c*}` — damping enlarges the true basin, so `err_fall≈0.39` vs the analytic labels is the
**damping shell**, not misclassification. The test asserts this direction explicitly
(`y_roll.mean() ≥ analytic.mean()`). The conservative analytic boundary is the one to use as a *safety* envelope;
the dynamic one is the more accurate *descriptive* boundary.

## Files touched (full list)

| File | LOC | notes |
|---|---|---|
| `scenarios/humanoid/geometry.py` | +135 (new) | `RiemannianMetric`; symbolic + numeric curvature; Bakry–Émery |
| `scenarios/humanoid/viability.py` | +182 (new) | separatrix, rollout labeller, `LearnedBoundary`, `validate_boundary` |
| `tests/test_viability_boundary.py` | +165 (new) | 14 tests (unit + integration + performance) |
| `reports/2026-08-03-learned-viability-boundary-m0.md` | new | this report |

## CORE.YAML items touched
None. Non-core Python only. **No new dependency** — the learned boundary is numpy-only (no scikit-learn/torch);
SymPy/NumPy already present. §1 not triggered.

## Test results

- `pytest tests/test_viability_boundary.py -p no:randomly` → **14 passed in 62.9 s** (dominated by the symbolic
  curvature simplifications and the 41×41 rollout tests).
- Regression: `tests/test_humanoid_port_hamiltonian.py` → **12 passed in 10.0 s** (unchanged).
- `ruff check` on all three files → clean.

Layer coverage: **unit** (separatrix level; `in_roa`≡`H_d<c*`; control matches symbolic IDA-PBC; determinism;
Gauss curvature values + sign change; `R=2K`; flat metric `R=0`; Bakry–Émery `=k`; numeric↔symbolic curvature;
fit-before-predict) · **integration** (rollout↔analytic agreement ≥0.92; analytic-trained IoU ≥0.97;
data-trained IoU ≥0.9 with the shell direction asserted) · **performance** (below).

## Performance results

- `sample_viability` (41×41 = 1681 states, 1000 steps, vectorised): **median 25.5 ms**, IQR ≈0.1 ms,
  worst 26.0 ms (5 iters after warm-up) — budget 5 s, met by ~200×. RSS negligible (numpy scalar arrays).
- Symbolic Gauss curvature of the 2-link leg: ~2–4 s (SymPy `simplify`/`trigsimp`); the perf-sensitive path
  (rollouts) is the vectorised one above.

## Curvature results (grounding the geometry)

- 2-link leg Gauss curvature `K(q₂)`: **+6.19** at `q₂=0.05` (knee near-straight, geodesics focus) → **−5.91**
  at `q₂=3.0` (knee folded, geodesics diverge) — sign change verified; `R=2K` internal identity holds.
- Bakry–Émery Ricci of the shaped pendulum = the shaping gain `k` (entropy-relaxation rate bound).
- Numeric (finite-difference) scalar curvature matches the symbolic value to `<1e-2` — validates the `mj_fullM`
  path on the case where the closed form is known.

## §6.5 anti-patterns
None introduced. `RiemannianMetric` bundles state + behaviour (no free-function curvature dump); one
`LearnedBoundary` class, not a per-feature-set family; `ViabilityConfig` is a typed config, not string-typed
flags; no globals (a module logger only, per the allowed exception). Error paths explicit (`ValueError` on bad
shapes/`k≤0`, `RuntimeError` on predict-before-fit) — no `unwrap`/bare-except equivalent.

## Open issues / follow-up
- **M1 (roadmap):** a neural Lyapunov / barrier *certificate* seeded by `H_d` with a sampling/SMT verification
  pass — a certified boundary, not a classifier. Would introduce torch → a §1 core decision, to be escalated.
- **M2 (roadmap):** centroidal capturability boundary of the runner; the `L`-ports (foot placement, arm swing,
  shoulder + pitch hold) as inputs; co-learned with the policy as a safety envelope.
- The damping shell means "safety" and "descriptive" boundaries differ; M1's certificate should target the
  *conservative* (guaranteed-invariant) side.

## Provenance
Git SHA at start `66471ed9` (working tree also carries unrelated untracked seminar/experiment files, excluded).
Env: HyMeKo `.venv` (Python 3.11, NumPy, SymPy 1.14, MuJoCo 3.10 present but unused by these modules), macOS
(darwin 25.5). Deterministic: fixed `seed=0`, pinned `dt=4 ms`; rollouts have no RNG (grid + integrator only);
`LearnedBoundary` init seeded. No GPU. No dataset (analytic + simulated).
