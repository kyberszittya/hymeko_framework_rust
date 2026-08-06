# HSiKAN unification — phase 1: the `hymeko_neuro/core/` core

**Date:** 2026-06-24 · **Branch:** soma-vision · **Plan:** `docs/plans/2026-06-24-unify-hsikan-signed-kan-core/`

## Summary
Phase 1 of the unification: extracted the shared signed-KAN core into a new pure-torch package `hymeko_neuro/core/`,
isolated and tested. **No consumer touched** (`hymeko_rl` and `hymeko_neuro` are unchanged — that is phase 2/3).
The core is one `SignedKANLayer` parameterised over the four orthogonal axes the two implementations forked on:
aggregation backend, edge spline, skip/highway, incidence.

## Files added (all new, non-core)
- `hymeko_neuro/core/splines.py` — `catmull_rom`, `CatmullRomActivation` (the CR re-homed; byte-faithful to the
  parity-tested original), `EdgeActivation` base, `make_activation` (cr/relu/tanh Strategy).
- `hymeko_neuro/core/backends.py` — `AggregationBackend` (ABC) + `DenseBatchedBackend` (the dense einsum path; the
  sparse/Triton backends slot in at phase 3 via the same trait).
- `hymeko_neuro/core/layer.py` — `SignedKANLayer` (one conv: backend-agg → linear mix → spline → skip), the
  `HighwaySkip` (Schmidhuber gate, bias −2 = carry-dominant init), `_ResidualSkip`, skip Strategy.
- `hymeko_neuro/core/backbone.py` — `SignedKANBackbone` (incidence {fixed, learned, weighted} + layer stack + pool).
- `hymeko_neuro/core/__init__.py`, `hymeko_neuro/core/tests/test_core.py`.

## CORE.YAML items touched
None. New package; `parser`/`hymeko_core` untouched.

## Design notes
- **Skip defaults to `none`** → a layer reproduces a plain signed-conv (parity), so phase-2 adoption is
  behaviour-preserving by default; `highway` is opt-in with the canonical −2 bias init (not fudged to T≈1).
- **Transductive vs inductive is not a layer axis** — it is the consumer's input adapter; the core takes
  `(·, N, d)` features either way, so the same layer serves RL (features) and the vision line (embeddings).
- **Incidence `weighted`** = fixed structural mask (buffer) × learned per-arc weight (init 1.0 → parity); the
  real, free arc weights live only on real arcs (sparsity preserved) — the signed-hypergraph premise.

## Test results
`hymeko_neuro/core/tests/test_core.py`: **11 passed** (9.0 s). Covers: dense-backend aggregation correctness +
batched-shape guard; **Catmull-Rom parity vs `hymeko_neuro._catmull_rom_eval`**; skip modes (none/residual/
highway) forward; highway carry-dominant init (gate bias −2, mean T < 0.3); highway projection on dim mismatch;
backbone forward + mean/sum pool; incidence fixed-buffer / learned-param; `weighted` init parity + grad-masked-to-
real-arcs; adjacency/shape validation errors.

## Static analysis
- `ruff check hymeko_neuro/core/`: clean.
- `mypy --strict --ignore-missing-imports` on all 5 modules: clean.

## §6.5 anti-patterns
None. One transient duplication is *intended and planned*: `catmull_rom` / `CatmullRomActivation` now exist in
both `hymeko_neuro/core/` and `hymeko_rl/policy.py`; phase 2 deletes the `policy.py` copy (strangler-fig migration). The
parity test guards equivalence in the interim.

## Phase 2 — `hymeko_rl` migrated onto the core (DONE)
`hymeko_rl/policy.py` now delegates to the core:
- `_SignedConv` and the old `HSiKANBackbone` body **deleted**; `_catmull_rom` / `CatmullRomActivation` are now
  **re-exports** from `hymeko_neuro.core.splines` (so the Triton `cr_kernel` + parity tests keep their import path; the
  implementation lives once, in the core). The transient phase-1 duplication is resolved.
- `HSiKANBackbone` is now a thin `hg_state` adapter subclassing `hymeko_neuro.core.SignedKANBackbone` (builds the signed
  adjacency from the kinematic state, delegates everything else). It inherits `a_pos`/`a_neg`/`w_pos_arc`/
  `_effective_adj`/`node_activations`/`n_vertices`, so every external constructor/attribute is preserved.
- `hsikan_backbone` gained a `skip` param → **highway is now reachable** via `build_policy("hsikan", …, skip="highway")`.
- `policy.py`: ~305 → 187 LOC (≈118 lines of duplication removed).

**Behaviour-preserving — verified:**
- Full suite: **301 passed, 2 skipped** (`hymeko_rl/tests/` + `hymeko_neuro/core/tests/`, 4m12s).
- `test_policy_store` passes → **old trained HSiKAN checkpoints still load** (the core's `SignedKANLayer` keeps the
  same `layers.N.{w_self,w_pos,w_neg}` + `a_pos`/`a_neg` state-dict keys). This is the strongest parity evidence:
  a stored old-architecture policy round-trips into the migrated backbone.
- `ruff` + `mypy --strict` clean on `policy.py`.

## Next (phase 3)
Migrate `hymeko_neuro` onto the core (sparse backend), **gated on the OTC regression** (AUC ≥ 0.8738 / macro-F1 ≥
0.7651 within seed noise), with the Triton path kept as an optional registered backend. Higher risk (published
benchmark) — separate focused pass.
