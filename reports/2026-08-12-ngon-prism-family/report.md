# Regular-N-gon prism family — unify the triangle, add pentagon + hexagon (Track A2)

**Date:** 2026-08-12 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Plan:** [docs/plans/2026-08-12-ngon-prism-family/](../../docs/plans/2026-08-12-ngon-prism-family/) (plan.tex/pdf/tikz/mmd)

## Summary

Generalized the one-off triangular-prism branch into a **parametric regular-N-gon prism** generator and added
**O7-P** (pentagon, n=5) and **O8-H** (hexagon, n=6). The corner count `n` is now a shape axis: **triangle (3) →
pentagon (5) → hexagon (6) → … → circle**. Chosen over per-shape `elif` branches because it is the §6.5-clean move —
one generator, one equal-area circumradius helper, no per-polygon dispatch growth. The triangle (O6-T) becomes the
**n=3 special case**, and the refactor is **bit-parity-preserving** (its committed geometry is provably unchanged).

## Files touched

| File | Change |
|---|---|
| `hymeko_rl/env/object_spec.py` | `Shape.NGON` + `n_sides` field + `polygon_sides()` + precondition; shared `equal_area_regular_ngon_circumradius(r,n)`; generalized `footprint_radius`; `n_sides` threaded through `compose_kwargs`/`planar_env_kwargs`/`from_fields` |
| `hymeko_rl/env/planar_grasp_env.py` | `_tri_prism_vertices` → `_regular_prism_vertices(n,R,half)`; `disk_n_sides` kwarg on `compose_planar_scene` + `PlanarGraspEnv`; `triangle`\|`ngon` route through the shared generator |
| `data/robotics/galambos_env_o7_pentagon.hymeko` | **new** — `@dsk shape "ngon"; n_sides 5; radius 0.02` |
| `data/robotics/galambos_env_o8_hexagon.hymeko` | **new** — `@dsk shape "ngon"; n_sides 6; radius 0.02` |
| `hymeko_rl/coin_delivery/object_curriculum.py` | add `O7-P`, `O8-H` + docstring |
| `hymeko_rl/tests/test_object_spec.py` | ngon parse / helper / footprint / precondition / **byte-parity** |
| `hymeko_rl/tests/test_object_curriculum.py` | membership (8) + pentagon/hexagon intent |

## CORE.YAML items touched

**None.** `object_spec.py` and `planar_grasp_env.py` are `hymeko_rl/**` (Python), non-core. `compose_planar_scene` is
the **sanctioned** builder (the ownership guard forbids only *new callers*, not edits to the builder itself — the guard
tests stay green). No dependency added.

## Design: one generator, no per-polygon branch (§6.5)

The equal-area circumradius of a regular n-gon (area `½·n·R²·sin(2π/n)` set to `π r²`) is
`R_eq(r,n) = sqrt(2π r² / (n·sin(2π/n)))`, a **single shared helper** used by both `footprint_radius` (straddle
standoff) and the mesh generator (built geometry) — so standoff and geometry cannot drift. At **n=3** it reduces to the
former triangle expression `sqrt(π r²/(3√3/4))` bit-identically.

**Bit-parity (regression-tested):** `_regular_prism_vertices(3, …)` emits the exact former triangle vertex string
(same apex-up angles `π/2 + k·2π/n`, same extrusion order) → the committed O6-T mass (0.050264), geom (MESH), and
footprint (31.10 mm) are **unchanged**. `test_triangle_mesh_is_byte_identical_after_ngon_refactor` enforces this.

Footprints (equal area to the coin, r=0.02), verified: **n=3 → 31.10 mm, n=5 → 22.99 mm, n=6 → 21.99 mm**, → 20 mm as
n→∞. All mass = O0 (O7-P Δ=4.2e-7, O8-H Δ=8.1e-7).

## Test results

Runner `pytest -p no:randomly`. Python 3.11.15, mujoco 3.10.0, torch 2.12.0, numpy 2.4.6.

- **Unit** — object_spec + object_curriculum + planar_grasp_env + ownership-guard: **82 passed, 1 failed**. The single
  failure is the **same pre-existing, deterministic base-env spawn-distribution test**
  (`test_planar_grasp_env.py::test_env_shapes_and_coin_placed_in_reach`) documented for the O6-T change — unrelated to
  this refactor (spawn logic untouched; fails identically on the prior tree). All N-gon tests pass:
  - `test_triangle_mesh_is_byte_identical_after_ngon_refactor` (parity), `test_equal_area_circumradius_reduces_to_triangle_at_n3`,
    `test_polygon_sides_and_ngon_footprint_shrinks_toward_circle`, `test_ngon_requires_n_sides_and_rejects_degenerate`,
    `test_ngon_family_pentagon_hexagon_at_o0_mass[O7-P/O8-H]`, membership (8 variants), `Shape.from_str("ngon")`.
  - `test_object_ownership_guard` green (no new `compose_planar_scene` caller).
- **Static** — `ruff` clean; `mypy --strict` clean on `object_spec.py` (the new logic). `planar_grasp_env.py` has 5
  `mypy --strict` errors, **all pre-existing** (mujoco untyped import; a pre-existing un-annotated fn at :663; gym/ndarray
  typing in `step()` at :1102–1117) — **none on the N-gon additions** (verified by grep).

## Generation smoke — full 8-variant curriculum

Ran `r11_7a_u6a_generation_smoke.run()` over all 8 variants (48 rollouts) with an O0–O6 regression check vs the O6-T
run (`ngon_generation_smoke.json`):

- **Verdict `R11_7A_OBJECT_VARIANT_GENERATION_SMOKE_PASS`** (gate_pass=True); **0 / 48 model/contract failures**;
  `static_contracts_ok = True` for all 8. Taxonomy: 28 `OK_CERTIFIED_CAPTURE` + 20 `capture_no_certified_grasp`.
- **⭐ Regression O0–O6: ALL MATCH** — every prior variant's certified-capture count is reproduced exactly
  (O0 4, O1-L 4, O2-M 4, O4-S 5, O5-R 2, O6-T 5). The unification is **behaviourally inert** for the existing objects,
  at the physical-rollout level (not just the mesh string). This is the strongest safety evidence.
- O7-P / O8-H static: mass 0.050265 ≈ O0, geom MESH, handle/collision/exact-zero all ✓.

### Benchmark row (honest — generation + capture only)

| Object | Shape | corners | footprint | Certified capture | mass = O0 |
|---|---|---|---|---|---|
| O0 coin | cylinder | ∞ | 20.0 mm | 4/6 | ref |
| O4-S | box (square) | 4 | 25.1 mm | 5/6 | ✓ |
| O6-T | ngon | 3 | 31.1 mm | 5/6 | ✓ |
| **O7-P** | **ngon** | **5** | **23.0 mm** | **3/6** | **✓ (Δ4e-7)** |
| **O8-H** | **ngon** | **6** | **22.0 mm** | **1/6** | **✓ (Δ8e-7)** |

**Honest reading.** Generation and contracts are perfect for the whole N-gon family (0/48 failures, mass parity,
all static contracts). But **certified-capture rate declines with corner count** on the *fixed, coin-tuned* straddle /
capture policy: triangle 5/6 → pentagon 3/6 → hexagon 1/6. This is not a generation failure — it is the capture policy
(tuned for the coin's footprint) generalizing less well to the tighter, rounder, higher-corner prisms. O7-P shows a
clean **capture-seed pattern** (all three positions certify on seed 1, all fail on seed 0) — the same
"certified capture ≠ seed-robust" theme seen on the box. Per-shape capture-policy tuning is the open axis (out of scope
here); the architecture / generation transfers cleanly across the whole corner-count axis.

## Performance

- **Peak RSS: 298.8 MB** — well under the 2 GB budget (hard cap 16 GB §4).
- **Wall: 1974.6 s (~32.9 min)** for 48 CPU rollouts (macOS-26.5.2-arm64, Python 3.11.15). Diagnostic wall (a generation
  smoke, not a reportable benchmark); dominated by re-running the 6 prior variants for the regression check.
- Determinism: seeds fixed (`_SEEDS = (0,1)`).

## §6.5 anti-patterns

**Removed one.** The per-shape geometry dispatch is now unified: instead of adding pentagon/hexagon as new `elif`
branches (a Cartesian per-shape surface), a single `_regular_prism_vertices(n_sides, …)` + one circumradius helper serve
all n, and the triangle folds into it. No new function family, no global state, no string-typed internal config (`n_sides`
is an int; `Shape.NGON` at the boundary).

## Open issues / follow-up

- The pre-existing spawn-distribution failure (above) and the 5 pre-existing `planar_grasp_env` mypy errors remain — out
  of scope; flagged, not fixed.
- `Shape.TRIANGLE` is retained as a back-compat alias (O6-T's scene says `shape "triangle"`); it routes through the same
  generator as `ngon` n=3. A future cleanup could migrate O6-T's scene to `shape "ngon"; n_sides 3` and drop the alias.
- Teacher-free delivery / retrieval for the N-gon family is **out of scope** (generation + capture only), as with O6-T.
