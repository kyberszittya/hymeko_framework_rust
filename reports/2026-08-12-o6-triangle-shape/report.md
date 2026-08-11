# O6-T — Triangle: the first corner-bearing object (Track A2 probe)

**Date:** 2026-08-12 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Plan:** [docs/plans/2026-08-12-o6-triangle-shape/](../../docs/plans/2026-08-12-o6-triangle-shape/) (plan.tex/pdf/tikz/mmd)

## Summary

Added **O6-T**, an equilateral triangular prism, as the first *corner-bearing* manipuland in the multi-object
curriculum, and characterized it through the existing exact-zero generation smoke. This is a **Track A2 (shapes)
probe** — the frozen roadmap's `NOW` is Track C0; this is a user-directed shape excursion (`megpróbáljuk más
alakzatokkal?`).

**No new geometry code.** `Shape.TRIANGLE`, `ObjectSpec.from_hymeko`, `compose_kwargs`, `footprint_radius`, and the
`compose_planar_scene` `"triangle"` branch (equal-area equilateral prism mesh, density-derived mass/COM/inertia)
already existed. The object is declared as another HyMeKo scene and added to the curriculum; the existing generation
smoke picks it up automatically (no harness edit, no duplication).

The triangle is the first manipuland with **sharp corners**: three flat faces meeting at three vertices, a footprint
circumradius ≈ 31.1 mm (vs the coin's 20 mm and the square's ≈ 25 mm diagonal), equal projected area ⇒ **mass = O0**
(clean shape-only ablation).

## Files touched

| File | Change | LOC |
|---|---|---|
| `data/robotics/galambos_env_o6_triangle.hymeko` | **new** — O6-T scene (`@dsk shape "triangle"` radius 0.02) | +33 |
| `hymeko_rl/coin_delivery/object_curriculum.py` | add `ObjectVariant("O6-T", …, "shape-corner")` + docstring | +8 / −3 |
| `hymeko_rl/tests/test_object_curriculum.py` | **fix pre-existing red** membership + new triangle intent test | +14 / −1 |
| `docs/plans/2026-08-12-o6-triangle-shape/` | **new** — 4-format plan | — |
| `reports/2026-08-12-o6-triangle-shape/` | **new** — this report + smoke JSON | — |

## CORE.YAML items touched

**None.** Verified against `CORE.YAML`: core = the Rust crates (`hymeko_core/query/client/daemon/parser`),
`docs/spec/*`, `rtl/*`, pinned deps. All targets are `hymeko_rl/**` (Python), `data/robotics/*.hymeko`, `tests/**`,
`reports/**`, `docs/plans/**` — non-core (`on_unknown_path: treat_as_non_core`). No dependency added or changed.

## Pre-existing failing test — fixed as part of this change

Per CLAUDE.md §9, declared explicitly: `test_object_curriculum.py::test_all_variants_load_from_hymeko_with_stable_handle`
was **already red on the clean tree** — it asserted the curriculum was `["O0","O1-L","O2-M","O4-S"]`, but `O5-R` had
been added earlier (for R12.2) without updating the assertion. Baseline before this change: **1 failed, 27 passed**.
This change corrects the membership pin to the full intended set `["O0","O1-L","O2-M","O4-S","O5-R","O6-T"]` and adds
the triangle intent test. The corrected assertion is a genuine regression test: it fails against the prior tree.

## Test results

Runner: `pytest -p no:randomly` (tools.yaml). Python 3.11.15, mujoco 3.10.0, torch 2.12.0, numpy 2.4.6.

- **Unit** — `test_object_curriculum.py` + `test_object_spec.py`: **29 passed** (was 27 passed + 1 pre-existing fail).
  - New `test_o6_triangle_is_corner_prism_at_o0_mass`: geom type **MESH**, `body_mass` = 0.050264 kg ≈ O0
    (0.050265, |Δ| = 1.5e-6 ≪ 1e-4 tol), `footprint_radius` = 0.031102 m > coin 0.020 m. Would fail on the prior tree
    (no O6-T variant).
  - Existing triangle coverage in `test_object_spec.py` (`Shape.from_str("triangle")`, footprint circumradius > disk)
    still green.
- **Static analysis** — `ruff check` clean; `mypy --strict` clean on the changed non-test module.
- **Integration (generation smoke)** — full 6-variant curriculum incl. O6-T: **see below.**

### A second, separate pre-existing failure (NOT introduced here, NOT fixed here)

Widening the test run surfaced `test_planar_grasp_env.py::test_env_shapes_and_coin_placed_in_reach` as **also red on
the clean tree** (matches the memory note "pre-existing planar spawn-distribution test failure is UNRELATED, fails
identically on clean tree"). Verified it is not mine and not flaky:
- `git status` shows `test_planar_grasp_env.py` and `env/planar_grasp_env.py` **both unmodified** by this change; my
  edits are confined to `object_curriculum.py` + `test_object_curriculum.py` + the new scene, none imported by the
  failing assertion.
- The test is **deterministic** (`env.reset(seed=seed)` for `seed in range(16)`), and fails **identically 3/3 re-runs**
  — it is a genuine base-env spawn-distribution expectation gap, not a flake.
Left **as-is** (out of scope: it concerns the base env's spawn distribution, unrelated to the O6-T shape). Flagged as a
known pre-existing failure; not fixed here to avoid scope creep into env logic.

## Generation smoke — full curriculum incl. O6-T

Ran the existing `r11_7a_u6a_generation_smoke.run()` over the **full 6-variant curriculum** (via import; the historical
`smoke.json` was left untouched). 6 variants × 3 positions × 2 seeds = **36 rollouts**. Result
(`o6_generation_smoke.json`):

- **Verdict `R11_7A_OBJECT_VARIANT_GENERATION_SMOKE_PASS`** (gate_pass=True).
- **0 / 36 model/contract failures**; `static_contracts_ok = True` for all 6 (handle / collision / exact-zero-reset).
- Taxonomy: **24 `OK_CERTIFIED_CAPTURE` + 12 `capture_no_certified_grasp`** — no reach failures, no exceptions, no
  contract trips.

| Variant | geom | mass (kg) | handle/collision/zero | certified capture (of 6) |
|---|---|---|---|---|
| O0 coin | cylinder | 0.05027 | ✓/✓/✓ | 4 |
| O1-L size | cylinder | 0.05027 | ✓/✓/✓ | 4 |
| O2-M heavy | cylinder | 0.10053 (2×) | ✓/✓/✓ | 4 |
| O4-S box | box | 0.05027 | ✓/✓/✓ | 5 |
| O5-R rect | box | 0.05027 | ✓/✓/✓ | 2 |
| **O6-T triangle** | **mesh** | **0.05026** | **✓/✓/✓** | **5** |

**O6-T captures 5/6 — tied best with the box, better than the coin (4/6).** The single miss is at `center/seed0`
(`capture_no_certified_grasp`) — the placement where O0 itself struggles (center is the known-hard cell), **not a
triangle-specific failure**: offcenter 2/2 + far 2/2 + center 1/2. Reach ran on every rollout (object-invariant).

### Benchmark row (honest — generation + capture only)

| Object | Shape | Reach | Certified straddle | Certified capture | mass = O0 | Teacher-free delivery |
|---|---|---|---|---|---|---|
| O0 coin | cylinder | ✓ (1.0) | ✓ | 4/6 | ref | ✓ (prior R11.6C) |
| O4-S box | box | ✓ | ✓ | 5/6 | ✓ | existence ✓ / not stable |
| **O6-T** | **triangle (mesh)** | **✓** | **✓** | **5/6** | **✓ (Δ1.5e-6)** | **not yet run** |

The teacher-free delivery / retrieval column for O6-T is **out of scope for this probe** (no bank, no retrieval, no K6
delivery). This step establishes: the first sharp-cornered object **generates cleanly and captures at least as well as
the box** through the identical frozen exact-zero pipeline, with mass parity to O0 (clean shape-only ablation).

## Performance

- **Peak RSS: 301.6 MB** — well under the 2 GB budget (hard cap 16 GB §4). Live sampling mid-run showed ~297 MB flat.
- **Wall: 1286.4 s (~21.4 min)** for 36 CPU rollouts on the Mac runner (macOS-26.5.2-arm64, Python 3.11.15). This is a
  diagnostic wall (a generation smoke, not a reportable benchmark); the re-run cost is dominated by re-executing the 5
  non-triangle variants for regression coverage, not by the triangle.
- Determinism: seeds fixed (`_SEEDS = (0,1)`); no system entropy.

## §6.5 anti-patterns

None introduced. No new function family, no Cartesian surface, no harness duplication (the existing smoke's `run()` is
reused via import), no new geometry branch (the triangle path pre-existed), no global state, no string-typed config
(the shape is `Shape.TRIANGLE` at the boundary).

## Open issues / follow-up

- The generation smoke writes to a **hardcoded** historical path (`reports/2026-08-06-…/smoke.json`); this run used the
  reusable `run()` entry to write to the O6 report dir instead, leaving the historical artifact untouched. A small
  follow-up could make the output path a CLI arg (additive) so re-runs don't clobber history.
- If the triangle certifies straddle + captures cleanly, a bounded teacher characterization (reach / capture /
  teacher-K6|capture over a few seeds) would fill a full benchmark row — a **separate** step, per the plan's scope.
- O3 ellipse/capsule remains the roadmap's proper A2 shape-family (needs a new `Shape` member + geom branch).
