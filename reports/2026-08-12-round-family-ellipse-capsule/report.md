# The round family — ellipse (O3-E) + capsule/stadium (O9-K) (Track A2)

**Date:** 2026-08-12 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Plan:** [docs/plans/2026-08-12-round-family-ellipse-capsule/](../../docs/plans/2026-08-12-round-family-ellipse-capsule/) (plan.tex/pdf/tikz/mmd)

## Summary

Added the **round / smooth-non-circular** shape family — the complement of the polygon corner family: **O3-E**, a flat
elliptical prism, and **O9-K**, a flat stadium (discorectangle) prism = the planar analogue of a capsule. Both are
smooth-rimmed (no corners), orientation-dependent, and **equal-area to the coin (mass = O0)**. O3-E fills the roadmap's
long-parked "O3 ellipse" slot.

**Key recon result — native geoms are wrong here.** MuJoCo's native `ellipsoid` and `capsule` are 3-D rounded bodies
that **bulge in z**; even matched in the plane they have a different *volume* than the flat coin cylinder, so they give
**no mass parity** (measured: ellipsoid 0.0335, capsule 0.0344 vs coin 0.0503 kg) and break the flat-topped-puck
invariant. So both round shapes are built as **flat mesh prisms** (a smooth 2-D boundary extruded to ±half-thickness),
exactly like the polygon prisms — giving equal-area mass parity and consistent flat-face contact.

**⭐ Headline finding (honest).** The round shapes **build valid, mass-parity models** (all unit tests pass) but
**systematically fail the coin-tuned straddle *certification***: the smooth curved rim yields a contact normal
`n_dot ≈ +0.07` (zone_cross) / `−0.85` (zone_par) that never reaches the certification threshold (`n_dot ≈ −0.96`) the
flat-faced / cornered and small-circular shapes meet. Certification rate at 5 seeds incl. S1_SEED: **O3-E 0/5, O9-K 0/5**
(vs the triangle O6-T which certifies at S1_SEED with `n_dot = −0.962`). This is a real physical result — the *generation
/ architecture* transfers to the round family, but the coin-tuned *straddle certification* does not — and it exposed (and
this change fixes) a latent crash: the generation smoke's rig builder assumed S1_SEED always certifies.

## Files touched

| File | Change |
|---|---|
| `hymeko_rl/env/object_spec.py` | `Shape.ELLIPSE` + `Shape.CAPSULE`; preconditions (both need `radius_y`; capsule needs `radius > radius_y`); footprint branches. **No new field** — both reuse `radius`/`radius_y` |
| `hymeko_rl/env/planar_grasp_env.py` | `_ellipse_prism_vertices` + `_stadium_prism_vertices`; `compose_planar_scene` `ellipse`/`capsule` branches (reuse `disk_radius_y`, **no new kwarg**; existing branches untouched) |
| `data/robotics/galambos_env_o3_ellipse.hymeko` | **new** — `shape "ellipse"; radius 0.025; radius_y 0.016` |
| `data/robotics/galambos_env_o9_capsule.hymeko` | **new** — `shape "capsule"; radius 0.026533; radius_y 0.013266` |
| `hymeko_rl/coin_delivery/object_curriculum.py` | add `O3-E`, `O9-K` + docstring |
| `hymeko_rl/experiments/r11_7a_u6a_generation_smoke.py` | **robustness fix**: `run()` records `rig_acquisition_failed` instead of crashing when a variant's model builds but never certifies a straddle; `_summarize` gate is now model-health (not certification) + reports the failed set |
| `hymeko_rl/tests/test_object_spec.py`, `test_object_curriculum.py`, `test_generation_smoke_robustness.py` (**new**) | from_str / footprint / precondition / mass=O0 intent / membership-10 / smoke-gate robustness |

## CORE.YAML items touched

**None.** `object_spec.py`/`planar_grasp_env.py` are non-core; `compose_planar_scene` is the sanctioned builder (no
new caller — ownership guard green). No dependency added.

## Design

Both round shapes are **flat mesh prisms** reusing `radius`/`radius_y` (no new field, no new builder kwarg), and add
only new `elif` branches — existing shapes' code paths are **untouched** (lower regression risk than the N-gon refactor).

- **Ellipse** (O3-E): boundary `(rx·cos t, ry·sin t)`, 128 segments. rx=`radius`=0.025, ry=`radius_y`=0.016
  (rx·ry = 4e-4 = r², aspect 1.56:1). Compiled mass **0.050246** (Δ_O0 = 2.0e-5), footprint 25.0 mm.
- **Capsule / stadium** (O9-K): rectangle + two semicircular caps radius b, half-length a; area
  `4(a−b)b + πb² = πr²`. a=`radius`=0.026533, b=`radius_y`=0.013266 (aspect 2.0), 48 arc-segments/cap. Compiled mass
  **0.050260** (Δ_O0 = 5.9e-6), footprint 26.5 mm. Flat parallel sides + round ends (distinct from the all-curved
  ellipse).

Both Δ_O0 < 1e-4 (mass-parity tolerance) — clean shape-only ablations.

## Test results

Runner `pytest -p no:randomly`. Python 3.11.15, mujoco 3.10.0, torch 2.12.0, numpy 2.4.6.

- **Unit** — object_spec + object_curriculum + planar_grasp_env + ownership-guard + smoke-robustness: **90 passed,
  1 failed**. The single failure is the **same pre-existing, deterministic base-env spawn-distribution test** (documented
  for O6-T; spawn logic untouched here). New round-family tests all pass: `from_str("ellipse"/"capsule")`, footprint,
  preconditions (missing `radius_y` / capsule a≤b assert), O3-E/O9-K intent (geom MESH, mass=O0, footprint),
  membership-10. Two new smoke-robustness tests pass (generation gate passes + records a non-certifying object;
  a real model/contract break still fails the gate). Existing shapes' mass/geom verified unchanged (O0/O4-S/O6-T/O8-H).
- **Certification diagnosis** (`cert_sweep`): O3-E 0/5 seeds, O9-K 0/5 (incl. S1_SEED); O6-T control certifies at
  S1_SEED (`n_dot = −0.962`). The ellipse's per-attempt `n_dot` at S1_SEED is `+0.072` (zone_cross) / `−0.855`
  (zone_par), both below the certification threshold → the smooth-rim contact-normal is the mechanism.
- **Static** — `ruff` clean; `mypy --strict` clean on `object_spec.py`; no new `mypy` errors on the `planar_grasp_env`
  or smoke additions (the 5 pre-existing `planar_grasp_env` errors remain).

## Generation smoke — full 10-variant curriculum (with the robustness fix)

Ran `run()` over all 10 variants (60 rollouts) with an O0–O8 regression check vs the N-gon run
(`round_generation_smoke.json`):

- **Verdict `..._GENERATION_SMOKE_PASS`** (gate_pass=True — the generation gate is model health, not certification);
  **0 / 60 model/contract failures**; all static contracts (handle / collision / mass) OK for all 10.
- **`rig_acquisition_failed: [O3-E, O9-K]`** — the round family is recorded as *generated + contract-OK but never
  certifies a straddle at S1_SEED* (no crash, thanks to the fix). O3-E/O9-K: `rig_acquired=False`,
  handle/collision OK, mass parity, geom MESH.
- **⭐ Regression O0–O8: ALL MATCH** — every prior variant's certified-capture count is reproduced exactly, so the two
  new `elif` branches did not perturb any existing shape.

### Benchmark row (honest)

| Object | Shape | family | footprint | Generates | Certified straddle | Certified capture | mass=O0 |
|---|---|---|---|---|---|---|---|
| O0 coin | cylinder | round-limit | 20.0 mm | ✓ | ✓ | 4/6 | ref |
| O6-T | ngon (n=3) | corner | 31.1 mm | ✓ | ✓ | 5/6 | ✓ |
| O8-H | ngon (n=6) | corner | 22.0 mm | ✓ | ✓ | 1/6 | ✓ |
| **O3-E** | **ellipse** | **round** | **25.0 mm** | **✓** | **✗ (0/5 seeds, n_dot +0.07/−0.85)** | **0/6** | **✓ (Δ2e-5)** |
| **O9-K** | **capsule/stadium** | **round** | **26.5 mm** | **✓** | **✗ (0/5 seeds)** | **0/6** | **✓ (Δ6e-6)** |

**Reading.** The round family **generates cleanly** (valid mass-parity mesh, 0 contract failures) but **does not certify
a straddle** under the coin-tuned certification — the curved-rim contact normal never reaches the threshold. The polygon
/ box / coin families certify at S1_SEED; the smooth round family does not. So on this substrate the round shapes are a
**generation success + a certification wall** — a different, more fundamental limit than the N-gon's declining capture
*rate* (which still certified). Per-shape certification tuning (a round-aware `n_dot`/axis criterion, or an
orientation-varying straddle) is the open axis; the finding here is that the coin certification does **not** transfer to
smooth rims.

## Performance

- **Peak RSS: 303.2 MB** — well under the 2 GB budget (hard cap 16 GB §4).
- **Wall: 1745.6 s (~29.1 min)** for 60 CPU rollouts (macOS-26.5.2-arm64, Python 3.11.15). Diagnostic wall (generation
  smoke, not a reportable benchmark).
- Determinism: seeds fixed (`_SEEDS = (0,1)`); certification sweep seeds 14250–14255.

## §6.5 anti-patterns

None introduced. Additive branches; reuse of `radius`/`radius_y` (no new field/kwarg); no global state; `Shape.ELLIPSE`/
`Shape.CAPSULE` at the boundary (no string-typed internal config). The native-vs-mesh decision is documented in-code.

## Open issues / follow-up

- **The round-family certification wall is the substantive open item:** the coin-tuned straddle certification
  (`n_dot ≈ −0.96` threshold) does not accept the smooth curved rim (`n_dot ≈ +0.07 / −0.85`). A round-aware
  certification (a curvature-appropriate `n_dot`/axis criterion, or an orientation-varying straddle that grips the minor
  axis) would let the round family acquire — that is the open axis. Out of scope here (it touches the certification
  contract, not the shape). The shapes themselves are correct and ready for it.
- Higher-eccentricity ellipses / other stadia are now trivial (just new scene literals).
- `Shape.CAPSULE` is a *flat stadium prism* (documented), not a 3-D capsule — the 3-D capsule is a deliberate non-choice
  (z-bulge breaks mass parity + the flat-puck invariant).
- Teacher-free delivery / retrieval for the round family is **out of scope** (generation only, since capture doesn't
  certify).
- The pre-existing spawn-distribution failure + the 5 pre-existing `planar_grasp_env` mypy errors remain (flagged, not fixed).
