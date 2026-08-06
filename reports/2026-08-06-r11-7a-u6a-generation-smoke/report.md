# R11.7A U6A — Object-Variant Generation & Physics Smoke

**Date:** 2026-08-06 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher` · **Milestone base:** `8c5a6937`
(U1–U5 unification). U6 runs on this separate commit chain.
**Scope:** the first multi-object *generation* gate — does the exact-zero reach→capture pipeline build and RUN on
non-coin objects, generated from HyMeKo, with correct contracts? No bank generation, no tuning, no delivery.

---

## Object curriculum (HyMeKo declarations, single-axis ablations)

Each variant is a full `.hymeko` scene identical to `galambos_env.hymeko` except its `@dsk` object declaration; the
`ObjectSpec` is read via `EnvSpec.from_hymeko` (a curriculum entry is *another HyMeKo file*, not a Python branch).
O3 (ellipse/capsule) is parked — it needs a new `Shape` member + a `compose_planar_scene` geom branch, which we do
not change mid-measurement.

| id | ablation | scene | shape | radius | density | mass (kg) | intent |
|---|---|---|---|---|---|---|---|
| O0 | reference | `galambos_env.hymeko` | cylinder | 0.020 | default (1000) | 0.050265 | frozen control |
| O1-L | size | `galambos_env_o1_large.hymeko` | cylinder | 0.024 | 694.4444 | 0.050265 | radius ×1.20, **mass = O0** (density lowered), inertia recomputed |
| O2-M | dynamics | `galambos_env_o2_heavy.hymeko` | cylinder | 0.020 | 2000 | 0.100531 | geometry = O0, **mass ×2** |
| O4-S | shape | `galambos_env_o4_square.hymeko` | box | 0.0177245 (half) | default | 0.050265 | square prism, equal projected area ⇒ **mass = O0** |

`O1-L` density `694.4444 = 1000·(0.02/0.024)²` (exact mass match); `O4-S` half-extent `√π·0.02/2` (equal-area
square). Contract test `test_object_curriculum.py` (6/6) confirms each realizes its intended axis.

## Method

The exact-zero rig is built per variant via `_rig(object_spec=…)`, which threads the object through
`acquire_snapshot → acquire_certified_straddle → _make_env` (the U6 plumbing) and uses the footprint-aware
`CoinStraddleTargets.for_object` standoff (U3b). Each variant is then run through `reach_capture_descriptor` on
**3 deterministic coin placements × 2 seeds = 24 rollouts** (placements: `bank_c1_-0.03_+0.02` center 0.046 /
`bank_c3_r6_a-30` mid 0.077 / `bank_c2_+0.025_-0.025` far 0.109 from the zone — TRAIN/DEV only, the sealed TEST
split is untouched). Each rollout is assigned exactly one primary failure taxon (or a certified capture).

## Result — certified-straddle acquisition (confirmed, pre-gate)

Every family acquires a certified both-contact straddle at the frozen S1 seed through the generated path — the
critical de-risking that the non-circular / heavier / larger object flows through the whole stack:

| id | acquired | certified | mass (compiled) | geom_type | physical-state hash | n_dot |
|---|---|---|---|---|---|---|
| O0 | ✓ | ✓ | 0.05027 | 5 (cyl) | `16778d7df544b9e8` (= frozen ref) | −0.997 |
| O1-L | ✓ | ✓ | 0.05027 | 5 (cyl) | `111da454c2611668` | −0.995 |
| O2-M | ✓ | ✓ | 0.10053 | 5 (cyl) | `fd138a7d3bced14c` | −0.98 |
| O4-S | ✓ | ✓ | 0.05027 | 6 (**box**) | `2d189e3058ba1b2b` | −0.974 |

O0 reproduces the frozen hash (parity preserved); each variant has a distinct physics hash (genuine physical
difference); the box (geom_type 6) certifies — a non-circular object straddled by the same certified-grasp gate.

## Gate — R11_7A_OBJECT_VARIANT_GENERATION_SMOKE_PASS ✓

**Verdict: PASS.** 0 / 24 MODEL_OR_CONTRACT failures. All static contracts hold for every variant.

| criterion | result |
|---|---|
| HyMeKo-generated (`EnvSpec.from_hymeko`) | ✓ all 4 |
| stable "disk" handle | ✓ all 4 |
| mass/inertia/friction differ per intent | ✓ O1-L/O2-M/O4-S all differ |
| collision contract == O0's (contype/conaffinity) | ✓ all 4 identical |
| exact-zero reset q=[0,0,0,0] | ✓ all 4 |
| reach runs | ✓ all 24 |
| shape-aware capture certificate well-formed | ✓ all 24 |
| **0 MODEL_OR_CONTRACT failures** | ✓ **0/24** |

**Capture outcomes (informational — U6A does not gate on capture success):** 17/24 certified captures, 7
`capture_no_certified_grasp` (a CONTACT_RETENTION-class finding, *not* a model/contract fault). Per variant:

| variant | certified capture | non-certified |
|---|---|---|
| O0 | 4/6 | 2 (both **center**) |
| O1-L | 4/6 | 2 (center, far) |
| O2-M | 4/6 | 2 (both **center**) |
| O4-S (**box**) | **5/6** | 1 (center) |

**Key finding — the non-certified captures are a PLACEMENT characteristic, not variant-specific.** 6 of the 7 are
at the *center* placement (short transport, coin near the zone), and **O0 itself fails capture 2/2 there** — so
the center placement is hard for the certified-grasp gate for *every* object, the coin included. The variants
track O0 closely (all 4/6, box 5/6); no family is systematically worse than the reference. The box (O4-S)
capturing best (5/6) is a strong signal that a non-circular object flows cleanly through the exact-zero
straddle→capture stack.

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, numpy 2.4.6, macOS Darwin 25.5.0 (Apple Silicon), venv
`hymeko_framework_rust/.venv`, `OMP_NUM_THREADS=1`. Seeds {0,1} per placement; S1 cradle seed 14250. Deterministic.
Peak RSS 312 MB (≪ 16 GB cap); wall ≈ 15 min (24 rollouts, each a full reach + CEM capture solve). Result:
`reports/2026-08-06-r11-7a-u6a-generation-smoke/smoke.json`. New/edited files listed in the U6A commit.
