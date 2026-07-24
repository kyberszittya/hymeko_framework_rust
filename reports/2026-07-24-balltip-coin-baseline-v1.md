---
title: BALLTIP_COIN_BASELINE_V1 — frozen deployed baseline for the collision-on ball-tip coin embodiment
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: BALLTIP_COIN_EMBODIMENT_OPERATIONAL / BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0 / CLAMP_REMAINS_TASK_SPECIFIC_STRONG_BASELINE / BALLTIP_SELECTED_FOR_OBJECT_GENERALIZATION
frozen: true
---

# BALLTIP_COIN_BASELINE_V1 (frozen, 2026-07-24)

The **deployed** controller for the collision-on ball-tip coin embodiment. This is the DEPLOY policy; the B5 SAC/TD3
checkpoints are a recorded NEGATIVE result and are **NOT** deployed. Selected for object-shape generalization (it avoids
clamp-specific cupping). The clamp remains the task-specific strong coin baseline.

## Deployed controller (frozen)
```
state → ball proposal update-0 → fixed b=8 structured search → committed push/brake/release option → frozen settling pi_0 → K6
```
No RL in the loop. The search selects θ around the proposal center; the committed macro executes; the frozen clamp pi_0
settles after a valid handoff.

## Robot specification
- Spec: `data/robotics/galambos_planar_balltip_v1.hymeko` (spherical fingertips)
- Fingertip radius: **0.020** (collision = visual, one geom); golden `galambos_inertia` preserved (mass/inertia = frozen arm)
- Inter-arm collision: **ENABLED** (physically honest); filtering **DISABLED** (no exploit) — built via `build_variant_rl("balltip_nofilter")`
- Embodiment routing: `CoinRL4Dof(geom="POINT", arm_mjcf_transform=ball-tip)`; matched-start transplant shares nq=7

## Checkpoints (SHA-256, first 16)
| role | file | sha256/16 |
|---|---|---|
| proposal (ball update-0) | `carry_proposal_balltip_v1.pt` | `88679107b06c78d4` |
| settling pi_0 (frozen, shared) | `frozen/pi0_shared_clip_actor.pt` | `1902454ca7a74c27` |

## Fixed search configuration
- Budget b = **8**; horizon = 160; executor = `search_select` → `structured_random_around` (std_amp 0.6, std_dur 2.0)
- Action language: push→brake→release, θ = 15 (12 amplitudes ±3.0, 3 durations [2,18])
- Search is deterministic given (state, θ_center, search seed)

## Certificate (K6, unchanged)
- strict step: `dtz ≤ CENTER_TOL(0.02) ∧ speed < SETTLE_VEL(0.06)`, increment-or-reset
- K6 = `max_strict ≥ HELD_DWELL(6) ∧ touched`; entry_tol 0.05

## Object (canonical coin — the O0 reference for OBJECT_TO_TARGET_VARIANTS)
- from `galambos_env.hymeko` (EnvSpec.from_hymeko): `coin_shape="cylinder"`, **`disk_radius 0.02`**, `disk_half 0.02`;
  planar slide-x/slide-y/hinge-z. (The `compose_planar_scene` default 0.035 is NOT used — the hymeko scene declares 0.02.)
- target zone: cylinder site at `(zone_x, zone_y)` per the hymeko scene, `zone_half` per spec
- object variation (O1–O4): `CoinRL4Dof(disk_radius_override=…, coin_shape="cylinder"|"box")` — overrides ONLY the
  manipuland on the EnvSpec (flows to compose + arm-clearance); canonical scene/robot/reward untouched.

## Deterministic eval panel
- held-out states: seeds **14000–15200**, `build_boundary_panel(want=24)`, families {contact_retention, transport, braking}, strict_primary (0,)
- search seeds: fixed per state (paired protocol `8000 + i·131 + j`, j∈{0,1,2})

## ⚠ Distribution caveat (found in O1, 2026-07-24)
The measured numbers below are on the **transplant** handoff distribution (canonical clamp reconstruct → transplant qpos)
— correct for the B1 *robot comparison* but NOT the ball's own true deploy distribution. On **fresh per-object
reconstruction** (pi_0 replayed directly on the ball) the frozen proposal scores ≈0 (O1), while the expert still solves
0.5–0.69. So this baseline is **operational on the transplant distribution**; its fresh-reconstruct deploy performance is
an OPEN item (re-fit the proposal on fresh-reconstruct states for a true-deploy baseline). See `reports/2026-07-24-object-variants-o1.md`.

## Measured performance (deployed = proposal update-0 + b=8; TRANSPLANT distribution — see caveat)
- **Paired b=8 K6 = 0.236** (≈ 5.7/24) — the honest search-seed-paired number (the SAME protocol used for the RL claim)
- single-search-seed diagnostic (9000+i): 0.333 — a DIAGNOSTIC, not the claim
- full structured expert ceiling (192-shot): **16/24 (0.667)** — reachable only by non-local search, not at b=8
- comparison: clamp deployed ≈ 5/24; ball-tip deployed ≈ 6/24 (comparable; ball has the higher ceiling)

## Recorded NEGATIVE RL result (NOT deployed)
- `coin_balltip_b5_sac.py` + `coin_balltip_b5_eval.py`: option-level stochastic-Gaussian SAC, reward-certified
  (`delivers=True`, R_K6 9.19 ≫ −0.79), init from this update-0.
- Seed-aware paired ΔK6 (vs this own update-0): **SAC [−0.083, −0.097] median −0.09**, hier-bootstrap CI (−0.222, +0.035);
  TD3 [−0.139, −0.097]. Both SAC seeds negative → `BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0`. Local RL degrades the distill
  init; the deploy→ceiling gap needs non-local search, not local actor-critic. Checkpoints
  `carry_rl_balltip_{sac,td3}_seed{0,1}_bestval.pt` are archived, **not deployed**.

## Videos
- delivery (deployed controller success): `reports/2026-07-24-balltip-regression/video/balltip_4way_state3_seed14001.mp4`
  (ball panel — the deployed proposal+b8 delivering; clearance positive, no exploit)
- failure / exploit contrast: `…/balltip_4way_state14_seed14005.mp4` (filtered exploit shown for contrast; the honest
  collision-on ball does NOT deliver there — the remaining failure mode)

## No-regression gate
The deployed controller is the update-0 + b=8 above. Any future ball-tip coin change must reproduce Paired b=8 K6 ≥ 0.236
on the frozen eval panel with the frozen checkpoints/search config, and must NOT deploy an RL checkpoint that fails the
seed-aware paired gate (both seeds' ΔK6 > 0 with the hierarchical CI clearing 0).

## Frozen statuses
`BALLTIP_COIN_EMBODIMENT_OPERATIONAL` · `BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0` ·
`CLAMP_REMAINS_TASK_SPECIFIC_STRONG_BASELINE` · `BALLTIP_SELECTED_FOR_OBJECT_GENERALIZATION`
