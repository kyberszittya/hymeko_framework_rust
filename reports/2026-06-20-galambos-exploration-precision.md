# Galambos: precision (center bonus) + anti-stall — what worked, what didn't

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*

## Summary

User-directed reward changes on the harder declarative task: drop `action_cost` (the
freeze incentive), add an **anti-stall** term (penalise an idle arm) and a graded
**center bonus** (reward closeness to the exact zone centre). Goals went **1/8 → 2/8**.
Honest attribution: **the center bonus worked (precision); the anti-stall did not** — the
freezes are unreachable-far-coin give-ups, not a missing anti-freeze penalty.

## Changes (all declarative)

- **Removed `action_cost`** from `galambos_task.hymeko` — it rewarded stationarity (the
  freeze-then-timeout failure).
- **Added `center_bonus`** (weight 5.0): graded +1 at the exact centre → 0 at the zone
  edge (`zone_half` tolerance). The precision signal the binary `in_zone` lacked.
- **Added `arm_motion`** (weight 0.5): −max(0, v_min − arm joint speed). Reads a new
  `PlanarGraspMetrics.arm_speed`.
- Kept `grasp_approach` (= both arms' proximity to the coin), `reach_distance` (coin→zone),
  `both_contact`, `in_zone`.

## Result (harder task, 150 it + curriculum, via `from_hymeko`)

| metric | harder baseline | + center + anti-stall |
|---|---|---|
| Goals (8 ep) | 1/8 | **2/8** |
| Goal-episode disk→zone | ~0.035 (barely in) | **0.034, centred tightly** |
| Fully-stationary timeouts | 1/7 | **4/7** |

**The center bonus works.** The two goal episodes centre tightly (disk→zone 0.034, well
inside the 0.04 zone) — precision improved and goals rose.

**The anti-stall did not.** Stationary timeouts went *up* (1 → 4), all on the far-coin
spawns (dz 0.11–0.17): the arm approaches (min_tip ~0.09) but never engages, then stalls.
A small idle penalty does not overcome "this far coin isn't profitably reachable," so the
policy still freezes there. The stationarity is a *symptom* of unreachable-far-coin
give-up, not a missing anti-freeze term — the user's "no-exploration" hypothesis is not
the binding constraint here.

## Decision

Kept the new reward (net better: 2/8 vs 1/8, driven by the center bonus). The `arm_motion`
term is retained but flagged ineffective; a clean **center-only ablation** (one retrain)
would confirm the attribution — not run here to avoid another budget hit. The real lever
for the far-coin freezes is training budget / curriculum reach-out / the structural
two-sided pinch, not a reward knob.

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_rl/env/planar_grasp_env.py` | +~6 | `PlanarGraspMetrics.arm_speed` + arm dof plumbing |
| `hymeko_rl/env/reward.py` | +~25 | `_term_arm_motion`, `_term_center_bonus` |
| `data/robotics/meta_reward.hymeko` | +8 | `@arm_motion`, `@center_bonus` kinds |
| `data/robotics/galambos_task.hymeko` | +5/−3 | drop `action_cost`; add center + anti-stall |
| `hymeko_rl/tests/test_planar_grasp_env.py` | +~30 | directed tests for both terms + updated parse/registry |

## Test results

- Full `hymeko_rl` suite — **123 passed**. `hymeko validate galambos_task.hymeko` — ✅.
  `ruff` + `mypy --strict` — clean.
- GIFs: `reports/gifs/galambos_explore/` (two tightly-centred goals + a frozen far-coin
  timeout).

## CORE.YAML / dependencies

**None.** All `hymeko_rl/` + `data/robotics/` (non-core).

## Open / follow-up

- **Center-only ablation** to confirm the precision attribution (1 retrain).
- Far-coin engagement: budget / reach-out curriculum (anneal the coin spawn from near to
  far) / the structural two-sided pinch (`both_contact` still 0).

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). CPU MuJoCo, no GPU. Seeds fixed.
Checkpoint `ppo_explore.pt`.
