# Galambos: 100-seed eval + the arm-arm penalty exposes a degenerate policy

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*

## Summary

A proper 100-seed evaluation corrected the headline (the 5/8 anecdote was noise: real rate
**25 %**), and a user-requested arm–arm collision penalty revealed *why* even 25 % was
misleading — the policy was **mashing the two arms together** (72.5 % mutual contact) and
shoving the disk as a clump, not grasping it. The penalty eliminates the clash (→ 0 %) but
the honest two-arm score is **13 %**, undertrained. The penalty is kept: it makes the setup
legitimate.

## 1. Multi-seed evaluation (the honesty fix)

New `hymeko_rl/eval_planar_grasp.py` — rolls the deterministic policy over N held-out seeds,
reports goal/death/timeout + the goal-rate 95 % Wilson CI.

| checkpoint | goals/100 | 95 % CI | deaths | note |
|---|---|---|---|---|
| `ppo_strategy` (no penalty) | **25** | 17.5–34.3 % | 0 | the "5/8" was an 8-seed lucky draw |
| `ppo_noclash` (arm-arm penalty) | 13 | 7.8–21.0 % | 20 | honest two-arm score, undertrained |

## 2. The arm-arm penalty (user request) — and what it exposed

Declarative `@arm_collision` term (−1 while a left-arm geom touches a right-arm geom),
detected in `compute_planar_metrics` (a new `PlanarGraspMetrics.arm_self_contact`), wired in
`galambos_task.hymeko` at weight 1.0. It works exactly as intended:

| policy | arm–arm contact (steps) |
|---|---|
| no penalty (`ppo_strategy`) | **2302 / 3176 = 72.5 %** |
| arm-arm penalty (`ppo_noclash`) | **0 / 3894 = 0.0 %** |

**The finding:** the 25 % policy achieved its goals by clamping the two arms together and
using them as one pushing surface — degenerate for a *two-finger* task. The penalty makes
that impossible, so the policy must use two separated arms — much harder, and at 150 iters it
manages 13 % (with 20 deaths as the separated arms over-push the disk out of bounds). The
penalty is therefore **correct and kept**: it converts a clumped-pusher hack into a real
(if currently weak) two-arm task. The drop is honest difficulty, not a regression.

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_rl/eval_planar_grasp.py` | +60 (new) | N-seed eval + Wilson CI |
| `hymeko_rl/env/planar_grasp_env.py` | +~10 | `arm_self_contact` metric (left↔right geom contact) |
| `hymeko_rl/env/reward.py` | +12 | `_term_arm_collision` |
| `data/robotics/meta_reward.hymeko` | +4 | `@arm_collision` kind |
| `data/robotics/galambos_task.hymeko` | +3 | `@noclash` in the reward spec |
| `hymeko_rl/tests/test_planar_grasp_env.py` | +~12 | arm-collision term + metric field |

## CORE.YAML / dependencies

**None.** All `hymeko_rl/` + `data/robotics/` (non-core).

## Test results

- Full `hymeko_rl` suite — **128 passed**. `hymeko validate galambos_task.hymeko` — ✅.
  `ruff` + `mypy --strict` (changed) — clean.

## Open / follow-up

- **Retrain the legitimate (no-clash) setup with more budget** — the two-arm task needs more
  than 150 iters; the 13 % should recover well above it. This is the real next run.
- The 20 deaths suggest adding a mild out-of-bounds shaping or keeping the disk in-workspace
  is worth it for the separated-arm policy.
- An explore→exploit `log_std` schedule (declarative in the strategy) for the over-pushing.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). CPU MuJoCo, no GPU. Eval seeds
2000–2099 (held out from training seed 0 + diagnostic 1000–1007). Checkpoints `ppo_strategy.pt`
(25 %, clumped), `ppo_noclash.pt` (13 %, legitimate two-arm).
