# Galambos: a harder task + the environment as a `.hymeko` (MDP-as-data)

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plans: [harder-task](../docs/plans/2026-06-20-galambos-harder-task/), [declarative-env](../docs/plans/2026-06-20-galambos-declarative-env/)*

## Summary

Two user-driven changes: (1) the Galambos task was generalized from a degenerate
fixed-zone/coin-between-arms setup to a **harder** one — small randomized zone in
both-arm reach, coin able to spawn outside the band, wider arm stance; (2) the
**environment scene geometry is now declared in a `.hymeko`**, and the env is built by a
three-path `PlanarGraspEnv.from_hymeko(robot=, env=, task=)`. The whole MDP — robot,
environment, reward — is now described in HyMeKo.

## 1. Harder task (less degenerate)

The old task: fixed central zone, coin always between the arms → optimum is "nudge to the
middle." Now:
- **Randomized small zone** (`zone_half` 0.055 → 0.04) re-placed each episode within both
  arms' reach (`model.site_pos` moved at reset; the coin→zone obs channel already lets the
  policy see the per-episode target).
- **Coin spawns over the reachable table**, possibly *outside* the between-arms band
  (`|x|` up to 0.20) — forces a reach-out-and-corral.
- **Wider stance**: bases ±0.14 → ±0.18 in `galambos_planar.hymeko` (source change; the
  emitter + env follow). Reach centres are read from the model, so sampling tracks the stance.

The shoulder-mobility regression still holds on the wider arm (the emitter's parent→child
contact excludes generalize).

## 2. The environment as data (the user's idea, three-path form)

Every part of the MDP already came from `.hymeko` except the scene geometry. Now:

- **`meta_env.hymeko`** (new vocabulary) — `@env_spec` + config term kinds `@target_zone`,
  `@coin_spawn`, `@workspace`, `@success` (`<isa> env.param`), mirroring `meta_reward`.
- **`galambos_env.hymeko`** (instance) — the scene geometry as field-carrying config terms
  bundled in an `env_spec` (zone half/region/randomize, coin region/clearance, workspace
  bounds, success steps).
- **`hymeko_rl/env/env_spec.py`** — `EnvSpec.from_hymeko` reads the bundle via the existing
  `_profile.read_bundle` and regex-parses each term body (the same machinery as
  `RewardSpec`).
- **`PlanarGraspEnv.from_hymeko(robot=, env=, task=)`** — composes robot + `EnvSpec` +
  `RewardSpec` into one env, mirroring `ArmReachEnv.from_hymeko`. `train_planar_grasp` now
  builds the env this way.

A new test case = a new `galambos_env.hymeko` (bigger zone, different spawn) with **no
Python change** — the manifesto's thesis applied to the RL environment.

## Files touched

| File | Δ | Note |
|------|---|------|
| `data/robotics/galambos_planar.hymeko` | +2/−2 | wider stance ±0.18 |
| `data/robotics/meta_env.hymeko` | +40 (new) | environment vocabulary |
| `data/robotics/galambos_env.hymeko` | +30 (new) | the scene as data |
| `hymeko_rl/env/env_spec.py` | +80 (new) | `EnvSpec` reader |
| `hymeko_rl/env/planar_grasp_env.py` | +~55 | randomized zone, wider coin, region kwargs, `from_hymeko` |
| `hymeko_rl/train_planar_grasp.py` | +1 | build via `from_hymeko` |
| `hymeko_rl/tests/test_planar_grasp_env.py` | +~60 | zone-randomize, outside-band, EnvSpec, from_hymeko tests |

## CORE.YAML / dependencies

**None.** All `hymeko_rl/` + `data/robotics/` (non-core). No new dependency.

## Test results

- `pytest hymeko_rl/tests/test_planar_grasp_env.py` — **17 passed** (incl. EnvSpec parses
  `galambos_env.hymeko`, `from_hymeko` env matches the spec, zone randomizes within reach
  and is observed, coin can spawn outside the band).
- `hymeko validate data/robotics/galambos_env.hymeko` — ✅.
- `ruff` + `mypy --strict` on changed code — clean.
- Full `hymeko_rl` suite — **121 passed**, no regressions.

## Elegance pass

The scene parameters had pushed `PlanarGraspEnv.__init__` to **16 arguments** — the
config-struct anti-pattern (CLAUDE.md §6.5#6). Collapsed the scene geometry into the
`EnvSpec` config struct passed as one `env=` argument; `__init__` is now **6 arguments**
(`robot, reward_spec, env, frame_skip, max_steps, difficulty`), and `from_hymeko` is a
two-liner (`env=EnvSpec.from_hymeko(env)`). `EnvSpec` is the single carrier for the scene,
in memory and on disk.

## Re-baseline (harder declarative task)

_150 it, curriculum 60, via `PlanarGraspEnv.from_hymeko`._

| | fixed-zone (old) | harder (declarative) |
|---|---|---|
| Goals (8 ep) | 5/8 | **1/8** |
| zone | fixed centre, half 0.055 | randomized, half 0.04 |
| coin | between arms | reachable table (can be outside the band) |
| stance | ±0.14 | ±0.18 |

The harder task is, as expected, much harder: the policy reduces disk→zone on most
episodes (it reaches out and pushes the coin closer) but rarely settles it in the small,
moving zone within budget. **1/8 is the honest re-baseline of a strictly harder
benchmark, not a regression of the framework** — the easier 5/8 task and its GIFs remain
valid. Closing the gap is a training-budget / structural-pinch question, deliberately not
chased here. GIFs: `reports/gifs/galambos_harder/` (the goal + two reach-out near-misses).

## Open / follow-up

- A true two-sided pinch remains the structural lever (`both_contact` still 0).
- The same three-path `from_hymeko` could back `ArmReachEnv` too (it currently reads robot
  + obs/task but not a unified env scene); a small unification.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). CPU MuJoCo, no GPU. Seeds fixed.
