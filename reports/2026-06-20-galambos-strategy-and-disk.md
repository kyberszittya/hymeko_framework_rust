# Galambos: smaller declarative disk + a declarative explore/exploit strategy — and it works

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plan: [docs/plans/2026-06-20-galambos-strategy/](../docs/plans/2026-06-20-galambos-strategy/)*

## Summary

Three user-directed changes: (1) the disk is smaller (it's a disk, not a coin) and now a
**declarative** field of the environment; (2) proximity is confirmed disk-*centre*-based;
(3) the PPO **explore/exploit strategy** is now a `.hymeko`. Re-baselining the **harder**
task with a wider-exploration strategy: **5/8 goals** — the arms now contact and pull far
disks into the small randomized zone. The win came from one `.hymeko` edit (the exploration
tactic), not Python — validating the declarative-strategy idea.

## Changes

- **Smaller declarative disk:** `EnvSpec.disk_radius` (default 0.02, was 0.035), declared in
  `galambos_env.hymeko` (`@disk { radius 0.02; }`); `compose_planar_scene` takes it from the
  spec. Proximity (`grasp_approach`, `disk_to_zone`, `in_zone`) is to the disk *centre*
  (`disk_pos` = disk body centroid).
- **Declarative strategy:** new `meta_strategy.hymeko` (vocab: `@exploration` /
  `@exploitation`), `galambos_strategy.hymeko` (the values), `hymeko_rl/strategy_spec.py`
  (`StrategySpec.from_hymeko` → `PPOConfig` + `log_std_init` + `curriculum_iters`).
  `build_policy` gained `log_std_init`; `train_planar_grasp` reads the strategy.
  The whole pipeline — robot + env + reward + **strategy** — is now data.

## Result (harder task, via the declarative strategy)

Exploration tactic in `galambos_strategy.hymeko`: wider action noise (`log_std_init -0.5`,
std ≈0.6), `ent_coef 0.01`, reach-out curriculum 60.

| metric | harder, prior strategy | + wider-exploration strategy |
|---|---|---|
| Goals (8 ep) | 1–2 / 8 | **5 / 8** |
| Contact happens? | no (0 steps) | **yes** (ep0 7, ep1 39, ep7 85) |
| Far-disk centring | — | ep5 0.172→**0.012**, ep3 0.145→**0.021**, ep1 0.171→**0.037** |
| Training return | ~−47 | **−65.9 → −3.6** |

The arms now *engage* the disk (contact) and pull far-spawned disks precisely into the
small moving zone. **Honest note:** my prior worry that high action noise would kill
precision was wrong here — the noise was what got the policy to engage far disks, and the
dense reward kept the centring tight. The 3 failures are now *over*-pushing (ep6 shoved the
disk away, 0.085→0.189) — the opposite of the earlier freezing, fixable by annealing the
exploration noise down late (explore early, exploit late).

GIFs: `reports/gifs/galambos_strategy/` (4 goals + 1 timeout) — far disks contacted and
pulled into the zone.

## Files touched

| File | Δ | Note |
|------|---|------|
| `data/robotics/galambos_env.hymeko` + `env_spec.py` | +6 | `@disk { radius }` + `EnvSpec.disk_radius` |
| `data/robotics/meta_strategy.hymeko` | +24 (new) | strategy vocabulary |
| `data/robotics/galambos_strategy.hymeko` | +35 (new) | the explore/exploit values |
| `hymeko_rl/strategy_spec.py` | +85 (new) | `StrategySpec.from_hymeko` |
| `hymeko_rl/policy.py` | +4 | `build_policy(log_std_init=)` |
| `hymeko_rl/train_planar_grasp.py` | rewrite | builds env + policy + PPO from `.hymeko` |
| `hymeko_rl/tests/{test_planar_grasp_env,test_strategy_spec}.py` | +~50 | disk-radius, strategy parse/build tests |

## CORE.YAML / dependencies

**None.** All `hymeko_rl/` + `data/robotics/` (non-core).

## Test results

- `hymeko_rl` planar + strategy tests — **23 passed**; `hymeko validate` on all three new
  `.hymeko` — ✅; `ruff` + `mypy --strict` — clean.

## Open / follow-up

- **Anneal exploration noise down late** (explore early → exploit late) to fix the 3
  over-pushing failures — a `log_std` schedule (declarative in the strategy).
- Contact happens but `both_contact` (two-sided pinch) is still incidental — a real pinch
  remains the structural lever.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). CPU MuJoCo, no GPU. Seeds fixed.
Checkpoint `ppo_strategy.pt` (5/8, best on the harder task).
