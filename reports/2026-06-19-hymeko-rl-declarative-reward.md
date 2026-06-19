# Report — declarative reward: `step()`'s reward driven by the `.hymeko` task profile

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ✅ complete — the env's reward is assembled from a declared `RewardSpec` read
from a `.hymeko` task profile; the hard-coded `-dist` is gone. The whole reaching MDP
(robot + observation + reward) now comes from `.hymeko`. One pre-existing flaky PPO test
flagged (unrelated; root-caused below).

## Summary
Fourth and final increment of the Kato "one `.hymeko` source → whole MDP" line. The reward
was the last procedural piece — `step()` returned `-dist` literally. It is now a
**declarative reward spec**: an ordered list of `(term_kind, weight)` pairs, each term mapped
to an extractor (Strategy) over the live env state, summed as `Σ weight·term`. The terms +
weights are read from a `.hymeko` task profile's `reward_spec`, so a new reward needs only a
new `.hymeko`.

The behaviour is **bit-identical** to the former `-dist` (the default `REACH_REWARD` is
`reach_distance` weight 1.0), so BC/PPO behaviour is unchanged — verified by an equivalence
test and by `np.array_equal` on the observation.

## Design (symmetric with the observation spec)
New `.hymeko` artifacts (both validate via `hymeko validate`):
- **`data/robotics/meta_reward.hymeko`** — the reward vocabulary: a `reward_term` root, a
  `reward_spec` bundle type, and term kinds `@reach_distance` (default weight 1.0),
  `@success_bonus`, `@action_cost`. Companion to `meta_observation` / `meta_task`.
- **`data/robotics/arm_reach_task.hymeko`** — the reaching task profile: a `@goal` (`move_to`
  flange→target) and a `@reach_reward` (`reward_spec` bundling `@dist: reach_distance` weight
  1.0). The HyMeKo form of the env's `-dist`.

New Python (`hymeko_rl/env/`):
- **`reward.py`** — term extractors (`reach_distance` = `-dist`, `success_bonus`,
  `action_cost`), a `_REWARD_TERMS` registry, `RewardSpec(terms)` with `evaluate(env, dist,
  action) = Σ w·term` and DbC validation, `read_reward_terms` (terms + weights from a
  profile), and `REACH_REWARD`.
- **`_profile.py`** — `read_bundle(profile, spec_kind)`, the **shared** narrow profile reader
  for both specs (a spec bundle + its ordered members). `observation.read_channels` was
  refactored to use it — the obs and reward readers no longer near-copy the parse (§6.1).

Wiring (`arm_reach_env.py`): `__init__` takes `reward_spec` (default `REACH_REWARD`);
`step()` returns `self.reward_spec.evaluate(self, dist, ctrl)`; `from_hymeko` gains
`task_profile=` → `RewardSpec.from_hymeko`, completing one-source **scene + obs + reward**.

## Files touched (untracked `hymeko_rl/`; tracked `data/robotics/`)
| File | Change |
|---|---|
| `data/robotics/meta_reward.hymeko` | **new** — reward vocabulary |
| `data/robotics/arm_reach_task.hymeko` | **new** — reaching task + reward profile |
| `hymeko_rl/env/reward.py` | **new**, ~110 — terms, registry, `RewardSpec`, reader |
| `hymeko_rl/env/_profile.py` | **new**, ~45 — shared `read_bundle` |
| `hymeko_rl/env/observation.py` | `read_channels` now delegates to `read_bundle` (−25 lines) |
| `hymeko_rl/ppo.py` | `PPOConfig.seed`; `train_ppo` seeds the env reset (flaky-test fix) |
| `hymeko_rl/env/arm_reach_env.py` | `reward_spec` param, `step()` reward, `from_hymeko(task_profile=)` |
| `hymeko_rl/tests/test_reward.py` | **new**, ~95 — 7 tests |

**CORE.YAML items touched:** none. (Confirmed `data/robotics`, `meta_task` are not CORE.)

## Tests (`pytest -p no:randomly`)
- **`test_reward.py` — 7 passed:** `REACH_REWARD` = `-dist`; weighted multi-term sum
  (distance + success bonus + action cost); unknown/empty rejected (DbC); `read_reward_terms`
  recovers the profile's `(reach_distance, 1.0)`; weight + order parsed from a tmp profile;
  missing `reward_spec` rejected; and an **equivalence** test — `step()`'s reward
  `== pytest.approx(-info["dist"])`.
- **`test_observation.py` — 7 passed** after the `read_bundle` refactor (reader behaviour
  unchanged).
- **Equivalence (bit-exact):** the spec-driven `node_features` is `np.array_equal` to the
  legacy layout across 5 seeds, and the reward is `float(1.0·−dist) == −dist` — so PPO/BC
  inputs are byte-for-byte unchanged.
- **Regression:** `test_reach_bc`, `test_arm_world`, `test_arm_reach_from_hymeko`,
  `test_policy`, `test_hypergraph_state`, and all `test_ppo` pass. **Full `hymeko_rl` suite:
  55 passed, 0 failed** (after the flaky-test fix below).

End-to-end (one source): `ArmReachEnv.from_hymeko(anthropomorphic_arm.hymeko,
obs_profile=arm_reach_observation.hymeko, task_profile=arm_reach_task.hymeko)` → obs channels
**and** reward terms read from `.hymeko`; `step()` reward `−0.49 == −dist`; expert reaches
`0.490 → 0.059 m`.

## Pre-existing flaky test — diagnosed AND fixed
`test_ppo.py::test_ppo_improves_return` was flaky — **1 pass / 1 fail** on identical seeded
code. Root cause (diagnosed, then confirmed by the fix): `train_ppo`'s `env.reset()` at
`hymeko_rl/ppo.py:163` (and the in-loop reset at `:123`) was **unseeded**, so the reaching
targets drew from system entropy → episodic returns varied run-to-run → the marginal
assertion `mean(history[-3:]) > history[0]` flipped. A §3 "no reliance on system entropy"
gap, **independent of the reward change** (obs bit-identical via `np.array_equal`, reward
`== -dist`).

**Fix (non-CORE, `hymeko_rl/ppo.py`):** `PPOConfig` gained `seed: int = 0`, and `train_ppo`'s
initial reset is now `env.reset(seed=cfg.seed)` — the subsequent in-loop resets advance the
same generator deterministically. **Verified reproducible:** two seeded runs give
`np.array_equal` histories; the run is now deterministic and PPO improves (`history[0]=-32.74
→ mean[-3:]=-28.57`), so the assertion passes **3/3** across repeats (was 1/2). The gym-
idiomatic split holds — the caller seeds torch, `train_ppo` seeds the env RNG. No sibling
PPO test regressed.

## Static analysis
- `ruff check` (5 changed/new files) → **clean**.
- `mypy` on `reward.py` / `_profile.py` → only the transitive `mujoco` `import-untyped`
  baseline; none on my annotations.
- **§6.5 anti-patterns:** none. Reward terms unified by Strategy + registry (not per-term
  functions — #1/#9); the obs and reward readers now share `read_bundle` instead of near-
  copying (#3); weights are declarative (no string-typed config — #7).

## Performance
Not a perf change. `evaluate` is a tiny weighted sum (1 term by default); equivalence
verified bit-exact. `read_reward_terms` runs once at construction.

## Dependencies
None.

## Open issues / follow-ups
1. ~~**PPO determinism**~~ — **done** (seeded `train_ppo`'s env reset; runs reproducible, test
   3/3). Two prior nondeterminism sources remain idiomatically caller-owned: torch (the caller
   `manual_seed`s) and any non-deterministic torch CPU op (none observed here).
2. **`AgentSpec.from_hymeko`** — can now compose `ObservationSpec.from_hymeko` (obs_dim) +
   `RewardSpec.from_hymeko` (reward) + the kinematic vertex count into a full MDP spec.
3. **Per-channel/term vertex bindings** — the profiles declare effector/target vertices
   (`(+ flange, - target_frame)`); the env still uses its procedural ee/target. Honouring the
   declared bindings is the remaining faithfulness step (shared with the obs gap).
4. **Structured profile read** — `read_bundle` is a narrow text parse; replace with the
   engine snapshot once B-003 lands.

## Provenance
- Git: `hymeko_rl/` untracked; `data/robotics/{meta_reward,arm_reach_task}.hymeko` are new
  tracked files. CLI built from `7d16ad0` + the non-CORE emit fixes.
- Platform: Windows 11, MuJoCo 3.9.0, Python 3.12.
- Seeds: equivalence/reward tests seed 4; reach smoke seed 2.
