# Edge-weighted reward hyperedges — RL scenario descriptions reworked

**Date:** 2026-06-24 · **Branch:** soma-vision · **Plan:** `docs/plans/2026-06-24-kato-collab-dual-discriminator/` (§7)

## Summary
Reworked the RL reward scenario descriptions so the **weight lives on the hyperedge arc** (the bundle's
incidence) rather than as a `weight` attribute on the term node — Kato's data-model directive. A reward term is
now a typed signal; the `reward_spec` bundle declares each term's weight as a numeric arc annotation
`(+ approach 4.0, …)`. The bridge reader prefers the arc weight and falls back to the legacy body `weight` (so
un-migrated files keep working and the migration is parity-exact).

No grammar change: the arc-weight syntax already parses (a signed arc ref's optional annotation `OptValue`
admits `Value::Num`; the weight lands in `RefAtom.anno.value`). **No CORE edit, no §1 escalation.**

## Files touched
- `hymeko_rl/env/_profile.py` (+8/−5) — `read_bundle` now returns `(name, kind, body, arc_weight)` quadruples;
  arc regex captures the optional trailing weight.
- `hymeko_rl/env/reward.py` (+9/−4) — `read_reward_terms` prefers the arc weight, falls back to body `weight`, then `1.0`.
- `hymeko_rl/env/observation.py`, `env/termination.py`, `env/env_spec.py`, `strategy_spec.py` (+1/−1 each) —
  unpack the 4-tuple (weight ignored; these bundles are unweighted).
- `data/robotics/galambos_task.hymeko` — 11 term weights moved to the bundle arcs.
- `data/robotics/pick_place_task.hymeko` — 7 term weights moved to the bundle arcs.
- `data/robotics/arm_reach_task.hymeko` — imported-term arc weight `(+ arr.dist 1.0)`.
- `hymeko_rl/tests/test_reward.py` (+34) — 3 new tests (arc weights; arc>body>default precedence; reworked-profile
  regression with exact pre-rework values).

## CORE.YAML items touched
None. Confirmed against the frozen `parser` grammar that arc weights already parse.

## Test results
- `test_reward.py` (incl. 3 new) + the 5 `read_bundle` callers (`test_pick_place_task`, `test_observation`,
  `test_strategy_spec`, `test_scenario`, `test_galambos_task_graph`): **28 passed** (12.0 s).
- `test_task_hyperedges` + `test_planar_grasp_env` (Galambos env build + task graph from the reworked file):
  **28 passed** (9.6 s).
- `test_reward` + `test_arm_reach_from_hymeko` (reach task arc weight): **15 passed** (7.6 s).
- **Parity:** `test_pick_place_task` (env reward == procedural) passes unchanged — arc weights yield identical
  values. The new `test_reworked_task_profiles_have_arc_weights` pins the exact pre-rework weights as a regression.

## Static analysis
- `ruff check`: clean on all 7 changed `.py`.
- `mypy --strict --ignore-missing-imports`: clean on the 6 changed modules. (Bare `--strict` surfaces only the
  pre-existing `mujoco` missing-stub `import-untyped` noise, not introduced here.)

## §6.5 anti-patterns
None introduced. The arc-weight parse is centralized in the one shared `read_bundle` (no duplication); the reader
is backward-compatible (no string-typed config, no new branch-per-variant).

## Open / follow-up
- `meta_reward.hymeko` still defines a default `weight 0.0` on the term vocabulary; now redundant under the
  arc-weight model. Optional vocab cleanup (offered, not done — it touches the shared vocabulary).
- The other half of "weights on hyperedges" — the HSiKAN signed incidence `A±` carrying real arc weights (not
  binary) — is part of the dual-discriminator backbone work, gated on Kato's 3 design decisions.
