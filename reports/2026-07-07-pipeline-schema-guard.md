# Live tensor-contract guard — `verify_schema` wired into the rollout → replay → critic → eval path

**Date:** 2026-07-07 · Git SHA `4320202` (working tree dirty). Non-core (`hymeko_rl`). RL stays frozen — this is a
correctness guard on the RL *plumbing*, not an RL run.

## Summary

Wired `TensorContractMonitor`'s schema/field-order check into the **live** off-policy pipeline, as directed. A
new `PipelineSchemaLedger` (in the `task_monitor` package — the runtime sibling of monitor #8) carries the
canonical transition schema derived from the env and records the schema actually seen at each of the five
pipeline stages, aborting the instant any field ordering or dimension drifts:

    rollout → replay serialization → replay loading → critic training → evaluation

Each `record()` validates against the canonical schema **immediately** and raises `PipelineSchemaError` at the
first offending stage — so a mismatch between how a transition is written and how the critic reads it aborts
rather than silently training a wrong-but-running critic (the class of provenance/anchor-path bug the earlier RL
smoke hit). After all five stages are seen once, `verify_or_abort` confirms completeness and seals the ledger to a
no-op. `verify_schema` defaults **on** for every off-policy run.

## The five stage hashes (live, real galambos CTDE env)

From the wiring smoke (`obs=(6,8)` → 48 flat, `priv_dim=5`, `action=4`):

| stage | fields | field-order hash |
|---|---|---|
| rollout | obs, action, reward, next_obs, done, priv, next_priv | `21a9ab6770a378dd` |
| replay_serialize | obs, action, reward, next_obs, done, priv, next_priv | `21a9ab6770a378dd` |
| replay_load | obs, action, reward, next_obs, done, priv, next_priv | `21a9ab6770a378dd` |
| critic_train | obs, action, priv | `e69f698c10024439` |
| eval | obs, action | `1c1b5d2e284af370` |

The three **full-transition** stages hash **identically** → the transition schema is consistent write→store→load.
`critic_train` / `eval` are the expected consumer subviews (obs precedes action precedes priv, in canonical
order). **n_envs=1 and n_envs=2 produce identical hashes** → the single-env and vectorized rollout paths are
schema-consistent with each other.

## Design (hierarchy-consistent, single-source)

- `TransitionSchema.from_env(env, action_dim, priv_enabled)` — the canonical ordered `(name, dim)` layout:
  `obs, action, reward, next_obs, done [, priv, next_priv]`. `obs` dim = `prod(observation_space.shape)`
  (robust to the 2D node-feature obs); `priv` dim = `env.privileged_dim`. Single-source from the env, like the
  rest of the monitor.
- `PipelineSchemaLedger.record(stage, fields)` — validate-and-abort per stage. **Full stages** (rollout /
  serialize / load) must equal the canonical schema exactly; **consumer stages** (critic / eval) must read a
  subset of the fields in canonical order with matching dims. Re-recording a stage with a different schema
  (mid-run drift) also aborts.
- `ReplayBuffer.column_schema()` — the buffer now reports its own serialization column layout (the serialization
  schema is owned by the buffer, not hand-duplicated in the trainer).
- `OffPolicyConfig.verify_schema: bool = True` — the guard flag; off only for a bespoke env whose transition
  layout intentionally departs from canonical.

Where each stage is recorded in `train_offpolicy`: serialization + eval statically at buffer creation (they are
static input contracts); rollout at both `buf.add` (single-env) and `buf.add_batch` (vectorized) sites;
replay-load + critic-read inside `_update_once` right after the sample; seal after the first update.

## Files touched (my additions)

| file | change |
|---|---|
| `hymeko_rl/eval/task_monitor/pipeline.py` | **NEW (135 LOC)** — `TransitionSchema`, `TransitionField`, `PipelineSchemaLedger`, `PipelineSchemaError`, `flat_dim`, `STAGES` |
| `hymeko_rl/eval/task_monitor/__init__.py` | export the pipeline symbols + docstring |
| `hymeko_rl/train/ddpg.py` | `verify_schema` config field; import; ledger construction; record at rollout (×2), replay-load, critic-train; seal-on-complete (~55 LOC added) |
| `hymeko_rl/train/replay.py` | `ReplayBuffer.column_schema()` (~10 LOC) |
| `hymeko_rl/tests/test_task_monitor.py` | +8 pipeline-guard tests (flat_dim, from_env, pass+seal, 5 abort paths) |
| `scratchpad/schema_guard_smoke.py` | live wiring smoke (not committed) |

`ddpg.py` and `replay.py` carried **pre-existing uncommitted changes** at session start (both were `M`); the
line counts above are my ledger additions, not the full working-tree diff.

**CORE.YAML items touched:** none (`hymeko_rl` is not core). No new dependencies.

## Tests

- **Unit:** `pytest hymeko_rl/tests/test_task_monitor.py` → **22 passed** (14 monitor + 8 pipeline-guard). The 8
  cover: `flat_dim` batched/unbatched, `TransitionSchema.from_env` (priv + non-priv, 2D-obs flatten), pass+seal,
  and 5 abort paths — priv-dim mismatch, full-stage field-order swap, consumer field-order swap, incomplete
  coverage, mid-run re-record conflict.
- **Live wiring smoke:** `schema_guard_smoke.py` — `train_offpolicy` on the real galambos CTDE env, both
  `n_envs=1` and `n_envs=2`, 300 steps each. Both print `[schema] tensor-contract PASS — 5/5 stages` and complete
  with no `PipelineSchemaError`. Plumbing only — no RL/performance claim.
- **Regression (guard defaulted on):** `pytest hymeko_rl/tests/test_ddpg.py test_offpolicy_framework.py`
  → **31 passed** (269 s). Cart-pole (non-priv path) and the priv path both unaffected — the guard's canonical
  schema matches the real tensors, so no healthy run false-aborts.
- **Static:** `ruff` clean on all touched files.

The abort path is proven by unit test (five distinct drifts raise `PipelineSchemaError`); the healthy live path is
proven by the smoke + regression suite. A live abort can't be injected without a real schema bug, since
`train_offpolicy` builds the ledger from the same env it runs — which is the point.

## Performance

Negligible: each stage records once (a tuple of small ints + one hash), then the ledger seals and every guarded
site is a `sealed` boolean check. No effect on steps/s (the smoke ran at the env's normal rate). No measurable RSS
change.

## Required report fields (from here on)

Per the directive, every future experiment report must now include: **reward, ft_dom, monitor_pass, monitor_score,
violation_reason, reward-vs-monitor consistency, critic-vs-monitor consistency (if a critic exists), and tensor
contract pass/fail.** The tensor-contract line is now produced live by this guard (the `[schema] … PASS` log, or a
`PipelineSchemaError` abort); the monitor fields come from `TaskMonitor.evaluate_policy` /
`RewardConsistencyMonitor` (report `2026-07-07-v2-task-monitor.md`).

## Status

Monitor V0 frozen. The live tensor-contract guard is wired and defaulted on. **The monitor is still an external
verifier — NOT in the reward.** Next RL (research, not now) remains gated on the monitor acceptance criteria; any
run will now also carry the live schema guard.
