# MetaWorld Stage B — bounded training-smoke setup (GATED, no training run)

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration`
**Status:** setup/dry-run only. A gated, bounded training-smoke harness for the reward ablation is in place and the
dry-run validates end-to-end. **No training was launched.**

---

## Summary

Reward-computation-level ablation is robust (`mw_in_place` dominant SUPPORTED 5/5, `mw_grasp` inert
NOT_SUPPORTED 5/5, cross-view 25/25). Stage B asks the *policy-learning* question: **does a policy trained under
`mw_in_place_off` behave differently from one trained under the original reward?** This task builds the harness to
answer it later — it does **not** train.

`hymeko_rl/experiments/exp_metaworld_reward_stageb.py` provides: reward profiles (via `ablate_reward_spec`), a
reward-override MetaWorld env (`HymekoRewardMetaWorld` — the step reward becomes the HyMeKo `Σ weight·component`), a
`dry_run` that validates all plumbing and exits before the optimizer, a `--launch-training` safety gate, a
MetaWorld-appropriate reward certification (does the reward rank success above failure?), and a bounded REINFORCE
smoke behind the gate. It reuses the Stage-A machinery and the MetaWorld ENVS — no copied reward/eval code, no new
trainer duplication (the repo trainers require 2-D hypergraph obs; MetaWorld is flat, so a small flat-obs actor is
new, not a duplicate).

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/experiments/exp_metaworld_reward_stageb.py` | **new** — `StageBConfig`, `HymekoRewardMetaWorld`, `dry_run`, `launch` (gated), `_scripted_delivers` (certification), `_GaussianMLP` + `_train_reward_smoke` (bounded REINFORCE) |
| `hymeko_rl/tests/test_metaworld_stageb.py` | **new** — 7 tests (gate, profiles, paths, eval cmd, REINFORCE plumbing, dry-run) |
| `reports/figures/2026_07_09_metaworld_stageb_dry_run/stage_b_dry_run.json` | dry-run validation artifact |

CORE.YAML / `pyproject.toml` / `metaworld_reward.hymeko` / FANUC `PickPlaceEnv` / coin-collab untouched. No dependency added.

## Reward profiles compared

| profile | dropped term | intent |
|---|---|---|
| `original` | — | the full declared HyMeKo reward (control) |
| `mw_in_place_off` | `mw_in_place` | remove the **dominant** driver (the primary Stage-B question) |
| `mw_grasp_off` (optional) | `mw_grasp` | remove the **inert** term (negative-control training arm) |

## Budget (bounded smoke — proves plumbing, not skill)

`total_env_steps=3000`, `max_steps=180`, `eval_episodes=24`, **1 seed**, `wall_time_cap_s=600` (hard 10-min cap),
REINFORCE `lr=3e-4`, `hidden=64`, live log every 5 episodes.

## Exact dry-run command (validated, run above — NO training)

```
python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --dry-run \
  --profiles original mw_in_place_off mw_grasp_off \
  --out reports/figures/2026_07_09_metaworld_stageb_dry_run
```

Dry-run output (this run): every profile's reward is finite; certification is **discriminating** (both success and
failure episodes appear under action noise); paths do not collide with the Stage-A `cip_reward_ablation` family;
`trained=False`; no checkpoint written.

| profile | delivers | discriminating | success/rollouts | mean HyMeKo return (success / failure) |
|---|---|---|---|---|
| `original` | True | True | 3/8 | **970.2** / −248.0 |
| `mw_in_place_off` | True | True | 4/8 | **18.3** / −312.5 |
| `mw_grasp_off` | True | True | 3/8 | **992.0** / −168.8 |

**Honest Stage-B-relevant reading:** `mw_in_place_off` *technically* still delivers (success ranks above failure),
but its success-return margin **collapses ~50×** (18.3 vs 970.2). The prediction this sets up: a policy trained
under `mw_in_place_off` has a **much weaker learning signal toward delivery** than under the original reward —
which is exactly the behavioural difference Stage B will measure.

## Exact launch command — ⛔ NOT RUN

```
# NOT RUN — Stage B training. Requires the explicit safety flag; run only on your go-ahead, watched live.
python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --launch-training \
  --profiles original mw_in_place_off \
  --out reports/figures/<stamp>_metaworld_stageb_train
# mw_in_place_off's reward still certifies here; if a future check flags it uncertified, add --allow-uncertified
# (training on a possibly-non-delivering reward IS the mw_in_place_off hypothesis — a deliberate, waived choice).
```

## Safety-gate behavior

1. **`--launch-training` is mandatory.** `launch(cfg, launch_training=False)` raises `StageBGateError` and writes
   nothing (test-verified). The CLI defaults to `--dry-run` when no mode is given.
2. **Reward certification is a second gate.** Before training a profile, `_scripted_delivers` checks the reward
   ranks success above failure (the MetaWorld analogue of `reward_oracle.certify`'s `delivers` — the galambos
   abstract-MDP oracle does not model this task, so the pattern is extended, not the oracle reused). A
   non-delivering reward requires `allow_uncertified=True` / `--allow-uncertified` — an explicit, logged waiver, so
   training on a broken reward is a deliberate act, not an accident (CLAUDE.md §3).
3. **Dry-run never reaches the optimizer.** It steps the env only for env/reward validation.

## Expected metrics (collected LATER, per profile, when training runs)

monitor pass rate · progress_score · near_fraction · obj_to_target_delta · total_reward under its own reward ·
total_reward recomputed under the original reward · reward-monitor disagreement · collapse/no-collapse flags ·
emitted CIP DAG · LiNGAM-SH weighted mechanism fit · cross-view verification. The post-training evaluation reuses
the Stage-A `run_reward_ablation_comparison` pipeline; the per-profile eval command is generated by
`StageBConfig.eval_command`.

## Tests

7 `test_metaworld_stageb.py` tests: import has no training side-effects; missing `--launch-training` blocks
training (no checkpoint); both profiles loadable + distinct (`mw_in_place` zeroed in `off`, kept in `original`);
output paths do not overwrite Stage A; post-eval command generated; **REINFORCE plumbing correct on synthetic
data** (returns-to-go + a policy-gradient step updates the actor — this test caught a real bug: a reparameterized
`rsample()` cancels the score-function gradient, so the actor uses a detached `sample()`); dry-run validates
without training. 17/17 reward-ablation + Stage-B green; 90/90 causal/LiNGAM-SH green. ruff / radon (no block ≥ C) /
mypy `--strict` (my file) clean. **Pre-existing, unrelated:** `test_quadruped_from_hymeko.py` 6 failures on clean
HEAD (quadruped_env) — untouched.

## Runtime-verified vs unverified (honest)

- **Verified now:** reward profiles, env construction, the reward-override signal, certification (discriminating),
  logging paths, eval-command generation, the REINFORCE update on synthetic data, the safety gate.
- **Unverified until first `--launch-training`:** the REINFORCE optimizer *loop on the env* (never run per the
  no-training constraint). Per CLAUDE.md §3, the first `--launch-training` run **is** the production-scale smoke: 1
  seed, short budget, live-logged — watch the loss/return the whole way.

## Next decision required

**Do you authorise the bounded training smoke** (`--launch-training`, 1 seed, ≤10 min, live-logged)? The dry-run
already predicts a ~50× weaker delivery signal under `mw_in_place_off` — the smoke would confirm the plumbing runs
end-to-end and give the first behavioural read; a multi-seed Stage-B run would follow only if the smoke is clean.

## Constraints honored

No training launched · no production ablation · FANUC v2 / coin-collab v2b / `CORE.YAML` / `metaworld_reward.hymeko`
/ `pyproject.toml` untouched · no existing report/artifact overwritten · quadruped failures left untouched
(mentioned only as pre-existing) · no policy-learning claim made.
