# Multi-seed off-policy architecture-eval harness (+ artifact timestamps)

**Date:** 2026-06-22 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-22-offpolicy-multiseed-eval/` (tex/pdf/tikz/mmd, compiles)

## Summary
Built the env-agnostic harness the Galambos single-seed report (`2026-06-22-galambos-structure-vs-capacity.md`,
follow-ups #1/#2) named: an **off-policy** learner (SAC/DDPG/TD3) on a **real-topology** task across `>=5`
seeds, **HSiKAN vs a params-matched MLP**, scored by **curve-max** (the roadmap's fix for the noisy
final-snapshot metric) with median/IQR/worst. Plus the user's request: **timestamps stamped on generated GIFs
and result tables**.

This is the *harness + verification + production-scale smoke*. The 10-cell multi-seed run is **not yet
launched** — it writes checkpoints/policies, so it is gated on go-ahead (CLAUDE.md §11).

## What changed (files)
| file | change | LOC ± |
|---|---|---|
| `hymeko_rl/offpolicy_eval.py` | NEW — env+algo Strategy registries, `compare_offpolicy`, curve-max agg, CLI | +205 |
| `hymeko_rl/ddpg.py` | additive `eval_fn` on `train_offpolicy`; `_backbone` now forwards `hidden` to the MLP | +18 −5 |
| `hymeko_rl/sac.py` | additive `eval_fn` on `train_sac` | +6 −3 |
| `hymeko_rl/evaluate.py` | `now_stamp()` + `_stamp_frames()`; `stamp` on `_write_gif`/`render_episode_gif`/`compare_gif` | +44 −4 |
| `hymeko_rl/render_reach.py` | `encode(..., stamp=)` stamps frames (gif+mp4) | +8 −3 |
| `hymeko_rl/reach_arch_compare.py` | `timestamp` field on the report dict | +3 −2 |
| `hymeko_rl/tests/test_offpolicy_eval.py` | NEW — unit + regression + integration (8 tests) | +110 |
| `hymeko_rl/tests/test_evaluate_stamp.py` | NEW — stamping units (4 tests) | +50 |
| `hymeko_rl/tests/test_render_reach.py` | roundtrip test opts out of the stamp (`stamp=""`) | +3 −2 |

## CORE.YAML items touched
**None.** `hymeko_rl/` is non-core; no pinned-dependency change.

## Design notes
- **One `eval_fn` seam, not a new trainer.** `train_sac`/`train_offpolicy` were already duck-env-agnostic; the
  only cart-pole coupling was the `eval_balance` curve metric. An additive `eval_fn=None` param (default →
  unchanged `eval_balance`) lets a real-topology task inject `greedy_return_eval` (mean episode return via the
  reused `evaluate.evaluate` + the actor's `action_mean`). No copied loop (§6.5 #1/#3).
- **`_backbone` MLP-width bug fixed.** The off-policy `_backbone` called `_BACKBONES["mlp"](flat_dim)` and
  **dropped `hidden`**, pinning the MLP baseline at width 64 — so it could not be params-matched. Now it
  forwards `hidden`. Default behaviour unchanged (the prior effective width *was* 64). This is what makes the
  standing rule ("always run the params-matched control") actually achievable off-policy. Regression test added.
- **Strategy registries**, not a Cartesian function family: `TASKS` (name→env factory) and `_ALGOS`
  (name→(build, train, config)). One build call site, one train call site for all three algos.
- **Timestamps.** `now_stamp()` is the single source ("YYYY-MM-DD HH:MM:SS"); GIFs get a bottom-right baked
  label by default (`stamp=""` opts out — used by the encoder-fidelity roundtrip test); result-table report
  dicts get a `timestamp` field. Stamping is inherently non-deterministic (the whole point); these are
  visualization/result artifacts, not reproducibility fixtures.

## Test results
- New + touched, isolated: `test_offpolicy_eval.py` (9), `test_evaluate_stamp.py` (4), `test_render_reach.py`,
  `test_render_planar.py`, `test_sac.py`, `test_ddpg.py` — **all pass**.
- Full suite: `pytest -p no:randomly hymeko_rl/tests/` → **226 passed, 1 skipped, 2 failed in 200 s**.
- **The 2 failures are pre-existing and unrelated to this change:** `test_strategy_spec.py` expects
  `galambos_strategy.hymeko` at `n_iters=150 / curriculum_iters=60`, but that file is committed (292388b) at
  `300 / 200` (the prior session's retune; the Galambos report cites "300 iters"). Neither the test nor the
  `.hymeko` is in this diff. Fix is a one-line expectation bump (150→300, 60→200) **or** revert the strategy
  retune — your call which is the spec; I did not touch someone else's test silently.
- **Static gates:** `ruff check` clean; `mypy --strict --ignore-missing-imports` clean on all 6 touched source
  files. (The pre-existing `render_reach.py:142` imageio `type: ignore[arg-type]` is outside this diff and is
  needed when the optional `demo` group is installed.)

## Performance — the production-scale smoke (§3)
1-seed SAC on the real Galambos env, both backbones, `total_steps=2000`:
- **Wall 118.1 s for 2 cells** (~59 s/cell @ 2000 steps). Single-thread torch.
- No crash; both backbones train end-to-end on the 6-vertex hypergraph.
- The curve-max-vs-final spread confirms the metric choice: `sac/mlp` curve-max **−193** but final **−664**
  (a late dip the noisy final-snapshot would have mis-scored).
- **Params at equal `hidden=64` are NOT matched:** HSiKAN **14,728** vs MLP **7,816** (~1.9×). Probe →
  **`mlp@hidden=96 = 14,792`** matches HSiKAN@64 within 0.4%. The full run must use `--hidden 64
  --mlp-hidden 96`.

**Full-run projection.** ~30k steps/cell × 10 cells (2 backbones × 5 seeds) ≈ **~15 min/cell → ~2.5 h** for
SAC alone (linear in update-steps from the smoke). This exceeds the >2× reconciliation trigger vs a quick
toy run, so it is correctly treated as a queued experiment, not a unit test (§3, §11).

## Halt — awaiting go-ahead before the multi-seed run (§11)
The decisive run writes checkpoints + `.hymeko` policies + the report jsonl (persistent state). Proposed command:

```
python -m hymeko_rl.offpolicy_eval --task galambos --algo sac \
    --mode full --hidden 64 --mlp-hidden 96 \
    --out reports/2026-06-22-offpolicy-galambos-sac.jsonl
```

Open decisions for you: (a) **SAC only** (~2.5 h, strongest/cheapest learner) or **add TD3/DDPG** (~3× wall);
(b) keep `total_steps=30000` or shorten; (c) Galambos only, or also queue the quadruped (14-vtx) once this
lands — the harness is task-agnostic so it's a one-line `TASKS` entry.

## Addendum (2026-06-22, later) — scope expanded to a 4-task campaign + launched
On user direction the eval became a **4-task campaign**: `cartpole` (inverted pendulum, 2-vtx control floor),
`galambos` (6-vtx), `arm6dof` (anthropomorphic reaching, 6-DOF), `quadruped` (14-vtx). Harness additions
(all tested, ruff+mypy clean):
- All four wired into the `TASKS` Strategy registry; `--task` now accepts multiple (one resumable command).
- **Per-cell resume journal** (`compare_offpolicy(..., journal=)`): finished cells skipped on rerun (§4).
- **Auto params-match** (`match_mlp_hidden`): MLP width tuned per task to match HSiKAN (`--mlp-hidden auto`).
  Matches: cartpole hsikan@64=13186≈mlp@110; galambos 14728≈mlp@96; arm6dof 14988≈mlp@92; quadruped 14096≈mlp@102.
- **Table generator** `hymeko_rl/offpolicy_tables.py` (Markdown + `booktabs` LaTeX from the journal; verdict =
  HSiKAN−MLP median gap vs cross-seed IQR). Shared aggregator `aggregate_records` (no dup).
- Stale `test_strategy_spec.py` **fixed** (bumped to 300/200 to match the committed strategy).

Per-task smokes (SAC, 1 seed, 2 cells @2000 steps): cartpole 77.7 s, galambos 118 s, arm6dof 126.5 s,
quadruped 150.6 s — all run end-to-end. **Launched (user choice: SAC only, 5 seeds, ~9 h):**
```
python -m hymeko_rl.offpolicy_eval --task cartpole galambos arm6dof quadruped --algo sac \
    --mode full --seeds 5 --journal reports/2026-06-22-offpolicy-campaign.jsonl \
    --out reports/2026-06-22-offpolicy-campaign.json
```
**In flight (background task `btc6x5tiv`):** log `reports/2026-06-22-offpolicy-campaign.log` (growing),
journal `reports/2026-06-22-offpolicy-campaign.jsonl` (one line/cell). cartpole capped at 15k steps (`_TASK_STEPS`).

Companion handout for Kato: `reports/2026-06-22-sac-td3-hsikan-overview.{tex,pdf}` (SAC/TD3 + HSiKAN application).

## Follow-ups
- When `btc6x5tiv` finishes: `python -m hymeko_rl.offpolicy_tables --journal reports/2026-06-22-offpolicy-campaign.jsonl
  --md ... --tex ...` → tables; fold the verdict into the Galambos three-task picture + the RL roadmap memory;
  fill the extended-abstract scaffold.
- TD3 arm of the campaign (deferred): rerun the same command with `--algo td3` (journal keeps SAC cells).
