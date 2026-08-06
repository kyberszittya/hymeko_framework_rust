# Vectorized PPO rollout — 3.1× faster balance training (and the policy-efficiency numbers)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** [docs/plans/2026-06-21-vectorized-ppo-rollout/](../docs/plans/2026-06-21-vectorized-ppo-rollout/)

## Summary
"Get the numbers and make even more performance." Both done, **profile-gated** per §3 (no optimization
without a measurement of the hot spot).

**The numbers (measured, not asserted):**
- *Phase decomposition* of one single-env iter: `ac.act` (batch-1 policy forward) = **2.84 s / 1024**,
  `env.step` = 0.41 s — the forward is 87% of the rollout, the rollout dominates the iter. The cost is
  per-step **torch dispatch overhead**, not FLOPs (the 2-vertex graph is trivial).
- *Forward vs batch size* (per call / per sample): B=1 2.19 ms / 2.19 ms; B=8 2.11 ms / **0.26 ms**;
  B=32 2.79 ms / **0.087 ms**. Per-call cost is ~flat to B=8 → **dispatch-bound** confirmed.
- *HSiKAN-vs-MLP ablation* (120 iters, seed 0): HSiKAN 144.0 upright-steps @ 26 243 params / 452.7 s;
  MLP 115.8 @ 9 091 / 180.6 s. HSiKAN balances better on the 2-vertex graph but at ~2.9× params / ~2.5×
  wall — recorded honestly.

**The optimization:** vectorized rollout — run N envs lock-step and batch the policy forward, so a rollout
of `n_steps` transitions takes `n_steps/N` ticks. Plus two free wins: single-thread torch at the binary
entry (~10%, the batch-1 tiny-tensor case) and emitting the cart-pole MJCF **once** shared across workers
(avoids N `hymeko` CLI subprocess calls at construction).

## Results
| | rollout | update | iter | wall (120 it) | upright-steps |
|---|---|---|---|---|---|
| single-env | 2.47 s | 0.61 s | **3.08 s** | 452.7 s | 144.0 |
| vec N=8 | 0.55 s | 0.59 s | **1.14 s** (2.7×) | — | — |
| **vec N=16** | **0.36 s** | 0.57 s | **0.93 s** (3.3×) | **147.4 s (3.1×)** | **161.4** |

The rollout drops **6.8×** (2.47→0.36 s) at N=16; `_update` is constant (same total transitions). **Learning
is preserved** — vec N=16 reaches 161.4 upright-steps (≥ the single-env 144.0). The 3.1× full-run speedup is
confirmed by a fresh 120-iter run, not single-shot variance.

### Honest caveat
vec `final_return` (the per-iter mean episodic return) under-reports (37 vs the eval's 161) because with
N=16 the per-iter horizon is `1024/16 = 64` ticks while a balanced episode lasts ~161 steps, so few episodes
*complete* per iter. GAE bootstraps the truncated fragments correctly (the policy learns), and `upright_steps`
(full-episode eval) is the true metric. Documented, not hidden.

### Earlier mis-measurement (corrected)
A first end-to-end benchmark showed N=16 *slower* — a measurement artifact: it rebuilt N envs **per measured
round**, and each `InvertedPendulumEnv()` shells out to the `hymeko` CLI, so it timed 16–32 subprocess calls,
not the rollout. Timing the **real** `_collect`/`_collect_vec`/`_update` separately, and sharing one emitted
MJCF, exposed the true 3.1–3.3× win. (CLAUDE.md "analyse, don't declare" — the convenient first number was
wrong; the discriminating measurement collapsed it.)

## Files touched
| File | Δ |
|---|---|
| `hymeko_rl/ppo.py` | +75 (`_collect_vec` + `train_ppo` `n_envs`/`make_env` branch; single-env path untouched) |
| `hymeko_rl/env/inverted_pendulum_env.py` | +12 (`mjcf=` share param + `emit_cartpole_mjcf`) |
| `hymeko_rl/train_inverted_pendulum.py` | +12/−4 (`n_envs`, shared MJCF, `--envs`, threads-at-entry) |
| `hymeko_rl/tests/test_ppo_vec.py` | new (+115) |

**CORE.YAML items touched:** none. **New/removed deps:** none.

## Test results
- `test_ppo_vec.py` + `test_ppo.py` (reach regression) + `test_inverted_pendulum_env.py`: **26 passed, 1
  skipped** (the psutil RSS branch), 107 s. The reach regression passing proves the single-env path is
  byte-unchanged; `test_collect_vec_faster_than_single` asserts the vec rollout median < single-env median.
- Coverage: `_collect_vec` branches (shapes, truncation bookkeeping, per-env GAE), the `train_ppo`
  `n_envs`/`make_env` validation, and the vec-learning integration each have a test.
- Static: `ruff` clean; `mypy --strict` clean on the changed modules apart from the endemic `mujoco`
  import-untyped (as elsewhere in `hymeko_rl/env`).

## Performance / resources
Peak RSS rises only by N small `MjData` + an (N,2,2) obs batch — a few MB; the run stayed well under the
16 GB cap. Single-thread torch is set at the CLI entry only (§6.5 #11 "set once at main" exception); library
callers keep the default.

## §6.5 anti-patterns
None. `_collect_vec` is added beside `_collect` (the single-env path is directly tested by `test_ppo.py` and
must stay); the two share `_gae` (per-env) and `_update` (unchanged) — no duplicated rollout *math*. The
vec/single choice is a parametric `n_envs` switch, not a forked function family.

## Provenance
Git SHA `292388b` (dirty). Windows 11; Python 3.13 `.venv`; torch 2.12.0+cu132 (CPU, 1 thread), mujoco
3.9.0, gymnasium 1.3.0. Seeds: train 0 (per-env `seed+i`), eval 10 000/20 000. No GPU.

## Open issues / follow-ups
1. **MLP at parity under vectorization** — a multi-seed vec HSiKAN-vs-MLP run would now be cheap (~150 s each)
   and is the rigorous version of the single-seed ablation above.
2. **Per-iter return metric for vec** — optionally report a running upright-steps eval per K iters instead of
   the horizon-truncated `final_return`, so the training curve reads true.
3. **Apply the same vec path to reach** (`run_ppo`) — `ArmReachEnv` already satisfies `RolloutEnv`; only the
   per-env curriculum (`on_iteration`) needs threading into the worker envs first.
