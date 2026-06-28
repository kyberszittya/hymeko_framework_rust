# DDPG — the first off-policy actor-critic on the HyMeKo cart-pole (P2)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** [docs/plans/2026-06-21-rl-algorithm-architecture/](../docs/plans/2026-06-21-rl-algorithm-architecture/) (P2) ·
**Survey:** [reports/2026-06-21-offpolicy-rl-survey.pdf](2026-06-21-offpolicy-rl-survey.pdf)

## Summary
DDPG is implemented and works on the cart-pole — the first off-policy method beside the on-policy PPO
baseline. A deterministic actor `μ(s)` is improved *through* a Q-critic `Q(s,a)` (the deterministic policy
gradient), the critic regresses to a Polyak-target Bellman backup over a replay buffer, and exploration is
additive Gaussian action noise. The state encoder is the *same* swappable backbone (`mlp`/`hsikan`/
`signedkan`) the PPO policy uses — the architecture is orthogonal to the algorithm.

**Smoke (mlp, seed 0, 20k env steps):** untrained **27.3** → trained **199.4** upright-steps/200; learning
curve `[121, 198, 200, 200, 199]` at 4k/8k/12k/16k/20k steps — **solved by ~8k steps**.

## The off-policy win, measured honestly
| | env steps to solve | wall | n_params |
|---|---|---|---|
| **DDPG** (mlp, UTD 1, 1 thread) | **~8 000** | 215 s / 20k steps | 4 545 |
| PPO (mlp, vec N=16) | ~2 000 000 (120 it × 16 × 1024) | 147 s / 120 it | 9 091 |

DDPG is **~250× more sample-efficient** (the central point of the survey) — it reaches 200 in thousands of
real steps where PPO needs millions. PPO still wins on **wall-clock** here, because it is vectorised while DDPG
runs one gradient step per env step single-threaded; DDPG's wall would drop with a vectorised collector and/or
higher UTD (REDQ-style). So: DDPG wins samples, PPO wins wall — exactly the on/off-policy trade.

## What was built
- **`hymeko_rl/replay.py`** — `ReplayBuffer`: a fixed-capacity struct-of-arrays ring (§5), O(1) `add`, uniform
  `sample`; peak RSS bounded regardless of run length.
- **`hymeko_rl/ddpg.py`** — `DeterministicActor` (`μ = scale·tanh(head(backbone))`), `QCritic`
  (`Q = head(concat(backbone(s), a))`), `build_ddpg` (independent backbones of the requested kind),
  `DDPGConfig`, `train_ddpg` (Polyak targets, Gaussian exploration, the DPG/Bellman updates), `run_ddpg`, CLI
  (`python -m hymeko_rl.ddpg --policy {mlp,hsikan,signedkan}`).
- **`hymeko_rl/train_inverted_pendulum.py`** — `eval_balance` retyped to a `GreedyPolicy` Protocol so it scores
  *either* a PPO `ActorCritic` or a DDPG `DeterministicActor` (one eval for both algorithms).
- The **truncation correctness** point is preserved: time-limit truncation is stored as **non-terminal**
  (`done = terminated`, not `terminated or truncated`), so the Bellman backup bootstraps past the time limit.

**Reuse (§6.1):** the `RolloutEnv` step API, the `_BACKBONES` registry (so `hsikan`/`signedkan` slot into the
Q-critic too), and `eval_balance`. No duplication of the env or the backbones.

## Test results
- `test_ddpg.py` — **5 tests**: replay ring caps size + overwrites oldest; bad-batch/param rejection; actor
  bounded to ±scale + critic scalar (mlp **and** hsikan); `train_ddpg` runs end-to-end on a tiny budget with a
  finite curve. `test_ppo.py` — **6 pass** (PPO regression: the `eval_balance` retype is behaviour-preserving).
  Total 11. `ruff` clean; `mypy --strict` clean (one documented `# type: ignore` on `backward`).

## Honesty / scope
- **Single seed**, easy task. DDPG did *not* show its notorious fragility here (cart-pole is forgiving) — but
  that fragility is real on harder tasks; **TD3** (twin critics + delayed updates + target smoothing) is the
  reliable successor and is **additive over this** (P3, next).
- The **architecture** comparison (hsikan/signedkan vs mlp) is *not* run here — cart-pole cannot test structure
  (established); `build_ddpg` supports all kinds and the hsikan path is unit-tested, but the verdict belongs on
  a real-topology task (6-DOF arm / Galambos) with the params-matched control.
- Exploration is a minimal hardcoded Gaussian; the declarative exploration vocab (P1) would wire it as
  `gaussian @ action_select` — deferred, not blocking.

## Provenance
Git SHA `292388b` (dirty). torch 2.12.0+cu132 (CPU, 1 thread). DDPG smoke: mlp, seed 0, 20k steps, 215 s.

## Open issues / follow-ups
1. **TD3** (P3) — twin critics, delayed policy, target smoothing; the robust deterministic baseline.
2. **SAC** — the strong stochastic baseline; the fair opponent for the HSiKAN backbone.
3. Vectorised/higher-UTD DDPG collector to close the wall-clock gap to PPO.
4. **Safe RL** (queued, per the survey §8): a `meta_constraint` cost vocabulary + a Lagrangian on the
   off-policy update — the declarative "constraints as data" continuation.
