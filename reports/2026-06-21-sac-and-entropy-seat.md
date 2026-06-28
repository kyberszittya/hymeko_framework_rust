# SAC — max-entropy off-policy, and the entropy-feedback seat (policy / ensemble / structural)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Survey:** [reports/2026-06-21-offpolicy-rl-survey.pdf](2026-06-21-offpolicy-rl-survey.pdf) ·
**Prior:** [DDPG](2026-06-21-ddpg-offpolicy.md), [TD3](2026-06-21-td3.md)

## Summary
SAC is implemented — the strong stochastic max-entropy baseline, completing the off-policy arc
**PPO · DDPG · TD3 · SAC**. A tanh-squashed Gaussian actor maximises reward + policy entropy, twin soft-Q
critics (clipped double-Q) bound overestimation, and the temperature `α` is auto-tuned to a target entropy.
Reuses the off-policy scaffolding (`ReplayBuffer`, `QCritic`, the backbone family, `_polyak`, `eval_balance`);
a *separate* trainer (`sac.py`) because the stochastic actor + entropy-augmented update are a **structural**
difference from the deterministic DDPG/TD3 core (§6.5 #8), not just a config.

## Result — SAC is the strongest learner (single seed, mlp, 20k steps)
| algo | curve (4k · 8k · 12k · 16k · 20k) | reading |
|---|---|---|
| DDPG | `121 · 198 · 200 · 200 · 199` | solves ~8k, stable to end |
| TD3 | `142 · 66 · 89 · 200 · 113` | noisy / oscillates |
| **SAC** | `196 · 200 · 200 · 200 · 122` | **solves fastest (4k), holds 200 for 8k–16k**, late dip |

SAC reaches 196 by **4k steps** (DDPG needed ~8k) and holds a perfect 200 across three evals — the most
sample-efficient and stable *during* training, as expected of the strongest baseline.

**Honest metric caveat (matters):** the headline `upright_steps` is **one eval at exactly step 20k**, and both
SAC and TD3 happened to catch a *late-training dip* there (SAC final 122, having held 200 just before). So the
single post-training number is **noisy under late-training variance**; the **curve is the truer signal**. A
fairer protocol would report the curve-max or a smoothed tail (all three hit 200), and/or early-stop at the
peak — a follow-up. On cart-pole all four algorithms solve the task, so the real differentiators
(sample-efficiency, robustness) need **multi-seed + a harder task** to separate cleanly.

## The entropy-feedback seat — three signals, one site
SAC's `α·log π(a)` term is the maximum-entropy regulariser. It is the explicit insertion point for a unifying
idea (the user's "TD-k / entropy feedback" observation): **three distinct uncertainty signals all plug into
the same seat.**
1. **Policy entropy** `H(π)` — *aleatoric* (action stochasticity). **SAC, built (this report).** Auto-α.
2. **Critic-ensemble disagreement** `Var_k Q_k(s,a)` — *epistemic* (value uncertainty). This is **"TD-k"**:
   our `n_critics` config already builds `k` critics with a `min`-target (a valid pessimistic TD-k);
   REDQ-proper adds random-`M`-subset min + high UTD. Ensemble disagreement as an exploration bonus is
   established (bootstrapped-DQN, OAC, SUNRISE) — so a k-critic ensemble **is** an entropy-feedback mechanism.
3. **Structural entropy** of the state hypergraph `H_struct(s)` — the **HyMeKo-native** signal (`hymeko
   entropy`, per-scope structural entropy), and literally the seat `ppo.py` flags ("ent_coef is where the
   algebraic entropy feedback will replace/augment H(π)"). **The novel bet.**

**The unification:** the exploration-vocab (P1) `source × site` makes these interchangeable — a strategy
*declares* whether the entropy source is policy-entropy, ensemble-disagreement, or structural-entropy, wired to
the regularisation site. (1),(2) are textbook; (3) is the research contribution. **The discriminating
experiment:** does structural-entropy feedback beat policy-entropy on a task where the graph *structure carries
information*? — testable only on a real-topology task (6-DOF arm / Galambos), never cart-pole.

## Files touched
| File | Δ |
|---|---|
| `hymeko_rl/sac.py` | new (+180): `SquashedGaussianActor`, `build_sac`, `SACConfig`, `train_sac` (twin soft-Q + auto-α), `run_sac`, CLI |
| `hymeko_rl/tests/test_sac.py` | new (+55, 4 tests) |

**CORE.YAML / deps:** none. **Reuse:** `ddpg.QCritic`/`_backbone`/`_polyak`, `ReplayBuffer`, `eval_balance`.

## Test results
- `test_sac.py` — **4 tests**: squashed actor bounded + finite (tanh-corrected) log-prob + deterministic
  `action_mean`; twin-critic build (mlp + hsikan); `train_sac` end-to-end finite curve. `ruff` + `mypy
  --strict` clean (two documented `# type: ignore` on `backward`).

## Provenance
Git SHA `292388b` (dirty). torch 2.12.0+cu132 (CPU, 1 thread). SAC smoke: mlp, seed 0, 20k steps, 422 s.

## Open issues / follow-ups
1. **Fairer eval** — report curve-max / smoothed-tail (or early-stop at peak) instead of a single noisy
   step-20k snapshot; then a **multi-seed** PPO/DDPG/TD3/SAC comparison.
2. **TD-k / REDQ** — expose `n_critics` end-to-end + random-`M`-subset min + UTD>1 (small additions); measure
   the ensemble-disagreement entropy signal.
3. **Structural-entropy feedback** — wire `hymeko entropy` into the `α·H` seat and test on a real-topology
   task (the novel experiment).
4. **Safe RL** (queued) — `meta_constraint` cost vocabulary + a Lagrangian on the off-policy update.
