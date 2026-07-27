# Coin-R8 — tip-referenced bounded-residual RL under the CORRECTED authorization gate

**Date:** 2026-07-27 · **Branch:** `recovery/coin-r8-tip-referenced-residual-rl` (worktree `hymeko_coin_r8_wt`)
**Final verdict:** `BOUNDED_RESIDUAL_RL_GENERALISES_TO_HELDOUT_BUT_DOES_NOT_YET_DELIVER`

## Why R8 exists (the corrected gate)

R1–R7 tried to solve coin delivery *deterministically* (θ-prediction, closed-loop coast, adaptive coast, brake-to-stop,
velocity servo) and each failed held-out for a distinct measured reason. The RL-authorization gate had gone **circular** —
it demanded deterministic 4/4 delivery *before* RL, but deterministic 4/4 was exactly what could not be built. R8 replaces
it with the corrected gate: **a safe deterministic scaffold is required; deterministic 4/4 is NOT an RL prerequisite.**

```
S0 physical feasibility → S1 safe scaffold → S2 update-zero identity → S3 learnability → S4 matched SAC/TD3 dev → S5 frozen held-out
```

RL is authorised once **S1 ∧ S2 ∧ S3** pass. Every integrity constraint stayed hard throughout: no teleport / hidden force /
teacher fallback / oracle injection; unchanged torque / slew / joint-velocity / collision / motion contracts; the reward is
independent of the frozen K6 certificate; the Bellman action is the actor emission only; held-out `s4`/`s7` were excluded
from all training, replay, hyperparameter, threshold and selection decisions until the single S5 eval; update-0 reproduces
the scaffold; release stays R6-certificate-gated; the oracle is a feasibility witness only.

## Gate-by-gate result

| gate | result | evidence |
|---|---|---|
| **S0** physical feasibility | PASS | frozen K6 delivery reachable on all 4 cradles (prior arc) |
| **S1** safe deterministic scaffold | PASS | tip-referenced joint-velocity transport; joints regulated, coin within limits (`8e73b261`) |
| **S2** update-zero identity | PASS (bit-exact) | zero-residual adapter reproduces the scaffold over the full trace, max diff `0.00e+00` (`a7bd37a8`) |
| **S3** learnability | `RESIDUAL_LEARNABILITY_SIGNAL_PASS` | d_fwd_vel EFFECTIVE; 50.8% safe-positive candidates; robust top-decile enrichment 1.379 (IQR [1.30,1.53], 0.76 of 25 splits) (`eb1cbd7c`) |
| **S4** matched SAC/TD3 (dev) | TD3 ≫ SAC | TD3 median Δ **+0.123** over scaffold (boot-CI [0.116,0.133], 3/3 seeds); **s3 K6-delivered all seeds**; SAC Δ +0.02 (`ea2eb8eb`) |
| **S5** frozen held-out | `RESIDUAL_RL_IMPROVES_HELDOUT_WITHOUT_DELIVERY` | dev-champion (TD3 s2), **one-shot** on s4/s7: return −0.169→−0.038 (Δ **+0.131**); s4 266→30mm, s7 51→45mm; safe 2/2; 0/2 K6 |

### The formulation

The Bellman action is a **bounded residual** `a ∈ [-1,1]^3` (forward-velocity reference / squeeze / stop-gain), applied as a
**constant per-option residual** over the frozen tip-referenced scaffold via `ConstantResidualActor` + `ResidualTipAdapter`.
One `env.step` = one full scaffold+residual rollout = one semi-MDP option (`terminal=1`, so `γ^τ·Q_next` is zeroed — the
K6-independent option-consequence return IS the target). This plugs into the framework `train_semi_mdp` (`option_rl`)
unchanged; the env holds **only** dev cradles, so nothing held-out can enter the replay or the critic. The `replay_audit`
proves, over a probe grid, that the recorded Bellman action equals the clipped emission and the executed residual equals
`clip(a)·bounds` — no torque or corrected target is ever presented as the action; at `a=0` the residual is zero (the S2
identity). Update-0 is the actor distilled to a zero mean = the safe scaffold.

## What this means (measured / inferred / hypothesis)

- **Measured.** A bounded residual over the safe scaffold, trained by TD3 on the two development cradles only, transfers to
  the two held-out cradles: it cuts the mean held-out gap-to-zone from 16.9 cm to 3.8 cm (Δ +0.131), one-shot, while staying
  inside every motion-contract limit. On dev it delivers K6 on 1 of 2 cradles; SAC does not (matched, same seeds/config).
- **Measured limit.** It does **not** deliver K6 on either held-out cradle — s4 misses the 20 mm tolerance by ~10 mm, s7 by
  ~25 mm. This is improvement, not delivery. No overclaim: the verdict is *improves-without-delivery*.
- **Inferred.** This is the **first held-out generalization in the R1→R8 arc**. R1 (flat amortization), R2 (relational),
  R3 (physical-intent decoder) and R4–R7 (deterministic feedback laws) all held flat at 0/2 held-out. The distinguishing
  change is not representation, search or decoder — it is that a **learned, bounded residual over a scaffold that already
  works** amortizes where full-θ / full-intent learning did not. The scaffold carries the safety and the bulk of the
  transport; RL only has to nudge, and that nudge transfers.
- **Hypothesis (not yet tested).** The remaining ~10–25 mm to held-out K6 is consistent with S3's finding that the value
  landscape is *shallow* (weak ordering, top-decile signal only) and with the *constant*-per-option residual being less
  expressive than a time-varying one (S3: coherent 0.55 > constant 0.51). A per-step closed-loop residual and/or a richer
  initiation fingerprint are the next discriminating experiments — a new stage, not a reinterpretation of this one.

## Artifacts

`reports/2026-07-27-coin-r8-residual-rl/`: `contract.json`, `s2_update_zero_identity.json`,
`s3_{sensitivity,candidate_search,rankability,learnability}.json`, `training_contract.json`, `replay_audit.json`,
`sac_results.json`, `td3_results.json`, `matched_comparison.json`, `heldout_frozen_result.json`, `s5_heldout.png`,
`ckpts/{sac,td3}_seed{0,1,2}_best_val.pt`, `final_verdict.json`.
Code: `hymeko_rl/coin_delivery/theta_option/{tip_transport,residual_adapter,residual_option_env}.py`,
`hymeko_rl/experiments/coin_r8_residual_rl.py` (`--s2/--s3/--s4/--s5`). Reports: `2026-07-27-coin-r8-s3-learnability.md`.

## Tests / gates

Unit + integration: `hymeko_rl/tests/test_coin_r4_closed_loop.py` — 19 fast + 1 slow GOLDEN; S2 identity (bit-exact),
residual bounds/identity-at-zero, residual-option env contract (dev-only + Bellman==residual), distill-to-scaffold.
Static: `ruff` clean; no function ≥ CC 15 (`radon`). CORE.YAML items touched: **none**. New dependency: none.
Live observability: `train_semi_mdp` logs dev score / delivery / train-recent / Q / replay every 75 options.

## Provenance

Worktree `hymeko_coin_r8_wt` @ `recovery/coin-r8-tip-referenced-residual-rl`; isolated from the main checkout to avoid a
HEAD collision with a concurrent coin-toss session. Seeds 0/1/2; RL carve-out (multi-seed median/IQR, not bit-exact).
Wall: S2 ~90 s, S3 ~100 s, S4 ~224 s (3 seeds × 2 algos), S5 ~91 s. Panel build dominates each. Machine: Apple-Silicon
CPU venv (torch CPU). Held-out cradles `s4`,`s7` evaluated exactly once.
