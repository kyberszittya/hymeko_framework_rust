---
title: Actor-update-path audit V1 — the Markov critic repair is confirmed; actor conversion is blocked
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: MARKOV_CRITIC_REPAIR_CONFIRMED + ACTOR_CONVERSION_BLOCKED (supersedes REPAIRS_NOT_SUFFICIENT)
tags: [coin, markov, actor-update, bc-anchor, trust-region, mechanism-audit, verdict-correction]
---

# ACTOR_UPDATE_PATH_AUDIT_V1 — REPAIRS_NOT_SUFFICIENT was misleading; the actor path blocks the fix

A review critique of the ablation was correct: the `REPAIRS_NOT_SUFFICIENT` verdict did not follow from the data. The
critic learned terminal proximity, but the transactional wrapper rolled back ~99 % of policy steps — the improved critic
was essentially never let through to the actor. This audit isolates the mechanism on a trained Markov critic, with **no
new campaign and no new env steps for the analysis** (one training run to obtain the critic, then gradient/Q analysis on a
gate-active replay batch).

## What was measured (Arm A critic, 4000 updates, 128 gate-active states)
| probe | result |
|---|---|
| trained actor's counter-use `‖a(strict=1) − a(strict=5)‖` | **0.0005** — strict-blind despite the critic representing proximity |
| Q-loss gradient vs BC-loss gradient (cosine) | **−0.395** — the BC anchor *opposes* the counter-using Q gradient |
| A: one Q-only step | ΔQ +0.02, counter-use 0.0006 (a single lr-1e-5 step barely moves) |
| B: one Q+BC step | ΔQ +0.02, counter-use 0.0006 |
| C: Q + trust region | **REJECTED** (scale = None) |
| D: Q + BC + trust region | **REJECTED** (scale = None) |
| **UNCONSTRAINED Q-only, 100 steps (no BC, no trust)** | **ΔQ +29.0, counter-use 0.0005 → 0.024 (45×), drift 6.7 — raises Q AND grows counter-use** |
| trust-region cap that binds on a Q-only step | **cum_p95** (cumulative-drift p95) |

## Mechanism — `BC_ANCHOR_AND_TRUST_REGION_BOTH_BLOCK`
1. **The critic gradient IS informative.** Unconstrained, the actor converts the critic's terminal-proximity signal: Q
   rises +29 and the policy *starts using the counter* (counter-use grows 45×). So it is not "the gradient is
   uninformative" and not "the task is a wall".
2. **The BC anchor opposes it.** The BC target is `pi_0(obs48)`, which is **strict-blind** (identical action for strict 1
   and 5). Its gradient has cosine **−0.395** with the counter-using Q gradient — it actively pulls the policy back to
   ignoring the counter. This is a three-way conflict: Q-loss says *use the counter*, BC says *ignore it*, trust region
   *rolls back* the counter-dependent move.
3. **The trust region rolls it back.** Every constrained proposal (C, D) is rejected; the binding cap is the cumulative
   anchor drift p95. Across the full ablation this produced ~99.3 % rollback and an identical drift plateau (~0.031) in
   all six runs — an artificial wall, not a natural optimum.

## Corrected verdict
Replace **`REPAIRS_NOT_SUFFICIENT`** with:
- **`MARKOV_CRITIC_REPAIR_CONFIRMED`** — the representation repair worked; the critic authorizes and represents
  distance-to-terminal (Q rises toward K6 in all runs; unconstrained actor converts it).
- **`ACTOR_CONVERSION_BLOCKED`** — the improved critic is prevented from reaching the policy by (a) the strict-blind BC
  anchor and (b) the transactional cumulative-drift trust region.

It is **not** "local improvement exhausted" and **not** a Markov failure. The next binding contract is the actor-update
path, which the ablation did not vary.

## Confounds in the original ablation (acknowledged)
The obs55 trainer reimplemented parts of the historical loop rather than reusing them exactly: a uniform `_sample_nstep`
(not the phase-balanced replay), a generic `_sample_actor` (not the gate-only BC bank), and a hand-built anchor bank (not
`build_anchor_bank`). So the run mixed the strict-counter repair with replay-distribution, BC-anchor, and trust-anchor
changes — it was not a clean single-variable ablation, and the anchor mismatch may mis-measure the drift the trust region
gates. The critic-repair conclusion (Q represents proximity; unconstrained actor converts it) is robust to these; the
"no K6 improvement" outcome is now explained by the actor-path blockers above, not attributed to the repair.

## Next lever (needs your go — no reward edit, no new campaign yet)
Under the now-healthy Markov critic, on a graded-ladder eval bank: (1) make the BC anchor **strict-aware** or decay it
near the terminal (so it stops fighting the counter), and (2) relax or replace the cumulative-drift trust region with a
monitored standard TD3 actor step; re-test whether K6 converts. Reuse the exact historical `build_anchor_bank` /
phase-balanced replay / gate-only BC bank to make it a clean single-variable change.

## Files & commits
- entry: `experiments/…/rl_entry/coin_actor_update_path_audit.py`; result `…/actor_update_path_audit_v1.json`.
- trainer hook: `train_arm(..., return_artifacts=True)` in `coin_markov_ablation_train.py`.
- supersedes the verdict in `reports/2026-07-23-strict-counter-markov-repair-ablation-v1.md` (frozen; not modified).

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Arm A critic at 4000 updates (single-thread). No env
steps in the analysis; no training beyond obtaining the critic; canonical reward/task unchanged. No CORE.YAML items.
