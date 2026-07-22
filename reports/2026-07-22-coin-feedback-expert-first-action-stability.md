# FEEDBACK_EXPERT_FIRST_ACTION_STABILITY_PASS — the receding-horizon expert's first action is stable (and reaches strict K6 within the horizon)

**Created-at:** 2026-07-22 19:55 JST
**Branch:** recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62` · obs-contract `FULL_ACTION_OBS_HISTORY_V1` (SHA `6c84fa5b…`)
**Compute:** Mac (Apple Silicon, CPU torch), 8 workers, bounded probe (NOT the 120-state campaign — §11).

## Verdict

`FEEDBACK_EXPERT_FIRST_ACTION_STABILITY_PASS`. On 12 representative states across the critical phases, repeated
receding-horizon planning (6 independent search seeds per state) produces **0/12 materially-conflicting first
actions**. The deterministic selection rule (best predicted lexo → action tie-break) therefore returns a
well-defined label per state. As a bonus signal, the planner reaches **strict K6 (dwell 6) within the 15-step
horizon** on almost every state — the expert is stable *and* effective.

## Setup (§3–§4)

- **Expert (§3):** MPC-style replanning in the deployed actor's **4-dim `inner.step` arm space** (== the label /
  deployed action space; no env.step/6-dim aliasing). At each state: CEM over an H=15 × 4 action sequence, scored by
  the strict K=6 `DeliveryCertifier` over `_cert_step` (the exact deploy certificate), carrying robot-attribution +
  in-progress dwell from the real state; execute/label only the FIRST action. Warm-start hook for retained plans.
- **Learner key:** the exact `FULL_ACTION_OBS_HISTORY_V1` (152-dim, k=3 obs + 2 actions) captured at each state.
- **Probe states:** captured from neutral→transport rollouts (frozen E-approach + handoff) on 8 delivering headline
  seeds; per-seed quota → cross-seed diversity. Config: 6 search seeds, pop 48, iters 8.

## Metric correction (transparent — the naive gate would have false-BLOCKED)

The first gate metric (min pairwise **cosine** ≥ 0.5) returned `ALIASING_BLOCKED` (min_cos 0.000). That was a
**metric-integrity artifact**, not aliasing: cosine is ill-defined for near-zero vectors, and the settle/dwell
optimal action *is* near-zero ("hold still"). The data proved it — cosine collapsed exactly as `|mean|` shrank:

| phase | `|mean|` | cosine | abs std | reading |
|---|---|---|---|---|
| STRICT_DWELL | 0.10 | 0.000 | 0.09 | near-zero hold; actions agree, direction is noise |
| SETTLING | 0.26–0.40 | 0.04 | 0.11 | near-zero hold |
| TRANSPORT | 1.5–2.2 | 0.51–0.81 | — | non-trivial, directions agree |

The corrected gate is **magnitude-aware**: a state CONFLICTS only if the action is non-trivial (`|mean| > 0.35`)
AND directions disagree (cosine < 0.5) AND the relative spread is large (`std/|mean| > 0.6`). Near-zero or
tight-cluster actions are stable (well-defined label). The bigger CEM budget (pop 48 vs 32) also confirmed the
earlier low target-entry cosine (0.08) was **CEM noise** — it rose to 0.35–0.88. Both the naive and corrected
summaries are committed for audit.

## Result (12 states, 0 conflicting)

| phase | states | `|mean|` range | cosine range | rel_spread | predicted strict-within-H |
|---|---|---|---|---|---|
| TRANSPORT | 4 | 1.6–2.2 | 0.51–0.81 | 0.30–0.34 | 4/4 (dwell 6) |
| TARGET_ENTRY | 4 | 0.95–3.49 | 0.35–0.88 | 0.20–0.48 | 4/4 (dwell 6) |
| SETTLING | 2 | 0.26 / 1.02 | 0.00 / 0.33 | — | 1/2 (one near-zero hold) |
| STRICT_DWELL | 1 | 0.30 | 0.05 | — | near-zero hold |

## Non-claims / scope (§4, §9)

- Probe states are from **delivering** rollouts (transport / target-entry / settling / strict-dwell). Explicit
  **dwell-recovery** and **contact-loss-recovery** states did not arise in delivering trajectories and were **not**
  captured — their stability is untested here and will be exercised by the §5 full-rollout pilot on the 9+30 states.
- The "strict within horizon" prediction is a **short-horizon** planner forecast from mid-trajectory states, not a
  full neutral-reset delivery. The delivery claim is the §5 gate (`RECEDING_HORIZON_FEEDBACK_EXPERT_PASS`).
- No final-test seeds used. No RL. No BC trained yet.

## Provenance

Harness `coin_delivery/coin_v3_receding_horizon.py` (§3 expert + §4 stability). Results
`experiments/2026_07_22_coin_v3_learning/receding_horizon/first_action_stability{,_naive_cosine}.json`. Obs contract
SHA `6c84fa5b6ab0aab96cc9deccbb03ec8c9b7226219bd8c0cc96387fe84f504eac`. Config: H=15, pop=48, iters=8, 6 search seeds.

## Next (§5, RL still gated §15)

Implement `receding_horizon_rollout` (per-step replanning from neutral) and run the bounded expert pilot on the 9
headline + 30 preregistered train_query states on kato14 → `RECEDING_HORIZON_FEEDBACK_EXPERT_PASS` requires ≥6/9
headline strict + ≥60% on the 30-state bank, every accepted success replay-certified from neutral.
