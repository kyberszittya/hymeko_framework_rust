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

---

# §5 — Closed-loop receding-horizon feedback-expert pilot

**Created-at:** 2026-07-22 20:25 JST · **Compute:** KATO14 (26 workers), obs contract `FULL_ACTION_OBS_HISTORY_V1`
(SHA `6c84fa5b…`), bundle `6664ac459cca8f62`. Pilot result JSON SHA `fab6ea8172ef9b6…`. RL **not** started
(`rl_started=false` in the result). **No physical / observation / action / reward / graph / success contract changed.**

## §5 Verdict

`RECEDING_HORIZON_FEEDBACK_EXPERT_FAIL`. The closed-loop expert meets the headline threshold (**6/9**, ≥6 ✓) but
misses the train_query threshold (**16/30**, needs ≥18 ✗). Every counted success is replay-certified from true neutral
(planning-success == replay-certified-success in both banks — **no replay nondeterminism**).

## Setup

`receding_horizon_rollout`: true neutral → frozen E-approach to the bilateral-grasp handoff (declared teacher
approach; the strict-K=6 scorer has no approach gradient) → then the §3 CEM expert **replans every step** in the
deployed actor's 4-dim `inner.step` space, executes only the first action, discards the suffix, replans. Config
**H=15, pop=40, iters=6, elite=8, plan_seed_base=0, max_steps=360**. Banks: headline (9, sha `1e1d5ab87ff2c854`) +
preregistered train_query 6000–6029 (30, sha `652fde8fd1ebf18e`). Streaming `ObsHistoryV1` == batch (verified §2).
Replay-cert: reset neutral + replay the executed sequence with no replanning, verify strict K=6.

## Results

| bank | expert engaged | planning | **replay-certified** | threshold |
|---|---|---|---|---|
| headline (9) | 8/9 | 6/9 | **6/9** {1011,1045,1164,1174,1278,1568} | ≥6 ✓ |
| train_query (30) | 29/30 | 16/30 | **16/30** | ≥18 ✗ |

## Failure taxonomy

| class | headline | train_query | stage |
|---|---|---|---|
| target_exit_failure | 0 | **7** | settle (reached zone, exited un-settled) |
| dwell_recovery_failure | 0 | 3 | settle (reached K2–K5, lost it) |
| target_entry_failure | 0 | 2 | transport (never centered) |
| contact_loss_recovery_failure | 1 | 1 | recovery (lost grasp, no re-grasp) |
| no_bilateral_grasp | 2 | 1 | approach (never grasped) |

Previously-unobserved recovery cases **did arise and are reported, not claimed solved**: `contact_loss_recovery_failure`
(headline 1447; train_query 1× 6023) and `dwell_recovery_failure` (train_query 3×). Aggregate diagnostics
(train_query): **contact-loss-after-acquisition 22/30**, **target-exit-after-entry 15/30**, 12 states reached the zone
but failed. The headline's 3 misses are the known-unrecoverable 1358/1202 (dwell 0 even for the strengthening) + 1447.

## Dominant limitation (measured, not asserted): planning HORIZON (myopia)

10/14 train_query failures are at the **settle stage** (target-exit 7 + dwell-recovery 3). The mechanism: with H=15
the horizon **cannot contain** reach→brake→6-step-dwell (a coin still approaching needs ~20–40 steps to arrive AND
hold six). So the top strict/dwell objective is never satisfiable inside a plan; the search falls back to the
`−min_dtz` tiebreak → it rushes the coin close-and-fast → overshoot/exit (target-exit) and grip break (contact-loss
22/30). The headline's *easy* seeds tolerate the myopia (6/9); the harder train_query distribution does not. This is
**horizon**, not budget (the §4 first actions are stable at this budget), not observation aliasing (§4 audit below),
and not nondeterminism (replay == planning).

## Stability audit (§4 data preserved unchanged; raw distributions)

12 probe states — `min / median / max`: action mean magnitude **0.264 / 1.613 / 3.488**; min pairwise cosine
**0.000 / 0.543 / 0.875**; absolute std **0.257 / 0.558 / 0.762**; std÷|mean| **0.198 / 0.354 / 1.345**. The only
cosine≈0 states are the **2 near-zero holds** (|mean|≤0.35); all **10 high-magnitude** actions agree in direction
(cosine 0.327 / 0.588 / 0.875). Near-zero holds are demonstrably not confused with aliased high-magnitude actions.
The magnitude-aware gate is **unchanged** from §4 and was not altered after seeing §5.

## Proposed single bounded corrective (NOT run; success contract unchanged)

Re-run the same pilot with **H=30** (double the planning horizon; keep pop≈48). Hypothesis: a horizon that contains
reach→brake→6-dwell lets the strict/dwell objective dominate, converting the 10 settle-stage failures (target-exit +
dwell-recovery) into successes and pushing train_query ≥18/30. Everything else — env, obs contract, action space,
reward, strict-K=6 certificate, banks — held fixed. Secondary limit (contact-loss/no-grasp, ~2–3/bank) is a separate
no-re-grasp-gradient issue not addressed by horizon.

## Decision

`RECEDING_HORIZON_FEEDBACK_EXPERT_FAIL` → **RL not started** (§15 holds). Dominant limitation = planning horizon.
One bounded corrective proposed (H=30). Awaiting authorization before running it or proceeding.

## Provenance

`coin_delivery/{coin_v3_receding_horizon,coin_v3_feedback_pilot}.py`; pilot result
`experiments/2026_07_22_coin_v3_learning/receding_horizon/feedback_pilot_result.json` (SHA `fab6ea81…`). Headline
bank sha `1e1d5ab87ff2c854`, train_query pilot sha `652fde8fd1ebf18e`. Config H=15/pop=40/iters=6, plan_seed_base=0.
