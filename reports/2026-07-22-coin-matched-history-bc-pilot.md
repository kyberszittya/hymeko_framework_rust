# COVARIATE_SHIFT_DOMINATES — open-loop-trained BCs clone accurately but drift off-manifold in closed loop; richer input does not rescue it

**Created-at:** 2026-07-22 23:02 JST · branch recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62`
· obs contract `FULL_ACTION_OBS_HISTORY_V1` SHA `6c84fa5b…`. No RL, no DAgger, final-test untouched.

## Verdict

**Primary: `COVARIATE_SHIFT_DOMINATES`.** Secondary: `OPEN_LOOP_BC_ROUTE_EXHAUSTED` (all open-loop variants ≤ the
handoff baseline across contracts and seeds). Supervised cloning is excellent and improves with richer input, but
closed-loop delivery collapses for **every** input contract, and executed-action history does **not** materially
improve rollout — the failure is state-distribution drift, not action-channel exposure bias.

## §2–§4 matched setup

One frozen corpus: the **57 certified strengthened open-loop trajectories** (teacher=search), trajectory-level split
(train 10463 / val 2069 transitions). Identical architecture (MLP 256×256, input width only varies), optimizer, LR,
update budget, phase-balanced sampler, 3 training seeds, evaluation banks (headline 9 + validation 30). No H=30
per-step feedback labels; no handoff/open-loop mixture. D = the frozen handoff-feedback baseline (instantaneous,
SHA `cc8c31fd`).

## §5 supervised vs closed-loop (best of 3 seeds)

| config | input | val MSE (teacher-forced) | grasp | **headline (closed-loop)** | validation |
|---|---|---|---|---|---|
| A instantaneous | obs_t (48) | 1.06e-3 | 9/9 | **0/9** | 2/30 |
| B obs-history | obs_t,t-1,t-2 (144) | 7.1e-4 | 9/9 | **1/9** | 4/30 |
| C obs+action-history | +a_{t-1},a_{t-2} (152) | **3.4e-4 (best)** | 9/9 | **1/9** | 2/30 |
| D handoff baseline | obs_t (48) | — | 9/9 | **3/9** | 2/30 |

**Supervised fit improves monotonically with richer input** (C best, 3.4e-4) — the action channel *does* help
predict the demonstration action. **Closed-loop delivery does not follow**: A 0/9, B 1/9, C 1/9 — all far below the
handoff baseline (3/9), all with 9/9 grasp. The supervised↔closed-loop gap is the covariate shift.

## §6 exposure-bias / prefix-replay (best C, seed 2)

Hand off to C from E-approach+handoff states at progressively later phases (near-distribution takeover):

| handoff phase | C delivers |
|---|---|
| TRANSPORT | **0/5** |
| TARGET_ENTRY | **0/4** |
| SETTLING | 1/2 |
| STRICT_DWELL | 0/1 |

C cannot continue the demonstrated trajectory even from clean **mid-transport** states; it completes only when handed
off already near settling. **The first recoverable divergence is at TRANSPORT** — the policy leaves the demonstration
manifold as soon as it drives the transport→settle segment.

## §8 action-history operationally load-bearing? (answered by A vs C)

A (no action history, 0/9) ≈ C (action history, 1/9) in closed loop, while C is clearly better *supervised*. Per §8's
interpretation, "no closed-loop degradation without the action channel means action history was not operationally
load-bearing." The action channel improves the teacher-forced fit but is inert once the policy's own actions populate
the history — consistent with `COVARIATE_SHIFT_DOMINATES` rather than `AUTOREGRESSIVE_EXPOSURE_BIAS_BLOCKED` (which
would require A, lacking the action channel, to survive where C collapses; it does not).

## Why not the other pre-registered classes (§9)

- **not `HISTORY_CONDITIONED_BC_PILOT_PASS`** — no open-loop config reaches ≥4/9 headline (max 1/9) or exceeds the
  3/9 handoff baseline.
- **not `ACTION_HISTORY_MATERIALLY_IMPROVES_BC`** — C does not exceed A/B in *closed-loop* validation (only supervised).
- **not `AUTOREGRESSIVE_EXPOSURE_BIAS_BLOCKED`** — the failure is not specific to C's own-action feedback; A fails
  equally without any action channel → general state-distribution drift.

## Interpretation & RL-gate (§11)

The BC prerequisite via imitation of the certified open-loop distribution is **experimentally exhausted for this
matched pilot**: accurate supervised cloning does not yield a closed-loop policy, and no observation- or
action-history contract closes the gap. This meets the §11 case-B condition to **reconsider** the RL gate under the
revised question: *can reward-driven SAC/TD3 improve the best clonable initialization by learning recovery behaviour
under its own state distribution?* The best clonable initialization is the **handoff baseline (3/9)** — **not** a
competent 6/9 BC; any RL claim must respect that starting level (§11). Per §10 I launch **no** corrective
automatically; the first recoverable divergence (TRANSPORT) is reported for whatever intervention you choose.

## Provenance

`matched_bc_pilot.py` (+ `matched_bc_pilot_results.json`), `prefix_replay.py` (+ `prefix_replay_result.txt`),
`eval_action_history_bc_delivery` added to `full_action_bc.py`. Corpus = 57 open-loop traj; D handoff SHA
`cc8c31fd`. Obs contract SHA `6c84fa5b…`, bundle `6664ac459cca8f62`. Ran on Mac; kato14 clean.
