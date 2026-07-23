---
title: Strict-counter state contract V1 — the certifier terminal state is hidden from the critic
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: HIDDEN_CERTIFIER_STATE_NONMARKOV
tags: [coin, markov, strict-counter, critic, td3, non-markov, measurement-only, no-training]
---

# STRICT_COUNTER_STATE_CONTRACT_V1 — HIDDEN_CERTIFIER_STATE_NONMARKOV

Measurement-only. No training or environment change. This narrow check asks whether the strict-dwell counter (the
distance-to-terminal, 0..K=6) is observable to the policy and critic. It is **not** — and the consequence is a critic
that maps identical observations to TD targets differing by the full terminal bonus.

## 1. Handoff strict-counter distribution (dev banks)
| bank | n | handoff `_strict` distribution |
|---|---|---|
| transport | 15 | {0: 15} |
| braking | 10 | {0: 10} |
| settling_dwell | 6 | **{1: 6}** |
| ALL | 31 | {0: 25, 1: 6} |

Only the settling starts carry dwell from the pi_0 prefix (all at 1); nothing reaches 2–5 at the handoff. This is
exactly the 6/31−1/31 gap: the 6 settling starts (handoff `_strict`=1) that the arc-canonical `rl._strict` counts and an
offline dwell-from-0 does not.

## 2. Observability audit (measured, not inferred from dtz/speed)
| consumer | contains the counter? | evidence |
|---|---|---|
| actor obs (node_features 48) | **NO** | node_features has no `_strict`/`_success`/dwell field |
| critic conditioning | **NO (exact)** | conditioning = onehot3(control_mode)+onehot2(contact)+event5; `control_mode` collapses `strict≥1 → "settling_dwell"` — it does not distinguish 1..5 |
| replay transition | **NO** | late replay stores (state=obs+conditioning, action, reward, next); no counter |
| target-network input | **NO** | target critic sees the same state vector as the online critic |

`control_mode(dtz, speed, prev, strict)` returns `"settling_dwell"` for **all** `strict∈{1,2,3,4,5}` (unit test
`test_control_mode_collapses_strict_counter`). So the only strict-derived signal reaching the critic is a binary
"settling phase" flag — the exact distance-to-terminal is discarded.

## 3. Paired-state probe — identical physical state, strict 0..5
From fresh deterministic reconstructions (identical qpos/qvel **and** node_features history buffer), the strict counter
set to 0..5, one frozen-pi_0 holding step:

| strict | obs_48 hash | control_mode | reward | terminated | TD target (V=1) |
|---|---|---|---|---|---|
| 0 | `a8b7bf…` | settling_dwell | −0.75 | False | 0.24 |
| 1 | `a8b7bf…` | settling_dwell | −0.75 | False | 0.24 |
| 2 | `a8b7bf…` | settling_dwell | −0.75 | False | 0.24 |
| 3 | `a8b7bf…` | settling_dwell | −0.75 | False | 0.24 |
| **4** | `a8b7bf…` | settling_dwell | **+29.25** | False | **30.24** |
| **5** | `a8b7bf…` | settling_dwell | −0.75 | **True** | −0.75 |

- **obs_48 IDENTICAL** across strict 0..5 (same hash).
- **reward DIFFERS** — the graded terminal (+30) fires at strict=4.
- **termination DIFFERS** — terminated only at strict=5.
- **control_mode does NOT distinguish strict** (all "settling_dwell").
- **TD targets differ by ~31** (30.24 vs −0.75) for **identical critic input**.

(Note a reward/termination off-by-one worth flagging: `terminal_deliver_graded` fires at `_strict→5` while termination
is at `_strict→6` — the +30 lands one step before the terminal.)

## 4. Verdict
**`HIDDEN_CERTIFIER_STATE_NONMARKOV`.** The counter (or an exactly-sufficient history) is available to neither policy nor
critic. Two states that are identical to the critic have bootstrapped targets separated by the terminal bonus, so no
critic on this state representation can be consistent — the value function is non-Markov w.r.t. the K=6 terminal.

## 5–6. Both evaluation semantics (never mixed)
| semantics | rate | definition |
|---|---|---|
| **CONTINUATION_STRICT** | **6/31** | inherit `handoff_strict` (arc-canonical `rl._strict`) — the *historical continuation* result |
| **RESET_AT_HANDOFF_STRICT** | **1/31** | strict=0 at the late-controller boundary — the *late-skill-from-zero-dwell* result |

Both are reported; neither is "the" number. 6/31 credits the late controller with the prefix's carried dwell; 1/31
measures only the late segment's own dwell.

## 7. Reclassification of local critic / TD3 findings
| finding class | classification |
|---|---|
| PHASE_SWITCHED_TD3 (stage 1/1b/1c), TRANSACTIONAL_TD3, TRANSPORT_DWELL_TD3, PHASE_GATED_RESIDUAL_CRITIC | **VERDICT_REQUIRES_RERUN** — the critic state HID the strict counter; identical observations mapped to TD targets differing by up to the terminal bonus (~31). The critic could not represent distance-to-terminal, so its bootstrapped targets mixed physically-similar states with different terminal proximity. "No local improvement" / "critic route blocked" are **confounded** by non-Markov critic state, not established as task walls. |
| CHUNK / PRIMITIVE / repaired-planner supervised baselines | **UNAFFECTED** — supervised regression / CEM search, no bootstrapped critic target. |
| Reconstruction / audit success rates | governed by the CONTINUATION vs RESET semantics above (§5–6), not by this Markov defect. |

The corrective (for a future campaign, not this check): add the exact strict counter (or a one-hot of 0..K, or a
countdown-to-terminal scalar) to the critic state (and ideally the actor state) before re-running any local-improvement
RL. Only then is "local improvement exhausted" a testable claim.

## Files
- entry: `experiments/…/rl_entry/coin_strict_counter_contract.py`; test `test_control_mode_collapses_strict_counter`.
- result: `experiments/…/rl_entry/strict_counter_contract_v1.json`, this report.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; reward v3; corrected traces (start_id + env_strict).
Measurement-only, deterministic (1-thread). No CORE.YAML / reward / certifier / controller changes.
