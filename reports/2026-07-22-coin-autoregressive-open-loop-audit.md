# AUTOREGRESSIVE_DATA_LEAKAGE_AUDIT_PASS — encoding is causal; action-history is only MODESTLY load-bearing and NOT trajectory-ID memorization

**Created-at:** 2026-07-22 22:40 JST · branch recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62` ·
obs contract `FULL_ACTION_OBS_HISTORY_V1` SHA `6c84fa5b…`. No RL; no BC trained; no final-test access.

## Verdict

`AUTOREGRESSIVE_DATA_LEAKAGE_AUDIT_PASS`. The history encoding is provably causal — no temporal leakage. Two
supporting diagnostics **temper** the reframing honestly: the action-history is only *modestly* load-bearing (6–14%
conflict reduction, most at settling), the observation-history alone does nothing, and it is **not** trajectory-ID
memorization.

## §1 claim registry corrected (history preserved)

`OPEN_LOOP_TEACHER_NOT_CLONABLE` → narrowed to `OPEN_LOOP_TEACHER_NOT_CLONABLE_UNDER_INSTANTANEOUS_OBSERVATION`;
added open hypothesis `OPEN_LOOP_TEACHER_MAY_BE_CLONABLE_UNDER_ACTION_HISTORY` and
`FEEDBACK_MPC_SUCCESSFUL_BUT_LABEL_MULTIVALUED` (the H=30 expert passed its rollout gate; only its per-step settle
labels are multivalued). Prior commits/reports unchanged.

## §2 admissibility class

`OPEN_LOOP_AUTOREGRESSIVE_LABELS_V1` — admissible **only** for the exact `FULL_ACTION_OBS_HISTORY_V1` input (152-dim,
current+2 obs + 2 executed actions, deterministic padding; SHA `6c84fa5b…`, graph `sem:469094de…`). `OPEN_LOOP_PLAN_ONLY`
remains correct for instantaneous / observation-only-frame-stack inputs. No future action/obs, seed, timestep, or
planner state may enter the input.

## §3 temporal-leakage audit (PASS)

Controlled-perturbation proof on all 96 base trajectories (19334 samples): feature[t] = [obs_t, obs_{t-1}, obs_{t-2},
a_{t-1}, a_{t-2}] (152 = 3×48+2×4, **no slot for a_t**). Perturbing a[t] **and every future obs/action** leaves
feature[≤t] **byte-identical** (causality); a[t] legitimately enters feature[t+1] as the past action; the label is not
recoverable from the feature; variable-length trajectories carry no length signal. All 6 checks pass.

## §5 is the action-history load-bearing? (OPEN_LOOP_HISTORY_ONLY, 57 traj / 12532 samples)

Conditional action dispersion (k-NN, lower = more consistent), by phase & representation:

| phase | instantaneous (48) | obs-only k=3 (144) | **FULL history (152)** | Δ from action channel |
|---|---|---|---|---|
| TRANSPORT | 0.0583 | 0.0588 | 0.0552 | −6% |
| TARGET_ENTRY | 0.0471 | 0.0468 | 0.0445 | −5% |
| SETTLING | 0.0637 | 0.0619 | 0.0531 | **−14%** |
| STRICT_DWELL | 0.1791 | 0.1829 | 0.1694 | −7% |

**Honest reading:** the *observation*-history alone (instant→obs-k3) does **nothing** (consistent with the earlier
frame-stack-k3 failure); the *action* channel (obs-k3→full) gives a **modest** 5–14% reduction, largest at settling.
This is a real but **modest** effect — *not* the dramatic consistency I described last turn (that number was on the
handoff-dominated *mixed* set). Crucially, the open-loop is **already fairly consistent given the instantaneous
observation** (dispersion ≈0.058 vs transport action magnitude ≈1.5 ⇒ ~4%). That tempers the reframing: the earlier
reactive-BC failure was likely more **covariate-shift** (feedforward cloning drifts off-trajectory) than
instantaneous-observation multivaluedness.

I therefore do **not** claim `ACTION_HISTORY_WAS_LOAD_BEARING` — the effect is modest, not material across all four
phases. The decisive test is the autoregressive rollout (§7–§11), where covariate-shift/exposure-bias is exercised.

## §6-A trajectory-identity control (NOT memorization)

Restricting k-NN neighbours to **different trajectories** (different seeds), FULL-history action std:

| phase | within-any | **cross-trajectory** |
|---|---|---|
| TRANSPORT | 0.0552 | 0.0482 |
| TARGET_ENTRY | 0.0445 | 0.0464 |
| SETTLING | 0.0531 | 0.0538 |
| STRICT_DWELL | 0.1694 | **0.0909** |

Cross-trajectory consistency is **as good as or better than** within-trajectory → the history is a **reusable control
state**, not a trajectory identifier. So `ACTION_HISTORY_TRAJECTORY_ID_MEMORIZATION` is refuted at the dataset level.
(§6-B action-history perturbation and §6-C prefix-swap are model-based and belong to the §7 training stage.)

## Where this leaves the route

- Leakage: clean. Memorization: refuted. Action-history benefit: real but modest (5–14%), settling-weighted.
- The open-loop is fairly clonable given instantaneous obs already; the open question is **autoregressive
  covariate-shift**, which only the §7–§11 training + closed-loop rollout can answer.
- Recommendation: proceed to the bounded autoregressive BC pilot (§7–§11) with the honest expectation that the
  history gain is modest and exposure bias (§8) is the primary risk — not a foregone win. RL gate holds (§15).

## Provenance

`leakage_audit.py` (+ result `leakage_audit_result.json`), `history_mechanism.py`, `claim_registry.json`,
`admissibility_OPEN_LOOP_AUTOREGRESSIVE_LABELS_V1.json`. Density-controlled estimators retained from `050fd5f`.
Obs contract SHA `6c84fa5b…`. kato14 clean (analysis ran on Mac).
