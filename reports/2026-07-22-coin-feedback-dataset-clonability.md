# FEEDBACK_TEACHER_STILL_MULTIVALUED — the stochastic H=30 feedback teacher is MORE multivalued at settling than the open-loop mix (in full obs-history space)

**Created-at:** 2026-07-22 22:25 JST · **Branch:** recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62`
· obs contract `FULL_ACTION_OBS_HISTORY_V1` (SHA `6c84fa5b…`). No RL; no BC trained; no final-test access.

## §7 Verdict

`FEEDBACK_TEACHER_STILL_MULTIVALUED`. The clonability gate FAILS on its core conditions: the H=30 feedback dataset
does **not** have materially lower conflict than `OPEN_LOOP_MIXED_LEGACY` — at **settling** it is markedly **higher**.
Per the directive I stop here and do **not** train the pilot BC.

## §3 dataset (feedback-only, admissible labels)

H=30 canonical expert (pop 40, iters 6) on the 9 headline + 30 train_query pilot states; only replay-certified
trajectories stored. **27 certified trajectories, 578 H=30-expert feedback labels + 2547 E-approach labels, 0
open-loop suffix labels.** Tarball SHA-256 `c7bf94335b4f09bead7a…` (host-local; manifest committed). Every label is a
state-feedback action (E-approach prefix or per-step-replanned H=30 expert); the quarantined open-loop CEM suffix
actions never enter it.

## §5 conditional action-conflict (density-CONTROLLED — the critical methodology fix)

The naive k-NN comparison is confounded by dataset size (B = 19334 samples ≫ C = 3125): a bigger dataset has closer
neighbours and looks artificially low-conflict. Two matched estimators remove the confound:

| phase | estimator | **B_OPEN_LOOP_MIXED** | **C_H30_FEEDBACK** | gate wants |
|---|---|---|---|---|
| TRANSPORT | size-matched k-NN (n=496) | conflict 0.000, cos-dis 0.007 | conflict 0.069, cos-dis 0.050 | C < B ✗ (≈, C slightly higher) |
| SETTLING | size-matched k-NN (n=286) | conflict **0.000**, cos-dis **0.004** | conflict **0.266**, cos-dis **0.470** | C < B ✗✗ (C much higher) |
| STRICT_DWELL | size-matched k-NN (n=145) | conflict 0.000 | conflict 0.000, cos-dis 0.250 | ≈ |
| SETTLING | fixed-radius r=9 (standardized) | conflict 0.000 | conflict **0.306** | C < B ✗✗ |

At **matched density**, the open-loop mixed labels are near-perfectly consistent in full obs-history space
(settling cosine ≈ 0.996) while the H=30 feedback settling labels disagree (cosine ≈ 0.53).

## §6 diagnostic predictability (held-out by trajectory, k-NN regressor)

Feedback-action MSE: instantaneous obs (48) **0.1248**; `FULL_ACTION_OBS_HISTORY_V1` (152) **0.1162** (only ~7%
better); phase-conditioned **0.1171** (no gain). Per-phase k-NN irreducible action std: TRANSPORT 0.077, **SETTLING
0.136** (~2×), DWELL 0.019. The settling irreducible variance is the smoking gun — even the dataset's own nearest
neighbours disagree on the settling action.

## Mechanism (measured, and the important reframing)

- The **open-loop mixed labels are CONSISTENT in full obs-history space** (settling cos ≈ 0.996). The earlier BC
  failure (`FULL_ACTION_BC_COMPETENCE_BLOCKED`) was measured in *instantaneous*-observation space; adding the
  **previous-actions** channel of the history (which encodes where you are in a time-indexed suffix) makes even the
  open-loop suffix a near-deterministic function of the history. Frame-stack k=3 failed earlier because it lacked the
  action channel. **So "open-loop is unclonable" was an instantaneous-observation artifact, not intrinsic.**
- The **H=30 feedback teacher is a stochastic per-step CEM optimizer.** Across trajectories, at similar histories it
  selects **different (all-delivering, §4-benign) settle actions** → settling multivaluedness (cos 0.53, irreducible
  std 0.136). The receding-horizon replanning did not *reduce* label multivaluedness; at settling it *increased* it
  versus the smooth deterministic open-loop suffix.

Net: the load-bearing change for label consistency is the **full obs-history representation**, not the feedback
teacher; and the stochastic feedback teacher adds settling multivaluedness. This is why the §7 gate fails.

## §7 gate conditions

| condition | result |
|---|---|
| H30 feedback TRANSPORT conflict materially < mixed | ✗ (≈/slightly higher) |
| H30 feedback SETTLING conflict materially < mixed | **✗✗ (much higher)** |
| no high-magnitude phase with severe opposing labels | settling cos 0.53 = moderate spread, not opposing-severe |
| obs-history predicts feedback better than instantaneous | ✓ but marginal (0.116 vs 0.125) |
| not dominated by duplicated states | n/a (distinct-seed trajectories) |
| every label has planner provenance + stability | ✓ (planner diagnostics stored) |
| no OPEN_LOOP_PLAN_ONLY action in supervised loss | ✓ (0 open-loop) |

Fails on the two load-bearing conditions → `FEEDBACK_TEACHER_STILL_MULTIVALUED`.

## Honesty / non-claims (§9)

- The settling multivaluedness is **benign** (all constituent actions deliver, §4) — a MSE-BC *might* still clone the
  convex-ish valid set. But the directive's gate is conservative (conflict must be *lower* than the mix); it is not,
  so I do not proceed to BC. This is a proxy result, not a trained-BC result.
- Not a physical/contact claim; not an RL claim. RL gate holds (§15).

## Recommended next direction (for the user to decide — not started)

The mechanism points two ways, both without RL and without open-loop action labels in the loss:
1. **De-randomize the feedback teacher** — fix the CEM seed per state / average the elite / take the deterministic
   MPC action, so the settle labels become single-valued; re-measure conflict.
2. Given open-loop is *consistent in full-history space*, a **full-history BC** could be re-examined — but that would
   require revisiting the OPEN_LOOP_PLAN_ONLY quarantine, which is your decision, not mine.

## Provenance

`conflict_analysis.py` (size-matched k-NN + fixed-radius standardized), `conflict_result.json`, dataset manifest
(`feedback_dataset_manifest.json`, tarball SHA `c7bf9433…`). Obs contract SHA `6c84fa5b…`, bundle `6664ac459cca8f62`.
kato14 left clean.
