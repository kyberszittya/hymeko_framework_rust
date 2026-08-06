# FULL_ACTION_BC_COMPETENCE_BLOCKED — the strengthened teacher is open-loop; its labels are not a function of the observation, so no reactive/short-memory BC clones the delivery

**Created-at:** 2026-07-22 19:20 JST
**Branch:** recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62`
**Compute:** dataset generation + DAgger on KATO14 (32 cores); BC training/eval on Mac (Apple Silicon, CPU torch).

## Verdict

`FULL_ACTION_BC_COMPETENCE_BLOCKED`. The best deployed full-action BC reaches **3/9 headline** — and only when
trained on the **feedback-policy (handoff) demos alone**; the full mix (feedback + open-loop search) *regresses* to
0–1/9, and observation memory does not help. 3/9 is below the §11 gate (≥6/9 headline, ≥60% final-test). The block
is **not** a physical ceiling — the strengthened teacher delivers 6/9 headline and 24/30 held-out by open-loop
search (`STRENGTHENED_CANONICAL_DYNAMIC_EXPERT_PASS`). It is a measured **structural mismatch between the teacher and
the policy class**: the 57/96 open-loop CEM-suffix demonstrations are a feedforward, time-indexed *sequence*, not a
state-feedback law, so their labels are not a function of the observation and are un-clonable (and actively
poison the mix). The feedback (handoff) demos *are* clonable — the reactive BC reproduces the handoff teacher's own
~3/9 coverage exactly. Reaching 6/9 requires converting the open-loop-only states to a **feedback** expert, or RL.
Isolated by five discriminating tests, not a first-pass negative. RL remains gated (§15); this report hands the user
the mechanism and the concrete unblock paths.

## What was run (all per the §1–13 protocol)

- **Seed banks frozen** (§1): train_query 6000–6119 (120), validation 7000–7029 (30, no labels), final_test
  8000–8049 (50, **untouched**), all disjoint + committed. Headline 1011–1568 is the historical regression panel.
- **Dataset** (§2–4, `FULL_ACTION_DATASET_CERTIFIED`): 96/120 train_query states certified as natural neutral→K=6
  replays (no injection); 19334 transitions; teacher handoff 39 + open-loop CEM-search 57; all 7 runtime-predicate
  phases populated; phase-balanced sampler equalises each phase to 1/7.
- **Initial BC** (§5–7, `INITIAL_FULL_ACTION_BC_COMPLETE`): phase-balanced, 3 seeds. Clones accurately (val MSE
  ~6e-4, every phase < 3e-3), 9/9 first-contact + grasp from neutral, but **1/9 headline**. Per-seed rollout: the
  coin is pushed *partway* then **rolls back out** of the strict zone.
- **DAgger iter-1** (§8–10): searched a strict-K=6 corrective suffix from the BC's **own** reached transition state
  (not the E-approach's), replay-certified from neutral; 70/120 certified, 7 BC-already-delivers, 40 search-failed,
  3 replay-fail. Retrain on base ∪ DAgger (166 traj): **0/9 headline**, and val loss **rose** 6e-4 → 1.8e-3.

## BC delivery across configurations (best of 3 seeds; headline = held-out 1011–1568)

| configuration | data | headline | validation (30) | note |
|---|---|---|---|---|
| initial reactive | base (96) | 1/9 | 2/30 | clones val 6e-4, 9/9 grasp |
| + DAgger iter-1 | base ∪ 70 | **0/9** | 2/30 | val loss ↑ 6e-4→1.8e-3 (label conflict) |
| uniform sampling | base ∪ 70 | 1/9 | 1/30 | phase-balancing not the cause |
| frame-stack k=3 | base ∪ 70 | 1/9 | — | observation memory does not help |
| **handoff-only** | 39 feedback | **3/9** | 2/30 | feedback demos clonable; best deployable reactive BC |
| strengthened teacher (reference) | — | 6/9 | — | open-loop search; DEMONSTRATION only, not deployable |

## Discriminating tests (the mechanism, measured not asserted)

| # | test | result | what it rules in/out |
|---|---|---|---|
| 1 | observation audit | `node_features` (48) carries coin **position** but **no coin velocity** | settle is a POMDP for a reactive policy: same coin position, opposite braking depending on unobserved motion |
| 2 | k-NN action spread, transport phase | base 0.111 → base∪DAgger **0.148** (+33%) | the DAgger (open-loop) labels are **more** one-to-many at fixed obs — they conflict |
| 3 | uniform vs phase-balanced sampling | 1/9 vs 0/9 (≈equal) | phase-balancing is **not** the cause |
| 4 | frame-stack k=3 (recovers coin velocity from history) | **1/9** headline, k-NN spread unchanged (0.080→0.083) | observation **memory does not resolve it** — the ambiguity is not (only) missing velocity |
| 5 | handoff-only demos (feedback-policy) vs mix | **3/9** (2 of 3 seeds; union {1011,1045,1174,1447,1568}) vs mix 0–1/9 | open-loop demos **poison** the clone; feedback demos **are** clonable, up to the handoff's own ~3/9 coverage |

**Isolated mechanism.** The strengthened teacher's transport is 57/96 **open-loop CEM suffixes** — a feedforward,
time-indexed action *sequence*, not a state-feedback law. Cloning it as any observation(-history)→action map is
ill-posed: two trajectories passing through the same observed state (even with velocity, via frame-stacking) at
different suffix-times prescribe different actions, so MSE regression averages the conflict → the coin is pushed
partway and rolls out (test 1's failure signature). DAgger adds *more* open-loop labels → more conflict (test 2) →
worse (0/9). Memory does not help (test 4) because the ambiguity is in the **teacher** (feedforward), not the
observation. This is the SUCCESSFUL_SUFFIX_COVERAGE_GAP (P1) resolved to its root cause.

## Non-claims (§4, §9)

- **Not** a physical/contact ceiling: search delivers 6/9 headline, 24/30 held-out.
- **Not** a fit failure: every configuration clones to val MSE 1–2e-3 and reaches 9/9 grasp.
- **Not** a first-pass negative: four policy/data configurations + five discriminating measurements converge.

## Unblock paths (for the user to direct — no RL launched, §15 holds)

1. **Feedback transport expert for DAgger** (within "no RL"): re-solve a short-horizon CEM at each transport step
   (receding-horizon / MPC) and clone the FIRST action — this makes the expert a function of state, clonable by a
   reactive BC. Cost: a search per step (expensive; needs a bounded-horizon approximation on KATO14).
2. **RL** (§15-gated): a genuine state-feedback policy with its own memory handles both the open-loop-teacher issue
   and the observation POMDP. The BC route is now *measurably* exhausted — the documented precondition for lifting
   the gate is met, pending the user's decision.
3. **Observation fix + feedback expert**: add coin velocity to `node_features` (a bundle change → §1 escalation) AND
   use path (1); the observation fix **alone** is insufficient (test 4).

## Provenance

Results + logs under `experiments/2026_07_22_coin_v3_learning/bc_configs/` (committed text). Best deployable reactive
BC = handoff-only seed 0 (3/9), checkpoint SHA-256 `cc8c31fd926b1a0058c19e7e905c899a6a6f6855e0592ca4c5f544632a34c2b1`
(binary local, referenced by SHA). DAgger labels tarball SHA `36d8ac4d52194cdd641e02d534bc75fba48eb1f6b6734e312121daa70fe13d5c`,
dataset tarball SHA `6da0089f97a571d54320ecda9ff8b676bfd0ef9542d655262bd8d682596cfd1a`. Driver
`experiments/2026_07_22_coin_v3_learning/train_full_action_bc.py`; harness
`coin_delivery/{full_action_bc,coin_v3_dagger,coin_v3_suffix_search,coin_v3_dataset_gen}.py`. All demos
replay-certified from neutral; the final_test bank (8000–8049) was never touched. Known harness nit: the
`--eval-validation` reload rebuilds `FullActionBC()` at the default 48-dim, so it skips the k=3 validation number
(the in-loop headline eval used the live 144-dim model — 1/9 stands).
