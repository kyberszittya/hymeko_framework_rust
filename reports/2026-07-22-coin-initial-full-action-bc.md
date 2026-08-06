# INITIAL_FULL_ACTION_BC_COMPLETE — phase-balanced BC clones accurately, delivers 1/9; failure isolated to transport-settle precision

**Created-at:** 2026-07-22 18:35 JST
**Branch:** recovery/coin-hymeko-bundle-and-results · bundle `6664ac459cca8f62` · dataset SHA `6da0089f…`

## Verdict

`INITIAL_FULL_ACTION_BC_COMPLETE`. The phase-balanced full-action BC trains to accurate cloning (val MSE ~6e-4,
every phase < 3e-3) and reaches **9/9 first-contact + grasp** from a true neutral reset, but strict-K=6 delivery
caps at **1/9** on the headline panel — below the §7 initial checkpoint (≥5/9). The gap is isolated, not diffuse:
it is transport-settle **precision on the BC's own state distribution**, the covariate-shift signature that
open-loop-suffix cloning provably cannot close. This routes to §8–10 DAgger.

## Dataset (§2–4, FULL_ACTION_DATASET_CERTIFIED)

96/120 train_query states certified (80%), 19334 transitions, all from natural neutral→K=6 replays (no injection).
Teacher: handoff 39, CEM-search 57. Phase-balanced sampler equalises every runtime phase to 1/7; STRICT_DWELL rises
2.81%→14.29%, BILATERAL 5.70%→14.29%. Tarball SHA-256 `6da0089f97a571d54320ecda9ff8b676bfd0ef9542d655262bd8d682596cfd1a`.

## BC training + standalone eval (§5–7)

Trajectory-level split (86 train / 10 val-loss; never split a trajectory across sets). 3 seeds, phase-balanced,
300 epochs × 200 steps. Deployment eval rolls `u = bc(node_features)` from neutral, graded by the strict K=6
certificate — no teacher, no scripted base.

| seed | train MSE | val MSE | first-contact | grasp | **DELIVER** | delivered |
|---|---|---|---|---|---|---|
| 0 | 4.7e-4 | 6.4e-4 | 9/9 | 9/9 | **1/9** | {1011} |
| 1 | 1.8e-4 | 5.4e-4 | 9/9 | 9/9 | **0/9** | {} |
| 2 | 3.6e-4 | 6.1e-4 | 9/9 | 9/9 | **1/9** | {1278} |

Per-phase val MSE (seed 2): APPROACH 2.2e-4, CONTACT_ACQ 1.1e-3, BILATERAL 2.9e-3, TRANSPORT 4.8e-4, TARGET_ENTRY
5.5e-4, SETTLING 7.3e-4, STRICT_DWELL 5.2e-4. Cloning is accurate in every phase — the failure is **not** a fit
failure.

## Where it dies (deployed rollout, seed-2 BC, per headline seed)

| seed | min_dtz | max_dwell | end_dtz | delivered |
|---|---|---|---|---|
| 1011 | 0.0323 | 0 | 0.1786 | no (reached near-zone, rolled out) |
| 1045 | 0.0641 | 0 | 0.1884 | no |
| 1164 | 0.0753 | 0 | 0.2171 | no |
| 1174 | 0.0744 | 0 | 0.1927 | no |
| 1202 | 0.1358 | 0 | 0.2666 | no |
| **1278** | **0.0169** | **6** | 0.0172 | **YES** |
| 1358 | 0.1654 | 0 | 0.2427 | no |
| 1447 | 0.1402 | 0 | 0.2119 | no |
| 1568 | 0.0689 | 0 | 0.0938 | no |

center_tol=0.02. The BC touches the coin on every seed and transports it *partway*; only 1278 reaches the strict
center. On the near-misses the coin approaches (1011 to 0.032) then **rolls back out** (end 0.18) — a settle/braking
precision failure, not an approach failure.

## Mechanism (measured vs inferred)

- **Measured:** approach + grasp solved (9/9); cloning accurate every phase; deployment fails in transport→settle,
  coin pushed partway and drifts out.
- **Inferred (consistent with P1/P4 diagnosis):** 57/96 demonstrations are **open-loop CEM suffixes** — a
  feedforward action *sequence*, not a feedback controller. Cloning obs→action from an open-loop rollout yields a
  policy that only tracks the exact demonstrated state path; under the BC's own compounding error the coin lands
  off that path where the cloned action is wrong. This is the SUCCESSFUL_SUFFIX_COVERAGE_GAP (P1: handoff delivers
  3/9 from expert grasp, 1/9 from BC grasp) made concrete at the settle stage.
- **Not claimed:** that this is a physical/contact ceiling. The strengthened teacher delivers 6/9 headline from the
  *E-approach* transition; the BC-reached transition is a different, un-searched state distribution. §4 forbids
  reading un-searched states as a physical wall.

## Next (§8–10 DAgger, RL still gated §15)

Success-certified DAgger **on the BC's own divergences**: roll the BC on train_query, and for each non-delivering
BC-reached transport state run the CEM suffix search **seeded from that reached state** (not the E-approach
transition), replay-certify (BC-approach-to-reached-state + certified suffix must deliver from neutral), add the
certified (obs, action) labels, retrain phase-balanced. Iterate until headline ≥6/9 or the stop criterion. This is
compute-bound → kato14, same harness as the strengthening.

## Provenance

BC checkpoints `experiments/2026_07_22_coin_v3_learning/bc_initial/bc_seed{0,1,2}.pt`
(SHA-256 7fedb45… / a2226b5… / dd2e2a2…), results `bc_headline_results.json`. Harness
`coin_delivery/full_action_bc.py` (`train_bc_phase_balanced`, `eval_bc_delivery`).
