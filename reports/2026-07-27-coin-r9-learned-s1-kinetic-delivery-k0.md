# K0 — the KINETIC snapshot / replanning contract for the first learned s1 delivery

**2026-07-27 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (seed 14250) · held-out s4/s7 untouched · f1–f4 SEALED**

## Summary

This is milestone **K0** of the plan `docs/plans/2026-07-27-coin-r9-learned-s1-kinetic-delivery/` (the first learned s1
delivery: `frozen APPROACH → learned KINETIC coin-following → G0 release → frozen coast → frozen K6`). K0 builds the
**contract** the learning stack (K1–K4) sits on — no training — and answers the one question that de-risks the whole campaign:
**is a legal receding-horizon teacher a strict-K6 positive control when replanned from the state the learner will occupy?**

**It is.** A full deterministic CEM replanned from the frozen KINETIC-entry state delivers strict K6 on s1
(`entry+full_cem`: dwell 32, min_dtz **15.7 mm** ≤ 20 mm zone, motion-contract held). The teacher can label the KINETIC
transport segment. Verdict: **`REPLAN_FROM_KINETIC_ENTRY_DELIVERS_S1`**.

**This is the interface-side positive control for a learned skill, NOT a learned delivery.** What runs in the loop today is
the teacher/CEM replan, not a policy — so the milestone is deliberately *not* `LEARNED_S1_DELIVERY`. What K0 establishes is
that the learning target is physically and interface-side reachable: from the exact state the learner will occupy, a legal
causally-replanned torque primitive exists that reaches strict K6. That rules out three unpleasant possibilities — the KINETIC
entry is not a hopeless state; no new APPROACH is needed; the missing skill genuinely lives in the *local transport/release
decision* (and is state-dependent feedback, not a cradle-specific constant θ — the squeeze→0 result below shows the delivering
action *recognises the moving state and must not re-clamp*).

Delivered, tested, and gated:

- **`TransportSnapshot`** — a duck-typed, deterministically-branchable snapshot of an *arbitrary* mid-transport state (the θ
  teacher's `CradleSnapshot` interface without the B_v straddle gate), so the proven teacher replans from a moving, gripped
  coin. `branch()` is bit-identical across calls.
- **`roll_until`** — a higher-order capture loop that mirrors the frozen `velocity_rollout` step kernel *bit-for-bit* (tested)
  but exposes the mid-loop `prev_tau` the frame-hook cannot see.
- **`freeze_kinetic_entry`** — the deterministic, content-hashed frozen KINETIC-entry state on s1 (`state_hash
  ce343c478d2a0cb7`, step 4, dtz 75.75 mm, v_par 0.249, admissible).
- **`kinetic_observe`** — the single canonical 41-D policy-input extractor, **batch == streaming** (tested bit-for-bit) with
  **no future / teacher / K6 leak** (schema + purity tested).
- **`receding_horizon_relabel`** — replan the teacher from a transport snapshot, export **only the first causal torque
  action** as a label, with the strict-K6 verdict. The teacher's lookahead is baked into the label, never into the policy input.

## The K0 measurement — where the strict-K6 positive control lives

`python -m hymeko_rl.experiments.coin_kinetic_positive_control` (deterministic; 1 seed; 56.3 s; peak RSS 0.23 GB). Artifact:
`reports/2026-07-27-coin-r9-learned-s1-kinetic-delivery-k0/positive_control.json`.

| start × budget | K6 | dwell | min_dtz | peak_qdot | reading |
|---|---|---|---|---|---|
| handoff + canonical θ | **True** | 24 | 18.6 mm | 2.00 | proven control **reproduced** (scout: 18.5 mm) |
| handoff + full CEM | **True** | 24 | 18.6 mm | 2.00 | teacher delivers from the at-rest handoff |
| entry + canonical θ | False | 0 | 54.4 mm | 2.02 | from-rest θ **stalls** from the moving entry (firm-grip clamp) |
| **entry + full CEM** | **True** | 32 | **15.7 mm** | 2.01 | **teacher replanned from the KINETIC entry DELIVERS** |
| entry + warm local CEM (narrow) | False | 0 | 20.7 mm | 2.01 | narrow window near canonical is *close* but misses |
| handoff + KINETIC scaffold | False | 0 | 50.6 mm | 2.01 | matches G0's ~48–51 mm hand-tuning wall |

**The delivering θ from the entry is `[0.0, 0.271, −0.057, 16.54, 8.80, 3.46]` — squeeze = 0.** From the moving entry the
delivering strategy **drops the grip entirely** (no clamp) and pushes, letting the coin slide-and-coast to a moving release —
exactly the "light-contact kinetic sliding transport → moving release → passive coast" mechanism the campaign brief predicts.
This also explains the two negatives: the from-rest canonical θ (squeeze 0.05) and the narrow warm-window (which stays near
squeeze 0.05–0.065) apply a light-but-nonzero grip that clamps the moving coin and stalls it. The full box-wide CEM finds
squeeze → 0. **Calibration for K1:** warm-start the relabel CEM from the *entry-delivering* θ (squeeze ≈ 0), not the from-rest
canonical, and use a box-wide (not narrow) search — the narrow window reached 20.7 mm, just outside the 20 mm zone.

This is a first-pass measurement on 1 seed; it is reported as an established **positive control** (a delivering trajectory
demonstrably exists from the frozen entry), not as a tuned relabeler. K1 calibrates the relabel budget/warm-start against it.

## Files touched (all new; no existing file modified)

| file | lines | role |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_contract.py` | 338 | the K0 contract module |
| `hymeko_rl/experiments/coin_kinetic_positive_control.py` | 122 | the positive-control smoke |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | 141 | unit + integration tests |
| `docs/plans/2026-07-27-…/plan.{tex,tikz,mmd,pdf}` | 201/65/56/(pdf) | the K0–K4 plan (4 formats) |
| `reports/2026-07-27-…-k0.md` (this) + `…-k0/positive_control.json` | — | report + artifact |

No existing function/signature was changed → no regression surface in the frozen scaffold (the default
`HybridApproachController` still has `kinetic_transport=False`, update-zero; K6 monitor, motion contract, physics untouched).

## CORE.YAML items touched

**None.** Verified against `CORE.YAML`: all new code is under `hymeko_rl/` and `tests/`/`docs/plans/`/`reports/`
(allowlisted); no `hymeko_core`/`hymeko_query`/`hymeko_client`/`hymeko_daemon`/`parser` file and no pinned dependency touched.
No new/removed dependencies.

## Test results

`pytest -p no:randomly hymeko_rl/tests/test_coin_kinetic_contract.py` — **8 passed in 94.05 s**, peak RSS 0.26 GB.

- **Unit (pure, no physics):** `test_feature_schema_frozen_and_no_future_leak` (41-D schema, unique, no forbidden
  future/teacher/K6 name), `test_transport_admissibility_algebra`, `test_windowed_delivery_cfg_warmstarts_at_canonical_theta`.
- **Integration (fast committed-bank s1 snapshot):** `test_roll_until_mirrors_velocity_rollout_trajectory` (roll_until ==
  velocity_rollout coin_trace, bit-for-bit — proves the capture loop is a faithful mirror, not a reimplementation),
  `test_transport_snapshot_branch_bit_identity`, `test_kinetic_observe_batch_equals_streaming` (batch == streaming, bit-for-bit
  at 4 sampled steps), `test_freeze_kinetic_entry_deterministic_and_admissible` (stable hash + admissible dual contact),
  `test_receding_horizon_relabel_first_action_only_deterministic` (length-4 slew-normalised first action only; deterministic).

Coverage rule (§3): every new public/private function is exercised by a new test; the batch==stream and
roll_until-mirror tests are regression tests that would fail against a leaky or divergent implementation. No behaviour of any
existing function was changed (no regression test owed elsewhere).

## Performance results

| quantity | measured | budget (plan) | status |
|---|---|---|---|
| positive-control smoke wall | 56.3 s | ≤ 5 min | ✅ |
| smoke peak RSS | 0.23 GB | < 1 GB (cap 16) | ✅ |
| unit+integration suite wall | 94.0 s | < 120 s | ✅ |
| suite peak RSS | 0.26 GB | < 1 GB | ✅ |
| one governed rollout | ~0.8 s | (anchor) | — |
| full teacher CEM (pop32×iters6) from a snapshot | ~17 s | (anchor) | — |
| acquire s1 certified straddle | 21.2 s | (anchor) | — |

**Downstream reconciliation (per the >2× wall-estimate halt rule).** The plan estimated ~5–10 s/label for a warm local
relabel; the smoke shows the *narrow* warm window does **not** deliver, while a full/wide CEM (~17 s) does. So the honest
relabel cost is **~17 s/label** with a box-wide CEM, or cheaper if warm-started from the now-known entry-delivering θ (to be
measured in K1). A ~200-label bank ⇒ ~1 h, checkpointed — I will re-measure the warm-started-from-entry-θ cost at the top of
K1 before committing the bank run, not carry the 5–10 s estimate forward.

## Static analysis

- `ruff check` — **All checks passed** (new files).
- `radon cc -a` on `kinetic_contract.py` — **average A (2.83)**; `radon cc -nc` — no block graded C or worse (all under the
  warn-at-10 / fail-at-15 cyclomatic budget; longest function well under 80 LOC).
- `mypy --strict` on `kinetic_contract.py` — clean except the codebase-wide `mujoco` `import-untyped` note (present in all 40
  modules; mujoco ships no stubs). **Suppressions introduced:** three `# type: ignore[arg-type]`, each scoped to a single
  teacher-call line in `receding_horizon_relabel`, with an inline reason — `TransportSnapshot` structurally duck-types the
  `CradleSnapshot` rollout interface (`branch/stack/prev_tau/lo/hi/q_hold`); the alternative (re-annotating the shared,
  effectively-frozen `forward_displacement` teacher signatures to a Protocol) was rejected as a shared-file edit for a
  behaviourally-inert typing concern. No `#[allow]`/`# noqa`/broad-except introduced. §6.5 anti-patterns: none introduced
  (dataclasses + small pure functions + one higher-order `roll_until`; no Cartesian API, no string-typed config, no globals —
  the only module state is `const`-like frozen anchors).

## Experiment provenance

- **Git SHA:** `c7bf6274558146be9f6d595f487c96a13e4334a1` (`recovery/coin-r9-causal-residual-delivery`). Working tree dirty:
  the new files above are untracked (not yet committed); no tracked file modified.
- **Env:** Python 3.11.15, mujoco 3.10.0, numpy 2.4.6, macOS-26.5.2-arm64 (Apple Silicon), venv
  `…/hymeko_framework_rust/.venv`. CPU host quiet during the measurement window; single-process, no GPU.
- **Seeds:** cradle seed 14250 (s1); teacher CEM seed `20260727` (frozen in `ForwardConfig`); all rollouts deterministic.
- **Dataset:** `reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json` (BANK), md5 `ee9bd4958e4ab544`.
- **Frozen KINETIC entry hash:** `ce343c478d2a0cb7` (reproducible from the committed code).
- **Artifact:** `reports/2026-07-27-coin-r9-learned-s1-kinetic-delivery-k0/positive_control.json` (the 6-row verdict table).

## Open issues / follow-ups (K1 — small and step-wise, not a data factory)

1. **Re-measure relabel cost FIRST (before any bank run):** run 8–16 representative relabels warm-started from the
   entry-delivering θ `[0.0, 0.271, −0.057, 16.54, 8.80, 3.46]` with a **box-wide** (not narrow) legal window; report cost +
   success-rate; reconcile the ~17 s/label estimate (the >2× wall-estimate halt rule). Only then size the bank.
2. **Step-wise bank (diversity, not sample count):**
   - **K1-A — 32 labels:** schema, diversity, replan success-rate, cost audit.
   - **K1-B — 128 labels:** first feedback-clone + closed-loop smoke on s1.
   - **K1-C — 256–512 labels only if needed:** DAgger-visited states (not merely more teacher-neighbour samples).
   The load-bearing bank metrics are **not** the count: how many distinct physical states; how many successful replans; how
   wide the v_par / slip / force / geometry range; how much the first optimal action varies across them. If 128 strongly
   varied first-action labels already carry a clone to the close-and-moving release manifold closed-loop, no data factory is
   needed.
3. **Label / actor-input firewall (already defended in K0, re-assert in K1):** positive warm-start = entry-delivering θ;
   search = a sufficiently-wide legal box; label = first executed action only; actor input = the 41-D causal observation only.
   The CEM trajectory and terminal K6 result must never enter the actor feature (tested: `kinetic_observe` cannot read them).
4. **Feedback bank construction (K1):** branch perturbed *control* from the frozen entry (v_floor, kinetic_squeeze, L/R
   imbalance, prev-action noise, teacher-trace neighbourhoods) through real physics — physically legal, snapshot-branched,
   **not** state-edited — relabel each admissible state, log the rejection rate.
5. **Entry-timing question:** the frozen entry is captured at the first KINETIC step (step 4), where `prev_tau` still carries
   the APPROACH's firm grip — the exact condition under which the from-rest canonical θ clamps. K1 should test whether a
   slightly-later entry (grip already relaxed) widens the deliverable relabel set, or whether the squeeze→0 relabel handles it.
6. **G0 landing guard (deferred):** promote the inline k-NN LOO landing predictor (`coin_release_guard.py --g0`) to a fittable
   `LandingGuard` object once the bank exists (the plan's release-guard input, "if safely integrable").

Not opened: s4/s7 (validation-only), f1–f4 (SEALED). No five-cradle atlas. No hand-tuned profile tuning (the stop conditions).

## Gate ladder from here

- **`REPLAN_FROM_KINETIC_ENTRY_DELIVERS_S1`** *(this milestone — done)*: a legal causally-replanned torque primitive from the
  frozen entry reaches strict K6. Interface-side positive control for the learned skill; the teacher still runs the loop.
- **`LOCAL_KINETIC_FEEDBACK_SKILL_PASS`** *(next real gate, not BC-loss / action-R²)*: the learned actor, running **without the
  teacher**, from entry-neighbourhood perturbations stably holds-or-correctly-releases the moving coin, improves reach to the
  release manifold, does not re-clamp, and reaches strict s1 K6 at least once.
- **`FIRST_LEARNED_S1_K6_DELIVERY`** *(campaign gate)*: the learned KINETIC policy is the sole thing in the loop and delivers,
  with the full success-gate (safety, deterministic replay, provenance, video, event-aligned trace).

## Verdict

`REPLAN_FROM_KINETIC_ENTRY_DELIVERS_S1` — the K0 contract is built, tested, and gated; the receding-horizon teacher is a
strict-K6 positive control from the frozen KINETIC entry (min_dtz 15.7 mm), so the KINETIC coin-following skill has a physical
expert to imitate. Not yet a learned delivery (the teacher/CEM still runs the loop). K1 (feedback bank) is unblocked.
