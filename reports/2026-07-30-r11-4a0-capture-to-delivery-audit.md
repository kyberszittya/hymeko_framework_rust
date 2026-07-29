# R11.4A0 — capture→delivery contract audit and corrected delivery taxonomy

**2026-07-30 · branch `feature/r11-4a-target-conditioned-delivery-teacher` · parent R11.3 `5b3995e1` · diagnosis-only, no controller-behaviour change · no CORE.YAML items · no new dependencies**

## Summary

Investigating why the R11.4A target-conditioned delivery teacher couldn't recover a settle-failure, a measured audit of the capture→delivery interface overturned two hypotheses (including one of mine) and found a **small, local contract gap**, not a deep robotics problem:

> The pipeline accepted "the capture primitive ran" as "delivery-ready", even when the coin was **never bilaterally grasped**. The frozen downstream then either entered KINETIC/R2 (a genuine grasp→delivery) **or** the APPROACH servo merely *nudged* the ungrasped coin into the 20 mm K6 zone — a K6 flag with **no delivery-mode transition**.

The real delivery stack (KINETIC + `HANDOFF_RESET` + R2) is **valid and works**: handed a bilateral grasp it delivers tight K6 (8.76 mm). The blocker is **capture grasp reliability** — the RRT straddle isn't always inside the capture's grasp-success basin (**G_RRT ⊆ B_grasp**), and the grasp is **capture-seed-dependent**.

This report freezes the diagnosis + the corrected taxonomy **before** any behaviour change (Boundary 2 = `CONTACT_ACQUIRE_AND_HOLD`).

## The decisive measurements

| probe | capture contacts | delivery | outcome |
|---|---|---|---|
| `bank_c0_3` seed 1 | **2 (grasped)** | KINETIC ✓, `HANDOFF_RESET` fired ✓ | **K6 at 8.76 mm** (real R2 delivery) |
| `bank_c0_3` seed 0 | 0 (released) | never enters KINETIC | stuck APPROACH |
| `c1_diag` (R11.3 "K6") | 0 (never grasps) | never enters KINETIC | K6 at 19.14 mm — **nudge** (≤ `CENTER_TOL` 20 mm) |
| `c1_horiz` (failure) | 0 (never grasps) | never enters KINETIC | 32.83 mm — nudge fell short |

`CENTER_TOL = 0.02` (20 mm), so "strict K6" is a 20 mm settled zone that an APPROACH nudge can trip. The exact-replay showed the released captures **never reach bilateral contact at all** (`[0,…,0]`) — so it is neither a Zeno guard (no guard is even approached) nor a snapshot-selection bug (no pre-release grasped state exists to hand off).

## Hypotheses, settled honestly

- **Zeno-boundary (mine): refuted.** The `M3→M4` transition fires cleanly (`reset_fired=True`, `kin=True`) whenever handed a grasp. I elevated the deep interpretation before excluding the trivial one — the wrong instinct; the exact-replay discipline caught it.
- **Snapshot-selection (hand off the pre-release grasped state): refuted for these cases.** The failing captures never establish bilateral contact, so there is no grasped state to select; the terminal *is* the right endpoint when a grasp exists.
- **`K6_SUCCESS_WITHOUT_DELIVERY_MODE_TRANSITION`: confirmed.** Some R11.3 "K6" are ungrasped nudges, not deliveries.

## Corrected taxonomy + derived reclassification

`hymeko_rl/coin_delivery/delivery_teacher/delivery_contract.py` adds the **`DELIVERY_READY_GRASP_CERTIFICATE`** (bilateral contact held for N steps, bounded relative slip, bounded coin speed, safe, continuous episode) — the precondition to start/score the downstream; otherwise `CAPTURE_INCOMPLETE`. And the 5-class taxonomy: `K6_WITH_VALID_DELIVERY_MODE`, `K6_WITHOUT_DELIVERY_MODE_TRANSITION`, `CAPTURE_TO_DELIVERY_REGRASP_FAILURE`, `DELIVERY_FAILURE_AFTER_VALID_GRASP`, `SETTLE_FAILURE_AFTER_VALID_GRASP`.

A **versioned derived reclassification** of the R11.3 bank (`derived_reclassification.json`, proxy `contacts==2 ⇔ grasped ⇔ delivery-mode`, verified on the exact-replay set) — the R11.3 bank is **not rewritten**:

| corrected class (scenario-best, 64) | n |
|---|---|
| K6_WITH_VALID_DELIVERY_MODE | **7** |
| K6_WITHOUT_DELIVERY_MODE_TRANSITION (nudge) | 1 |
| CAPTURE_TO_DELIVERY_REGRASP_FAILURE (never grasped) | 30 |
| DELIVERY_FAILURE_AFTER_VALID_GRASP | 12 |
| SETTLE_FAILURE_AFTER_VALID_GRASP | 14 |

**The former "8/64 K6" is really 7 genuine deliveries + 1 nudge.** And **26/64 scenarios did grasp but the delivery/settle fell short** — those are genuinely delivery-teacher-addressable (once certified grasped), which is why the delivery+settle teacher (already built, non-invasive-validated) is not wasted; it is simply gated behind a certified grasp now.

## Files

| file | role |
|---|---|
| `delivery_teacher/delivery_contract.py` | DELIVERY_READY_GRASP_CERTIFICATE + 5-class corrected taxonomy + derived-reclassification mapping |
| `delivery_teacher/regrasp_characterize.py` | instrumented frozen delivery: reaches-KINETIC / bilateral-dwell / addressable classification |
| `delivery_teacher/phase_energy.py` | non-invasive phase-marked energy ledger (bit-exact hook, validated) |
| `delivery_teacher/handoff_reset.py` | the certified `CAPTURE_TO_DELIVERY_REGRASP` transition capture |
| `delivery_teacher/{solver,record}.py` | target-conditioned delivery+settle teacher (built; gated behind a certified grasp) |
| `experiments/r11_4a0_reclassify.py` | derived reclassification of the R11.3 bank (no rewrite) |
| `tests/test_r11_4a_delivery_teacher.py` | contract + taxonomy + non-invasive + known-seed mechanism replay |

## Tests & static gates

- **`ruff` clean · `radon cc -a -nc` no C+ block · `mypy --strict` clean** on the pure delivery modules (`delivery_contract` etc.); the mujoco↔teacher modules are the boundary.
- **15 fast tests pass**; slow physics tests cover the non-invasive gate + the known-seed grasp-split mechanism (`bank_c0_3` seed 1 grasped→KINETIC→K6 vs seed 0 released→stuck).

## Verdict + forward plan

**`R11_4A_DELIVERY_STACK_VALID_BLOCKER_IS_CAPTURE_GRASP_RELIABILITY`** — the delivery mode delivers tight K6 whenever handed a bilateral grasp; the R11.3 "delivery failures" and the nudge-K6 stem from the capture failing to establish/hold a grasp from the RRT straddle. This is a small, local interface gap — a very human robotics lesson: *grasp, hold, then hand off.*

**Boundary 2 (next, gated on this commit):** an explicit deterministic **`CONTACT_ACQUIRE_AND_HOLD`** completion primitive (controlled symmetric squeeze under existing torque/slew limits → independent L/R contact detection → damp relative coin motion → consecutive bilateral-contact dwell → low slip + low coin speed → `DELIVERY_READY_GRASP_CERTIFICATE` → KINETIC). No CEM, no RL, no snapshot injection, no target-directed push, no canonical joint-state hack. First close the known `bank_c0_3` seed-0→seed-1 split, then re-measure the 64-scenario grasp/delivery-addressable rates. Only after that does the delivery+settle teacher resume, restricted to certified delivery-ready states. No BC/RL/Hamiltonian.

## Provenance

Parent `5b3995e1`. Derived reclassification proxy verified on the exact-replay set (4 scenarios). CORE.YAML items: none. New dependencies: none. Env: Python 3.11 / mujoco 3.10.0 / macOS-arm64 (CPU). Deterministic (fixed seeds).
