# R11.6C — Exact-Zero Retrieval Composition

**Date:** 2026-08-06
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA:** `833b5e98`
**Verdict:** **`R11_6C_EXACT_ZERO_RETRIEVAL_COMPOSITION_PASS`**

---

## The milestone

From the exact-zero home `q=[0,0,0,0]`, with **no per-instance delivery CEM, no oracle, and no teacher at run
time**, the system carries the coin into the target zone under strict K6. The delivery θ comes only from the frozen
retrieval table shipped in R11.5R. This answers the program's central robotic question — *the system delivers the coin
autonomously* — and, just as importantly, the full-chain failure taxonomy names the **actual** next bottleneck.

Chain (only the delivery θ is new vs the certified pipeline):
```
exact-zero q=[0,0,0,0] → IC cert + admissibility → RRT reach → capture (1st certified grasp, NO oracle)
   → LIVE 30-D descriptor → frozen table θ = predict(x) → pure rollout_theta → strict K6 → certificate
```

---

## Result (panel: 44 train-like + 7 dev + 2 parked-stress; TEST split sealed)

Primary config `std_weighted3` (control `std_nearest` reported alongside):

| group | success rate | outcome classes |
|---|---|---|
| **train-like** | **1.000** (44/44) | 44 `exact_zero_delivered` |
| **dev** | **0.571** (4/7) | 4 `exact_zero_delivered`, 3 `delivery_failure_in_support` |
| parked | 0.000 (0/2) | 2 `capture_no_certified_grasp` |

- **48 `ExactZeroCoinDeliveryCertificate`s**; **0 nudge**; **0 safety regression**.
- **`max descriptor_drift = 0.00e+00` across all 53** — the live handoff descriptor reproduces the stored `x` exactly,
  everywhere. The plan's headline risk (descriptor drift) is fully closed; the reconstruct is deterministic at scale.
- Control `std_nearest` gives **identical** success rates (1.0 / 0.571 / 0.0, 48 certs); primary and control differ on
  only 2/53 scenarios (same success status). The weighting is near-inert in composition.

**Gate:** train-like 1.0 ≥ 0.50 ✓ · dev 0.571 > 0 ✓ · 48 ≥ 1 certificate ✓ · 0 safety ✓ → **PASS**.

### First `ExactZeroCoinDeliveryCertificate` (the milestone artifact)

`bank_c0_0`, seed 23: exact_zero_ic ✓ / teacher_free ✓ / cem_free ✓ / oracle_free ✓ / no_teleport ✓ / strict_k6 ✓ /
valid_delivery_mode ✓ / safe ✓; support_dist 0.0; **dtz 1.99 mm**; θ = (0.04, 0.45, −0.0725, 19.42, 12.40, 2.23).

---

## The taxonomy decides the next lever — and it is NOT densification

The composability round exists to locate the *actual* dominant failure before spending on any densification. It does:

**Zero of the failures are `RETRIEVAL_OUT_OF_SUPPORT`.** All 3 dev delivery failures are **in-support**:

| dev failure | support dist (radius 6.14) | gap_closed | dtz |
|---|---|---|---|
| `bank_c3_r7_a+45` | 2.805 (in) | 0.597 | 29.3 mm |
| `bank_c3_r9_a-30` | 3.072 (in) | 0.670 | 32.9 mm |
| `bank_c3_r9_a-15` | 2.698 (in) | 0.684 | 31.0 mm |

A *near* train demo exists for each (support 2.7–3.1, well under the 6.14 table radius), but its robust θ closes only
60–68 % of the gap — ~30 mm short of strict K6 — on the dev handoff. The single out-of-support delivery
(`bank_c2_+0.025_-0.025`, support 8.41) actually **succeeded**. So the delivery-stage bottleneck is **not** table
coverage.

**Readout (per the pre-registered decision tree):** the lookup returns a near, in-support θ, but the incoming handoff
differs enough that the θ under-delivers → the lever is **capture-handoff robustness / descriptor recalibration / a
delivery robust to residual handoff variation — NOT more coin–target points.** Densification is not indicated by this
round. (The 3 failures are all c3 far-angle geometry — the long-standing frontier.)

The 2 parked failures are `capture_no_certified_grasp` (upstream of delivery): the known capture-support gap
(`bank_c2_+0.015_+0.015`) and geometry limit (`bank_c3_r9_a-45`). These are a capture-stage frontier, also not a
retrieval-coverage problem.

---

## Faithfulness (the composition reproduces the isolated result)

Composition dev success **0.571 exactly matches** the isolated retrieval characterization's dev K6 (0.571), and
`drift = 0` everywhere. This proves the full chain is faithful: RRT reach + capture reconstruct the *identical* handoff
the delivery θ was certified on, so the isolated delivery result transfers to the composed chain with no degradation.
train-like 100 % is the strongest form of this — every frozen θ survives its live reconstructed chain.

---

## Interpretation

- **The composability question is answered: YES.** A fully teacher-free, CEM-free, oracle-free pipeline delivers the
  coin from exact-zero to strict K6 — 48 certified successes, 100 % on seen scenarios, 57 % generalization to unseen dev.
- **Densification is not the next lever.** The retrieval table already has near support for the failing dev scenarios;
  the residual is the delivery θ's sensitivity to the handoff, not coverage. The taxonomy (0 out-of-support failures)
  is the evidence that would have justified densification and does not.
- **Next levers (a design decision, HALT for review):** (a) capture-handoff robustness — reduce the dev handoff's
  deviation from the train handoffs the θ was certified on; (b) a delivery robust to residual handoff variation on the
  c3 far-angle geometry; (c) capture-support / geometry work for the 2 parked. All are targeted at the *measured*
  bottleneck, none is "perfect another surrogate."

---

## Boundaries & guards

- **TEST split sealed** — not evaluated here; may be spent once later, only after freezing whatever change comes next.
  Densification (if ever) must define a NEW untouched test panel (`test_frozen.json` records the current spent one).
- **Energy stays diagnostic** (frozen R11 contract → R11.8), a deliberate deviation flagged since R11.6A.
- No delivery CEM / no oracle / no teacher-lookup beyond the frozen table / no snapshot teleport / exact-zero IC — all
  asserted in the certificate; 48 valid certificates confirm the contract held.

---

## Files / tests / provenance

| File | Δ |
|---|---|
| `hymeko_rl/coin_delivery/exact_zero_composition.py` | `compose_one` / `reach_capture_descriptor` / `deliver_record`, `ExactZeroCoinDeliveryCertificate`, `CompositionOutcomeClass` (+ `RETRIEVAL_OUT_OF_SUPPORT`) (new) |
| `hymeko_rl/experiments/r11_6c_exact_zero_composition.py` | panel + fanout + gate (new) |
| `hymeko_rl/tests/test_r11_6c_composition.py` | 6 tests (classify / cert / gap_closed valid / gate) |
| `hymeko_rl/coin_delivery/delivery_bc/retrieval.py` | `support_distance` / `table_coverage_radius` (the OOD signal) |
| `reports/2026-08-06-r11-6c-composition/composition.json` | gate + all 53 records (new) |
| `docs/plans/2026-08-06-r11-6c-exact-zero-composition/` | §2 plan, 4-format, tectonic PDF (gitignored) |

- **Tests:** 17 pass across the R11.6C + retrieval suite; ruff + radon clean (all functions rank A/B).
  `valid_delivery = touched and gap_closed ≥ 0.5` calibrated on 4 certified deliveries
  (gap_closed ∈ [0.906, 0.979]; `forward_at_release` −7.1…+1.4 mm refuted).
- **CORE.YAML:** none. **Deps:** none. Reach / capture / descriptor / rollout / frozen-table reused read-only; the only
  new decision is the delivery θ from `frozen_policy.json`.
- **Env:** framework `.venv`, torch 2.12.0, macOS, CPU; 8-way fanout, `OMP_NUM_THREADS=1`, ~10 min wall,
  ~0.35 GB RSS/worker ≪ 16 GB. Deterministic (fixed per-scenario seed, `fresh_rig`, stable panel order).
- **Inputs:** frozen `frozen_policy.json` (table md5 `ecc823fe`, primary std_weighted3 / control std_nearest); panel from
  `dataset_b1` train+dev + the 2 parked; table radius 6.1431 (95th pct intra-table NN distance).

## Boundary

Exact-zero composition is now demonstrated. Remaining R11 ladder: R11.7 (other-shaped objects), R11.8 (Hamiltonian
energy). The next targeted lever (capture-handoff robustness vs the c3 far-angle delivery) is the immediate follow-up.
