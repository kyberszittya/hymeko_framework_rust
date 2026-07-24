# Overnight digest — genuine-contact V4 gate + 6D realistic-motion regression

**Date:** 2026-07-25 (overnight run)
**Branch:** `feat/architectural-assimilation-v1`
**Directive:** OVERNIGHT — genuine-contact V4 freeze + C2 ablation + parallel realistic-motion 6D regression.
**One-line outcome:** the *real* V4 harness bug was found and fixed; genuine contact is now measured; the actuation
stack is **motion-legal where contact occurs** but a single tip on a low-friction disk **does not sustain continuous
contact** → **V4 not frozen, C2 not launched, halted for one design decision**. The 6D realistic-motion regression ran
in parallel.

---

## 1. Genuine-contact gate validity — the load-bearing correction

The **prior V4 negative verdict is VOID.** Root cause (found + fixed):

- The old gate drove the arm with `GovernedArm.pd_step`, which calls `mujoco.mj_step` **directly** and **bypasses the
  coin env's `step_ablation`**. `_planar_metrics` (contact state) is only recomputed inside `step_ablation`, so it stayed
  stale → `contact_frames == 0` for **every** configuration → the "sustained-contact" gate actually measured a
  **free-space swing**. That whole negative proved nothing about contact.
- Corrected driver = `motion_robust_carry` (goes through `step_ablation`, so contact state updates), with a new
  **delivery-agnostic** `sustained_press` mode: acquire the coin, then keep pushing it in a **fixed, zone-independent**
  direction (flipped every 30 steps to keep it in reach). K6 / zone are **never** read in the gate or the selection.

**Genuine contact is now real:** normal force up to **17 N**, contact-conditioned joint velocity measured, governor
active ~50 % of sub-steps, active braking firing. Contact established on **3 / 4 states** (`any_contact 3/4`).

## 2. V4 verdict — clean partial, halt for a decision

```
VERDICT: GENUINE_CONTACT_INTERMITTENT_MOTION_LEGAL_BUT_NOT_SUSTAINED__HALT_FOR_GATE_DECISION
```

| over_hard_brake | sustained (≥10f/≥0.25) | any-contact | peak vel **in contact** (≤3.45) | integ | recover | Fn (N) | legal-in-contact |
|---|---|---|---|---|---|---|---|
| 1.5 | 1/4 | 3/4 | 1.92 | 0.0 | 19f ok | 17.2 | **True** |
| 2.0 | 1/4 | 3/4 | 1.92 | 0.0 | 20f ok | 17.2 | **True** |
| 3.0 | 1/4 | 3/4 | 1.94 | 0.0 | 21f ok | 17.2 | **True** |

- **Where contact occurs, the stack is motion-legal:** contact-phase peak joint velocity **1.92–1.94 rad/s ≪ 3.45**
  (the pre-declared 3.0 × 1.15 transient gate), integrated overspeed **0.0**, returns below the safe band in **~20
  control steps**, no runaway. This *corroborates* the earlier active-braking finding (6.71 → 2.28) — the governor keeps
  the contact regime legal.
- **But contact is NOT sustained:** only **1 / 4** states reach ≥10 contact frames (or ≥0.25 fraction). A **single tip on
  a low-friction disk cannot sustain continuous contact** — the coin squirts away. This is *physically consistent* with
  the coin task itself being **push-and-coast / intermittent-contact**, not grasp (`project-coin-neutral-start-delivery`:
  "Coin Delivery ≠ Grasp, push/coast VALID, no force-closure").
- The pre-declared **sustained** criterion (≥10 frames / ≥0.25 fraction) is **mismatched to that intermittent regime**.
  Relaxing it now to force a pass would be *changing a threshold after seeing results* — forbidden. So the gate halts.

## 3. Full V4 sweep + selection rationale

- Swept **only** `over_hard_brake ∈ {1.5, 2.0, 3.0}` (the single new intervention); `kv` held at the V3-frozen value.
  `over_hard_brake` is **dormant below `qdot_hard`** (verified in `test_over_hard_brake_adds_active_braking_above_hard`),
  so the V3 free-space / agility / tracking / reversal / settling gates are **inherited by construction** — no re-sweep.
- Pre-declared thresholds (fixed before the run, recorded in the artifact): nominal hard **3.0 rad/s**, transient
  tolerance **15 %** → absolute gate **3.45 rad/s**, integ **0.4**, genuine **≥10 frames or ≥0.25 fraction**, recovery
  **≤200 steps**, saturation **≤0.5**.
- **No configuration passes the sustained gate** → no lexicographic selection is made → **V4 is not frozen.**
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/dynamics_contract_v4.json` (full per-state table, contact
  force, recovery, governor/brake/saturation fractions).

## 4–6. C2 phase ladder + mechanism

**Not run.** The capability gate requires an immutable frozen V4 first; V4 did not freeze. C2 launching would violate the
directive ("Launch C2 only after V4 is committed and immutable" / "Do not launch C2 on a failed contract"). The C2
harness (`hymeko_rl/experiments/coin_c2_ablation.py`) is built, committed, and ready the moment a contact contract
freezes.

## 7–8. 6D-0 / 6D-1 realistic-motion regression — **VALIDATED**

```
VERDICT: MULTIMODAL_POLICY_SEARCH_VALIDATED_UNDER_REALISTIC_MOTION_LIMITS
```

**Executor realism (measured):** under `slew_joint_vel_limit = 2.0 rad/s` with physical horizons (via **208**, goal
**248** env-steps, derived from joint-budget / (slew·ctrl_dt) + accel/brake + settling), the executor's **sustained**
peak joint velocity is **2.069 rad/s** (at the cap; mean **0.735**). The raw peak 6.168 is a **2–4 step startup servo
transient** (far target from rest), which the motion contract treats as diagnostic-not-gated — analogous to the qacc
transients. Eligibility was **RECOMPUTED under the limited executor** (not reused from the unlimited run): rate **0.259**.

**6D-1 seed-hardening (6 paired seeds, fresh panels under slew, equal budget):**

| metric | value |
|---|---|
| per-seed Δ@B12 (K-mode − single-head) | +0.231, +0.455, +0.143, +0.500, +0.308, +0.400 |
| seeds positive | **6 / 6** |
| seed-median Δ (IQR) | **+0.354** [0.25, 0.44] |
| hierarchical seed→state bootstrap | median **+0.337**, 95 % CI **[0.192, 0.493]** (lower bound > 0) |
| critical pair (every seed) | K-mode ≫ single-head@wrong (0→6, 2→10, 1→9, 4→11 …) |
| budget curve K1 (single-head) | 0.348 → 0.515 → 0.545 → 0.561 → 0.576 |
| budget curve K-mode | 0.348 → 0.773 → 0.833 → **0.909** → 0.924 |
| timeout (separate failure category) | 0–1 per seed (physical-horizon-appropriate) |

The **strong multimodal advantage survives** the velocity limit + physical horizons — as predicted
(`project-option-rl-structured-temporal-runtime`: the advantage is about geometrically-separated **basins**, not speed).
Figure: `reports/2026-07-24-se3-obstacle-6d1/obstacle_6d1_realistic.png`.

**Obstacle-shift generalisation (fresh geometry under slew):** the K-mode edge is present but **weaker** under a bounded
geometry change (Δ@B12: wider 0.0, taller +0.143, narrow +0.125) at low eligibility (0.10–0.13) — the headline holds on
the hardening panel; the geometry-shift edge is a softer PILOT-level signal, flagged not overclaimed.

**6D-0 caveat (honest):** the quick pose-reach probe I added ran a *straight-line* via inside the **obstacle** env, so it
is not a clean obstacle-free 6D-0 test (0/12, 3 timeouts — the straight line intersects the box / the horizon can't close
from it). The dedicated `se3_reach_6d0` under slew is the correct 6D-0 rerun; not a real 6D-0 regression. Artifact:
`reports/2026-07-24-se3-obstacle-6d1/obstacle_6d1_realistic.json`.

## 9. Claims and non-claims

**Claimed (measured):**
- The old V4 negative is void — it measured free-space, not contact (`contact_frames == 0`).
- Genuine contact now occurs (Fn ~17 N), and **where it occurs the actuation stack is motion-legal** (contact-phase
  peak ≤ 1.94 ≪ 3.45, integ 0, recovers ~20 steps).
- The 6D executor is velocity-limited in the sustained regime (~2.07 rad/s), vs the 27 rad/s unlimited coin arm.
- **6D-1 multimodal policy search is validated under realistic motion limits**: 6/6 seeds positive, seed-median
  Δ +0.354, hierarchical bootstrap 95 % CI [0.192, 0.493] — the advantage is about route **basins**, not speed.

**NOT claimed:**
- V4 is **not** frozen; the actuation stack is **not** certified contact-robust under the *sustained*-contact criterion.
- No delivery / K6 / transport-capability claim (C2 not run).
- The single-tip / low-friction geometry is **not** shown to sustain continuous loading.
- The 6D obstacle-**shift** K-mode edge is only a soft PILOT signal (weak Δ at low eligibility), not a strong claim.
- 6D-0 pose-reach under slew is **not** cleanly regressed (my quick probe used a straight line through the obstacle).

## 10. Exact next gate (the design decision to halt on)

**Should the coin dynamics contract certify SUSTAINED or INTERMITTENT contact?** The coin *task* is intermittent
(push-and-coast). Two clean options, user's call:

- **(A)** Re-scope the V4 gate to certify the **intermittent-contact** regime the task actually uses — i.e. accept
  "motion-legal on the contact frames that occur" (already **True**: peak ≤ 1.94, integ 0, recovers) as the contract,
  with a pre-declared minimum *total* contact exposure across the panel rather than a per-state *sustained* run. This
  would let V4 freeze immediately and unblock C2.
- **(B)** Keep the **sustained**-contact criterion and change the **contact geometry** to one that grips (two-finger
  pinch / higher friction) so continuous loading is physical — a larger scenario change.

I did **not** pick one (it changes what the contract *means*). Per the directive I stopped after one documented driver
fix + rerun rather than tuning to morning.

---

### Commits (separate, per directive)
- `8fef09fb` — V4 contact-gate correction (void prior negative) + contact-conditioned metrics + honest partial verdict.
- `<test>` — sustained_press config + mj_contactForce normal-force reader tests.
- 6D realistic-motion result — committed on run completion.
- This digest — final.

### Preserved unchanged
V2 / V3_AGILE frozen contracts, `COIN_LEGACY_FAST_V1`, all prior coin results, the C2 harness.
