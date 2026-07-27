# R5 — adaptive coast identification & release control: STAGE-0 CONTRACT (frozen before the estimator)

**Created-at:** 2026-07-27 06:20 JST · **Branch:** `recovery/coin-r5-adaptive-coast-release` (from the R4 tip `c7a09b00`).
**Builds on:** the R4 substrate (`theta_option/closed_loop_{state,intent,rollout}.py`, `experiments/coin_r4_gates.py`) —
the continuous closed-loop rollout, the ResponseState, and the GOLDEN (bit-identical to the frozen option) are reused
unchanged. **CORE.YAML: none.** `forward_displacement.py` remains unmodified. This document freezes the R5 experimental
contract before the online estimator is written.

## 0. Why R5 (what R4 closed)

R4 proved a **single hand-fixed coast model + deterministic correction is insufficient**: with `a_friction=0.55` the coast-in
release mistimes per cradle → C0 teacher **1/4** (destabilises 3/4 delivering plans) and C2 CL ties open-loop (1/4, held 0/2,
not load-bearing), while the *machinery* is exact (GOLDEN) and the oracle is 4/4. The measured physical picture:

- delivery is **PUSH → RELEASE → passive COAST → stop under friction** (not active push-brake);
- the decisive quantity is the **instantaneous effective dissipation** and the **release timing matched to it**;
- the effective coast deceleration varies **≈0.42–0.67 m/s²** cradle-to-cradle, so any single `a_friction` is wrong on some
  states — s3 (fast momentum, needs precise trim/hold), s4 (gentle, motion-limit), s7 (low momentum, undershoots on an early
  coast);
- object `forward_push_reach` is near-identical (0.072–0.080) → **not** different actuator regimes but different real
  **coast dynamics**, observable from the *trajectory* though not extrapolable from the cradle snapshot.

Port-Hamiltonian reading: the dissipation term is **not constant** — `R = R(x, m, contact-history)`. R5's controller
estimates the instantaneous dissipation online and matches the energy injection + release to it. This is the HyMeKo /
port-Hamiltonian mechanism (structured mode + measured energy flow → online dissipation estimate → intent correction →
certified mode switch), so R4's negative is a *sharpening*, not a retreat.

## 1. The single authorised change

Replace R4's **fixed** `a_friction` with an **online estimate `â_eff`** of the effective coast deceleration, measured from the
coin's own velocity response, and drive the RELEASE timing from the `â_eff`-predicted coast landing. Everything else frozen
(physics, 6-D option, R3 decoder, authority, K6, panel, held-out discipline, budget-8, GOLDEN machinery).

## 2. Online estimator (frozen form)

    â_eff = median{ −Δv∥ / Δt }   over the last k_est valid steps,

valid ⟺ (no active forward push this step) ∧ (v∥ > v_min for a stable ratio) ∧ (Δv∥ < 0, physically decelerating) ∧
(no large contact-driven forward drive). Robust median (or a small local filter); clamp to `[a_lo, a_hi]` (physical, e.g.
0.20–1.20); fall back to a prior `a_prior` until ≥ n_min valid samples. `d_coast = v∥² / (2·â_eff)`; RELEASE when the
predicted landing enters the K6 zone with a safety margin.

**Chicken-and-egg resolution (frozen).** `â_eff` cannot be read before some deceleration exists, but the first RELEASE
decision needs it. Therefore the phase order is **PUSH → (grip-held) BRAKE-PROBE → RELEASE**: a short *low-gain* velocity-
feedback brake window that (a) produces the deceleration to estimate `â_eff` and (b) **holds the grip so a fast cradle (s3)
cannot escape** (R4's failure mode). The probe brake contribution is modelled from `B_coin` and subtracted (or kept small so
`â_eff ≈ measured deceleration`). This respects the monotone PUSH→BRAKE→RELEASE order and lets `â_eff` inform the release.
The probe length is dev-tuned and must be short enough not to rob a coast-cradle of the momentum it needs.

## 3. Learned residual — LATER, bounded, on top of the analytic estimate

Only after the analytic adaptive controller passes A0/A1/A2: a **bounded learned residual** over the analytic estimate,
correcting chiefly **release timing, coast-distance prediction, squeeze-decay timing, small forward-deficit, lateral drift** —
*not* all seven intent roles with equal freedom (R4 localised the load to release/coast/squeeze-decay/small-forward). The
learning task is the LOCAL, stable `response-history → effective-dissipation / landing-error → release-time residual`, not
the R3 `cradle-snapshot → full 7-D intent` that did not extrapolate.

## 4. Gates (in order; RL stays blocked throughout)

- **A0 — adaptive teacher fixed point.** Teacher own-intent + the adaptive estimator must keep **K6 = 4/4** with a
  near-no-op correction (no destructive override). This is the R4-blocker de-risking test: does *measuring* `â_eff` (vs
  fixing it) preserve the precisely-tuned teacher deliveries? A0 fail ⇒ estimator/probe bug, not a scientific negative.
- **A1 — development.** R3-predictor intent + adaptive release ⇒ s1/s3 = **2/2**.
- **A2 — one frozen panel.** s1/s3/s4/s7 = **4/4**, held-out **2/2**, budget ≤ 8, motion/contract clean, provenance valid.
- **Only after A2:** residual-intent SAC/TD3 (the R6–R10 axis), gated exactly as before. **SAC/TD3 remain BLOCKED until A2.**

## 5. Mandatory tests (before any gate)

Estimator: (a) recovers a known constant deceleration from a synthetic decaying-velocity window (within tolerance); (b)
rejects invalid samples (accelerating / low-speed / active-push steps); (c) robust to one outlier (median); (d) clamps to
`[a_lo,a_hi]` and uses the prior below n_min. Controller: (e) reduces to R4's coast-in when `â_eff` is forced to the fixed
value (regression); (f) monotone PUSH→BRAKE→RELEASE with the grip-held probe; (g) the GOLDEN still holds (constant controller
≡ `rollout_primitive`); (h) determinism; (i) no teacher fallback; (j) budget ≤ 8 provenance.

## 6. Decision tree

| result | verdict | RL |
|---|---|---|
| A2 4/4, held-out 2/2 | `ADAPTIVE_COAST_RELEASE_LOAD_BEARING` → the right closed-loop level | **authorised** (do not start it in the same session) |
| A2 3/4 or held-out 1/2 | `ADAPTIVE_COAST_IMPROVES_BUT_GATE_OPEN` | blocked |
| A2 2/4, held-out 0/2 | `ADAPTIVE_COAST_ALONE_INSUFFICIENT` → the bounded learned residual (§3) is the next axis | blocked |
| A0 fail | estimator/probe/integration audit (not yet a scientific negative) | blocked |

## 7. Audit carried from R4 (recorded here so it is not read as a regression)

R4's `coin_r4_gates.py` open-loop baseline used a **uniform** search seed `90000`; the R3/R1/R2 panels use a **per-state**
seed `90000 + i·131 + K`. The decoder, R3 predictor (NW bw 3.0, same 6-dev set), option, and physics are byte-identical to
R3. s3's *predicted*-intent delivery is search-seed-fragile (delivers at 90131+K, misses at 90000, landing ~37 mm short) —
a narrow-basin jitter sensitivity, not a regression. The canonical R3 open-loop baseline remains 2/4.

## 8. What this session did / did NOT do

**Did:** recorded the R4 seed audit; froze this R5 contract (design only). **Did NOT:** write the estimator, run A0/A1/A2, or
train. **SAC/TD3 remain BLOCKED** until A2. Build order for the (gated) implementation: (1) the online `â_eff` estimator +
its unit tests; (2) the grip-held brake-probe + adaptive-release extension of the R4 controller (reduces to R4 when `â_eff`
is fixed); (3) `coin_r5_gates.py` A0 → A1 → A2, reusing the R4 harness + honesty controls.
