# R6 — grip-held brake-to-stop & certified release: STAGE-0 CONTRACT (frozen before the controller)

**Created-at:** 2026-07-27 13:30 JST · **Branch:** `recovery/coin-r6-brake-to-stop-certified-release` (from R5 tip
`5933944a`). **Builds on** the R4/R5 substrate (continuous closed-loop rollout, ResponseState incl. the stopping guard
`d_stop`, CoastEstimator, PhaseMachine, R3 decoder, GOLDEN) reused unchanged. **CORE.YAML: none.**
`forward_displacement.py` unmodified.

## 0. Why R6 (what R4/R5 closed)

R4: a *fixed* coast model mistimes release per cradle (C0 1/4). R5: an *online* coast estimate cannot help because the
released-coast dissipation is a **post-release** quantity — unobservable at the moment the release must be decided
(`ADAPTIVE_COAST_RELEASE_HITS_AN_OBSERVABILITY_WALL`, `â_eff` saturates). The wall is real, not a hyperparameter.

**The fix removes the ill-posed prediction.** Two hybrid modes with different dissipation `R = R_m(x, history)`:

    HOLD / GRIP :  R_hold ≈ grip + contact + controller dissipation   (OBSERVABLE now)
    COAST      :  R_coast ≈ floor/contact-history dissipation         (appears only AFTER release)

Do **not** estimate `R_coast` before release. Keep the coin in the controllable/observable **HELD** mode, brake it to rest
*inside the zone under held bilateral contact* (so a fast cradle like s3 cannot escape a mistimed release), decay the
squeeze / stored contact energy, and switch to RELEASE **only from a certified rest state**.

## 1. The strategy (the single change)

    PUSH / HELD-TRANSPORT → GRIP-HELD BRAKE-TO-STOP → STOP-HOLD → SQUEEZE / STORED-ENERGY DECAY → CERTIFIED RELEASE → K6 DWELL

- **Transport→brake trigger = the R4 `d_stop`.** HELD-TRANSPORT (grip held, forward push) while `d_stop = v∥²/2·a_brake <
  dtz − zone`; enter GRIP-HELD BRAKE when `d_stop ≥ dtz − zone` (brake now to stop in-zone). `a_brake` is the measured
  `brake_opposed_reach` (large ⇒ brake late, near the zone), grip held throughout ⇒ **s3 stays controllable/observable**.
- **BRAKE** = the frozen option's velocity-feedback brake `d_brake = −v_coin/‖v_coin‖`, the decoder producing the
  slew-admissible torque from the *current* authority; squeeze only as much as holds contact, decaying as speed drops.
- **RELEASE only after the certificate passes** (§2), then monotone RELEASE → frozen K6 dwell.

Monotone phase PUSH→BRAKE→RELEASE preserved (BRAKE is the long stop-hold; RELEASE is once and late).

## 2. The release certificate (the load-bearing new element)

`v_coin ≈ 0 while gripped` is **not** sufficient (the pin/preload audit: zero coin velocity can hide large contact force,
stored elastic energy, non-null object wrench, actuator preload → the coin shoots out on release). Release only when ALL
hold for **N consecutive frames**:

    C_release = C_zone ∧ C_velocity ∧ C_spin ∧ C_wrench ∧ C_contact_vel ∧ C_qdot ∧ C_squeeze_decayed

concretely: `dtz < zone_tol`; `coin_speed < settle`; `|spin| < spin_tol`; **realised fingertip-only object wrench**
`‖Σ_side (fn·n + ft·t)‖ < wrench_tol` (balanced grip, no net push); contact-relative `‖v_n,v_t‖ < crv_tol`;
`max|qdot[:4]| < qdot_tol`; squeeze/preload decayed below `sqz_tol`. Only then RELEASE; then verify the frozen K6 dwell
(unchanged). All tolerances dev-tuned; **held-out never used for gain/threshold selection.**

## 3. Gates (in order; RL blocked throughout)

- **B0 — held-brake mechanism** (dev): forward progress + controlled deceleration + contact retention + no motion breach.
- **B1 — certified release**: stops in-zone + wrench/energy decays + **no ejection after release**.
- **B2 — development**: s1/s3 = **2/2** K6.
- **B3 — frozen panel**: s1/s3/s4/s7 = **4/4**, held-out **2/2**, budget ≤ 8, motion/contract clean, provenance valid.
- **Only after B3:** residual-intent SAC/TD3 (R6+ axis). **SAC/TD3 BLOCKED until B3.**

## 4. Decision tree

| result | verdict | RL |
|---|---|---|
| B3 4/4, held 2/2 | `BRAKE_TO_STOP_CERTIFIED_RELEASE_LOAD_BEARING` | authorised (not this session) |
| B3 3/4 or held 1/2 | `BRAKE_TO_STOP_IMPROVES_BUT_GATE_OPEN` | blocked |
| B3 2/4, held 0/2 | `BRAKE_TO_STOP_ALONE_INSUFFICIENT` | blocked |
| B0/B1 fail | mechanism / certificate audit | blocked |

## 5. Port-Hamiltonian reading

The controller no longer relies on a not-yet-reached mode's dissipation. It regulates the **current, observable HELD-mode
energy flow**, dissipates the kinetic + internal/stored energy, and switches to RELEASE only from a certified state — the
general pattern *stay in the controllable/observable contact mode until the energy + safety conditions for the next-mode
transition are verified*. `R = R_m(x, history)` with the mode `m` explicit.

## 6. What this session does / does NOT do

**Did:** froze this contract. **Building next (gated):** (1) `release_certificate.py` — the certificate + N-frame monitor +
the fingertip-wrench measurement (reuse `primary_fingertip_contacts` / `measure_contact_velocities`); (2) the brake-to-stop
controller (transport→brake on `d_stop`, held grip, certified release) on the R4/R5 substrate; (3) B0→B3, reusing the R4
harness + honesty controls. **SAC/TD3 remain BLOCKED until B3.** Change nothing in the physics, K6, decoder, or held-out
discipline.
