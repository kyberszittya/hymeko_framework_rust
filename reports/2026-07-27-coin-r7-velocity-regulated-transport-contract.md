# R7 — velocity-regulated held-contact transport + saturating non-reversing stop: STAGE-0 CONTRACT

**Created-at:** 2026-07-27 13:55 JST · **Branch:** `recovery/coin-r7-velocity-regulated-transport` (from R6 tip `90142eea`).
**Reuses** the R6 release certificate + `d_stop` intuition, the R4/R5 continuous rollout scaffolding, the frozen
PHYSICS (`step_ablation` + `govern_torque` + slew/motion contract + K6) and the measured object authority `B_coin`
(`identify_Bcoin`). **Replaces only the per-step CONTROL law** (the frozen accelerating-push / opposing-velocity-brake that
R6 proved structurally inadequate). **CORE.YAML: none.** `forward_displacement.py` unmodified.

## 0. Why R7 (what R6 closed)

R6 proved the missing element is **not** the release monitor or the `d_stop` trigger (both correct and kept) but the
**transport/stop primitive**: the frozen option's open-loop *accelerating* push over-drives the held coin (blow-up to
vpar 1.4) and its opposing-velocity brake *reverses* it (vpar → −1.0). Neither holds a bounded transport speed nor arrests
at zero (`BRAKE_TO_STOP_NEEDS_VELOCITY_REGULATED_TRANSPORT`, 0/4 across the brake-gain sweep). So R7 introduces a **new
per-step control primitive** — a velocity servo on the measured object velocity — inside the same frozen physics.

## 1. The primitive (velocity-regulated held transport)

Regulate the coin's target-directed velocity directly (a reference that decays to zero at the zone):

    v_ref  = clip(k_d · d_remain, 0, v_max)                       (d_remain = dtz − zone)
    a_cmd  = clip(k_v · (v_ref − v∥), −a_stop_max, a_push_max)

**Mandatory non-reversing stop** — the command may bring positive motion to rest but NEVER to escape:

    if v∥ > 0 ∧ (v∥ + a_cmd·Δt) < 0 :  a_cmd = −max(v∥ − v_deadband, 0)/Δt

Map `a_cmd` (a desired along-track coin-velocity change `Δv = a_cmd·Δt`) to a slew-admissible Δτ through the measured object
authority (the least-norm inverse of the along row of `B_coin`), plus a **minimum grip** in the contact channel:

    Δτ = (b_along · Δv)/(b_along·b_along)  +  I_squeeze·squeeze_dir ,  then clip to the per-joint slew box,

with `I_squeeze = I_min + k_s·contact_risk` (only as much as retains contact). Lateral / spin corrections are a separate,
smaller-authority channel. Everything is then executed through the frozen governed step (motion contract enforced).

## 2. Phase machine (monotone)

    HELD_REGULATE → STOP_HOLD → SQUEEZE_DECAY → CERTIFIED_RELEASE → K6_DWELL

Inside **HELD_REGULATE** the velocity servo handles accel / cruise / decel as positive or negative `a_cmd` (NOT a phase
reversal). STOP_HOLD holds the coin at rest in the zone; SQUEEZE_DECAY drops the grip to the minimum; the R6 release
certificate (`release_certificate`, unchanged — the pin/preload guard) latches CERTIFIED_RELEASE only from a certified,
low-stored-energy rest; then the frozen K6 dwell decides.

## 3. Gates (in order; RL blocked throughout)

- **V0 — primitive algebra (pure unit tests):** bounded `v_ref`/`a_cmd`; the non-reversing stop never turns positive motion
  negative; `Δτ` slew-admissible (in the box); deterministic.
- **V1 — mechanism** (all 4 teacher states): `|v∥| ≤ v_max`, **no sign reversal**, no illegal contact loss, the coin
  **approaches the zone** (the R6 blow-up/reversal is gone).
- **V2 — certified stop** (dev): zone ∧ low velocity/spin ∧ low wrench proxy ∧ squeeze decayed ∧ release certificate latches.
- **V3 — development:** s1/s3 = **2/2** K6.
- **V4 — frozen panel:** s1/s3/s4/s7 = **4/4**, held-out **2/2**, budget ≤ 8, motion/contract clean, provenance valid.
- **Only after V4:** residual-intent SAC/TD3. **SAC/TD3 BLOCKED until V4.** Held-out never tunes a gain/threshold.

## 4. Decision tree

| result | verdict | RL |
|---|---|---|
| V4 4/4, held 2/2 | `VELOCITY_REGULATED_CONTACT_TRANSPORT_LOAD_BEARING` | authorised (not same session) |
| V1 ok but V4 3/4 / held 1/2 | `VELOCITY_REGULATED_TRANSPORT_IMPROVES_BUT_GATE_OPEN` | blocked |
| V1 fails (blow-up/reversal persists) | primitive / servo audit | blocked |

## 5. Port-Hamiltonian / CIP reading

The controller actively regulates energy **in and out** under held contact — it injects only enough to reach `v_ref` and
removes it without reversal, switching mode only from certified low stored energy. This is a candidate **shared CIP core**
primitive, general beyond coin (pick-place: object-velocity regulation before place; humanoid: COM/EE velocity regulation
before contact; AIBO: body-velocity regulation before a stable stop):

    VelocityRegulatedContactTransport · SaturatingNonReversingStop · StoredEnergyReleaseCertificate

Proven on coin first, then validated with the other embodiment adapters to show it is genuinely generic.

## 6. What this session does / does NOT do

**Did:** froze this contract. **Building next (gated):** (1) `velocity_transport.py` — the pure servo + non-reversing stop +
`B_coin` least-norm inverse + V0 unit tests; (2) the R7 rollout inside the frozen physics + the HELD_REGULATE phase machine
reusing the R6 certificate; (3) V1 → V4. **SAC/TD3 BLOCKED until V4.** Nothing in the physics, K6, certificate, or held-out
discipline changes.
