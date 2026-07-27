# R6 — grip-held brake-to-stop & certified release: B0 mechanism de-risk (RL still blocked)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r6-brake-to-stop-certified-release` (from R5 tip `5933944a`) ·
Contract: `reports/2026-07-27-coin-r6-brake-to-stop-certified-release-contract.md`.

**Result in one line:** the R6 mechanism (release certificate + held-transport→d_stop-brake→squeeze-decay→certified-release)
is built and unit-tested, but the **B0 mechanism de-risk fails 0/4** for a *structural* reason: driving the held coin with
the frozen option's **accelerating** PUSH over-drives it (s3 blows up to `vpar 1.4 m/s`), and the velocity-feedback brake then
**reverses** it (`vpar → −1.0`, the coin flies backward past the zone). The certificate never arms because the coin never
reaches a certified rest. The strategy needs a **velocity-regulated transport + a non-reversing stop** that the frozen 6-D
option's open-loop accelerating push does not provide. Verdict **`BRAKE_TO_STOP_NEEDS_VELOCITY_REGULATED_TRANSPORT`**.
**SAC/TD3 remain BLOCKED** (B2/B3 not reached).

## 1. What was built (staged; on the R4/R5 substrate)

- `theta_option/release_certificate.py` — the load-bearing new element. Per-frame predicate `C_zone ∧ C_velocity ∧ C_spin
  ∧ C_wrench ∧ C_contact_vel ∧ C_qdot ∧ C_squeeze_decayed` (the pin/preload lesson: `v≈0` while gripped is *not* safe —
  the fingertip object wrench is proxied by the coin's residual accel `‖qacc‖` + grip-pressure imbalance/magnitude), and a
  `ReleaseCertMonitor` that latches RELEASE only after **N consecutive** certified frames. **Unit-tested** (N-frame latching,
  tolerance shape).
- `theta_option/brake_to_stop.py` — `BrakeToStopController` (a structural variant, §6.5 #8): HELD-TRANSPORT (grip held)
  while `d_stop = v∥²/(2·a_brake) < dtz−zone`; GRIP-HELD BRAKE when `d_stop ≥ dtz−zone`; squeeze decays as the coin settles;
  RELEASE only when the certificate arms. Reuses SnapContext authority, the R3 decoder magnitudes, the monotone PhaseMachine,
  and the continuous `closed_loop_rollout` driver. All functions < CC 11; ruff clean.

## 2. B0 de-risk — the traces (teacher & predictor intents, all 4 states)

Both teacher and predictor intents give **0/4**; the certificate never arms on any state. The failure is a coin blow-up +
brake reversal, visible in the per-step trace:

| state | held-transport | brake onset | outcome |
|---|---|---|---|
| s3 | dtz 100→82 mm, vpar 0→0.37 (t1–16) | t17 (dtz 51) | vpar **explodes to 1.40**, reaches zone (dtz 5) then **reverses to −1.08**, flies to **387 mm** |
| s4 | dtz 96→43 mm, vpar builds then **reverses at t33** | never (d_stop stays < dtz−zone) | coin reverses to −0.93, flies to **240–267 mm** |

Diagnosis: `a_brake` measured from `brake_opposed_reach/dt` is **≈7.5 m/s²** (a one-step *reachable* Δv, an optimistic upper
bound), so `d_stop = v²/(2·7.5)` is tiny (≈4 mm at 0.2 m/s) and the transport→brake trigger fires **far too late** — the
frozen PUSH keeps *incrementing* torque every step (there is no velocity regulation), so the held coin accelerates
unboundedly. When the brake finally engages, the strong velocity-feedback brake over-corrects through zero and **reverses**
the coin.

## 3. Why this is structural (not a hyperparameter)

The frozen 6-D option provides an **open-loop accelerating push** (ramped-then-constant Δτ) and a **velocity-feedback brake**
that opposes `v_coin`. Neither regulates the coin to a *target speed* or arrests it *at* zero: sustained push ⇒ blow-up;
strong opposing brake near `v=0` ⇒ reversal (momentum + accumulated torque carry it backward). Brake-to-stop requires (a) a
**speed-regulated held transport** (drive the coin toward the zone at a bounded speed, not an accelerating push) and (b) a
**non-reversing stop** (decelerate to exactly zero, e.g. a saturating/deadband brake). Both are new control primitives beyond
the frozen option's per-step map. An `a_brake_gain` sweep confirms it is the *primitive*, not the trigger timing: **0/4 at
every gain** — low gain `0.04` avoids the blow-up (peak coin speed ≤ 0.87 m/s) but the coin **stalls short** (dtz 43–100 mm,
certificate never arms); high gain `0.25` **reverses** (s3 → 358 mm, peak 1.34). No braking *time* both stops the coin and
keeps it in the zone, because the accelerating push and the opposing-velocity brake cannot hold a bounded transport speed nor
arrest at zero.

## 4. Where this points

The R6 **certificate** is sound and reusable (it correctly demands a certified rest before release — the pin/preload guard).
The missing piece is the *transport/stop primitive*. Two redirections (fresh contract):

1. **Speed-regulated held transport + saturating stop** — a small new per-step primitive: hold a bounded coin speed toward
   the zone under grip, then a deadbanded brake that cannot reverse (clamp the brake so it only removes forward momentum).
   This keeps the certificate and the d_stop trigger, replacing only the push/brake law. Most direct.
2. **Learned residual** over the analytic brake-to-stop that regulates the transport speed + stop — the R6+ residual-intent
   axis — but only after (1) clears B0/B1.

## 5. Status

`release_certificate` + `brake_to_stop` built, unit-tested (12 fast tests incl. the R5 estimator + R6 cert-monitor; the
slow GOLDEN unchanged); ruff clean; < CC 11. **Verdict:** `BRAKE_TO_STOP_NEEDS_VELOCITY_REGULATED_TRANSPORT`. **SAC/TD3
BLOCKED** (B2/B3 not reached). **Exact next action:** a fresh contract for the speed-regulated held-transport + saturating
(non-reversing) stop primitive (§4.1), keeping the R6 certificate and `d_stop` trigger; B0→B3 then RL. The arc's cumulative
lesson stands: the missing component is **history-dependent, mode-aware energy management under held contact** — R6 localised
it further to the *transport/stop control primitive*, which the frozen open-loop option does not supply.
