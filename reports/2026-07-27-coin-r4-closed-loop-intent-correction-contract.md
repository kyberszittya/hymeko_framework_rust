# R4 — closed-loop basin-aware intent correction contract (FROZEN before any controller code)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r3-physical-intent-decoder` (R4 built on a fresh branch) ·
**Base:** `e3624d14` (R3 = `PHYSICAL_INTENT_FACTORISATION_ALONE_INSUFFICIENT`; decoder VALID, open-loop amortisation FAILS).
This document **freezes the R4 experimental contract before any controller code exists**. R4 is authorised **only** as the
isolated *open-loop → closed-loop* axis. This session writes **only this contract**; the controller is built in a separate,
gated session.

## 0. Why R4 (what R1–R3 closed)

| axis | result |
|---|---|
| physical delivery (frozen option) | **4/4 — SOLVED** |
| R1 canonical representation | 2/4 (`FLAT_R1_LEARNED_AMORTISATION_FAILS`) |
| R2 HyMeKo relational organisation | 2/4 (`RELATIONAL_ORGANISATION_ALONE_INSUFFICIENT`) |
| R2 basin audit | wrong basin, **not** search geometry |
| R3 physical-intent decoder | decoder VALID (D0 4/4) but 2/4 (`PHYSICAL_INTENT_FACTORISATION_ALONE_INSUFFICIENT`) |
| **open-loop amortisation of a fixed decision** | **the wrong abstraction level** |

R1, R2, R3 all land at **2/4, held-out 0/2** while every *forward* factorisation (canonical frame, relational graph,
authority decoder) is verified correct in isolation. The bottleneck is the **amortisation itself**: a single open-loop
decision (θ **or** intent) predicted from the static cradle description does not cover held-out. The concrete witness is
**s4** — the predictor gives a dev-like strong transport (peak velocity 0.47) where the real cradle tolerates only a gentle
one (0.263, it rides the motion limit). *This cannot be decided reliably before the trajectory — but it is visible within
the first few physical steps.*

## 1. Invariant — the single thing that changes

> Replace the **open-loop fixed intent** with a **closed-loop basin-aware correction**: an initial conservative intent →
> R3 decoder → θ → a short physical segment → measured coin + contact response → a causal correction `response_error →
> Δintent` → re-decode → continue. The policy learns/encodes the *simple, general* rule (observed error → Δintent), **not**
> a full cradle→solution map.

**Frozen (identical to R1/R2/R3):** the physics; the 6-D PUSH→BRAKE→RELEASE velocity-feedback option; the **R3
deterministic authority-aware decoder** (`authority_decoder.decode_intent`); the frozen K6 (zone 0.02 m, settle 0.06 m/s,
dwell 6); the 4-state panel (s1,s3 dev; s4,s7 held-out); held-out discipline; the teacher sets; the authority measurement
(`identify_Bcoin` / `object_authority` / `contact_internal_authority`); the residual diagnostics. **Forbidden:** any new
STATE feature for the predictor, any held-out-derived feature, changed physics/option semantics, a changed reward, peeking
at K6/future, or invoking the teacher at deploy. The **budget-8** search applies to the *initial* segment's θ only;
corrections are deterministic re-decodes (**no larger search budget**).

## 2. The closed-loop loop

```
initial CONSERVATIVE intent  (dev-safe prior, NOT the failed open-loop predictor)
   → R3 decoder → θ₀ → budget-8 search (initial segment only)
   → execute a short window of the frozen option
   → MEASURE the causal physical response:  coin accel / peak speed, contact fₙ (retention), lateral drift, dtz, speed
   → correction law:  response_error → Δintent   (clip to the intent box)
   → re-decode θ for the next window  (deterministic, no extra search)
   → repeat until RELEASE / horizon / K6
```

The measured response uses only **past+present** physical signals — the same class the K6 monitor and the option's
velocity-feedback brake already read. No new predictor STATE features, no future/K6 leak.

## 3. The correction law (frozen role vocabulary; deterministic first)

Concrete, causal `response_error → Δintent` rules (the load-bearing element; start deterministic, a learned gain is a later
step gated on C0):

| observed | correction |
|---|---|
| measured coin accel / peak speed too high | ↓ forward_drive / peak_velocity, earlier brake_entry |
| contact fₙ weakening (retention loss) | ↑ squeeze, ↓ forward_drive |
| lateral drift growing | adjust lateral / balance |
| stopping distance too large (dtz not closing, speed high) | ↑ braking_demand |
| near zone ∧ speed low | release |

These map to the **existing 7 intent roles**; the decoder turns the corrected intent into the cradle-specific θ. The
initial intent is a **conservative dev-safe prior** (e.g. a low-peak-velocity plan), so the correction can only *tighten*
toward delivery — the s4 lesson (a gentle transport must be reachable) is honoured by starting gentle.

## 4. Reuse — audit, don't reinvent (§6.1)

Existing feedback machinery to **audit and reuse** (classify keep/suspect/generated before building): `rollout_primitive`'s
`frame_hook` (observe-only — extend to a control-callback), `coin_v3_receding_horizon.py` (deterministic receding-horizon
CEM replanning — the execution/replan pattern), `coin_residual_*` (residual control), `coin_primitive_mpc.py`,
`coin_feedback_chunk_v2.py` / `coin_v3_feedback_pilot.py`. **The NEW element is that correction targets the physical INTENT
via the R3 decoder**, not θ/actions directly — a much smaller, more general control surface. Do not duplicate the rollout,
the authority measurement, or the decoder.

## 5. Gates (in order)

- **C0 — teacher replay.** Closed-loop correction applied along the 4 teacher trajectories must **not break** the frozen
  4/4 teacher K6 (a correct feedback law is a no-op when the open-loop plan is already delivering). *C0 fail ⇒ the
  correction law destabilises a working plan — fix the law, not the data.*
- **C1 — development.** s1/s3 = **2/2** (conservative initial intent + closed-loop, budget-8 initial segment).
- **C2 — frozen panel.** s1/s3/s4/s7 = **4/4**, held-out (s4,s7) = **2/2**, no motion/safety regression, provenance valid.

**Only after C2 passes:** SAC/TD3 AUTHORISED.

## 6. Decision tree

| result | verdict | RL |
|---|---|---|
| **C2 4/4, held-out 2/2** | `CLOSED_LOOP_INTENT_CORRECTION_LOAD_BEARING` → the right abstraction level | **SAC/TD3 authorised** |
| C2 3/4 or held-out 1/2 | `CLOSED_LOOP_IMPROVES_BUT_GATE_OPEN` | blocked |
| C2 2/4, held-out 0/2 | `CLOSED_LOOP_ALONE_INSUFFICIENT` → the delivery genuinely needs exploration (RL) | blocked |
| C0 fail | `CORRECTION_LAW_DESTABILISES_TEACHER` — feedback-law audit | blocked |
| C1 dev regression | implementation / controller audit — not yet a scientific negative | blocked |

## 7. Mandatory tests (before any gate)

1. **Teacher no-op** — on a delivering teacher plan, the correction produces ≈0 Δintent (a correct law does not disturb a
   working trajectory) → C0 in unit form.
2. **Causality** — the correction at window k uses only measurements from windows ≤ k (no future/K6 leak).
3. **Bounded intent** — every corrected intent stays in the frozen intent box; every re-decoded θ stays in the θ box.
4. **Mirror equivariance** — the closed-loop θ sequence is equivariant (balance sign) under the physical mirror.
5. **Determinism** — same cradle + same initial intent + same law → identical θ sequence + diagnostics.
6. **Monotone corrections** — each rule moves its role in the documented direction (e.g. over-speed ⇒ forward_drive
   non-increasing), absent saturation.

## 8. What this session did / did NOT do

- **Did:** froze this contract (design only). **Did NOT:** write any controller code, extend `frame_hook` to control,
  implement the correction law, train, or run RL. **SAC/TD3 remain BLOCKED** until C2.
- The R3 decoder, physics, search, and evaluation discipline stay frozen.
- Build order for the next (gated) session: (1) extend the execution to a **control-callback** window loop over the frozen
  option (audit/reuse `frame_hook` + `coin_v3_receding_horizon` pattern); (2) `theta_option/intent_correction.py` — the
  causal `response_error → Δintent` law + the 6 mandatory tests; (3) `coin_theta_rl_benchmark --r4-teacher-replay` (C0),
  then `--r4-update0` (C1 dev), then the frozen panel (C2). Change nothing else.

> R1–R3 proved the open-loop amortisation is the wrong abstraction level: the system need not perfectly *predict* the whole
> trajectory in advance; it must *correct* the intent from the measured physical response as it goes. That leads directly
> back toward the 4/4 delivery.
