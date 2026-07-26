# R3 — physical-intent → deterministic authority-aware decoder contract (FROZEN before any decoder code)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r2-relational-update0` (R3 built on a fresh branch) · **Base:**
`42cd36d6` (R2 = `RELATIONAL_ORGANISATION_ALONE_INSUFFICIENT`; basin audit = `PHYSICAL_INTENT_DECODER_AUTHORISED`). This
document **freezes the R3 experimental contract before any decoder code exists**. R3 is authorised **only** as the isolated
*factorisation* axis. This session writes **only this contract** — the decoder is built in a separate, gated session.

## 0. Why R3 (what the arc has closed)

| axis | result |
|---|---|
| physical delivery (frozen option) | **4/4 — SOLVED** (`a3459629`) |
| more development coverage (N=2,4,6) | insufficient (`COVERAGE_ALONE_INSUFFICIENT`) |
| more raw θ-modes (K-head, raw 42-D) | insufficient (2/4) |
| canonical coordinatisation (R1) | load-bearing but insufficient (2/4, `FLAT_R1_LEARNED_AMORTISATION_FAILS`) |
| HyMeKo relational organisation (R2) | insufficient alone (2/4, `RELATIONAL_ORGANISATION_ALONE_INSUFFICIENT`) |
| budget-8 search geometry | **not** the blocker (frozen basin audit) |
| **direct cradle → 6-D θ regression** | **WRONG FACTORISATION** |

The frozen basin audit (`basin_audit.json`) is decisive: the R2 held-out proposals are in a **different basin** — s4 at
**8.1×SEARCH_STD** from the teacher (0/600 motion-compatible working θ; teacher rides the motion limit), s7 at
**5.3×SEARCH_STD**, budget-8 first restores K6 only at α=1.0 / α=0.8, and the gap is **multi-role** (s4: squeeze + forward
+ release; s7: balance + squeeze + release). The actor does not choose a *slightly-off* θ; it chooses a **different
physical strategy**. Amortising cradle → concrete 6-D torque-option parameters is the wrong factorisation.

## 1. Invariant — the single thing that changes

> Replace the **direct cradle → 6-D θ regressor** with **canonical structured state → cradle-AGNOSTIC physical intent →
> deterministic authority-aware decoder → the unchanged 6-D θ**. The *intent* is the same physical plan for every state;
> only the *decoding* to θ is cradle-specific, and it is **computed from already-measured authority quantities**, not
> amortised.

**Frozen (identical to the R1/R2 gates):** physics + `CradleSnapshot`; the 6-D **PUSH → BRAKE → RELEASE** velocity-feedback
option; the **budget-8 centre-inclusive** `fixed_search_select`; the frozen **K6** monitor (zone 0.02 m, settle 0.06 m/s,
dwell 6); the **frozen 4-state panel** (s1,s3 dev; s4,s7 held-out); held-out discipline (s4/s7 eval-only); the teacher
sets; the motion contract. **Forbidden:** any new state variable, any held-out-derived feature, a larger search budget,
changed action semantics, a changed reward, or invoking the teacher at deploy.

## 2. The physical intent (cradle-AGNOSTIC vocabulary)

A small, fixed, **state-agnostic** plan — the *same physical goal* on every cradle. Frozen vocabulary (7 roles):

| intent role | meaning |
|---|---|
| forward transport | desired forward impulse / drive toward the target zone |
| lateral correction | desired cross-track (spin/steer) correction |
| contact retention / squeeze | desired internal preload holding the coin in contact |
| desired peak velocity | the coin speed the plan aims to reach before braking |
| brake-entry condition | when PUSH → BRAKE hands over |
| braking demand | the velocity-feedback braking strength |
| release condition | when BRAKE → RELEASE hands over |

The intent is **not** cradle-specific motor code. It is the physical *what*; the decoder supplies the cradle-specific
*how*.

## 3. The deterministic authority-aware decoder (inputs are ALREADY measured)

The decoder computes the physical 6-D θ **deterministically** from the intent and the frozen measured quantities — **no
new features** (all exist in `directional_authority.py` / the R1 extractor):

| decoder input | source (already measured) |
|---|---|
| `B_coin = ∂(coin v_along, v_perp)/∂Δτ` | `directional_authority.identify_Bcoin` (deploy-faithful FD through rate-limiter + governor) |
| reachable object authority (forward / lateral± / brake) over the slew-admissible box | `object_authority` (`reachable` over `admissible_dtau_box`) |
| contact / internal authority (squeeze, balance) — B_τ **null-space** | `contact_internal_authority` |
| ± slew-headroom per arm | `slew_head_up/dn`, `prev_tau` |
| current contact geometry (normal/tangent frame, tip-coin) | R1 contact groups |
| target-frame position & velocity | R1 `dtz`, `coin_vel_along/perp` |

**Deterministic, not a free regressor.** E.g. the forward-drive component of θ is the Δτ (within the slew-admissible box)
that realises the *desired forward impulse* given **this** cradle's `B_coin`; squeeze maps into the B_τ **null-space** (so
it does not perturb `B_coin`); brake demand scales the velocity-feedback term. The *physical relationship* (how much
torque produces how much forward drive at this configuration) is cradle-specific and **computed**, which is exactly why it
can generalise where a memorised θ cannot.

## 4. The s4 lesson (why deterministic, not slack)

s4's motion-compatible working set is **essentially a point at the motion limit** (0/600 harvest). There, a "roughly
good" θ does not deliver — the decoder must reconstruct the **correct physical relationship** simultaneously in *squeeze +
forward drive + brake timing + release*. This is the argument for a **deterministic physical decoder** at the centre
rather than another slack regressor: only a decode that respects the measured authority can hit a point-like basin.

## 5. Gates (in order; nothing downstream until the prior passes)

- **D0 — teacher consistency (factorisation validity).** Extract the intent from each teacher trajectory; the deterministic
  decoder must map it back to a **K6-capable θ** on the *same* cradle (round-trip). Validates the intent→θ factorisation on
  teacher data **before** any generalisation claim. *D0 fail ⇒ the vocabulary or the decoder physics is incomplete — fix
  the decoder, not the data.*
- **D1 — development update-0.** s1/s3 = **2/2** at the unchanged budget-8 search.
- **D2 — frozen panel.** s1/s3/s4/s7 = **4/4**, held-out (s4,s7) = **2/2**, no motion/safety regression, provenance valid.

**Only after D2 passes:** SAC/TD3 AUTHORISED.

## 6. Decision tree

| result | verdict | RL |
|---|---|---|
| **D2 4/4, held-out 2/2** | `PHYSICAL_INTENT_DECODER_LOAD_BEARING` → the right factorisation generalises | **SAC/TD3 authorised** |
| D2 3/4 or held-out 1/2 | `DETERMINISTIC_DECODER_IMPROVES_BUT_GATE_OPEN` | blocked |
| D2 2/4, held-out 0/2 | `DETERMINISTIC_DECODER_ALSO_INSUFFICIENT` → deeper factorisation / basin-aware search | blocked |
| D0 fail | `INTENT_FACTORISATION_INCOMPLETE` — decoder physics/vocabulary audit (not a data problem) | blocked |
| D1 dev regression | implementation / decoder audit — not yet a scientific negative | blocked |

Any honest failure is reported **without** changing physics, teacher sets, search budget, or held-out discipline.

## 7. Mandatory tests (before any training/tuning)

1. **Intent round-trip** — extract-then-decode on a teacher θ recovers a K6-capable θ (D0 in unit form).
2. **Determinism** — same (intent, snapshot) → identical θ (the decoder is a pure function of measured quantities).
3. **No new information** — the decoder reads only the frozen measured quantities; assert no held-out-derived / raw-world /
   future input enters.
4. **Authority-consistency** — the forward-drive Δτ realises the requested forward impulse under this `B_coin` (FD check);
   squeeze lives in the B_τ null-space (`‖B_coin·Δτ_squeeze‖ ≈ 0`).
5. **Bounded legal θ** — the decoded θ is always in the frozen box; slew-admissible.
6. **Search-provenance regression** — unchanged budget-8, centre-inclusion, θ₀/θ_exec split, K6 monitor.

## 8. What this session did / did NOT do

- **Did:** froze this contract (design only). **Did NOT:** write any decoder code, extract intents, train, or run RL.
- The R2 branch, checkpoints, physics, search, and evaluation discipline stay frozen. **SAC/TD3 remain BLOCKED** until D2.
- Build order for the next (gated) session: (1) `theta_option/physical_intent.py` — the frozen intent vocabulary +
  teacher-intent extractor; (2) `theta_option/authority_decoder.py` — the deterministic authority-aware decoder + the 6
  mandatory tests; (3) `coin_theta_rl_benchmark --r3-decoder-d0` (teacher consistency), then `--r3-update0` (D1 dev),
  then the frozen panel (D2). Change nothing else.

> The coin's delivery is already solved; we now also know **why** direct θ-regression could not amortise it. The next step
> is not another neural trick but the correct **factorisation** of the working physical mechanism.
