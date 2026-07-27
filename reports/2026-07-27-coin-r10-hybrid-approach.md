# Coin-R10 — phase-structured hybrid program (APPROACH momentum-build mode)

**Date:** 2026-07-27 · **Branch:** `recovery/coin-r9-causal-residual-delivery` (worktree `hymeko_coin_r9_wt`)
**Status:** C2 mechanism — **`APPROACH_BUILDS_MOMENTUM_BUT_NO_S1_K6`** (C2-A/B/C PASS, C2-D FAIL — handoff localised).

## Contract (phase-hybrid, frozen)

The R9 diagnosis (reach 46 mm · M0 50 mm · C0 · C1) proved the s1 failure is a **missing trajectory phase**, not a bad base
or bound. R10 makes the trajectory an explicit hybrid program:

```
APPROACH_MOMENTUM_BUILD → HOLD/TRANSPORT → BRAKE/SETTLE → SQUEEZE_DECAY → RELEASE (R6 certificate)
```

**Frozen / untouched:** the 20 mm K6 tolerance, physics, torque/slew/joint/motion limits, the BRAKE/SETTLE law, and the R6
release certificate (sole release authority — no free release bit). **Dev = s1, s3 only**; s4/s7 validation-only; the
`f1–f4` blind panel stays `SEALED_NOT_EVALUATED`. Modes are **trajectory phases**, not cradle regimes; near/far differ only
in guard timing. Each semantic stage commits separately.

## The APPROACH_MOMENTUM_BUILD mode (`hybrid_approach.py`)

A new phase prepended to the frozen tip-transport scaffold. Its control goal is **different**: build a bounded forward
impulse and reach a causal launch state — *not* the scaffold's distance-proportional `v_ref = k_d·d_remain` (which slows near
the target; M0 ruled out re-tuning it). It commands a **distance-independent** forward joint-velocity `qdot_approach` (above
the S1-safe settle cap 1.0, under the motion-contract hard limit) + an acquisition squeeze, then **monotonically** hands off
to the frozen servo/brake. Exit by first-applicable causal guard — **SAFETY ≻ LAUNCH (v_par band) ≻ REACHABILITY (v_par² ≥
2·a·d_remain) ≻ BUDGET (∑v_par·dt) ≻ HORIZON** — no future/K6/oracle info; phase never regresses. With `enabled=False` it
reproduces the frozen scaffold.

## C2 mechanism gate (dev s1 only; 12-config sweep + disabled=scaffold)

| check | result |
|---|---|
| disabled = scaffold reproduction | peak v_par 0.085, dtz 57.7 mm (matches frozen scaffold) ✓ |
| **C2-A** peak v_par exceeds the 0.154 ceiling safely | **PASS** — max **0.316** (vs teacher 0.322), safe |
| **C2-B** exits via a causal guard | PASS (LAUNCH / REACHABILITY) |
| **C2-C** hands off to the frozen brake | PASS |
| **C2-D** full hybrid reaches s1 K6 | **FAIL** — best 50.0 mm, no K6 |

**The momentum-build mode is LOAD-BEARING at the mechanism level:** it lifts peak forward velocity from the scaffold's 0.085
(and the 0.154 residual ceiling) to **0.316 ≈ the teacher's 0.322**, safely — the first genuinely-discovered hybrid mode of
the campaign, forced by the C1 unexpressible-component result, not invented for aesthetics.

## Localisation of C2-D (handoff, NOT brake)

Per the gate rule, C2-A pass + C2-D fail ⇒ localise the **handoff / guard timing**. The APPROACH exits on LAUNCH at step 5
while the coin is still far, and the scaffold's distance-proportional servo then **immediately dissipates the built
momentum** before the coin reaches the zone (it *is* the settle law). The missing element is the **HOLD/TRANSPORT coast
phase** between APPROACH and BRAKE — carry the momentum to the target corridor, *then* brake — and possibly a coast-in
handoff (the R4/R5 push-then-coast question, gated by the R6 rest certificate). **BRAKE was not modified.**

## C2.5 handoff phase-existence audit (`--c25`) — which second mode?

To decide whether the transport mode is **HELD_MOMENTUM_CARRY** (grip retained) or a teacher-like **RELEASED_COAST** (the
teacher loses contact at step 34 and coasts into the zone at 36), a CARRY phase (held forward-effort × duration, or a
passive-release coast) was swept between APPROACH and the frozen BRAKE on s1:

| handoff | best | K6 |
|---|---|---|
| HELD carry (qref 1.0, 8 steps) | 48.4 mm | ✗ |
| **PASSIVE_RELEASE coast (10 steps)** | **33.0 mm** | ✗ |

**`HANDOFF_TRANSPORT_MODE_NOT_YET_IDENTIFIED`** — neither delivers, so the handoff is **not a few carry frames**. The
released-coast (33 mm) beats held-carry (48 mm), pointing to the teacher-like mechanism, but a **fixed** carry-then-release
duration is the wrong release *state* — it wants a **reachability-gated release** (release when `v_par² ≥ 2·a·d_remain`),
which forces the design question the R6 rest-certificate raises: a **pre-zone launch/release guard is distinct from the final
settle/K6 certificate**. Correct current verdict: `APPROACH_MOMENTUM_MODE_LOAD_BEARING` + `HANDOFF_TRANSPORT_MODE_NOT_YET_IDENTIFIED`.

## C2.6 robust coast-entry guard (`--c26`) — the R5 wall re-appears

Per the R5 lesson (the post-release coast friction is unobservable before release), the guard uses the **interval**, not a
point estimate: `d_stop ∈ [v²/2·a_max, v²/2·a_min]`, fire only when the *whole* predicted landing interval ⊆ the corridor.
12 passive-coast branches on s1 bound the deceleration at **a ∈ [0.51, 1.26] m/s² (spread 2.5)**. Deployed, the robust guard
**never fires** (s1 ends 49.1 mm): the interval width (~30 mm) exceeds the 20 mm corridor, so a firing state where the
long-coast (low-a) branch does not overshoot does not exist. Verdict **`RELEASE_GUARD_PREDICTION_INSUFFICIENT`** (not a
looser-guard problem — a looser guard would gamble against the R5 wall).

**This is the R5 observability wall at the hybrid-mode level:** an *open-loop* released-coast cannot reliably hit a 20 mm zone
when the post-release friction spans a factor ~2.5. The releasing itself is fine; the missing element is a **closed-loop
correction after the coast**.

## C2.7 post-coast feasibility (`--c27`) — guided-coast ruled out, re-acquire required

Before committing the re-acquire into a full program, the easier compromise was tested: a **GUIDED_COAST** (light contact —
low squeeze + tiny effort — so the coin coasts but keeps correction authority), the physically-attractive middle between
full-grip (over-dissipates, 48 mm) and full-release (no authority, 33 mm). Result: the best guided-coast reaches only
**50.9 mm** on s1 (even at squeeze 0.02) — **worse than full release** (the tips still drag/dissipate). **`GUIDED_COAST_
INSUFFICIENT_REACQUIRE_NEEDED`.** The easier mode does not exist in this physics; the harder closed-loop **RE-ACQUIRE** is
genuinely required.

## Verdict chain (measured)

```
APPROACH_MOMENTUM_MODE_LOAD_BEARING            (C2:   peak v_par 0.316 ≈ teacher 0.322)
RELEASED_COAST_MODE_PROMISING                  (C2.5: 33 mm, the closest handoff)
ROBUST_OPEN_LOOP_COAST_ENTRY_UNAVAILABLE       (C2.6: coast a∈[0.51,1.26], guard can't fit the 20 mm corridor = R5 wall)
GUIDED_COAST_INSUFFICIENT                       (C2.7: 50.9 mm, worse than full release)
⇒ POST_COAST_CLOSED_LOOP_REACQUIRE_REQUIRED
```

## C2.8 RE-ACQUIRE feasibility (`--c28`, R0/R1) — the hard primitive EXISTS

A RE-ACQUIRE phase (gently catch the coasting coin with a bounded forward joint-velocity + a **ramped, impulse-limited**
squeeze, gated on the coin's closing velocity and the reachable corridor) was measured on s1 across release timings:

| gate | result |
|---|---|
| **R0** geometric reachability (tips reach the coasting coin, safe) | **12/12** ✓ |
| **R1** gentle re-grip (bilateral contact re-forms, no fling, dtz push ≤ 5 mm) | **9/12** ✓ |
| **R4** full chain reaches s1 K6 | 0/12 — best chain **31.0 mm** |

**`REACQUIRE_FEASIBLE_SETTLE_NOT_YET`.** The hard contact primitive the guided-coast couldn't replace **exists in the physics
and is gentle** — the tips reliably catch the coasting coin and re-grip without knocking it away. The full chain reaches
**31 mm** (the closest of the entire arc: beats released 33, guided 51, held 48). The residual gap is **R3**: the frozen
settle, run from the reacquired state at ~31 mm, cannot null the last ~11 mm — the *distance-proportional under-transport
recurs at small scale*. So the re-acquire should catch **closer** to the zone (or add a tiny forward nudge before the settle).

## Next (narrow — R3/R4 only)

Tune the re-acquire *catch point* / settle authority to close the last ~11 mm on s1 (catch closer via a tighter
`reacq_corridor_hi` + `reacq_vclose`, or a small forward nudge into the settle), then R4 full chain s1 K6, then C3 (manual
hybrid, s1 ∧ s3 strict K6, safety 2/2). Settle/K6 cert + physics frozen; blind panel sealed. **Scientific through-line:** the
same soft-frictional uncertainty recurs at every abstraction level — R1–R7 (fixed law), R8–R9 (bounded residual), R10
(open-loop coast) — and the answer is always a next *closed-loop*, measurement-discovered mode that re-acquires control after
the uncertainty realises. Three modes are now **discovered from the physics**: APPROACH (C1 unexpressible component),
RELEASED_COAST (C2.5), RE-ACQUIRE (C2.8) — a hybrid system grown by measurement, not a hand-drawn state machine.

## Artifacts / gates

`reports/2026-07-27-coin-r9-causal-residual-delivery/r10c2_approach_mechanism.json`; code `theta_option/hybrid_approach.py`,
harness `--c2`; tests `test_coin_r9.py` (causal exit-guard logic + bounds/monotone-order). ruff clean; no fn ≥ CC 15;
CORE.YAML untouched; blind panel never evaluated.
