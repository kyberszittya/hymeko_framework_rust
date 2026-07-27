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

---

## R3-A → R3-C → C3-D → K6-decomposition (2026-07-27, added) — the true wall localised

**R3-A catch-point (H1 vs H2).** Swept release-timing × catch-corridor × closing-velocity with the controller unchanged.
Gentle catches reachable, but the frozen settle never closed s1 → **`MICRO_TRANSPORT_MODE_REQUIRED`** (H2): catch-timing
alone does not deliver; a transport d.o.f. is missing. (Corrected an earlier read: the *gentle* re-grips land ~47–53 mm, not
31 mm — the 31 mm chain was a non-gentle catch.)

**R3-B/C micro-transport.** One bounded `delta_forward` over the FROZEN settle after the fixed C2.8 re-acquire.
`R3-B0` update-zero holds bit-exact (`micro_forward=0` ≡ base). Best nudge 49.4 mm, no K6 → **`MICRO_TRANSPORT_INSUFFICIENT`**.
Deep finding: the gentle re-grip **stops** the coin (~48 mm), and transport-from-stopped hits the **R1 soft-frictional wall**
— grip-transport (tip push) does not translate into coin motion. Fundamental tension: gentle re-grip needs stopping;
delivery needs momentum *through* the re-grip.

**C3-D velocity-matched capture (rho = v_after/v_before).** Mass in-process CEM (288 evals) over 6 capture params for a
capture that **moves with the coin** (tip velocity ≈ coin `v_par`, grip after an onset delay). Result: **rho = 1.0** — the
momentum-preservation tension is **resolved**; a legal post-coast contact that fully preserves momentum EXISTS (vs the
stopping re-grip's rho≈0). But delivery still fails (47.4 mm): the preserved momentum cannot be *used* — grip-transport hits
the same R1 wall, and the capture fires near the coast landing (coin already slow). Only **free coast** moves the coin, and it
stops at ~31–58 mm short of the zone.

**K6-decomposition audit (`--k6decomp`).** Re-graded representative s1 trajectories by K6-Z (spatial ≤20 mm) / K6-V (dynamic
settle) / K6-D (persistence) + a pre-registered tolerance sensitivity table (a *re-evaluation*, not a controller search):

| trajectory | min_dtz | reached zone | terminal speed | dwell | K6-F |
|---|---|---|---|---|---|
| scaffold | 57.7 mm | **no** | 0.0002 | 0 | False |
| teacher | 18.5 mm | yes | 0.0 | 24 | **True** |
| hybrid_vmc | 48.7 mm | **no** | 0.0 | 0 | False |

Tolerance sweep (speed 0.06, dwell 6): teacher passes at 20/25/30 mm; the hybrid fails at **every** tolerance up to 30 mm
(needs ≥50 mm). Reading: the hybrid comes to **rest** (terminal speed ~0, so K6-V/K6-D are *not* the failing terms) but
**outside** the zone — the failure is **purely SPATIAL (K6-Z)**. The frozen K6 is **not** requiring the impossible (the
teacher clears it at 18.5 mm), and loosening it to 25/30 mm would rescue nothing while ≥50 mm would *redefine* the task.

**Architectural adoption (from the user's K6 discussion).** The **launch/release guard** ("safe to release now, will the
free coast land in the corridor?") and the **final K6 settle certificate** ("did the coin end at rest in the zone?") are two
separate objects; the campaign coupled them too tightly (a pre-release *rest* certificate is the wrong abstraction for a
delivery that is necessarily post-release motion). Correct pipeline: `APPROACH_MOMENTUM_BUILD → LAUNCH/RELEASE GUARD →
RELEASED_COAST → (optional closed-loop correction) → FINAL SETTLE → K6-F`. This is now adopted — but it is **orthogonal** to
the barrier: even with the separation, the coin rests at 48 mm, not 20 mm.

**Localised barrier (measured, not asserted).** The coin can be moved **only** by free coast (grip-transport / capture cannot
finely translate it — R1 wall, confirmed by R3-C *and* C3-D rho=1.0-but-no-K6). The free coast from the APPROACH momentum
stops at ~48–58 mm. The teacher delivers by **release timing/momentum** (its coast lands at 18.5 mm), which C2.6 found bounded
by **coast uncertainty** for a point-estimate guard. So the remaining s1 lever is **coast-landing precision** (a
per-cradle-tuned or learned release guard) — **not** more capture/transport tuning (dead end) and **not** K6 recalibration
(correctly calibrated). This is the strategic-pivot boundary the mass search was set to find.

Artifacts: `…/r10r3a_catchpoint.json`, `r10r3bc_micro_transport.json`, `r10r3d_velocity_matched_capture.json`,
`k6_decomposition.json`; harness `--c29/--c30/--c31/--k6decomp`. ruff clean; all fns < CC 15; CORE.YAML untouched; blind sealed.
