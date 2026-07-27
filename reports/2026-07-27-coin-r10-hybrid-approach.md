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

## Next

The most promising path is a **reachability-gated RELEASED_COAST**: APPROACH builds momentum → when the coin can coast into
the zone (reachability), release under a *launch* guard (separate from the R6 settle certificate) → coast → SETTLE. Contrast
with an active HELD_MOMENTUM_CARRY primitive if the coast proves too sensitive. Then C3 (manual hybrid program, s1 ∧ s3
strict K6, safety 2/2). The final settle/K6 certificate and the physics stay frozen; the blind panel stays sealed. This is
genuine hybrid-system identification — the modes are discovered from the unexpressible teacher components, not drawn on by hand.

## Artifacts / gates

`reports/2026-07-27-coin-r9-causal-residual-delivery/r10c2_approach_mechanism.json`; code `theta_option/hybrid_approach.py`,
harness `--c2`; tests `test_coin_r9.py` (causal exit-guard logic + bounds/monotone-order). ruff clean; no fn ≥ CC 15;
CORE.YAML untouched; blind panel never evaluated.
