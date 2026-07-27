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

## Next (C2 refinement → C3)

Add the **HOLD/TRANSPORT** mode (maintain momentum / bounded coast to the target corridor; no active slowing) between
APPROACH and BRAKE, and/or make APPROACH exit on REACHABILITY rather than an early LAUNCH; re-run C2-D; then C3 (manual
hybrid program, s1 ∧ s3 strict K6, safety 2/2, R6 unchanged). The blind panel stays sealed.

## Artifacts / gates

`reports/2026-07-27-coin-r9-causal-residual-delivery/r10c2_approach_mechanism.json`; code `theta_option/hybrid_approach.py`,
harness `--c2`; tests `test_coin_r9.py` (causal exit-guard logic + bounds/monotone-order). ruff clean; no fn ≥ CC 15;
CORE.YAML untouched; blind panel never evaluated.
