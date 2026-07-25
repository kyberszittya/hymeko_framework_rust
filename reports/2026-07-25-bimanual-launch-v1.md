# BIMANUAL_TARGET_DIRECTED_LAUNCH_V1 — cooperative force allocation; acquisition is the wall

**Date:** 2026-07-25
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4_INTERMITTENT_CONTACT` + coast target + B1 barrier. Deterministic
teacher, no RL.
**One-line outcome:** the two-arm cooperative **force-allocation mechanism is sound** (a 3×4 coin-twist Jacobian
commanding translation-toward-zone + zero-spin keeps ω_c ≈ 0), but **simultaneous two-contact acquisition is unreliable**
— both tips engage on ≤ 1/8 states — so the cooperative resultant is rarely exercised and no state passes the strict
target-directed gate. `BIMANUAL_ACQUISITION_UNRELIABLE__BOTH_TIPS_RARELY_ENGAGE`.

*(Framing, from the user: two-arm simultaneous contact is the essential task mechanism, not a last resort; single-tip was
the control that proved impulse magnitude, the coast model, the passive barrier, and localized the residual error to
contact-point force-direction. This experiment tests the original two-arm hypothesis directly.)*

---

## The mechanism (and why it's right in principle)

Two contacts give the coin **wrench**: the resultant `F_L + F_R` (translation) and the torque `τ_c` about the coin are
partly *separately* shapeable — which a single contact point cannot do (there the push-line is fixed by the accessible
point). Formulated cleanly: the coin is a planar body with twist (v_x, v_y, ω); the **3×4 coin-twist Jacobian** J_t maps
the two arms' 4-DoF action to the twist, and once both tips contact we solve
`Δq = Jₜᵀ(W Jₜ Jₜᵀ + λ²I)⁻¹ W([v_target·e_∥ ; 0] − twist)` — translate toward the zone with **zero spin**.

**Where both tips engage, ω_c stays ≈ 0** (0.004–0.019 rad/s) — the force-line passes through the COM, exactly as the
mechanism predicts. So the cooperative-resultant idea is correct.

## Benchmark (8 states; A0 single-tip vs A1/A2/A3 bimanual)

Pre-registered gate: `v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ signed_disp > 0 ∧ joint ≤ 3.45`.

| arm | target-directed | both-tips contact rate |
|---|---|---|
| A0 single-tip L3 | 0/8 | — |
| A1 bimanual symmetric | 0/8 | 0.125 |
| A2 bimanual state-dep | 0/8 | 0.0 |
| A3 bimanual balanced | 0/8 | 0.0 |

- **Best single cell:** s3 — A1 drives cross ratio 0.887 → **0.195** (below the 0.2 gate) with ω 0.004; but v_∥ (0.246)
  falls short of 0.85·v_target (0.487), so it does not pass the *full* gate — and it did so with a single effective
  contact (both-tips 0), i.e. the symmetric acquire *aimed* the push, it was not a true two-contact resultant.
- **Both-tips contact almost never happens** (≤ 1/8). The state-dependent (A2) and force-balanced (A3) variants do not
  help because the cooperative solve is only active when both tips are in contact — which is the rare case.
- On several states A2/A3 make cross-track *worse* (the two-sided acquire disturbs the coin before engaging).

```
VERDICT: BIMANUAL_ACQUISITION_UNRELIABLE__BOTH_TIPS_RARELY_ENGAGE
```

## Interpretation — the wall is the relational acquisition decision

The two-arm force-allocation is not disproven; it is *un-exercised*, because **placing both arms in simultaneous, useful
contact with a small disk is itself the hard problem** — and it is inherently **relational**: which two contact points are
*simultaneously reachable* by the two arms, from this morphology and start configuration, so that the resultant can be
aimed at the zone. A hand-coded symmetric / nearest-surface acquire does not solve it robustly.

This is exactly the structural decision the later prior study targets:
`left arm ↔ left contact arc, right arm ↔ right contact arc, pair ↔ common resultant, coin COM ↔ zone axis,
morphological reachability ↔ selectable pair`. The benchmark **saves per-state teacher records** (both-tips, simultaneity,
force imbalance, force-line-miss ω, cross ratio) — the deterministic teacher for **flat vs engineered-flat vs HyMeKo** on
*contact-pair / force-allocation selection*, testing whether the contact-relational graph picks a simultaneously-reachable,
target-directed pair from fewer demonstrations and on new morphologies.

## Claims / non-claims

**Claimed (measured):** the cooperative twist-allocation mechanism is sound (ω_c ≈ 0 where both engage; s3 cross ratio to
0.195); no arm passes the full target-directed gate on any state; simultaneous two-contact acquisition is the binding
constraint (both-tips ≤ 1/8).

**NOT claimed:** that two arms *cannot* close the gap — only that a hand-coded acquire does not reliably achieve
simultaneous two-contact. A better contact-pair selection / sequential reposition (and, per the research plan, a learned
structural policy) is the open path. The embodiment is **not** proven geometrically incapable.

## Exact next gate

1. **Simultaneous two-contact acquisition** as its own sub-problem: a contact-pair *reachability* planner (which
   L/R contact arcs are jointly reachable) + sequential reposition→co-contact→launch, so the cooperative solve is actually
   exercised across the panel.
2. Then the **structural-prior study** (Kato/HyMeKo): flat vs engineered-flat vs HyMeKo on the relational contact-pair /
   force-allocation decision, using these teacher records — sample efficiency + generalization to new configs.
3. **O3 stays paused** until the coin has a robust target-directed launch.

---

### Commits
- `4d2fbe7f` — bimanual controller (twist-Jacobian force allocation) + benchmark + teacher records + tests.
- (this report) — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
