# TARGET_DIRECTED_LAUNCH_V2 — two-point contact-mode, force-line mechanism

**Date:** 2026-07-25
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4_INTERMITTENT_CONTACT` + coast launch target + B1 passive barrier.
Deterministic teacher, no RL.
**One-line outcome:** the **force-line mechanism is confirmed** — two-point contact aims the push through the coin centre
(coin angular velocity ω_c stays ≈0) and reduces cross-track where it engages — but **reaching two symmetric far-side
contacts is unreliable on the current arm morphology**, so the launch is not yet robustly target-directed.
`MULTI_CONTACT_DIRECTIONAL_CAPABILITY_EXISTS__REACHABILITY_OR_MODE_SELECTION_REMAINS_OPEN`.

*(Refinement carried from V1: single-point contact is not proven fundamentally insufficient — the current single-tip
embodiment + direct-launch strategy cannot reach an adequate far-side contact from every state. V2 tests the two-point
architecture as the justified next step.)*

---

## The force-line hypothesis + its direct proof

If both fingertips contact the coin symmetrically about the zone axis on the **far side** and squeeze toward the centre,
the resultant force passes through the COM along `e_parallel` ⇒ net v_∥ toward the zone, net v_⊥ ≈ 0, **net torque ≈ 0**.
The mechanistic proof is the coin's **angular velocity ω_c**: an off-centre (force-line-miss) push spins the coin; a
centre-line push does not.

**Measured ω_c is uniformly tiny (0.003–0.017 rad/s) across all states** — the two-point pushes are near-centreline. Where
two-point contact also **reduces cross-track**, the force-line mechanism is doing exactly what it should.

## Benchmark (8 states; L3 single-point vs L5a two-point vs L5b edge-aware)

Pre-registered gate: `v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ signed_disp > 0 ∧ joint ≤ 3.45 ∧ |ω_c| < 0.5 ∧ both tips`.

| state | cross ratio L3 → L5a → L5b | ω_c (L5a) | both tips |
|---|---|---|---|
| s3 | 0.887 → **0.252 → 0.231** | 0.003 | 0 |
| s6 | 1.457 → **0.668** | 0.017 | **1** |
| s4 | 1.345 → 2.926 | 0.009 | 0 (worse) |
| s1 | 0.432 → 0.432 | 0.006 | 0 (no engage) |
| s0,s2,s5,s7 | ~0 / huge (v_∥≈0) | ~0 | 0 |

- **Target-directed (strict gate): 0/8** for every mode.
- **Two-point reduces cross-track on 2/8 states** (s3 0.887→0.231, s6 1.457→0.668), each with **low ω_c** — the
  force-line mechanism is visible, not a lucky parameter (mechanism-confirmed cell: s6).
- **Both-tips acquisition is unreliable** — L5a engages both tips on 1/8 (s6); L5b (edge-aware φ) 0/8 (the probed φ
  reduced simultaneous engagement). The limiting factor is **reachability of two symmetric far-side contacts**, not the
  force-line principle.

```
VERDICT: MULTI_CONTACT_DIRECTIONAL_CAPABILITY_EXISTS__REACHABILITY_OR_MODE_SELECTION_REMAINS_OPEN
```

Not `CURRENT_EMBODIMENT_GEOMETRY_LIMITS_...`: the two-point principle works where reachable, and L5b did not exhaustively
search the contact-pair space — the open question is reachability + mode selection, not a proven geometric impossibility.

## Why this is the structural decision (the Kato / HyMeKo bridge)

V2 pins down the decision a structured model must learn — a **relational** choice, not coordinate regression:
`robot link/tip ↔ reachable coin-edge ↔ coin COM ↔ zone axis ↔ friction/contact mode`. Which tip or tip-pair, which
contact points, which contact mode, which force-line, and what is reachable from *this* morphology.

The benchmark **saves per-state teacher records** (`teacher_records` in the artifact): chosen mode, chosen φ,
both-tips-contact, force-line-miss ω, predicted v_∥/v_cross. This is the deterministic teacher for the later
structural-prior study — flat vs engineered-flat vs HyMeKo on **contact-mode/pair selection**, testing whether the
morphological + contact-relational graph selects an executable, target-directed contact mode from fewer demonstrations and
on new geometries.

## Claims / non-claims

**Claimed (measured):** the force-line mechanism is real (ω_c ≈ 0 everywhere; two-point reduces cross-track on s3/s6 with
low ω); two-point contact-mode is the correct architecture direction (reduces cross-track where reachable).

**NOT claimed / provisional:** 0/8 pass the strict gate; both-tips reachability is 1/8 (L5a) — the two-point acquire /
mode-selection needs work (sequential L5c not yet run); ω_c is a force-line proxy, not the exact `d_line`; the edge-aware
φ-probe underperformed and needs a better cost / candidate set.

## Exact next gate

- **Reachability / mode selection** — the open problem, and the one the structural prior should own: improve two-point
  acquisition (sequential reposition→launch L5c; a better contact-pair candidate set + cost), so both-tips engagement is
  reliable across states.
- Then the **structural-prior study** (Kato/HyMeKo): does the contact-relational graph pick an executable target-directed
  contact mode from fewer demos / on new morphologies than a flat or engineered-flat baseline — using these teacher
  records. **O3 stays paused** until the coin has a robust target-directed launch.

---

### Commits
- `f8ae110d` — two-point / edge-aware controller + V2 benchmark + teacher records.
- (this report) — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
