# TARGET_DIRECTED_LAUNCH_V1 — aiming the proven impulse at the zone

**Date:** 2026-07-25
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4_INTERMITTENT_CONTACT` + the B1 passive barrier. No retraining. The
**only** thing that varies is how the tip aims the (already-proven) impulse.
**One-line outcome:** the launch-direction wall is **partially addressable** — contact-**point** selection (not
directed-velocity control) is the right lever and measurably reduces cross-track where the tip can reach the coin's far
side, but it is **not robust across states**: single-point contact geometry limits directional control on the harder
states. `DIRECTIONAL_CAPABILITY_EXISTS__STATE_CONDITIONING_REMAINS_INSUFFICIENT`.

---

## Diagnostics (the discriminating tests)

- **Contact Jacobian is rank-2** (singular values e.g. s1 [0.025, 0.008], cond 3–7 across states) — the coin *is*
  steerable in principle; this is **not** a rank-1 dead end.
- **Directed-velocity control (L1/L2) is ineffective.** L1 (target-frame parallel objective) is mathematically identical
  to L0; L2 (cross-track suppression) barely moves s1 (cross ratio 0.658→0.618 even at k_cross=20) and **worsens** s6
  (1.47→2.78). The cross-track is set by **where the tip contacts the coin**, not correctable by local velocity commands.
- **Contact-POINT selection (L3) is the right lever.** Acquiring the coin's **far side** (opposite the zone) so the push
  goes through the centre reduces cross-track where directed-velocity control could not — s1: cross ratio **0.658 → 0.432**,
  launch angle 33.3° → 23.4°, and v_parallel rises to 0.318 (> the 0.305 target) with more transport.

## Benchmark (8 states, L0–L3, pre-registered gate)

Pre-registered target-directed gate: `v_∥ ≥ 0.85·v_target ∧ |v_cross|/v_∥ < 0.2 ∧ signed_disp > 0 ∧ joint ≤ 3.45`.

| state | cross ratio L0→L2→L3 | v_∥ (L3) | v_target |
|---|---|---|---|
| s1 | 0.66 → 0.66 → **0.43** | 0.318 (> target) | 0.305 |
| s3 | 0.89 → 0.89 → 0.89 | 0.198 | 0.573 |
| s4 | 1.28 → 1.34 → 1.34 | 0.127 | 0.539 |
| s6 | 1.47 → 1.46 → 1.46 | 0.164 | 0.488 |
| s7 | 4.49 → 4.48 → 4.48 | 0.100 | 0.751 |
| s0,s2,s5 | ~0 / huge (v_∥≈0) | ≈0 | 0.52–0.62 |

- **Target-directed (strict gate): 0/8** for every stage — no state reaches both v_∥ ≥ 0.85·v_target *and* cross < 0.2.
- **L3 reduces cross-track on 2/8 states** (clearly s1). On the rest, the single tip either barely moves the coin toward
  the zone (s0/s2/s5: v_∥ ≈ 0) or pushes heavily diagonally (s6/s7: cross ≫ v_∥) — the far-side contact point is
  unreachable / geometry-blocked there.
- (The panel-*mean* cross ratio is meaningless — v_∥→0 sends cross ratio→∞ on s5; use the per-state table + median.)

```
VERDICT: DIRECTIONAL_CAPABILITY_EXISTS__STATE_CONDITIONING_REMAINS_INSUFFICIENT
```

## Interpretation

The magnitude of the impulse is not the problem (proven earlier) and neither is directed-velocity control (the local
contact Jacobian cannot re-aim a push whose direction is fixed by the contact point). The load-bearing lever is the
**contact point**: contacting the coin on the far side aims the push through the centre. That mechanism *works* (s1), which
is a genuine directional-capability result — but a **single** point of contact cannot be placed on the far side in every
state (workspace / approach-angle limits), so the launch is not yet target-directed robustly. This is exactly the
pre-registered boundary at which a **two-point / edge-aware** contact becomes warranted.

## Claims / non-claims

**Claimed (measured):** directed-velocity control (L1/L2) cannot re-aim the launch (cross-track is set by the contact
point; k_cross even worsens some states); contact-point selection (L3, far-side) is the correct lever and reduces
cross-track on states where the far side is reachable (s1: 0.658→0.432, v_∥ reaches target).

**NOT claimed / provisional:** no state passes the strict target-directed gate; L3 helps only 2/8; the single-point
geometry limit on the hard states is inferred from v_∥≈0 / cross≫v_∥, not from an exhaustive contact-point search.

## Exact next gate

- **L5 — two-point / edge-aware contact.** The single-tip far-side lever is proven but state-limited; a two-point pinch or
  an edge-aware contact can place the push-line through the coin centre from more approach angles. This is the one
  remaining scenario change, pre-registered as the last resort, now justified by L3's partial success.
- Keep the composed pipeline ready: **target-directed launch → predictable coast → B1 passive barrier**. Only once the
  launch is target-directed on a robust subset is a proposal / bounded search / RL layer justified. **O3 stays paused.**

---

### Commits
- `110190ec` — directed-launch controller (L0–L3) + benchmark + tests.
- (robust-stats + this report) — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 passive-barrier brake, all
prior results.
