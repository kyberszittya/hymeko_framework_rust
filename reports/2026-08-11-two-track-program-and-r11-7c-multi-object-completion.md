# Two parallel research ladders + R11.7C — Multi-Object Exact-Zero Delivery Completion

**Date:** 2026-08-11 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Type:** program framing / scope correction (no code, no experiment run) — a durable record so the multi-object
delivery program is **not lost** beside the R12 rotor program.

## Why this document exists (the scope correction)

The **coin (O0) line is functionally complete** — full teacher-free exact-zero pipeline, well-characterized retrieval
limit. But the claim *"we did the same for other objects"* is **not yet true**. What we have on non-coin objects are
**benchmark characterizations, not per-object polished autonomous policies**:

- **O1-L** (larger disk, size variant): reach 1.0, capture partial, teacher K6|capture strong; teacher-free retrieval weak.
- **O2-M** (heavier disk, dynamics variant): ran in the multi-object characterization; the **structured teacher
  generalizes strongly** to this family; teacher-free full-chain not established.
- **O4-S** (box, first non-circular): first teacher-free exact-zero **strict-K6 existence** success (`9dac7c0e`,
  dtz 17.72 mm); teacher works well, but **stable deploy retrieval** and **capture-seed consistency** are not solved.
- The **unified architecture is done**: HyMeKo-spec → `ObjectSpec` → generated MuJoCo rig → the *same* exact-zero
  pipeline (`R11.7A`, `8c5a6937` / `d52a74f9`).

⇒ The **"push different physical objects in and deliver them" program is NOT closed.** The coin is the first *complete*
benchmark; the box proved the architecture transfers to a non-circular object; O1/O2 proved the structured teacher
generalizes. The families still need to be carried through to **stable teacher-free full-chain delivery** with a proper
**held-out** measurement. This must not be dropped in favour of R12/rotor.

## Two parallel research ladders

There are now **two distinct ladders**, to be built side by side and composed later:

```
TRACK A — Manipulation benchmark (the physical task ladder)
  coin (O0)
    → scaled disk (O1)
      → dynamics variant (O2)
        → box (O4)
          → ellipse / capsule (O3)
            → more shape / mass / friction (O5+)
              → stable teacher-free exact-zero delivery

TRACK B — Architecture (the model ladder)
  flat baseline
    → HSiKAN
      → Steiner / block designs
        → quaternion / rotor
          → Rotor-Spike
            → dynamic HyMeKo incidence
              → k-actor × n-critic tensor RL
```

**The final question that unifies both:**

> **Which architecture learns to reliably deliver different physical objects from an exact-zero state to a target?**

This is a far better exam than measuring HSiKAN/Steiner/rotor on ever-more synthetic ranker datasets. **Track B's real
grade is Track A's benchmark** — specifically the *teacher-free* column, where the current amortized demo-retrieval
provably stalls (the coin's object-independent retrieval wall, re-proven on the box in R11.7B). The multi-object
teacher-free delivery gap **is** the benchmark the structured architectures must close.

## R11.7C — Multi-Object Exact-Zero Delivery Completion

**Goal.** Carry **O1, O2, O4, then O3** through the **same frozen protocol** to stable teacher-free full-chain delivery
and a real held-out evaluation — filling in the question marks in a stable benchmark table. This runs **in parallel with
(not before/after) R12.4** — Track A and Track B are independent and neither closes the other.

**Per-family measurements (frozen set):**
- reach rate
- capture rate
- teacher K6 | capture
- teacher-free K6 | capture
- overall exact-zero K6 (full chain)
- safety
- held-out generalization
- failure taxonomy (1 primary cause / rollout; reuse the U6B taxonomy:
  `MODEL_OR_CONTRACT / REACH_GEOMETRY / CAPTURE_PROPOSAL_TRANSFER / CERTIFICATE_GEOMETRY / CONTACT_RETENTION /
  DELIVERY_PROGRAM_TRANSFER / RETRIEVAL_OUT_OF_SUPPORT / TARGET_ENTRY`)

**Acceptance is a stable benchmark *table*, not 95% per object.** A per-family "done" gate (from the R11.7B NEXT
LEVERS): **FIXED sealed dev/test split, ≥3 seeds, capture ≥ 80%, conditional-K6 ≥ 50%, overall-K6 ≥ 40%** — reported
honestly, with the retrieval wall documented per family where it bites rather than papered over.

### The honest benchmark table (the question marks are the work)

| Object | Reach | Capture | Teacher K6\|cap | Teacher-free K6\|cap | Overall exact-zero K6 | Held-out | Status |
|---|---|---|---|---|---|---|---|
| **O0 coin** | ✓ 1.0 | ✓ (placement-dependent) | ✓ | ✓ (autonomous, train 44/44, dev 4/7) | ✓ (deploy = descriptor-nearest, dev 0.571) | characterized (retrieval-limited) | **FULL CHAIN ✓** — first complete benchmark |
| **O1-L** large disk | ✓ 1.0 | partial (4/6 smoke) | ✓ 0.82–1.00 | ✗/? (retrieval 0.0–0.11) | ? | ? | teacher generalizes; teacher-free delivery **OPEN** |
| **O2-M** heavy/dynamics | ✓ 1.0 | partial (4/6 smoke) | ✓ 0.50–1.00 | ? (retrieval ~0.50 where tested) | ? | ? | teacher generalizes; full-chain **OPEN** |
| **O4-S** box | ✓ 1.0 | partial (5/6 smoke; ~52% seed-consistency) | ✓ (7/8 train scenarios) | **existence ✓** (17.72 mm; an interpolation, not a stored θ) | not yet stable (top-1 0/6) | retrieval wall (coin-mirror) | existence proven; **robust deploy BLOCKED** |
| **O3** ellipse/capsule | — | — | — | — | — | — | object-family **branch not built** (needs `Shape` member + geom branch) |

*(O5-R rectangle exists as the R12 orientation object but has not been run through the exact-zero delivery chain; it
enters Track A at the O5+ "aspect-ratio" rung.)*

### Order of work (user-set)

1. **O4 box** — stabilize capture-seed consistency; then either close robust teacher-free retrieval **or explicitly
   accept the coin-mirror retrieval wall** and report the teacher-conditional table (do **not** re-grind the retrieval
   wall — R11.7B closed it object-independently: `BOX_DELIVERING_THETA_IS_OUTLIER`, `6363f3e2`).
2. **O1-L** size variant — stable exact-zero capture → stable teacher-free delivery → held-out.
3. **O2-M** dynamics variant — same full chain.
4. **O3** ellipse/capsule — first build the **object-family branch** (`Shape` member + geom branch), then capture +
   delivery + retrieval.
5. **O5+** later: differing aspect ratio, friction, density/mass, triangle/wedge, and eventually 3-D cuboid/wedge
   (where Track A meets Track B's R12.4 SO(3) substrate).

### Honesty carry-forward (do not re-derive the hard way)

- **Teacher K6 | capture is SOLVED and object-robust** (0.82–1.00 across size/dynamics/shape, `2c215634`) — physical
  solvability is not the bottleneck.
- **Teacher-free robust retrieval is the bottleneck, and it is the coin's KNOWN object-independent wall** (delivering θ
  is a handoff-specific transfer outlier no top-1 heuristic — nearest / k3-blend / physics-match — surfaces;
  `BOX_RETRIEVAL_SELECTION_LIMIT` → `BOX_DELIVERING_THETA_IS_OUTLIER`). So "completion" of the teacher-free column may
  mean **documenting the wall per family**, not hitting 95% — and that gap is precisely **Track B's exam**.

## Relationship to R12 (the guard)

- **R12.4 (Track B, 3-D rotor substrate)** and **R11.7C (Track A, multi-object delivery completion)** are **parallel**.
- Building the 3-D rotor branch does **NOT** close or replace the multi-object delivery program, and vice-versa.
- The R12.1 static-ranker negative (`STATIC_INCIDENT_..._NOT_SUPPORTED`) and the R12 planar closure say nothing about
  whether a better architecture can close Track A's teacher-free column — that is the whole point of running them together.

## Status

No code written, no experiment run — this is a framing/record commit. Working tree clean apart from pre-existing
untracked artifacts unrelated to this arc. When R11.7C implementation begins, it gets a 4-format plan under
`docs/plans/` per the operating contract (Section 2) before any code.
