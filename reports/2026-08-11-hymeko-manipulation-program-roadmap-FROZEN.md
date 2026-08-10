# HyMeKo manipulation program — FROZEN roadmap (2026-08-11)

**Branch:** `feature/r11-4a-target-conditioned-delivery-teacher` · **Type:** canonical program freeze (no code, no run).
This is the master index tying R11 / R11.7 / R12 together. Detail lives in the linked per-arc reports; this document
is the *structure and ordering* the user froze on 2026-08-11.

## The research story (one paragraph)

HyMeKo generates the physical manipulation world from **one structural description**: `HyMeKo file → ObjectSpec →
canonical rig generator → exact-zero manipulation pipeline`. The system first delivers a **coin**, then **different
physical objects**, to a target from an exact-zero home. We then lift the flat baseline into **dynamic-hypergraph,
rotor-based, and multi-actor / multi-critic** architectures, and ask whether the **structural prior actually improves
physically-relevant generalization**. The coin line does not disappear — it stays the **canonical reference task**.

**Three parallel tracks, composed later. The multi-object teacher-free completion (Track A) is still open and must not
be lost beside the rotor/HSiKAN program (Tracks B/C) — the two must feed each other.**

---

## 0. DONE and banked

### R11 — Coin delivery ✓
`exact-zero HOME → reach → capture → teacher-free delivery → strict K6`. Teacher coverage strong; exact-zero teacher-free
successes exist; the retrieval ceiling is **causally characterized** (residual = local handoff×target coverage/selection
limit). **Do not grind coin retrieval heuristics further** (`R11.6C` `833b5e98`; `R11.6D` closed = target-geometry +
descriptor-nearest dev 0.571).

### R11.7A/B — Multi-object foundation ✓
`HyMeKo file → ObjectSpec → canonical rig generator → exact-zero pipeline` (`8c5a6937` / `d52a74f9`). O0/O1/O2/O4
characterized: reach ~object-invariant; **structured teacher generalizes across size/mass/shape**; box gives the first
**non-circular teacher-free exact-zero strict-K6** (`9dac7c0e`, 17.72 mm); demo-bank retrieval unreliable across several
families (`BOX_DELIVERING_THETA_IS_OUTLIER` `6363f3e2`). This is already a real multi-object benchmark.

### R12.1–R12.3 — Planar structured / orientation arc ✓ (closed)
Static HSiKAN/Steiner incidence does **not** beat flat (`STATIC_INCIDENT_..._NOT_SUPPORTED`); orientation-specific
physics exists; absolute orientation mostly redundant; relative geometry carries a small real signal; in SO(2)
`sin/cos ≈ quaternion ≈ rotor`. **Official transition:** *the planar substrate cannot meaningfully distinguish the rotor
hypothesis from a symmetry-aware relative-angle representation; a fair rotor test requires genuinely non-commutative 3-D
manipulation physics.*

---

## TRACK A — Multi-object manipulation (STAYS ACTIVE)

### A1 · R11.7C — Multi-Object Exact-Zero Delivery Completion
Fill the question marks; measure every object on the same frozen protocol:
`reach rate / capture rate / teacher K6|capture / teacher-free K6|capture / overall exact-zero K6 / safety /
held-out generalization / failure taxonomy`.

| Object | State |
|---|---|
| O0 coin | ✓ complete |
| O1 scaled disk | partial |
| O2 heavy / dynamics | partial |
| O4 box | existence ✓, stability still missing |
| O3 ellipse | not yet built |

### A2 · O3 ellipse / capsule (the next shape family — an intermediate rung)
`circle → ellipse → rectangle/box`. Orientation already matters here, but there is **no full 3-D tumbling yet** — the
deliberate bridge between the symmetric coin and the fully non-commutative 3-D substrate. Needs a new `Shape` member +
a geom branch (currently parked in `object_curriculum.py`).

### A3 · Physical variations (later)
`size · mass · friction · aspect ratio · shape · contact geometry` → a proper **object-generalization benchmark**.

---

## TRACK B — Genuine structured / rotor architecture

### B1 · R12.4A — Canonical SO(3) benchmark (the next major new build)
**Start with a 3-D task where orientation is causally necessary, NOT with a neural model.**
E.g. `cuboid/wedge → face contact → tipping → edge-supported rotation → corner/edge transition → target pose`.
Mandatory: non-axisymmetric object · non-commuting rotations · face→edge→corner mode switch · orientation-dependent
stability · ω / angular momentum · translation-only policy insufficient.

**State-ablation gate (must pass first):** `x < x,v < x,v,R < x,v,R,ω`. **If R and ω are not load-bearing, the
benchmark is wrong — rebuild it.**

### B2 · R12.4B — Quaternion vs rotor, done properly
Not `4 quaternion floats vs 4 rotor floats` but **quaternion-as-coordinate vs rotor-as-geometric-operator**: relative
rotor, inverse/composition, frame transforms, bivector / rotation-plane info, rotor action on the contact vectors.
The question: **does the geometric-algebra operation add anything beyond the encoding?**

### B3 · R12.4C — Rotor Spike (a dynamic mechanism)
Events: large ΔR · face→edge · edge→corner · contact-normal crosses principal axis · ω sign reversal · angular-momentum
jump · hybrid-mode transition. Payload: relative-rotor delta · contact-frame transition · Δ angular momentum ·
structural surprise · confidence change. (Connects to the Highway-Spikes idea.)

### B4 · R12.5 — Dynamic HyMeKo (hypergraph no longer static)
`face-contact → H_face · edge-contact → H_edge · corner/tumbling → H_corner`. The HyMeKo IR generates — from the **same
physical world** — model, contact structure, modes, certificates, and HSiKAN incidence. A far stronger test than R12.1's
fixed incidence.

### B5 · HSiKAN returns — in a new role
The static-ranker branch is a closed negative. Still open: **dynamic HSiKAN critic · mode-conditioned HSiKAN ·
rotor-aware HSiKAN · contact-hypergraph HSiKAN.** Controls kept honest: `MLP · random-sparse · degree-matched-random ·
task-derived HyMeKo · Steiner/block-design`.

---

## TRACK C — k-actor × n-critic tensor RL (comes after B)

**Actors:** push · tip · rotate · edge-guide · catch · stabilize · release · recovery.
**Critics:** task-success · safety · contact-stability · orientation · angular-momentum · robustness · energy ·
target-pose.

$$ Q_{a,c,p,h,\mu,e}(s) $$

`a` actor · `c` critic · `p` phase · `h` handoff / contact regime · `μ` mechanical mode · `e` embodiment / object /
environment. **Steiner returns here as a routing / incidence structure** — `actor ↔ critic ↔ hybrid-mode ↔ spike-channel`
— NOT the role where it scored zero in R12.1.

---

## Frozen ordering

```
DONE
  R11        coin exact-zero delivery
  R11.7A/B   HyMeKo-generated multi-object benchmark
  R12.1–3    planar structure / orientation characterization

NOW
  R11.7C     multi-object exact-zero completion
               O1 / O2 / O4 stabilization
               O3 ellipse / capsule

NEXT MAJOR
  R12.4A     canonical SO(3) manipulation task   (state-ablation gate first)
  R12.4B     quaternion vs rotor-as-operator
  R12.4C     Rotor Spike

THEN
  R12.5      dynamic HyMeKo incidence
  R12.6      k-actor × n-critic tensor RL         (= Track C)

LATER
  Steiner / block-design routing
  cross-object / cross-embodiment transfer
  energy / Hamiltonian layer
```

## Guards (binding)

- **Tracks A and B/C run in parallel; neither closes the other.** Building the rotor branch does not complete the
  multi-object teacher-free delivery, and the planar/R12.1 negatives say nothing about whether a better architecture
  closes Track A's teacher-free column. **The two must feed each other.**
- **The coin stays the canonical reference task** — it is not superseded, it is the control.
- **NOW = R11.7C** (Track A). **NEXT MAJOR = R12.4A** (Track B), and its **state-ablation gate is a HALT condition**:
  if R/ω are not load-bearing, the 3-D benchmark is rebuilt before any representation question.
- Each implementation step gets a 4-format plan under `docs/plans/` (operating contract §2) before any code.
