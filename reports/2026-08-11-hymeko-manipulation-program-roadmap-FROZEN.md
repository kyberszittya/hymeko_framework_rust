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

**Three tracks, composed later. Build order (user 2026-08-11) is `C → A → B`:** first the **reduced tensor-RL engine**
(Track C0, `μ`-free, on the substrate we already have), then the **multi-object completion** as its testbed (Track A),
finally the **3-D rotor substrate** (Track B), which re-enriches the engine with real mechanical modes `μ` (Track C1).
**The multi-object teacher-free completion is still open and must not be lost beside the rotor/HSiKAN program — the tracks
feed each other.**

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

## TRACK C — k-actor × n-critic tensor RL (REORDERED to run FIRST, as a reduced engine — user 2026-08-11)

**Build order reordered to C → A → B.** The full tensor indexes `μ` = mechanical mode, which only exists once Track B's
3-D face/edge/corner physics exists — so `μ` cannot be built before B. The dependency-coherent resolution: build a
**reduced, substrate-independent engine first**, validate it on the substrate we already have, then let Track B
re-enrich it with real `μ` later.

### C0 — reduced tensor-RL engine (NOW; `μ` dropped)
Build the k-actor × n-critic engine substrate-independently and validate it on the **existing** planar coin /
multi-object world (O0/O1/O2/O4):
- **Actors (planar-available):** push · stabilize · release · phase actors (reach / capture / deliver). The
  3-D-specific actors (tip · rotate · edge-guide · catch · recovery) wait for B.
- **Critics (planar-available):** task-success · safety · contact-stability · target-pose. (orientation is
  planar-limited per R12.2–3; angular-momentum / energy wait for B's dynamics.)
- **Q-tensor with `μ` DROPPED:** $Q_{a,c,p,h,e}(s)$ — `a` actor · `c` critic · `p` phase · `h` handoff / contact regime ·
  `e` embodiment / object / environment.
- **Steiner returns here as routing / incidence** — `actor ↔ critic ↔ phase ↔ spike-channel` — NOT the R12.1
  static-ranker null role.

**Why this is a strong move, not a detour:** the reduced engine's testbed *is* Track A's multi-object benchmark, so C0
directly attacks Track A's open **teacher-free column** — the exact question "does structured multi-actor / multi-critic
credit assignment deliver different objects where flat retrieval stalls?"

### C1 — full `μ`-enriched engine (AFTER B)
Once Track B exists, re-introduce `μ` (mechanical mode: face / edge / corner / tumbling), the 3-D actors (tip / rotate /
edge-guide / catch / recovery), and the angular-momentum / energy critics → $Q_{a,c,p,h,\mu,e}(s)$ (the original R12.6).

---

## Frozen ordering — REORDERED to C → A → B (user 2026-08-11)

```
DONE
  R11        coin exact-zero delivery
  R11.7A/B   HyMeKo-generated multi-object benchmark
  R12.1–3    planar structure / orientation characterization

NOW        (TRACK C — reduced engine)
  C0         reduced k-actor × n-critic tensor-RL engine, μ DROPPED  Q_{a,c,p,h,e}(s)
               + Steiner routing, validated on the EXISTING planar coin / multi-object substrate

THEN       (TRACK A — the engine's testbed)
  R11.7C     multi-object exact-zero completion  (O1 / O2 / O4 stabilization, O3 ellipse / capsule)

FINALLY    (TRACK B — the next major new build; re-enriches C with μ)
  R12.4A     canonical SO(3) manipulation task   (state-ablation gate first — HALT if R/ω inert)
  R12.4B     quaternion vs rotor-as-operator
  R12.4C     Rotor Spike
  R12.5      dynamic HyMeKo incidence
  C1 / R12.6 full μ-enriched k-actor × n-critic  Q_{a,c,p,h,μ,e}(s)

LATER
  cross-object / cross-embodiment transfer
  energy / Hamiltonian layer
```

*(Steiner / block-design routing is not a separate "later" item — it lives inside Track C as the actor↔critic↔phase
routing structure, from C0 onward.)*

## Guards (binding)

- **Build order is C → A → B (user 2026-08-11), and it is dependency-coherent, not arbitrary:** Track C's full form
  needs `μ` (mechanical mode) from Track B, so C runs first only as the **reduced, μ-free engine (C0)**; its testbed is
  Track A's multi-object benchmark; Track B then re-enriches C into its full form (C1 = R12.6).
- **The tracks still feed each other; neither closes the other.** The reduced engine (C0) directly attacks Track A's
  open teacher-free column; the planar / R12.1 negatives say nothing about whether this structured engine closes it.
- **The coin stays the canonical reference task** — not superseded, it is the control (and C0's first validation substrate).
- **NOW = Track C0** (reduced tensor-RL engine on the existing planar substrate). **THEN = R11.7C** (Track A testbed).
  **FINALLY = R12.4A** (Track B), whose **state-ablation gate is a HALT condition**: if R/ω are not load-bearing, the
  3-D benchmark is rebuilt before any representation question.
- Each implementation step gets a 4-format plan under `docs/plans/` (operating contract §2) before any code.
