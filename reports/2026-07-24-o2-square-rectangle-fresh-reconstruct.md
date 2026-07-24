---
title: OBJECT_TO_TARGET_VARIANTS_V1 — O2: square & rectangle on the fresh-reconstruct distribution
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: O2_STRUCTURALLY_SOLVABLE_DEPLOY_GAP / RETENTION_CAPABILITY_NOT_REWARD_LIMITED / TEACHER_REWARD_CHANGE_INSUFFICIENT / DETERMINISTIC_SHAPE_BLIND_PROPOSAL_INSUFFICIENT / ARCHITECTURAL_ASSIMILATION_REQUIRED
---

# O2 — square & rectangle (fresh-reconstruct, ball-tip embodiment fixed)

Diagnostic pass. Ball-tip robot FIXED (BALLTIP_COIN_BASELINE_V1); manipuland varied to equal-area boxes (square 1:1,
rectangle 2:1, rectangle 3:1) to the canonical r0.020 cylinder. All states are FRESH per-object reconstructions (pi_0
replayed on the ball+box), orientation-stratified by `disk_rz`, on one shared bank across arms. Reward + K6 certificate
FROZEN throughout (the HOLD reward is a separate pre-registered ablation, §HOLD). Frozen scene/robot/reward untouched.

## Measured facts — the five arms (K6 / 24, `reports/…/o2.json`)
| shape | pi_0 | zero-shot | fresh-refit | explicit | **expert** | contact-frac (refit / expert) |
|---|---:|---:|---:|---:|---:|---|
| square 1:1 | 0 | 3 | 4 | 0 | **18 (0.75)** | 0.17 / 0.21 |
| rect 2:1 | 0 | 2 | 2 | 1 | **13 (0.54)** | 0.18 / 0.30 |
| rect 3:1 | 0 | 4 | 3 | 1 | **16 (0.67)** | 0.15 / 0.32 |

**Primary verdict: `O2_STRUCTURALLY_SOLVABLE_DEPLOY_GAP`.** Every box is physically solvable (expert 0.54–0.75, square the
cleanest), yet every deployable arm is weak (fresh-refit 0.08–0.17, zero-shot 0.08–0.17, explicit ~0, pi_0 0). The limit is
NOT the embodiment or the object — it is that the current deploy stack cannot compactly REPRESENT and SELECT the solution
modes the search-expert finds. This is the O1 ball finding **extended to edged objects** (`EDGED_OBJECT_GENERALIZATION_FAILURE`).

## Mechanism — the strongest signal
**Expert contact-retention RISES with elongation (0.21 → 0.30 → 0.32); the deploy proposal stays flat (0.17 → 0.18 → 0.15).**
The longer edge is a physically more useful pushing/holding surface, which the search-expert exploits (more contact through
transport) but the deterministic deploy proposal does not sense or map onto a better option. The expert has implicitly
learned `edge-length + orientation + contact-point + moment-arm → the right push/brake/release mode`; the proposal sees a
~flat state → a single deterministic θ-center. (H1 edge-helps: supported for the expert. H5 long-axis rotation: see
`rot_range` per shape in the JSON.)

## The controller hierarchy (why the gap is structural, not a heuristic miss)
- simple explicit open-loop geometric rule → **not enough** (~0/24): a "push toward the zone" heuristic does not solve it;
- deterministic proposal + b=8 → **weak** (0.08–0.17);
- strong nonlocal search → **strong** (0.54–0.75).

So the winning policy is state- and contact-mode-dependent and multi-phase — not a single push law. `SEARCH_REMAINS_LOAD_BEARING`,
`SIMPLE_REFIT_INSUFFICIENT`. (The explicit arm is deliberately open-loop; it bounds "solvable without search" only weakly —
the expert is the real ceiling here. Noted as a limitation, not evidence against solvability.)

## Diagnostic 1 — geometry-aliasing probe (`o2_geometry_probe.json`)
_(fill on completion)_ Combined res_mse 0.946 vs square-only 0.13 suggested geometry aliasing. This control tests it directly:
per-shape fit MSE (is each shape individually hard, or is the aggregate driven by cross-shape averaging?), a shape-conditioned
probe (`obs` vs `obs + [hx, hy, hx/hy, sin rz, cos rz]` — does the geometry descriptor collapse the fit error?), and teacher
multimodality (nearest-obs θ-distance within-shape vs across-shape). **If obs+geom collapses the error ⇒ `GEOMETRY_ALIASING_CONFIRMED`.**

## Diagnostic 2 — HOLD_AWARE_REWARD_V1 ablation (`o2_hold.json`) — DONE
Separate, pre-registered; changes ONLY the teacher selection to a phase-dependent, farming-proof contact-retention score.
| | square | rect 2:1 | rect 3:1 | mean |
|---|---:|---:|---:|---:|
| canonical deploy K6 | 0.167 | 0.083 | 0.125 | **0.125** |
| HOLD deploy K6 | 0.125 | 0.042 | 0.083 | **0.083** |
| expert ceiling | — | — | — | 0.653 |

**Verdict: `RETENTION_CAPABILITY_NOT_REWARD_LIMITED`.** The contact-retention teacher did NOT improve K6 — it was slightly
worse (0.083 ≤ 0.125), far below the expert (0.653). This strongly excludes the simple explanation "good options exist, the
old reward just ranked them poorly."
- **Second, stronger signal:** the HOLD-teacher proposal fits at **res_mse 0.940** — essentially identical to the canonical
  teacher's **0.946**. Two *different* teacher-selection criteria, the same severe fit error ⇒ the bottleneck is not the
  label preference system but **the map itself**: a shape-blind state → a single deterministic θ cannot represent the option
  distribution across squares and different rectangles. `TEACHER_REWARD_CHANGE_INSUFFICIENT`.
- **Farming — two levels (kept distinct):** (1) teacher-selected options — the HOLD score subtracts toggles, so farming was
  never *selected* as beneficial; (2) deployed HOLD proposal — the rollouts show 5–11/24 farming. Reconciled: **the HOLD
  objective did not select farming as a beneficial strategy; farming reappeared after projection through the poorly-fitting
  deterministic proposal.** So this is a proposal-imitation failure, not a reward exploit. (camping 0–1, overshoot 1–3,
  never-release 0; pinning/high-force not audited this pass.)

## Combined architecture decision — reward is ruled out; the probe only sizes the FIRST fix
HOLD (reward) is settled: reward is NOT the limit (outcomes 1 and 2 excluded). The twin-teacher fit signal (0.946 ≈ 0.940)
already establishes `DETERMINISTIC_SHAPE_BLIND_PROPOSAL_INSUFFICIENT` and `ARCHITECTURAL_ASSIMILATION_REQUIRED`. **The
geometry-probe no longer decides *whether* an architectural change is needed — that is decided — only whether the FIRST fix
can be simple geometry conditioning or must go straight to structured multimodal/nonlocal:**
- probe collapses the fit error ⇒ `REPRESENTATION_LIMIT_DOMINANT` + `GEOMETRY_ALIASING_CONFIRMED` → first element =
  **StructuredObjectRepresentation** (geometry conditioning may be enough as the first step);
- probe helps only a little ⇒ `GEOMETRY_INFORMATION_NECESSARY_BUT_NOT_SUFFICIENT` + `MULTIMODAL_OPTION_STRUCTURE_DOMINANT`
  → the same geometric+physical state has multiple distant working strategies a single MSE deterministic head averages ⇒ a
  **multimodal/nonlocal proposal** is needed immediately, not just conditioning.
_(probe numbers filled on completion)_

## Assimilation requirements this O2 makes concrete (input to ARCHITECTURAL_ASSIMILATION_V1 — NO implementation here)
The next shared framework must AT LEAST provide (not "another shape-blind proposal training"):
```
StructuredObjectRepresentation   MultimodalProposal            NonlocalDecisionLayer
├── shape family                 ├── template probabilities    ├── retrieval
├── metric dimensions            ├── multiple θ modes          ├── adaptive search
├── orientation                  ├── mode-conditioned residual ├── mode-specific candidate bounds
├── mass and inertia             └── calibrated abstention     └── fixed-budget deploy policy
├── target-relative geometry
├── active contact edge / contact graph
└── phase and containment state
```
Direct preparation for the next tasks: **pick-and-place** (contact modes, object geometry, phase transitions);
**CIP-HyperSignedLiNGAM** (structured causal/hypergraph state, not flat obs); further control tasks (shared
proposal/search/certificate). **The ball-tip embodiment IS more general — but the current deploy architecture does not
generalize with it.** `ARCHITECTURAL_ASSIMILATION_REQUIRED`.

## Orientation stratification
Bins {axis_0, inter_22, diag_45, random}, exact `orient_rad` recorded per state (`o2.json` records). Per-arm per-bin K6 in
the records; the aggregate gap holds across bins (deploy weak, expert strong) — see JSON. `rot_range` per rollout tracks the
long-axis rotation mode (H5).

## Non-claims
- The explicit controller's ~0 does NOT prove "unsolvable without search" (it is a weak open-loop heuristic).
- `res_mse 0.946` alone does NOT prove architectural insufficiency — the geometry-probe is what converts it to a claim.
- Single-search-seed, 24 states/shape, first pass — the DIRECTION (expert≫deploy across all shapes) is robust; exact rates are one estimate.
- No transplant proxy used; fresh-reconstruct throughout.

## Recommendation for O3 (triangle)
Proceed to O3 (triangle) on the SAME fresh-reconstruct distribution with the same 5 arms + the two diagnostics. Triangle
adds vertex-first/edge-first asymmetry and a stronger multimodal expectation — it is the natural test of whether the
`MULTIMODAL_OR_NONLOCAL_OPTION_STRUCTURE` outcome dominates. But per the plan, complete O3/O4 (the bounded matrix) BEFORE
executing the assimilation.

## Files / provenance
- `coin_object_o2.py` (5-arm), `coin_object_o2_hold.py` (HOLD ablation), `coin_object_o2_geometry_probe.py` (diagnostics).
- frozen: pi_0 `1902454c`, transplant proposal `88679107`, fresh cylinder proposal `852cb529`; reward v2b; K6 cert; ball robot `galambos_planar_balltip_v1`.
- CORE.YAML: none touched. Regression: canonical E0 + coin tests green (§regression, on completion).
