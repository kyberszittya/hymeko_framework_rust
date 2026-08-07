# R11.7B — Multi-Object Characterization (object-family phase diagram)

**Date:** 2026-08-07 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Goal (user-scoped):** not to "solve" O1/O2, but to test whether the coin/box failure structure re-appears when
only **size** (O1-L) or only **dynamics** (O2-M) changes — the substrate for a real multi-object benchmark.

Four object families through the **identical** exact-zero ladder — reach → capture → structured-teacher feasibility
(`best_theta_full`, R=5) → frozen-bank teacher-free retrieval (LOO) → strict K6 — on 3 functionally-matched scenarios
(S0 short / S1 off-center / S2 far-angular) × 5 seeds. Sparse (best-θ/scenario) and dense (all K6 θ) audits computed
in one pass (no densification tuning). `characterization_{O0,O1-L,O2-M,O4-S}.json`.

## Phase diagram

| family | axis | reach | certified capture | **teacher-K6 \| capture** | retrieval-K6 \| capture (sparse) | (dense) | coverage (dense) | deliv-θ median rank |
|---|---|---|---|---|---|---|---|---|
| O0 | reference coin | 1.0 | 0.733 | 0.909 | 0.545 | 0.364 | 9/11 | 2.0 |
| O1-L | size (r ×1.20) | 1.0 | 0.600 | **1.000** | 0.111 | **0.000** | 5/9 | 2.0 |
| O2-M | dynamics (mass ×2) | 1.0 | 0.533 | 0.875 | 0.500 | 0.375 | 4/8 | 1.0 |
| O4-S | shape (box) | 1.0 | 0.733 | 0.818 | 0.364 | 0.364 | 7/11 | 1.0 |

## What is robust

1. **Reach is fully object-invariant** (1.0 for all four). Size, mass, and shape do not affect exact-zero reach.
2. **The structured teacher generalizes to every variant** — teacher-K6-given-capture **0.82–1.00** across size,
   dynamics, and shape. Physical solvability under the current controller is object-robust: whenever the object is
   captured, the structured CEM teacher finds a delivering θ almost always.
3. **Retrieval is the bottleneck** — the amortized demo-bank retrieval is variable and unreliable across families,
   never the clean top-1 policy the teacher's success suggests should exist. O1-L fails even on these easy scenarios
   (0.0–0.11); the box (O4-S) retrieves here (0.36) but *failed on the harder dev scenarios in the U6B pilot*.

⇒ **Outcome 1 (the strong result), stated precisely:** *the structured physical teacher generalizes across object
variants, while the amortized demonstration retrieval cannot reliably reconstruct the handoff×action transfer
interaction.* This is the motivated benchmark for a structured (HSiKAN / Steiner contact-hypergraph) evaluator: it must
score handoff × object × target interactions that flat descriptor retrieval cannot, across multiple object families,
where the *teacher already proves a solution exists*.

## Correction (honesty over a tidy story)

An earlier read from O1-L vs O2-M alone suggested "retrieval breaks under *geometry*, transfers across *dynamics*."
**The full four-family diagram refutes the clean split**: O4-S (a shape/geometry change) retrieves fine on these
scenarios (0.36, rank 1), while O1-L (also geometry — size) fails. On the comparable scenarios it is specifically the
**size** variant that fails, not geometry-in-general, and the box's failure was **scenario-dependent** (hard dev
scenarios, not these). So retrieval difficulty is **object × scenario-dependent, not a single clean axis.** The
geometry-vs-dynamics mechanism was over-claimed from two families and is withdrawn.

## Caveats

- **Small samples:** 8–11 captured snapshots per family, LOO over only 3 scenarios ⇒ the retrieval *rates* are noisy
  (e.g. O1-L 0/9 vs O0 4/11 is only marginally significant). The **teacher-generalizes** finding is the robust one;
  the per-family retrieval *ranking* is provisional and should not be over-read (`no-conclusions-from-first-pass`).
- Capture rate varies by scenario within a family (O1-L: S0 3/5, S1 1/5, S2 5/5) — capture-seed consistency remains a
  real secondary lever (parked, per user).
- The retrieval here is within-family LOO (each family retrieves from its own teacher θ); cross-object retrieval is a
  separate diagnostic, not run.

## Next

Three+ characterized families (O0 coin, O4 box, O1 size, O2 dynamics) now exist on one comparable ladder — the
benchmark substrate. The HSiKAN / Steiner contact-hypergraph question is now concrete: *can a structured representation
recover the handoff×object×target transfer that flat retrieval cannot, given the teacher proves it solvable?* Then the
k-actor × n-critic tensor, where the distinct mechanical regimes (size / mass / shape) give actors/critics something to
specialize on. Box capture-consistency stays a parked secondary engineering lever.

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, numpy 2.4.6, macOS (Apple Silicon), `OMP_NUM_THREADS=1`. Deterministic (fixed
seeds 0–4 per scenario). Harness `hymeko_rl/experiments/r11_7b_multiobject_characterization.py`. Scenarios: S0
`bank_c2_+0.015_+0.000`, S1 `bank_c2_+0.025_+0.000`, S2 `bank_c3_r6_a+15` (certifying band, center excluded).
