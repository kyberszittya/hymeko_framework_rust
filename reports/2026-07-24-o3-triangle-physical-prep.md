---
title: O3 — triangular-prism physical-panel preparation (foundation only; validated, no learning campaign)
date: 2026-07-24
branch: feat/architectural-assimilation-v1
status: PHYSICAL FOUNDATION VALIDATED (all gates green) — the O3 K-mode experiment is now unblocked at the physics layer
contract: O3_TRIANGLE_PHYSICAL_PREP
---

# O3 — triangular-prism physical-panel preparation

Per the instruction: prepare the O3 physical foundation in parallel with the video batch, **no teacher/RL campaign** until
the physical panel validation closes. This report is that validation — the runtime API stays frozen; nothing here trains.

## What was built (extend, not duplicate)
- **`hymeko_rl/env/planar_grasp_env.py`** — `coin_shape="triangle"` (committed `809bc1c2`): an equilateral triangular PRISM
  mesh manipuland, EQUAL-AREA to the `disk_radius` cylinder; MuJoCo density-derives mass/COM/inertia from the mesh.
- **`hymeko_rl/coin_delivery/triangle_footprint.py`** (new, pure geometry) — the **full-footprint** delivery certificate:
  every base vertex of the rotated+translated triangle inside the zone (stricter than centroid-in-zone), plus a signed
  `footprint_margin`, the `leading_feature` (vertex- vs edge-leading toward the zone), and a deterministic
  `orientation_strata` panel.
- **`hymeko_rl/experiments/o3_triangle_physical_prep.py`** (new) — the validation harness (no learning).

## Validation — all four gates green (`o3_physical_prep.json`, `all_ok: true`)
| gate | result |
|---|---|
| **full-footprint certificate** | centred triangle certifies; a **centroid-in-zone-but-corner-out** pose is REJECTED (stricter than centroid); far pose rejected. |
| **runtime mass / COM / inertia** | mass 0.050264 = equal-area cylinder 0.050265 (parity); COM at the body origin; body-frame inertia [1.278e-5, 1.278e-5, 1.216e-5] **rotation-invariant** across a 3-fold period. |
| **mesh-contact stability** | 3 seeds × 120 steps under random actions: no NaN/inf, max |qvel| 23.3 (< 50 bound) — the mesh settles, does not explode against the fingertips. |
| **orientation stratification** | 12 orientations over one 3-fold period → 7 vertex-leading / 5 edge-leading, per-orientation footprint margin computed (rotation-invariant at the zone centre, as expected). |

## What is deliberately NOT done yet (gated on this validation, now passed)
- The fresh-reconstruct **teacher** panel + the O3 K-mode experiment — a separate step, launched only now that the physics
  foundation is verified (per the instruction: no big teacher/learning campaign during video production / before physical
  validation).
- The O3 experiment will test the open architectural question in genuine CONTACT manipulation (does the multimodal policy
  search help here as it did on 6D-1's contact-free geometrically-separated basins, or does contact candidate-localization
  remain the wall — the O2 finding). The triangle's vertex/edge/orientation strata are the multimodality axis.

## Tests / provenance
- 6 new geometry tests (`test_triangle_footprint.py`) + 3 mesh-physics tests (`test_planar_grasp_env.py`, committed
  `809bc1c2`). Lint clean. CORE.YAML: none. Runtime API: frozen (this is env + geometry, no option_rl change).
- §6.5: extends `compose_planar_scene` + a new pure-geometry module; no new framework abstraction.

## Follow-up (the gated next step)
- [ ] O3 fresh-reconstruct teacher panel (triangle, orientation/vertex/edge-stratified) → the bounded O3 K-mode physical
  experiment (single-head vs K-mode at equal budget, full-footprint certificate) — the contact-manipulation test of the
  6D-1 conclusion.
