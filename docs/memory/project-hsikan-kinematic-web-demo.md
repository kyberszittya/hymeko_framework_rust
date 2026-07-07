---
name: project-hsikan-kinematic-web-demo
description: "In-progress HSiKAN kinematic regime-compass browser demo under demo_web/ — state, decisions, and the agreed next step."
metadata: 
  node_type: memory
  type: project
  originSessionId: 52a26962-051e-4198-ae39-e38e5e7038f0
---

Built 2026-06-08 (quick-prototype track, user opted out of §2/§3 plan+test+report). Robot-kinematics interpretability demo for HSiKAN.

**Where:** `demo_web/` (new dir at repo root — not the old `docs/demo/` WASM parser demo, not `signedkan_wip/src/demo/gui.py` Tk GUI).
- `export_kinematic_data.py` — stdlib URDF reader + simple-cycle enumerator → `kinematic_data.json` + `.js`. Tries real `signedkan_wip/src/kinematic/graph.py::kinematic_loop_summary` first (torch needed); falls back to stdlib enumerator. **This box has no torch**, so it used the fallback — which reproduces canonical counts exactly: four_bar 1×k4, stewart 15×k6, delta_3rrr 3×k6, serial_4/7 none.
- `index.html` — self-contained single page, no CDN. 3 panels: animated 3D mechanism (canvas, orbit/zoom), αₖ **regime compass** (cycle-arity bars, dominant glows gold — the centerpiece), signed topology graph (pulse traces loops). Data via `<script src=kinematic_data.js>` so double-click works (no CORS).
- `README.md` — run + real-vs-illustrative table.

**Verified:** exporter cycle counts match repo canon; `node --check` on inline JS; stubbed-canvas run 5 mechs × 30 frames, zero non-finite draws.

**Honesty boundary (kept explicit in UI/README):** exact = links, joints, signs, cycle histogram, family labels. *Illustrative* = the 3D motion (sinusoidal drive, not real FK). Compass shows the *structural* cycle fingerprint that trained αₖ converges to — NOT loaded model αₖ (couldn't, no torch).

**Agreed next step (asked user, awaiting answer):** either keep it a structural/illustrative showcase, OR make the arm motion driven by actual HSiKAN predictions via the `PositionRegHSiKAN` the MuJoCo demo (`signedkan_wip/demos/demo_kinematic_mujoco.py`) already trains. Also: run the exporter once in the user's torch env to stamp `kinematic_loop_summary` provenance on the compass numbers.

**If promoting to "real" artifact:** needs §2 plan dir (tex/pdf/tikz/mmd) + §3 tests + §9 report. Trained checkpoints exist: `checkpoints/kinematic/family_classifier_k{4,6}.pt`. Related: [[project-ac-hsikan-imdb]] (sentiment variant — no saved IMDB checkpoint, would need training).
