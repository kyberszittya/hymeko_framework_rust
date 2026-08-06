# HSiKAN — Kinematic Regime Compass (browser demo)

A self-contained, single-page interactive demo of the HSiKAN story for
robot kinematics: a robot's structure is a **signed graph**, its closed
loops are **cycles**, and HSiKAN's learned per-arity weight
**α<sub>k</sub>** — the *regime compass* — is the mechanism-family
fingerprint.

Three live panels:

1. **Live mechanism** — animated forward-kinematics of the selected
   mechanism (driven four-bar, waving serial arm, wobbling hexapod).
   Drag to orbit, scroll to zoom.
2. **α<sub>k</sub> regime compass** — the cycle-arity histogram over
   k = 3..6. Four-bar spikes at k=4, Stewart/delta at k=6, serial chains
   stay flat. The dominant bar (the arity α<sub>k</sub> concentrates on)
   glows gold.
3. **Signed topology graph** — links as nodes, signed joints as edges; a
   travelling pulse traces the closed kinematic loops.

## Run

Just open `index.html` — it loads its data from `kinematic_data.js` via
a `<script>` tag, so a bare double-click works (no server, no CORS).

Optionally serve over HTTP (uses `kinematic_data.json` via `fetch`):

```bash
python -m http.server 8000 --bind 127.0.0.1
#   http://127.0.0.1:8000/demo_web/index.html
```

## What is real vs illustrative

| Element                       | Source                                             |
|-------------------------------|----------------------------------------------------|
| Links, joints, joint signs    | **Exact** — parsed from the repo URDF fixtures     |
| Cycle-arity histogram (compass)| **Exact** — simple-cycle enumeration of the graph |
| Family / topology label       | **Exact** — ground truth for the canonical fixtures|
| 3D motion                     | *Illustrative* forward-kinematics (sinusoidal drive)|

The cycle counts match the canonical numbers used throughout the repo
(four-bar: 1×k4; Stewart: 15×k6; delta: 3×k6; serial: none).

## Regenerate the data

```bash
python demo_web/export_kinematic_data.py
```

Writes `kinematic_data.json` + `kinematic_data.js`. If the full research
stack (torch/numpy) is importable, cycle counts come from the canonical
`hymeko_neuro/experiments/kinematic/graph.py::kinematic_loop_summary`; otherwise
a dependency-free enumerator reproduces the same numbers.
