# HSiKAN Kinematic-Pose + MuJoCo Demos

Two short scripts that demonstrate HSiKAN end-to-end on a non-trivial
auxiliary domain (synthetic kinematic mechanisms), with optional
visualisation through MuJoCo's headless renderer.  No pre-trained
checkpoint required — the entire training + evaluation + render
pipeline completes in under 10 seconds on a single GPU.

---

## Quick start

```bash
# From the repo root.
# (Optional) Build the Rust cycle enumerator once:
cd hymeko_py && maturin develop --release && cd ..

# Demo 1: pose regression only — fastest path, table output
HSIKAN_TORCH_COMPILE=1 \
  python -m signedkan_wip.src.demo_kinematic_pose --arity 6

# Demo 2: pose + MuJoCo rendered simulation
HSIKAN_TORCH_COMPILE=1 \
  python -m signedkan_wip.src.demo_kinematic_mujoco --mech 4dof
```

---

## Demo 1 — `demo_kinematic_pose.py`

**What it does.** Trains a small `PositionRegHSiKAN` (805 parameters
at the default `h=16, n_layers=2, grid=3`) on 20 synthetic Stewart
platform fixtures (`arity=6`, k-cycles), then predicts per-vertex XYZ
coordinates of every link in 5 held-out test mechanisms.

**What you see.**
1. Training loss every 20 epochs.
2. Test MAE in unit-meter scale (typical: 3-5 cm error).
3. Single-mechanism inference latency (typical: 0.5-0.6 ms on cuda).
4. A per-vertex prediction table comparing predicted vs. ground-truth
   XYZ coordinates with L2 error per vertex.

**Sample output (Stewart platform, k=6):**

```
=== HSiKAN pose-regression demo ===
  device: cuda  arity: k=6  hidden: 16
  matched k=6 mechanisms: 20 train, 5 test  (n_nodes_max=14)
  model: PositionRegHSiKAN h=16 L=2 grid=3  (805 params)
    epoch  20  train_loss=0.1587
    epoch  80  train_loss=0.0172
  test MAE: 0.0377 ± 0.0218  (n_test=5)
  inference latency (median): 0.56 ms

  vid   predicted XYZ              true XYZ                  L2 err
  ----------------------------------------------------------------
    0   (+0.000, +0.000, -0.001)   (+0.000, +0.000, +0.000)   0.0006   ← base link
    1   (-0.000, -0.000, +1.001)   (+0.000, +0.000, +1.000)   0.0006   ← end-effector
    2   (+0.529, -0.020, +0.452)   (+0.500, +0.000, +0.500)   0.0593   ← strut
    ...
                                   mean L2 over all          0.0520
```

**Flags.**
- `--arity {4,6}` — mechanism family: `4` for 4-bar (under-constrained, MAE ~ 40 cm because synthetic positions are randomly rotated per-instance), `6` for Stewart/delta-3RRR (well-conditioned, MAE ~ 5 cm)
- `--n-train 80 --n-test 20 --epochs 80` — training-set size and epochs
- `--hidden 16` — hidden dim of the encoder
- `--show-vertices 8` — how many vertices to dump in the per-vertex table

---

## Demo 2 — `demo_kinematic_mujoco.py`

**What it does.**
1. Trains the same `PositionRegHSiKAN` predictor as Demo 1 (skipped if
   the chosen mechanism has no closed cycles, e.g. the `4dof` serial
   arm).
2. Loads the canonical 4-DOF serial arm or 4-bar mechanism into
   MuJoCo and rolls the simulation for `--duration` seconds with
   sinusoidal joint actuation (`d.ctrl[ai] = 0.6 * sin(2π · (0.5 + 0.2·ai) · t)`).
3. Renders frames headlessly (no display required) using
   `mujoco.Renderer` and saves them to `demo_out/frame_NNNN.png`.
4. Optionally bundles the frames into `demo_out/sim.mp4` if `imageio`
   and `imageio-ffmpeg` are installed (`pip install imageio
   imageio-ffmpeg`).
5. Reports per-body XYZ amplitude (range across the trajectory) and
   the HSiKAN single-mechanism inference latency.

**What you see in the rendered frames.**
- The **4-DOF arm** (`--mech 4dof`): a 5-link serial manipulator
  (base, shoulder, elbow, wrist, flange-with-blue-tip) sweeping
  through a continuous sinusoidal trajectory.  The flange end
  traces a roughly elliptical path with ~1.4 m × 0.6 m × 0.5 m
  amplitude under the default actuation.
- The **4-bar** (`--mech 4bar`): four planar links rotating around a
  closed loop.  Used to demonstrate the `arity=4` HSiKAN-graph case;
  closed-cycle topology is enumerable so the HSiKAN predictor also
  trains.

**Sample console output (4-DOF arm):**

```
=== HSiKAN + MuJoCo demo ===
  mechanism: 4dof  duration: 2.0s  device: cuda

[1/3] Training HSiKAN pose predictor (h=16, epochs=60) ...
  HSiKAN training skipped — mechanism has no closed cycles
  (serial arm).  MuJoCo render still proceeds.

[2/3] Running MuJoCo simulation + headless render ...
  rendered 63 frames @ 30fps to demo_out/
  encoded MP4: demo_out/sim.mp4 (29 KB)
  sim + render: 0.69s

[3/3] Trajectory summary:
  base_link             XYZ amplitude: (+0.000, +0.000, +0.000) m
  shoulder_link         XYZ amplitude: (+0.000, +0.000, +0.000) m
  elbow_link            XYZ amplitude: (+0.623, +0.277, +0.204) m
  wrist_link            XYZ amplitude: (+1.201, +0.509, +0.461) m
  flange_link           XYZ amplitude: (+1.396, +0.587, +0.552) m
```

**Flags.**
- `--mech {4dof,4bar}` — which canonical mechanism to load
- `--duration 2.0` — simulation length in seconds
- `--fps 30` — capture rate (default 30 fps)
- `--width 480 --height 360` — render resolution
- `--n-train 80 --epochs 60` — HSiKAN predictor training budget
- `--out demo_out` — output directory for frames + MP4

---

## Requirements

```bash
pip install torch numpy scipy scikit-learn mujoco pillow
# Optional, for MP4 encoding:
pip install imageio imageio-ffmpeg
```

The Rust cycle enumerator (`hymeko_py.cycles`) is *optional* — both
demos fall back to the pure-Python DFS if it is not installed.  The
Rust path is 30-50× faster on Slashdot-scale graphs but the
mechanism graphs in these demos (8-14 vertices) build cycle pools in
milliseconds either way.

Tested on:
- NVIDIA RTX 2070 SUPER (8 GB), CUDA 12.1, PyTorch 2.5
- MuJoCo 3.7
- Headless EGL rendering (no X server needed)

---

## What the demos do *not* show

- Backwards pass / training visualisation — these are forward-only demos
- Online inference (joint-feedback to robot) — the MuJoCo demo is
  open-loop sinusoidal actuation; closing the loop with HSiKAN-as-
  controller is named future work in the paper
- Real URDF imports — the `4dof` and `4bar` MJCF strings are inlined
  in `mujoco_bridge.py`; the more substantial `drchubo` / WAM imports
  are available through `urdf_to_hymeko.py` but not exercised here.
