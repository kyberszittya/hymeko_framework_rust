# Scoping note — CPML for pose detection and kinematic pose tracking

**Date:** 2026-05-29
**Status:** scoping / pre-plan. No code written. Sized for an "is this 1 week, 1 month, or a quarter?" decision.

## What exists today (the floor)

| Component | State | Where |
|:--|:--|:--|
| **CPML layer** | Fully implemented + 23 tests; in active use as inner shell of Gömb cascade | `signedkan_wip/src/core/cpml.py` (~985 LOC). Forward: cycles → degree-tiered aggregators → per-edge logits (BCE). |
| **Kinematic graph datasets** | Procedurally generated, four families (four_bar k=4 / stewart k=6 / delta k=6 / serial N-DOF k=0) | `signedkan_wip/src/kinematic/__init__.py`. 120 train + 40 test mechanisms, k-filtered. |
| **Per-vertex 3D position regression** | Working as `cell_pose(arity, hidden, n_epochs, device)` — MSE on synthetic xyz coordinates | `run_final_cell.py:750-814`. Model = `PositionRegHSiKAN` (MixedAritySignedKAN backbone + per-vertex Linear(h, 3) head). |
| **Kinematic classification + DOF regression** | Working as `cell_kinematic(...)` — family classification + DOF MAE | `run_final_cell.py:685-747`. Model = `GraphLevelHSiKAN`. |
| **URDF → SignedGraph parser** | drchubo (52-link Atlas-class) + WAM (7-DOF) imported | `scripts/scaling/urdf_to_hymeko.py`, `signedkan_wip/src/kinematic/__init__.py:urdf_to_signed_graph`. |

## What's missing (the ceiling)

| Gap | Cost estimate (LOC + time) |
|:--|:--|
| **CPML head for per-vertex 3D output** (currently CPML emits per-edge logits) | ~200-400 LOC + parity test + smoke. ~1 week. |
| **Time-series wrapper** (RNN/Transformer over frame sequences). No precedent in this repo. | ~500-1000 LOC. ~2-3 weeks. |
| **Sequence dataset loader** (current kinematic datasets are single-frame) | ~200 LOC. ~3 days. |
| **Vision pose detection** (HymeYOLO is object detection; no keypoint head) | ~1500+ LOC + retraining + new pose dataset. ~4+ weeks; novel research, not integration. |

## Three coherent project shapes

### Shape A — "CPML on static kinematic graphs" (1 week, Phase 1 only)

The minimal viable bridge. **Reuse**: cycle-pool input format, kinematic mechanism generators, URDF parser, per-vertex synthetic targets. **Build**: a `CPMLPose` head that takes CPML's per-tier per-vertex features and projects to (N, 3).

- Forward: `feats = cpml(node_x, cycle_pool, cycle_signs, tier_of)` then `Linear(d_final, 3)` per vertex.
- Comparison: drop-in for the existing `cell_pose` `PositionRegHSiKAN` model. Does CPML's tier-stratified routing help vs the flat MixedAritySignedKAN backbone on pose regression?
- Risk: CPML's tier routing was designed for edge classification; per-vertex regression may not benefit from degree-tiered routing (most useful when high-degree vertices anchor different scales of structure). Could be null.
- Deliverable: 1 report comparing CPMLPose vs PositionRegHSiKAN on `pose_k4` + `pose_k6` at 3-5 seeds.

### Shape B — "CPML for kinematic pose tracking" (3-4 weeks, Phases 1+2)

Adds time-series. Practical scenario: a robot arm moves through a sequence of poses; predict next-pose from a window of past pose graphs.

- Frame dataset: extend the kinematic generators to emit `(seq_len, frame_idx, graph_snapshot)` tuples by sweeping joint angles along motion trajectories. ~3 days.
- Temporal model: either (a) frame-wise CPMLPose + GRU over per-vertex outputs, or (b) per-vertex CPMLPose + Transformer over frame tokens. (a) is simpler to start.
- Loss: per-frame MSE summed over sequence, optional smoothness regularizer over `dθ/dt`.
- Metrics: per-frame MAE, sequence-level tracking error (cumulative drift over T frames).
- Risk: this is a real RL/tracking architecture, not a single-cell test. Easy to under-engineer the sequence model.
- Deliverable: a tracking benchmark + report on 5-seed paired comparison vs a no-temporal baseline.

### Shape C — "Vision-based pose detection" (1+ quarter, Phases 1+2+3)

The most ambitious: detect human/robot joint locations *in images*, then track them across frames. This is a genuinely new research direction in this repo.

- Vision pose head: HymeYOLO backbone + keypoint regression head (~1500 LOC).
- Pose dataset: real (COCO-pose, MPII) or synthetic-from-URDF rendering (~1000 LOC + a rendering pipeline if synthetic).
- Combine with Shape B for tracking.
- Risk: large enough to be its own project. Not "add a CPML layer"; rebuild substantial pose infrastructure.
- Deliverable: a real working pose detector + tracker.

## My recommendation

**Start with Shape A.** It's the smallest unit that answers a real question ("does CPML's tier-stratified routing help on per-vertex pose regression?") with existing data and limited new code. Results from Shape A inform whether Shape B is worth the 3-4 weeks (if Shape A is null on pose regression, Shape B probably is too — temporal wrapping won't fix a wrong inner model). Shape C is a separate research thread.

Concrete next steps if you greenlight Shape A:

1. Write a 4-format plan (`docs/plans/2026-MM-DD-cpml-pose-regression/`).
2. Implement `CPMLPose` class — subclass or adapt `CPML` with a per-vertex regression head.
3. Parity test: CPMLPose with appropriate init equals a uniform-prior baseline.
4. New `cell_cpml_pose(arity, hidden, n_epochs, device)` in `run_final_cell.py`.
5. 3-seed comparison: CPMLPose vs PositionRegHSiKAN on `pose_k4` + `pose_k6`. ~half day GPU.
6. Report: does tier-stratified routing add anything per-vertex?

## What this scoping note is *not*

- Not a 4-format plan (CLAUDE.md §2). That would come AFTER you greenlight the direction.
- Not a commitment to do this work — it's a sizing exercise so you can decide between continuing the vision push (current direction) and pivoting.

## Decision question for you

Currently the vision-HSiKAN push (per_channel sweep running as `b8fzfr013`) is the active thread. If at any point you want to pivot to Shape A, say the word and I'll write the formal plan; Shape A's GPU time fits in a single overnight.
