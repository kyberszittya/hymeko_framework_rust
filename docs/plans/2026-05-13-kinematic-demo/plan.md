# Kinematic-matching demo — plan

**Date:** 2026-05-13
**Audience:** robotics demo (preceding the comm-cliques demo for Niitsuma).
**Status:** plan v1.

## Scope

Build an interactive Gradio tab for **kinematic position regression**
on real URDFs, on top of the existing `PositionRegHSiKAN` and
`urdf_to_signed_graph` infrastructure.

The applied story: given a robot's joint topology + a joint
configuration, HSiKAN/Gömb predicts the 3-D position of every link.
Two sub-applications follow from that:

1. **Pattern identification** — classify which kinematic regime
   (gait / manipulation / idle / etc.) a trajectory belongs to via
   `GraphLevelHSiKAN` (already implemented).
2. **Marker positioning optimisation** — given a budget of *K*
   markers, pick the joint subset that minimises position-prediction
   error elsewhere. Reduces to a structurally-informed feature-selection
   problem; the cycle σ-products HSiKAN already computes are a strong
   prior.

## CORE.YAML items touched

**Empty list.** All work lives in `signedkan_wip/src/demo/` and
adjacent test files; no CORE crate is modified, no pinned dep changes.

## Affected files

New:

- `signedkan_wip/src/demo/kinematic.py` — `load_urdf_bundle`,
  `predict_positions`, `train_quick` analogues of the signed-link
  `inference.py`.
- `signedkan_wip/src/demo/kinematic_plotting.py` — matplotlib 3-D
  scatter + segment renderer (predicted vs. ground-truth skeleton, error
  heat-map).
- `signedkan_wip/src/demo/kinematic_registry.yaml` — catalogue of URDFs
  that ship with the repo (drchubo, WAM, mini_arm, …).
- `signedkan_wip/tests/test_demo_kinematic.py` — registry loader,
  URDF → graph round-trip, position-regression smoke on a
  small fixture.

Modified:

- `signedkan_wip/src/demo/gui.py` — add a new tab
  `"Kinematic position regression"` with a URDF dropdown + joint-config
  sliders + a 3-D plot.
- `signedkan_wip/src/demo/README.md` — describe the new tab + use case
  framing.

## Interface changes

- New public API in `demo.kinematic`:
  - `load_urdf_bundle(path: str | Path) -> KinematicBundle`
  - `predict_positions(bundle, joint_config: np.ndarray) -> PositionResult`
  - `quick_train(bundle, n_epochs=80, device="cpu") -> TrainedModel`
- Two registries — the existing `models.yaml` (signed-link) stays;
  `kinematic_registry.yaml` is the URDF catalogue.

## Test strategy

- **Unit:** registry parse + schema; URDF → SignedGraph round-trip on
  `mini_arm`; `predict_positions` shape correctness.
- **Integration:** `quick_train(mini_arm, n_epochs=20)` →
  `predict_positions` returns finite XYZ; smoke test runs in < 60 s
  on CPU.
- **No performance test** in v1 — this is a demo, not a benchmark
  target. If/when we promote to a research artifact, add latency +
  per-link-error budgets.

## Performance budget

- Single training cycle on a small URDF (mini_arm, ~7 links):
  < 60 s on CPU.
- Inference: < 50 ms per joint config on CPU.
- Peak RSS: under 2 GB (HSiKAN at hidden=16 on a small graph is tiny).
- GPU: not required for v1; if/when drchubo (52 links) is included,
  may move training to GPU but inference stays on CPU for
  responsiveness.

## Rollback path

Self-contained — drop `signedkan_wip/src/demo/kinematic*.py`, remove
the new tab from `gui.py`. Existing signed-link demo unaffected.

## Risk anticipation

- **HSiKAN may not match analytic FK on every URDF.** The model learns
  approximations of forward kinematics from data. Frame the demo as
  *"data-driven FK + downstream optimisation"*, not *"replaces analytic
  FK"*. Show the prediction error vs. ground truth honestly.
- **URDFs with many DoF need many training samples.** drchubo (52
  links) at random configs will need 1 k+ samples for convergence.
  Pre-train and ship checkpoints alongside `models.yaml`'s
  signed-link checkpoints.
- **Cycle σ-product semantics on kinematic graphs are looser than on
  signed-trust graphs.** Joint-pair signs (rigid / sliding / parallel)
  are a heuristic, not a hard rating. Note this in the README.

## Empty-plan-dir hygiene

If this demo is abandoned, delete `docs/plans/2026-05-13-kinematic-demo/`
before the next session.

## Why no TikZ/PDF/Mermaid plan

CLAUDE.md §2 requires four-format plans for non-trivial changes. This
is an *exploratory applied demo* that piggybacks on existing modules
(`PositionRegHSiKAN`, `urdf_to_signed_graph`). The interface surface
is small (three new functions + one tab), the rollback is trivial, and
no CORE.YAML item is touched. **If this demo becomes a research
artifact** (paper figure, ICRA/IROS submission, etc.), upgrade to the
full four-format plan at that time.

## Out of scope for v1

- Inverse kinematics from sparse marker positions (the "markers →
  joints" inverse problem). Possible v2 once forward prediction works.
- Trajectory-level pattern classification (`GraphLevelHSiKAN`). Lives
  in a follow-up tab.
- Multi-robot communication cliques (the Niitsuma demo). Separate plan
  doc once kinematic demo is presentable.
- Quantitative comparison with analytic FK or a learned-FK baseline.
  Honest visual error is enough for v1.

## Order of work

1. `kinematic.py` — bundle loader + `predict_positions` + `quick_train`.
2. `kinematic_plotting.py` — 3-D skeleton viz with per-link error
   coloring.
3. `kinematic_registry.yaml` — catalogue of in-repo URDFs.
4. `test_demo_kinematic.py` — registry + smoke tests.
5. `gui.py` — new tab wired up.
6. `README.md` update.
7. Smoke the GUI in a browser before reporting done.
