# SUPERSEDED 2026-05-11

This plan described the "cycle-density-heatmap as auxiliary YOLO feature"
interpretation of HyMeYOLO.  After consultation, the correct architecture
is **cycle/circle IS the detection primitive** (DETR-style queries),
which is already implemented in:

  - `signedkan_wip/src/vision/hymeyolo_q_smoke.py`     (single-object)
  - `signedkan_wip/src/vision/hymeyolo_hungarian.py`   (multi-object, Hungarian matching)
  - `signedkan_wip/src/vision/kcycle_detection.py`      (Delaunay-graph variant)

The canonical plans are:

  - `docs/plans_kcycle_vision_2026_05_07.md`
  - `docs/plans_kcvd_vs_yolo_2026_05_09.md`

This dir is retained for the LaTeX plan as an architectural negative
control reference (the auxiliary-feature interpretation); but its
implementation files were removed.

What was salvaged from the auxiliary-feature attempt:

  - `signedkan_wip/src/vision/cluttered_mnist.py`     (synthetic MNIST detection dataset)
  - `signedkan_wip/tests/test_cluttered_mnist.py`     (12 tests, all passing)

Future HyMeYOLO extensions (circles via chordless-cycle filter, Ricci-
curvature scoring) should be added to the existing canonical files.
