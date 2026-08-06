# Base-Soma (walks-only) vs Linear control on MNIST

**Date:** 2026-06-15 · **Branch:** `soma-vision`
**Plan:** `docs/plans/2026-05-14-gomb-soma/` (Phase 3-V bench / falsification)
**Artifact:** `reports/soma_walkconv_vs_linear_mnist_20260615.jsonl`

## Question

Does the *plain* walks-only base-Soma vision model (`WalkConvImageClassifier`,
the walk→polygon→triangle line **without** the Hodge/Ricci/SDRF machinery) do
real structural work — i.e. beat a parameter-heavier linear classifier on MNIST?
The harness states the falsification criterion itself: *"if base-Soma's 2 010
params can't beat a 7 850-param linear classifier on MNIST, the walks-only
sensorimotor hypothesis is in trouble."*

This closes the open Soma-vision question (BACKLOG #3 / audit gap): we already
knew the **full** RicciStim stack fails for vision; this tests whether the
**simpler** base is any better.

## Protocol

`train_mnist.py`, 5 seeds × 5 epochs, 5 000 train / 1 000 test (sub-sampled
MNIST), Adam lr 3e-3, CPU, fixed per-seed seeds. Both models on the identical
pipeline.

## Result — base-Soma is falsified for MNIST classification

| model | params | test-acc (mean ± pstd, n=5) | wall/seed |
|---|---|---|---|
| **base-Soma** (walk-conv) | 2 010 | **0.5186 ± 0.0204** | ~146 s |
| Linear(784→10) control | 7 850 | **0.9056 ± 0.0079** | ~8 s |

- **Paired Δ (base-Soma − linear) = −0.387 on every one of the 5 seeds.** It does
  not beat the linear control — by the harness's own criterion, the walks-only
  hypothesis is *in trouble*.
- Nuance (the one honest point in its favour): base-Soma is **2.2× more
  parameter-efficient** (0.258 vs 0.115 acc per 1 k params). But 0.52 absolute on
  10-class MNIST is weak, and the gap is large and consistent.

## Verdict

Combined with the full-stack result (RicciStim Cluttered-MNIST 0.14 mAP < 0.23
baseline) and the 2026-05-28 fair re-bench (hypergraph operators below a plain
MLP), this **completes the Soma-vision falsification**: *both* the elaborate
differential-geometry stack **and** the minimal walks-only base fail at
small-scale image classification. The structural prior, on this task, does not
help. The code is correct; the approach doesn't win here.

The remaining open, *un-falsified* axis is **brain-predictivity** (the Cichy-92
cortical benchmark) — accuracy and cortical alignment are distinct, and that is
the test the model was actually designed for (see
`docs/articles/cichy-cortical-prediction/`). It is blocked only on the real fMRI
data, not on compute.

## Correction (process honesty)

Mid-run I peeked at the background log and misattributed a ~0.91 test-acc to
base-Soma; that line was in fact the **linear** baseline (which runs after, also
over seeds 0–4). The final per-seed records show base-Soma at ~0.52. Stated
plainly so the record is right.

## Provenance

Working tree dirty (editor/seminar/soma-vision work in progress). MNIST via
torchvision (cached). CPU. Toolchain: torch (CORE-pinned), Python 3.12.
No CORE.YAML items touched; no new dependency.
