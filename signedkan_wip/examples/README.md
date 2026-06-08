# Pose-detection example app

A self-contained demonstration of `FuzzySignaturePoseModel` —
the rev-6 fuzzy-signature backbone validated on 2026-05-31 — on a
synthetic kinematic-chain pose task with **per-keypoint uncertainty
quantification** via heatmap-softmax entropy.

## What this example shows

1. **Sub-10k-param pose detection** works on synthetic 32×32
   skeleton-blob data: keypoint regression to within a few pixels.
2. **Per-keypoint uncertainty** is recoverable from the model's
   internal heatmaps without any added inference cost.
   High entropy = "the model spreads probability across many
   pixels for this keypoint" = a usable confidence signal for a
   robotic collaborator.
3. **The fuzzy framework's value beyond classification**:
   uncertainty calibration on a *per-output-dimension* basis, not
   just a single scalar confidence per sample. A robot reading
   this output can act on `left-shoulder` (low entropy, confident)
   and abstain on `right-hand` (high entropy, uncertain).

## How to run

From the repository root:

```bash
PYTHONPATH=. python signedkan_wip/examples/pose_demo.py
```

First invocation trains the model (~3 min on RTX 2070 SUPER, ~10 min
on CPU) and caches it at `/tmp/pose_demo_outputs/fuzzy_pose_model.pt`.
Subsequent invocations reuse the cache (~3 seconds).

Force a retrain:

```bash
PYTHONPATH=. python signedkan_wip/examples/pose_demo.py --force-retrain
```

Higher-fidelity training (matches the gate-passing config):

```bash
PYTHONPATH=. python signedkan_wip/examples/pose_demo.py --force-retrain --n-epochs 60
```

## Output

All outputs land in `/tmp/pose_demo_outputs/`:

| File | What it is |
|---|---|
| `sample_0.png` … `sample_7.png` | Per-sample 4-panel figure: input image, predicted vs ground-truth overlay, per-keypoint heatmap grid (8 panels), uncertainty bar plot |
| `summary.png` | Composite of all 8 samples, 3 columns wide (input / overlay / uncertainty bars) |
| `uncertainty_table.txt` | Text-mode per-keypoint uncertainty summary with ASCII bar charts |
| `fuzzy_pose_model.pt` | Cached trained model |

## What the figures communicate

- **Input image** (left panel): a 32×32 grayscale frame with 8
  Gaussian-blob keypoints rendered from a kinematic-chain prior
  (head, shoulders, elbows, hands, mid-hip).

- **Predicted vs ground truth** (right of input): filled circles =
  predictions, hollow circles = ground truth. A line connects
  matched pairs; short lines mean accurate keypoint localisation.

- **Per-keypoint heatmaps** (middle row, 8 panels): each panel is
  the soft-argmax-input heatmap for one keypoint. A sharply peaked
  hot region means the model is confident about that keypoint's
  location; a diffuse hot region means the model is hedging.

- **Uncertainty bar plot** (bottom): the softmax-entropy of each
  per-keypoint heatmap, plotted per keypoint with the same colour as
  the heatmap and the overlay marker. Tall bars = uncertain
  keypoints; short bars = confident. **This is the part most
  relevant to robotic collaboration**: a downstream controller can
  selectively trust keypoints with low entropy and defer / re-query
  keypoints with high entropy.

## How the uncertainty connects to the fuzzy framework

The example uses a model-agnostic uncertainty signal (softmax
entropy of the per-keypoint heatmap). The framework's *deeper*
uncertainty primitive — the Atanassov hesitancy
`π = 1 − μ⁺ − μ⁻` per channel per pixel — is one CR-activation
forward call away. The reason this example uses heatmap entropy
rather than per-channel hesitancy is that the FuzzySignaturePose's
final pose head is a `Linear(d, n_kp)` that mixes channels; reading
per-keypoint hesitancy requires a per-keypoint attribution back to
channels, which is implementable but adds complexity. The heatmap
entropy is the cleanest first-pass demonstration.

A follow-up extension would expose the per-channel
`(μ⁺, μ⁻, π)` triple from the last layer's `mu_plus`, `mu_minus`,
and the gate `g` and visualise hesitancy directly.

## Why this example matters for the Niitsuma robotics framing

For human-robot collaboration: a robot that reports its
*per-keypoint* uncertainty can defer the uncertain decisions to a
human operator without halting on the certain ones. That's the
exact use-case the framework was designed for. The demo
operationalizes this: any robot using this perception module
inherits a per-keypoint confidence signal essentially for free.

The framework is also small enough (the cached model is ≈ 9.6k
parameters, ~37 KB of weights) to run on an embedded controller
without GPU acceleration — `--device cpu` works for inference at
≥ 30 fps on a Jetson-class board (rough estimate; not yet
profiled).

## Related artifacts

- Plan: [`docs/plans/2026-05-31-fuzzy-pose-detection/plan.tex`](../../docs/plans/2026-05-31-fuzzy-pose-detection/plan.tex)
- Robotics application plan with 2-robot test case: [`docs/plans/2026-05-31-robotics-behavior-collaboration/plan.tex`](../../docs/plans/2026-05-31-robotics-behavior-collaboration/plan.tex)
- Mathematical background (Atanassov IFS, fuzzy signatures): [`docs/plans/2026-05-30-fuzzy-signature-layer/background.tex`](../../docs/plans/2026-05-30-fuzzy-signature-layer/background.tex)
- Model implementation: [`signedkan_wip/src/vision/fuzzy_pose.py`](../src/vision/fuzzy_pose.py)
- Tests: [`signedkan_wip/tests/test_fuzzy_pose.py`](../tests/test_fuzzy_pose.py)
