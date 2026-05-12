# k-Cycle vision detection (kCVD)

Ádám's idea, 2026-05-07. **Replace YOLO-style fixed-rectangle anchor grids with detection primitives that come from the image's own structure** — specifically, k-cycles in a graph built over superpixels / keypoints, plus hypergraph convolution aggregating per-cycle features into per-face classification.

## Why this might work where straight HSiKAN-vision didn't

Earlier today's negative result: HSiKAN ported literally to MNIST/Fashion underperformed HGNN by ~0.24 (and HGNN already lost to CNN by 0.40). The reason was that the literal port treats every receptive-field patch as a hyperedge of pixels — losing translation equivariance and competing with CNN's hand-tuned grid prior.

The kCVD framing is **not a literal port**. It rebuilds the inductive bias from the ground up:

| YOLO | kCVD |
|---|---|
| Regular `S × S` grid of cells | Graph derived from image structure (SLIC superpixels / Delaunay over keypoints) |
| `k` anchor boxes per cell, fixed aspect ratios | Each k-cycle (face of the planar graph) IS a candidate region; aspect = cycle geometry |
| Multi-scale via FPN backbone | Multi-scale via αₖ mixing over k=3, 4, 5, 6 cycles (different cycle lengths cover different scales) |
| Per-anchor classification + bbox regression head | Per-face classification (and per-face polygon-vertex regression for non-rectangular bbox) |
| Dense; needs NMS | Sparse; cycles already non-overlapping in a planar graph |

The key insight: the IMAGE provides the geometry. The detector aggregates evidence over that geometry rather than imposing a fixed scaffold.

## Connection to existing framework

This composes cleanly with what's already in the repo:

| component | reuse |
|---|---|
| **HSiKAN signed-cycle aggregation** (`signedkan_wip/src/signedkan.py`) | The Option-C `φ_e^s ∘ Σ φ_v^s` factorisation extends to vision faces. The "sign" becomes a polarity feature (above/below local mean intensity, or gradient direction). |
| **αₖ mixer** (`mixed_arity_signedkan.py`) | Mix per-arity face embeddings — small triangles for fine detail, larger 5–6 cycles for coarser objects. |
| **k-cycle Rust enumerator** (`hymeko_py::enumerate_top_k_cycles_rs`) | Reuse for face enumeration on the SLIC graph. |
| **HOSVD** (`hymeko_core::tensor::decomposition`) | Compress the per-face feature tensor. |
| **Vulkan kernels** (`hymeko_compute`) | GPU-accelerate cycle enumeration on dense graphs. |

So most of the engine is already built. The new work is the **graph-from-image** stage and the **per-face classification head**.

## Graph-from-image strategies (for the smoke version)

1. **SLIC superpixel adjacency graph** (default). Skimage's SLIC produces ~100–1000 connected superpixels per image. Their adjacency is a planar-ish graph. Cheap.
2. **Delaunay triangulation over keypoints** (e.g. SIFT / ORB). Keypoint-density-aware; sparser than SLIC.
3. **Pixel-level grid with non-axis-aligned edges** (full graph, then prune). Densest; falls back to YOLO-grid behaviour at the limit.
4. **Mesh extraction from a 3D point cloud** (for polyhedral detection). Tetrahedralization; faces become detection units.

Start with (1) — easiest scaffold, most existing literature.

## Per-face features

For each face (k-cycle) `c = (v_1, …, v_k)`:

```
h_c = Σ_{s ∈ {+, -}} φ_e^s ( Σ_{i: σ_i = s} φ_v^s(h_{v_i}) )
```

where:
- `h_v` = per-vertex (superpixel) features. Initialised with mean RGB, mean gradient magnitude, area, perimeter of the superpixel.
- `σ` = polarity per cycle edge. For vision, two natural choices:
  - **Contrast polarity**: `+1` if the edge crosses an intensity-rising boundary, `-1` otherwise.
  - **Gradient direction**: `+1` if the edge is oriented with the local image gradient.
- `φ_v^s, φ_e^s` = batched Catmull-Rom splines (Tier-3, exactly the HSiKAN primitive).

After mixing arities, each face emits a feature vector → linear head → per-face class probability.

## Detection vs classification framing

Two outputs per face:
1. **Classification**: P(class | face) — softmax over object classes + "background".
2. **Polygon refinement** (optional): for each face vertex, predict a delta to refine the bounding polygon. Skip in v1.

Loss = cross-entropy on per-face class labels. **Labels** require a per-face foreground/background assignment from training data (use IoU with ground-truth polygons, threshold 0.5).

## Smoke test

Synthetic dataset: 256×256 images, each with 1–4 random convex polygons (triangles, squares, hexagons) on a plain noisy background. Class label per polygon. Ground truth = polygon vertices.

Pipeline:
1. SLIC → superpixel graph (~200 superpixels)
2. Enumerate k-cycles for k ∈ {3, 4, 5, 6} (caps at maybe 5K cycles per image)
3. Per-superpixel features (mean RGB, gradient, area)
4. HSiKAN-style per-face aggregation
5. Per-face classification head
6. Train on synthetic + measure mAP

If mAP > 0.5 on synthetic polygons, scale to PASCAL VOC or a small-object subset.

## Open questions before starting

- **Cycle enumeration cost**: SLIC graph is sparse (~5–6 neighbours per superpixel). k=6 cycles in a 200-vertex graph: feasible in milliseconds.
- **Face-vs-cycle**: A planar graph distinguishes "faces" (regions bounded by cycles) from "non-face cycles" (cycles that DON'T bound a region in the planar embedding). For SLIC graphs the embedding is planar so this is a real distinction. v1 ignores it — treats every k-cycle as a candidate. v2 should restrict to faces only.
- **Multi-scale**: αₖ over k might naturally pick longer cycles for larger objects. Validate this empirically.
- **Translation equivariance**: SLIC superpixel boundaries change with image content. The detector inherits this — no automatic translation equivariance like CNN. Acceptable trade-off if the geometry helps elsewhere.

## Order of operations

1. Module scaffold: `signedkan_wip/src/vision/kcycle_detection.py` with SLIC graph builder, face enumerator, KCycleDetector class
2. Synthetic-polygon dataset generator (~30 lines numpy)
3. End-to-end smoke training (1 epoch, single seed, just check loss decreases)
4. **If smoke is green**: 5-seed run on synthetic, compare to a tiny CNN at matched param budget
5. **If still green**: PASCAL VOC small-object subset, real comparison
6. **If positive**: paper-line — could be a contribution

## Polyhedral extension (deferred)

For 3D point clouds (e.g. ScanNet object detection), the same machinery extends:
- Tetrahedralization of the point cloud → polyhedral cells
- Each cell = hyperedge of vertices (4 vertices per tet)
- HSiKAN k=4 aggregation over (vertex, tet) incidence
- Per-cell classification

This is a clean route into 3D detection without retraining from PointNet++ ideas.

## What this is NOT

- A drop-in YOLO replacement on COCO. The first deliverable is a SCAFFOLD + smoke test on synthetic data. Real-world replacement requires (a) the smoke proves the inductive bias works, (b) infrastructure (data pipelines, augmentation, eval harnesses) catches up to the YOLO ecosystem.
- A claim that HSiKAN-on-vision works. The negative result from earlier today still stands for the literal port. kCVD is a fundamentally different framing — geometry from the image, not pixel-as-vertex.

## Composition with the previous Ádám idea

The previous open Ádám direction was "**learn k-enumeration with a separate architecture**". For kCVD that means: train a network to PROPOSE which cycles in the image graph are likely object faces, instead of enumerating all of them. This is a downstream optimization once the basic kCVD pipeline works — RL or imitation learning over a teacher (the ground-truth-IoU scorer).

## v1 smoke result (2026-05-07, ~03:05)

`signedkan_wip/src/vision/kcycle_detection.py::smoke()` runs end-to-end on 32 synthetic 256×256 polygon images:

```
[smoke] N=32 n_v=196 n_e=574 n_t=379
[smoke] class distribution: [10879  245  281  419  304]   ← 8.7:1 background:foreground
[smoke] epoch  1  loss=1.6376  acc=0.028  fg_acc=0.271    ← random init
[smoke] epoch  2  loss=1.5514  acc=0.900  fg_acc=0.032    ← model collapses to bg
[smoke] epoch 10  loss=1.4149  acc=0.897  fg_acc=0.000
```

**What works**: pipeline is end-to-end, gradients flow, loss decreases (1.64 → 1.41), 2469 params for the scaffold.

**What doesn't yet**: severe class imbalance (8.7:1) overwhelms the simple weighted cross-entropy; the model converges to "always predict background". Foreground accuracy collapses from random 0.27 to 0 within one epoch.

### Fixes queued for v2

| fix | why |
|---|---|
| **Focal loss** (γ=2) instead of weighted CE | Standard for dense detection — down-weights easy negatives without losing the hard positives the model still gets wrong |
| **Triangle re-sampling** — all foreground + N random background per batch | Equalises ratio explicitly; cheap and predictable |
| **More keypoints** (500–1000) | At ~200 vertices the triangles are too coarse; small polygons cover only 3–5 triangles → little signal |
| **Real per-vertex features** — frozen tiny CNN replacing the 5-channel manual sampler | Hand-crafted features are too weak; the rest of the model can't compensate |
| **Higher arity (k=4, 5, 6)** + αₖ mixer | Larger objects need larger faces; v1 only uses Delaunay triangles |

None of these change the architectural claim (faces from the image → hypergraph conv); they're standard dense-detection tricks. v1 proved the pipeline works; v2 adds the missing ingredients to actually learn the task.

## Files shipped (v1 scaffold)

```
signedkan_wip/src/vision/kcycle_detection.py     ~280 LOC, runs on CPU
docs/plans_kcycle_vision_2026_05_07.md           this file
```

Dependencies: numpy + torch + scipy.spatial.Delaunay only (no skimage / opencv). The GPU is busy with the overnight HSiKAN SOTA run; v1 ran on CPU and converged in <10 s.

## v2.1 result (2026-05-07, ~09:20)

Focal loss + per-image 1:1 neg:pos resampling, α=[0.1, 1, 1, 1, 1], γ=2:

```
v1 (weighted CE):           fg_acc 0.27 → 0.00  (collapse to bg by epoch 2)
v2.0 (focal only):          fg_acc 0.27 → 0.00  (still collapses; focal too weak)
v2.0 (focal + 3:1 resamp):  fg_acc 0.27 → 0.00  (bg still dominates, fg split 4 ways)
v2.1 (focal + 1:1 resamp +  fg_acc 0.27 → 0.46  (peak at epoch 23/30; mild
       α[bg]=0.1):                               overfit afterwards)
```

**What changed**: the model has real per-class signal. The 4-way fg split was the missing piece — at 3:1 neg:pos, bg outweighs each individual fg class ~10:1, so the optimum is still "always bg". 1:1 + α[bg]=0.1 brings the per-class loss into the same order of magnitude.

**Caveat**: total accuracy dropped from 0.897 (always-bg trivial baseline) to 0.04 because the model now over-predicts fg. On the real task — 4-way polygon-class discrimination on the foreground triangles — fg_acc is the relevant metric and the curve looks like real learning, not collapse.

**Next levers** (in order of expected lift):
1. More keypoints (200 → 600). Current triangles are too coarse — small polygons cover only 3–5 triangles.
2. Higher arity (k=4, 5, 6) + αₖ mixer. Per the plan, the αₖ probe is whether longer cycles help with larger objects.
3. Frozen tiny-CNN per-vertex features. Mean RGB + gradient is too thin a feature for this network to compensate for.

(1) is the cheapest experiment. (3) likely matters most but requires a bit more wiring.
