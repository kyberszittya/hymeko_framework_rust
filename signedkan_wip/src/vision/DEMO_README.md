# HyMeYOLO Detection Demo

A tkinter + matplotlib GUI that shows the HyMeYOLO `+ricci-mod`
detector finding digits in Cluttered MNIST images. Pure stdlib +
matplotlib + torch — no streamlit / gradio / flask dependency.

## One-line launch

```bash
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
/home/kyberszittya/miniconda3/bin/python -m \
    signedkan_wip.src.vision.demo_hymeyolo_tk
```

First launch: ~50 s of CPU training (no checkpoint provided →
quick-trains a fresh 30-epoch model on 1000 images). Subsequent
launches with `--checkpoint <path>` skip training.

## What you'll see

Two side-by-side panels:

* **Left:** the input Cluttered MNIST image with **ground-truth
  boxes** drawn in **cyan**, each labelled with its digit class.
* **Right:** the same image with the model's **predicted boxes
  in red** (from the 4 box queries) and **circle-derived AABBs in
  orange** (from the 2 circle queries). Each prediction is
  labelled with its top-1 class + confidence.

Controls along the bottom:

* **New random image** — generates a fresh Cluttered MNIST stimulus.
* **Score threshold slider** (0..1) — hides predictions below this
  confidence.
* **Seed** spinbox — jump to a specific image deterministically.

The header shows which model is loaded (checkpoint vs
quick-trained), with the stage / schedule / warm-start metadata.

## Using a trained checkpoint

The full HyMeYOLO ladder produces strong models; the quick-train
fallback only does 30 epochs on 1000 images (~0.30 mAP). To use a
real Stage A-2 / A-3 checkpoint:

1. Train + save:

   ```bash
   ./signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh a2
   # ... but with --save-checkpoint added to the training CLI; for
   # a quick standalone train+save, run directly:
   /home/kyberszittya/miniconda3/bin/python -m \
       signedkan_wip.src.vision.train_circles_ricci \
       --n-images 5000 --epochs 100 --lr 0.003 --seed 0 \
       --configs '+ricci-mod' \
       --save-checkpoint /tmp/hymeyolo_demo_ckpts/
   # Produces /tmp/hymeyolo_demo_ckpts/ricci-mod_seed0.pt
   ```

2. Launch the demo against it:

   ```bash
   /home/kyberszittya/miniconda3/bin/python -m \
       signedkan_wip.src.vision.demo_hymeyolo_tk \
       --checkpoint /tmp/hymeyolo_demo_ckpts/ricci-mod_seed0.pt
   ```

## Tips

* The model is small (~1 M params); CPU inference is < 50 ms per
  image. The demo is responsive even without a GPU.
* When **threshold = 0.0**, every query box is drawn (4 box +
  2 circle = 6 boxes per image). As you raise the threshold,
  low-confidence queries drop out — letting you see the model's
  attention concentrate on real digits.
* When **threshold = 0.5+**, only confident predictions remain.
  Compare against the cyan GT boxes on the left panel.
* The **circle queries** (orange) are an architectural quirk:
  they're trained to predict circular contours that get
  axis-aligned for the box comparison. They often pick up
  curved-digit shapes (3, 8, 0) more confidently than box queries.

## Command-line reference

```
python -m signedkan_wip.src.vision.demo_hymeyolo_tk \
    [--checkpoint PATH]       # pre-trained .pt; default: quick-train
    [--n-train-quick N]       # if no checkpoint, train on N images (default 1000)
    [--device cpu|cuda]       # default cpu; cuda OK if free
```

## Architecture (briefly)

`RicciHyMeYOLOMulti`:
* Backbone: `TinyBackbone` (3-conv stack, 32 hidden channels).
* 4 box queries + 2 circle queries — learnable corner positions
  initialised by saliency-FPS warm-start.
* Per-query feature: bilinear-sample image features at the corner
  positions, aggregate via `HSiKANAggregator` (signed-cycle
  encoder over the k corners).
* Per-query 3 Ricci shape descriptors: scalar κ, mean cos θ,
  edge-length variance.
* Class head: linear over the concatenated query feature + 3
  Ricci scalars (with optional LayerNorm in Stage A-3+).
* Offset head: linear over query feature → per-corner (Δx, Δy)
  refinements.

The demo runs the forward pass once per image and decodes the
predictions for display. No post-processing (NMS) — the model
already outputs only 6 queries, so duplicates are rare.
