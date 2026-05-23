# Getting started — computer vision (HyMeYOLO)

A 15-minute tour of the CV pipeline: run the demo, train your
first model, climb the Stage A → D ladder.

> **Prereqs:** the [miniconda3 env](../signedkan_wip/) with
> `torch`, `torchvision`, `tkinter`, `matplotlib`. Repo root must
> be on `PYTHONPATH`. All paths below are relative to repo root.
>
> ```bash
> cd /path/to/hymeko_framework_rust
> export PYTHONPATH="$PWD"
> export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
> ```

---

## 1. The demo (zero training, ~1 min)

The fastest way to *see* what HyMeYOLO does. Use the launcher
script — it auto-detects any saved checkpoint and falls back to
quick-training a small model if none exist:

```bash
bash signedkan_wip/src/vision/launch_demo.sh
```

A Tk window opens with two panels:

- **Left** — Cluttered MNIST image + cyan ground-truth boxes.
- **Right** — same image + predicted boxes (red = box queries,
  orange = circle queries), each labelled with class + confidence.

Drag the **score threshold** slider down toward 0 to see all 6
query boxes; drag it up to see only confident detections. Use
the **seed** spinbox to jump to specific stimuli.

### Launcher modes

```bash
bash signedkan_wip/src/vision/launch_demo.sh list      # what ckpts exist
bash signedkan_wip/src/vision/launch_demo.sh quick     # force quick-train (no save)
bash signedkan_wip/src/vision/launch_demo.sh auto      # detect or quick-train (default)
bash signedkan_wip/src/vision/launch_demo.sh a2        # train Stage A-2 → save → launch
bash signedkan_wip/src/vision/launch_demo.sh b_resnet  # Stage B (ResNet-tiny)
bash signedkan_wip/src/vision/launch_demo.sh b_hsikan  # Stage B' (HSiKAN-CR)
bash signedkan_wip/src/vision/launch_demo.sh c_fpn     # Stage C (resnet + 2-level FPN)
```

The `a2` / `b_*` / `c_fpn` modes train + save under
`checkpoints/hymeyolo_demo/<stage>/ricci-mod_seed0.pt` before
launching (~10–15 min GPU); subsequent `auto` runs pick up the
most recent one without retraining.  Override the save root with
`HYMEYOLO_CKPT_ROOT=/some/path`.

> The launcher knows about Cluttered MNIST checkpoints only.  The
> Stage D VOC checkpoint produced by `train_voc_stagec.py` is a
> 20-class model at 224² and is not yet wired into this demo — see
> §5 below.

Direct invocation (bypassing the launcher):

```bash
python -m signedkan_wip.src.vision.demo_hymeyolo_tk \
    --checkpoint /tmp/hymeyolo_ckpts/ricci-mod_seed0.pt
```

Full demo docs: [signedkan_wip/src/vision/DEMO_README.md](../signedkan_wip/src/vision/DEMO_README.md).

---

## 2. Your first real training (~5 min CPU, ~1 min GPU)

The `train_circles_ricci.py` entry point trains 5 architecture
variants in parallel on Cluttered MNIST and emits a jsonl with
per-config metrics. The default is small enough for CPU.

```bash
python -m signedkan_wip.src.vision.train_circles_ricci \
    --n-images 500 --epochs 10 --lr 3e-3 --seed 0 \
    --jsonl-out /tmp/cmnist_quick.jsonl
```

What you'll see at the end:

```
config              start    end   drop    wall   box_acc circ_acc  mAP50 mAP50:95 mIoU
baseline           4.2812  3.1240  27%    12.5s   0.21    0.00      0.08  0.02     0.31
boxes-only         4.1985  2.8841  31%    15.2s   0.34    0.00      0.15  0.05     0.42
circles-only       4.5621  3.6710  20%     9.8s   0.00    0.18      0.05  0.01     0.28
boxes+circles      4.2143  2.7912  34%    18.7s   0.36    0.21      0.18  0.06     0.45
+ricci-mod         4.1822  2.6480  37%    22.4s   0.39    0.24      0.22  0.07     0.48
```

`+ricci-mod` is the production architecture. The other rows are
ablations — the existence of the `boxes-only` and `circles-only`
columns is how you know box+circle queries each contribute.

JSONL fields documented in
[reports/2026-05-16-hymeyolo-cluttered-mnist-sota.md](../reports/2026-05-16-hymeyolo-cluttered-mnist-sota.md).

---

## 3. The Stage A-2 SOTA recipe (~12 min GPU)

The reproducible "real" run: 5000 images, 100 epochs, cosine LR
schedule with linear warm-up, saliency-FPS query corner
warm-start. This is the config behind the headline
**0.7460 ± 0.035 mAP_50** result.

```bash
python -m signedkan_wip.src.vision.train_circles_ricci \
    --n-images 5000 --epochs 100 --lr 3e-3 \
    --ricci-scale 1.0 --warm-start \
    --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 \
    --configs '+ricci-mod' \
    --seed 0 \
    --save-checkpoint /tmp/hymeyolo_ckpts/ \
    --jsonl-out /tmp/cmnist_a2.jsonl
```

Now run the demo against the saved checkpoint:

```bash
python -m signedkan_wip.src.vision.demo_hymeyolo_tk \
    --checkpoint /tmp/hymeyolo_ckpts/ricci-mod_seed0.pt
```

The header now shows `source: checkpoint`, `epochs: 100`,
`backbone: tiny`, `fpn: none`.

For 5 seeds (the proper benchmark protocol):

```bash
bash signedkan_wip/experiments/run_hymeyolo_stage_a2_5seed_2026_05_16.sh
# results land in signedkan_wip/experiments/results/hymeyolo_stage_a2_*/
```

---

## 4. Climbing the ladder (Stage B + C)

The `--backbone` and `--fpn` flags swap the architecture without
changing the rest of the training recipe.  Three backbones ship
today:

| `--backbone` | What                                                | Params @ d=32 |
|:-------------|:----------------------------------------------------|--------------:|
| `tiny`       | 3-conv stack, ReLU (the Stage A default)            | ~25 k         |
| `resnet`     | ResNet-tiny — 2-conv residual blocks, ReLU          | ~107 k        |
| `hsikan`     | Same shape as `resnet` but Catmull-Rom basis activations (HSiKAN primitive) instead of ReLU | ~111 k |

The 5-seed Cluttered MNIST mAP_50 numbers
([reports/2026-05-17-hymeyolo-stage-c-5seed.md](../reports/2026-05-17-hymeyolo-stage-c-5seed.md)):

| Stage         | Backbone        | FPN         | 5-seed mAP_50      | σ      | wall  |
|--------------:|:----------------|:------------|-------------------:|-------:|------:|
| B `b_resnet`  | ResNet-tiny     | single      | 0.8955             | 0.0267 | 1627s |
| B `b_hsikan`  | **HSiKAN-CR**   | single      | **0.9032**         | **0.0087** | 3395s |
| C `c_fpn`     | ResNet-tiny     | 2-level FPN | 0.8926             | 0.0238 | 1779s |

`b_hsikan` has the highest mean *and* the lowest σ across the
three Stage B/C variants — a 3× variance reduction vs `b_resnet`
at the cost of 2.1× training wall (the per-channel univariate
basis function is more arithmetic-heavy than a ReLU).  Paired vs
`b_resnet` it's a **TIE** (z = +0.61) but the stability is real.

**Stage B (ResNet-tiny):**

```bash
python -m signedkan_wip.src.vision.train_circles_ricci \
    --n-images 5000 --epochs 100 --lr 3e-3 \
    --ricci-scale 1.0 --warm-start \
    --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 \
    --backbone resnet \
    --configs '+ricci-mod' \
    --seed 0 --save-checkpoint /tmp/hymeyolo_b_ckpts/
```

**Stage B' (HSiKAN-CR backbone):**

```bash
python -m signedkan_wip.src.vision.train_circles_ricci \
    --n-images 5000 --epochs 100 --lr 3e-3 \
    --ricci-scale 1.0 --warm-start \
    --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 \
    --backbone hsikan \
    --configs '+ricci-mod' \
    --seed 0 --save-checkpoint /tmp/hymeyolo_b_hsikan_ckpts/
```

This is the test of whether **HSiKAN's signed-cycle basis-function
primitive transfers to vision** — swapping ReLU for the same
Catmull-Rom basis the signed-link-prediction story uses.  Same
architecture as `b_resnet`, only the activation differs; the win
is a direct measure of the activation's contribution.

**Stage C (ResNet-tiny + 2-level FPN):**

```bash
python -m signedkan_wip.src.vision.train_circles_ricci \
    --n-images 5000 --epochs 100 --lr 3e-3 \
    --ricci-scale 1.0 --warm-start \
    --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 \
    --backbone resnet --fpn 2level \
    --configs '+ricci-mod' \
    --seed 0 --save-checkpoint /tmp/hymeyolo_c_ckpts/
```

> The 2-level FPN currently requires `--backbone resnet` (TinyBackbone
> has no `/4` tap).  `--backbone hsikan --fpn 2level` is the
> obvious next probe; it has not been tested yet — see the
> "Stage C''" follow-up in the Stage C report.

Pre-baked 5-seed launchers (use one of these for benchmark-grade
runs):

```bash
bash signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh b_resnet
bash signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh b_hsikan
bash signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh c_fpn
```

The demo automatically picks up the new architecture from the
checkpoint dict (`backbone`, `fpn` keys saved alongside the
state_dict).

---

## 5. Real images — Stage D on PASCAL VOC2007

Cluttered MNIST is a synthetic playground; **Stage D is the real
test.** Same architecture (Stage C — resnet backbone + 2-level
FPN), but on VOC2007 images at 224×224 with 20 classes and 12
queries.

```bash
# Smoke (50 images, 1 epoch, CPU OK — under 30 s):
python -m signedkan_wip.src.vision.train_voc_stagec \
    --n-images 50 --epochs 1 --input-size 96 --batch-size 4 \
    --seed 0 --device cpu

# Production single-seed (5011 trainval images, 30 epochs, ~60 min GPU):
python -m signedkan_wip.src.vision.train_voc_stagec \
    --image-set trainval --epochs 30 --input-size 224 --batch-size 8 \
    --n-box-queries 12 --lr 3e-3 --seed 0 \
    --save-checkpoint /tmp/stage_d_ckpts/ \
    --jsonl-out /tmp/stage_d_smoke.jsonl
```

The orchestrator runs the smoke first and only proceeds to the
5-seed when smoke mAP_50 ≥ 0.10 (CLAUDE §3 production-scale gate):

```bash
bash signedkan_wip/experiments/run_stage_d_voc2007_2026_05_18.sh
# results in signedkan_wip/experiments/results/stage_d_voc2007_*/
```

VOC2007 data lives under [data/torchvision/VOCdevkit/VOC2007/](../data/torchvision/VOCdevkit/VOC2007/) — already on disk.
Stage D plan: [docs/plans/2026-05-17-hymeyolo-stage-d-pascal-voc/plan.pdf](plans/2026-05-17-hymeyolo-stage-d-pascal-voc/plan.pdf).

---

## 6. Where to look for results

### Quick view: every run, latest per stage

```bash
python -m signedkan_wip.experiments.show_hymeyolo_results
```

Walks every HyMeYOLO run dir under
`signedkan_wip/experiments/results/`, parses the jsonl outputs,
and prints a one-line-per-(stage, dataset, label) summary
ordered by recency:

```text
stage          dataset                label             n  mAP_50_mean  pstdev    latest run
-----------------------------------------------------------------------------------------------
voc2007        voc2007_trainval       stage_c_voc       1       0.0073  0.0000    stage_d_voc2007_…
c_fpn          cmnist                 +ricci-mod        5       0.8926  0.0238    hymeyolo_ladder_c_fpn_…
b_hsikan       cmnist                 +ricci-mod        5       0.9032  0.0087    hymeyolo_ladder_b_hsikan_…
b_resnet       cmnist                 +ricci-mod        5       0.8955  0.0267    hymeyolo_ladder_b_resnet_…
stage_a2_5seed cmnist                 +ricci-mod        5       0.7460  0.0350    hymeyolo_stage_a2_5seed_…
…
```

Flags:

```bash
python -m signedkan_wip.experiments.show_hymeyolo_results --all          # one row per seed
python -m signedkan_wip.experiments.show_hymeyolo_results --stage b_hsikan
python -m signedkan_wip.experiments.show_hymeyolo_results --csv > out.csv
```

### Raw artefacts

| Where                                                            | What                                                  |
|------------------------------------------------------------------|-------------------------------------------------------|
| `signedkan_wip/experiments/results/<run-dir>/`                   | per-run logs, jsonl, orchestrator.log                 |
| `*.jsonl`                                                        | one JSON record per config / seed (mAP, loss curve)   |
| `reports/2026-05-16-hymeyolo-cluttered-mnist-sota.md`            | Stage A-2 SOTA writeup                                |
| `reports/2026-05-17-hymeyolo-stage-c-5seed.md`                   | Stage B/B'/C 5-seed paired analysis                   |
| `reports/2026-05-13-hymeyolo-kcycle-localization-bug.md`         | the +kcycle bug story (don't repeat it)               |
| `docs/plans/2026-05-16-hymeyolo-stage-*/plan.pdf`                | per-stage plans (claim, falsifier, rollback)          |
| `docs/plans/2026-05-17-hymeyolo-stage-d-pascal-voc/plan.pdf`     | the Stage D plan                                      |

Aggregate a single jsonl by hand:

```bash
python -c "
import json, pathlib, statistics as s
rows = [json.loads(l) for l in pathlib.Path('/tmp/cmnist_a2.jsonl').read_text().splitlines() if l]
for r in rows:
    print(f\"{r['label']:<15s}  mAP_50={r['mAP_50']:.4f}  loss_drop={r['loss_drop_pct']:.1f}%\")
"
```

---

## 7. One-paragraph mental model

The CV stack is a `RicciHyMeYOLOMulti` model: a backbone (tiny /
resnet) maps the image to features; optional FPN gives multi-scale
samples; learnable **query corner positions** (warm-started by
saliency-FPS) bilinear-sample those features; a per-query
**HSiKAN aggregator** encodes the signed-cycle structure across
the K corners; per-query **Ricci descriptors** (scalar curvature
κ, mean cos θ, edge-length variance) feed the class head; the
offset head refines each corner. Hungarian matching assigns
predictions to ground-truth boxes. The lift from Stage 0
(0.50 mAP) to Stage A-2 (0.75 mAP) came from cosine LR + warm
start; the lift from A-2 to Stage B (0.90) came from the deeper
backbone; Stage C adds multi-scale. Stage D tests whether any
of this transfers from synthetic clutter to natural images.

For the geometric / hypergraph theory behind HSiKAN and
Ricci-modulation, see `docs/book/` (mdBook) and
`docs/differential-geometry-primer.pdf`.
