# HyMeYOLO YOLO-parity ladder — workflow

**Status as of 2026-05-16 evening.** The ladder uses a single unified
orchestrator + a single unified paired analyser, parameterised by
stage name. Adding a new stage = config change, not a new file.

## Stages currently defined

| Stage | Levers added | Status | 5-seed mean mAP_50 |
|-------|--------------|--------|-------------------:|
| `baseline` | (none) — honest baseline, no warm-start, const LR, e=50 | shipped | 0.5041 ± 0.039 |
| `a1` | warm-start query corners | shipped | 0.6279 ± 0.052 |
| `a2` | + cosine LR + warmup + e=100 | shipped | 0.7460 ± 0.035 |
| `a3_lite` | + LayerNorm + WeightDecay + focal cls (3 of 4 A-3 levers; **no** GIoU) | single-seed smoke 0.749; 5-seed not run | (smoke only) |
| `a3` | + GIoU box (full Stage A-3, 4 levers) | **deferred** — GIoU branch is ~2.3× slower per epoch due to per-image `torch.stack` in `_box_loss_on_matched`; needs vectorisation before practical 5-seed launch | — |
| `b_resnet` | + ResNet-tiny backbone + A-3-lite levers | **shipped 2026-05-16 evening** | **0.8955 ± 0.027** |
| `b_hsikan` | + HSiKAN-CR backbone (Catmull-Rom basis-function activation at ResNet-tiny topology) + A-3-lite levers | **in flight 2026-05-16 night** (started 21:27, ETA ~23:50) | TBD |
| `c` | + FPN multi-scale heads | not yet implemented | predicted ~0.95 |

**Why `a3_lite` vs `a3`:** A 2026-05-16 evening smoke of the full
4-lever `a3` hit the 2400-second per-run timeout mid-training
(no jsonl row produced). Profiling identified the GIoU branch's
per-image AABB extraction (8 `min`/`max` reductions + 2
`torch.stack` per matched-batch element, called inside a Python
`for b in range(B):` loop) as the dominant overhead. The
`a3_lite` stage ships the other three levers at Stage A-2's
wall budget; the GIoU lever is queued for a Stage A-4 patch
that vectorises the AABB extraction over the batch dimension.

## Default behaviour as of 2026-05-16 evening

The CLI defaults of `train_circles_ricci.py` are now:

- `--warm-start` (saliency-FPS query corner init): **ON by default**
- `--schedule cosine` with `--warmup-epochs 10` `--min-lr-ratio 0.01`:
  **ON by default**
- The Stage A-3 levers (`--use-layernorm`, `--weight-decay`,
  `--cls-loss focal`, `--box-loss giou`): **OFF by default**, opt-in via the
  ladder orchestrator's `a3` stage.

Pass `--no-warm-start` / `--schedule constant` / `--warmup-epochs 0`
to reproduce the pre-2026-05-16-evening behaviour explicitly. All
historical orchestrators (pre-2026-05-16-evening) have been
defensively patched to do this so future re-runs match published
numbers.

## Running a stage

```bash
# Run a 5-seed for any defined stage:
./signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh <stage>

# Compare two stages' results paired-by-seed:
python -m signedkan_wip.experiments.analyse_hymeyolo_ladder_paired \
    <target-stage-dir> <baseline-stage-dir>
```

Each stage writes to
`signedkan_wip/experiments/results/hymeyolo_ladder_<stage>_<STAMP>/`
with one jsonl row per seed + an orchestrator log. The analyser
reads two such dirs, aligns by seed, and reports:

- per-stage mean ± pstdev / min / max
- paired Δ (target − baseline) per seed
- paired-mean Δ + pstdev + verdict against the pre-registered
  criterion (mean ≥ 0.03 AND z ≥ 2 → WIN; ≤ -0.03 at z ≤ -2 → LOSS;
  else tie)
- mAP_50 > 1.0 sanity (the metric-bug fix from 2026-05-16 morning
  must keep holding)

## Protocol parity rules

The ladder's headline claims are **paired-by-seed** comparisons. To
preserve that:

1. **Same `--n-images` / `--lr` / `--seed` set across stages.**
   The unified orchestrator defaults to `N_IMAGES=5000`,
   `LR=0.003`, `SEEDS="0 1 2 3 4"`.
2. **Same dataset realisation per seed.**
   `make_cluttered_mnist_hungarian_format` is deterministic from
   `--seed`. Don't override the seed list between stage runs.
3. **Same git SHA.**
   The orchestrator captures the SHA at launch into the master
   log. Any change to the model / loss / data path between stage
   runs *invalidates* the paired comparison.
4. **Same Python interpreter.**
   `miniconda3` with `torch 2.11`. The `.venv` (torch 2.4.1) is
   not interchangeable for this benchmark — the upstream `torch`
   release also affects roi_align / Adam / etc. in subtle ways.

## Adding a new stage

1. Add the new lever(s) to `train_circles_ricci.py` argparse +
   thread the kwargs into `train_one_config` and the loss / model.
2. Make defaults preserve the previous stage's behaviour.
3. Add a `case "<stage>"` block to
   `run_hymeyolo_ladder_5seed.sh` setting the stage's
   `STAGE_FLAGS`, `EPOCHS`, and `PER_RUN_TIMEOUT`.
4. Add a plan dir `docs/plans/2026-05-16-hymeyolo-stage-<stage>-*/`
   with the 4-format plan.
5. Run: `./run_hymeyolo_ladder_5seed.sh <stage>` after a smoke.
6. Analyse: `analyse_hymeyolo_ladder_paired.py <stage-dir>
   <previous-stage-dir>`.
7. Write the report at
   `reports/2026-05-16-hymeyolo-stage-<stage>-5seed.md`
   (Stage A-1, A-2, A-3 reports are the templates).
8. Update `docs/SOTA_RESULTS.md` § 0.5 + add a memory entry.

The plan dir + report + 4-format-plan-on-disk discipline is the
operating-contract bar (CLAUDE.md § 2 + § 9). Don't skip it.

## Smoke-then-launch pattern

Each new stage uses the same idiom:

1. **Single-seed smoke** (~18-20 min for e=100): same CLI as the
   5-seed but with one seed. Confirms the new code path doesn't
   crash, doesn't produce NaN, doesn't break the metric cap.
2. **Validate the smoke jsonl** — mAP_50 in [0, 1], `wall_s`
   roughly matches the stage's predicted wall.
3. **Launch 5-seed** via the ladder orchestrator.
4. **Analyse** when the 5-seed lands.

Smoke + 5-seed + report + SOTA update + memory = one half-day's
work per stage, fitting comfortably in an overnight budget.

## What `--rich` is and isn't

The 2026-05-16 `--rich` flag on `hymeko emit` (in `hymeko_cli`,
not in this train script) is a SEPARATE workflow that affects the
SDF/URDF emission path. It's unrelated to the ladder; mentioned
here only because it shares the date.

## Open questions for future stages

- **At what point does Cluttered MNIST saturate?** Stage B b_resnet's
  worst seed (0.843) is already above the prior best single-seed of
  any earlier variant. With 5-seed mean at 0.895, the benchmark's
  ceiling is closer than the original "0.85-0.92" estimate suggested.
  A wider-c_out follow-up (Stage B'') could probe saturation.
- **Does the CR primitive transfer from signed graphs to vision?** The
  b_hsikan stage answers this directly: same backbone topology as
  b_resnet, only swap is Catmull-Rom basis-function activation for
  ReLU. If b_hsikan ≈ b_resnet, CR is regime-general; if it loses
  meaningfully, the bounded-domain claim (CR's strength lives in the
  σ-cycle inductive bias on natively signed data, not in vision)
  applies.
- **Pure-backbone control for b_resnet attribution.** The b_resnet
  result shipped with the A-3-lite levers (LN + WD + focal) co-applied.
  A clean ResNet-only run (TinyBackbone + A-3-lite as the control)
  would isolate the backbone's contribution from the regularisation
  bundle. Stage B' if attribution becomes important.
- **Stage D (port to VOC subset / COCO-mini)** is the actual
  generalisation test. Cluttered MNIST is informative but
  bounded; YOLO-parity needs real-data measurement.

---

*Convention document, not a report. Update when a new stage
lands or when defaults change.*
