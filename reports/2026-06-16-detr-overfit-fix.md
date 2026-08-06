# DETR overfit fairness-guard fix — `l1giou` box loss + stable recipe

**Date:** 2026-06-16 · **Plan:** [docs/plans/2026-06-16-detr-baseline/](../docs/plans/2026-06-16-detr-baseline/) · **Slug:** detr-overfit-fix

## Summary

The MiniDETR budget-matched baseline (plan `2026-06-16-detr-baseline`) shipped with its
**overfit fairness guard `xfail`'d**: a from-scratch MiniDETR plateaued at
`mAP50_proxy ≈ 0.149` on 16 single-object images and could not overfit them. Until a
*competent* DETR exists, the headline "RicciStim (5 896 params, 0.228) is at parity with
DETR" is unprovable — a strawman baseline would be worse than none on this honesty-themed
line. This task **diagnosed the failure, fixed it, and removed the `xfail`.**

The xfail note guessed "IoU>0.5 localisation is the open task" but did not name the cause.
Instrumentation isolated it precisely: **classification was already solved** (`clsCorrect → 1.0`,
no no-object collapse); the **box regression saturated at IoU ≈ 0.46**, just under the 0.5
the metric requires, and stayed flat. This is the textbook pure-GIoU plateau — GIoU's
gradient vanishes as boxes approach overlap, with no L1 term to close the gap.

**Fix (two coupled parts, both measured):**
1. **`box_loss_kind="l1giou"`** — a new additive branch in `hungarian_set_loss` returning
   the canonical DETR box loss `corner-L1 + AABB-GIoU`. The L1 term supplies the
   non-vanishing gradient that drives IoU past 0.5. The existing `l1` and `giou` branches
   are **byte-for-byte unchanged** (the matching cost never depended on `box_loss_kind`),
   so RicciStim's `giou` parity — itself fairness guard #2 — is preserved.
2. **Stable recipe `lr=1e-3 + grad-norm clip 0.1`** — `l1giou` alone at the original
   `lr=3e-3` *diverged* past ~step 400 (mAP 0.18 → 0.023). Lowering lr to 1e-3 and adding
   DETR-canonical gradient clipping makes training monotone to a perfect overfit.

Under the fixed recipe the guard reaches **`mAP50_proxy = 1.000` by ~step 450** and **passes**.
The full RicciStim-vs-DETR head-to-head is now unblocked (a separate ~1–2 h GPU run).

## Diagnosis (measured, 16 single-object images, standard preset, GPU)

Instrumented per-50-step: `mAP` (eval metric), `matchIoU` (IoU of the Hungarian-matched
box vs its GT), `clsCorrect` (matched-query class accuracy), `noObjP` (no-object prob mass).

| recipe | step 200 | step 300 | step 450 | step 599 | verdict |
|---|---|---|---|---|---|
| **giou, lr 3e-3** (original) | mAP 0.139 / IoU **0.460** | — | — | (200 steps) | box plateau at IoU 0.46 |
| **l1giou, lr 3e-3** (no clip) | mAP 0.139 / IoU 0.460 | mAP 0.163 / IoU 0.664 | mAP **0.023** / IoU 0.362 | mAP 0.023 | IoU crosses 0.5 then **diverges** |
| **l1giou, lr 3e-3, clip 0.1** | mAP 0.153 / IoU 0.434 | mAP 0.153 / IoU 0.450 | mAP 0.153 / IoU 0.437 | mAP 0.153 | clip stops blow-up; lr too high to converge |
| **l1giou, lr 1e-3, clip 0.1** ✅ | mAP 0.282 / IoU 0.879 | mAP 0.585 / IoU 0.940 | mAP **1.000** / IoU 0.959 | mAP **1.000** | monotone perfect overfit |

`clsCorrect` reached 1.0 in every recipe by step ~100 — classification was never the
problem. The discriminator was the box term and training stability.

## Files touched

| File | Δ | What |
|---|---|---|
| [hymeko_neuro/experiments/vision/hymeyolo_hungarian.py](../hymeko_neuro/experiments/vision/hymeyolo_hungarian.py) | +37 / −20 | new `l1giou` box branch (additive); factored `_aabb_from_corners` helper; explicit `ValueError` on unknown `box_loss_kind` (§6.4); contract in docstring |
| [hymeko_neuro/experiments/vision/detr_baseline.py](../hymeko_neuro/experiments/vision/detr_baseline.py) | +13 / −2 | `train_one_seed` uses `l1giou`; `grad_clip=0.1` param + `--grad-clip` CLI; clip applied in loop |
| [hymeko_neuro/tests/test_detr_baseline.py](../hymeko_neuro/tests/test_detr_baseline.py) | +15 / −9 | overfit guard: `xfail` removed → `@pytest.mark.slow`; recipe → `l1giou` + lr 1e-3 + clip 0.1; recipe rationale in docstring |
| [hymeko_neuro/tests/test_hymeyolo_stage_a3.py](../hymeko_neuro/tests/test_hymeyolo_stage_a3.py) | +35 / −3 | `l1giou` added to the variant-finiteness loop; new `test_l1giou_box_term_is_l1_plus_giou` (additive identity); new `test_hungarian_rejects_unknown_box_kind` |

Total **100 insertions / 34 deletions**, 4 files. One-off diagnostic
(`scripts/dev/detr_overfit_diag.py`) written and **deleted** after use (no tree residue).

## CORE.YAML items touched

**None.** `hymeko_hungarian.py` / `detr_baseline.py` are `hymeko_neuro` application code,
not listed in any CORE.YAML crate / file / glob. No dependency change.

## Test results

- **Unit / regression (fast), `-m "not slow"`:** `test_hymeyolo_stage_a3.py` +
  `test_detr_baseline.py` → **18 passed, 1 deselected** in 7.96 s. Includes the two new
  tests: the `l1giou == l1 + giou` box-term additive identity (class weights zeroed to
  isolate the box term) and the unknown-kind `ValueError` guard.
- **Fairness guard (slow), the load-bearing gate:**
  `test_detr_overfits_small_set` → **PASSED** in 149.7 s (`final mAP50_proxy = 1.0`,
  gate `> 0.4`). Was `XFAIL` before this change — a regression that would have failed
  against the prior `giou`/`lr 3e-3` recipe (§3 regression-test rule satisfied).
- **Production-scale smoke (full train path):** `tiny`, n_train 300 / n_eval 200 / 3 ep,
  GPU → ran clean in **13.1 s**, **n_params 10 815** (order-matched to RicciStim ~5.9k),
  loss monotone 6.78 → 5.66. mAP low (~0.03) as expected at 3 ep — this validates the
  path, not a result. Artifact: `reports/detr_smoke_20260616.jsonl`.

## Performance

- Overfit guard: 149.7 s on the RTX 3070 (heavy by unit-test standards → marked `slow`).
- Extrapolated full head-to-head (next step, not run here): ~50 min `tiny`,
  ~1.5–2 h `standard` at 5000 img / 40 ep — consistent with the plan's budget.
- No RSS concern: small transformer over 64×64, VRAM ≪ 8 GB.

## §6.5 anti-patterns

**None introduced.** The new axis was added as a value of the *existing* `box_loss_kind`
dispatch (`l1`/`giou`/`l1giou`), not a new function name (avoids #1/#5); the box-term
logic is shared via `_aabb_from_corners` rather than duplicated (#0/§6.1). Pre-existing
ruff errors remain in untouched functions of `hymeyolo_hungarian.py` (`evaluate_multi`,
`main`: F841 unused locals, E702 semicolons at lines 321–437) and a pre-existing test
(lines 43/47) — **not introduced by this change** and out of scope; flagged here as
standing lint debt for a separate cleanup.

## Open / follow-up

1. **Run the head-to-head** — `standard` + `tiny` at 5000 img / 40 ep / seed 0, vs
   RicciStim 0.228 @ 5 896 params. The deliverable table. Honest framing: matched
   low-budget regime (DETR canonically wants 300+ ep).
2. **Multi-seed** before any published number (single seed here).
3. The `slow` guard is GPU-timed at 149 s; on CPU CI it will be substantially slower —
   keep it opt-in via `-m slow`.

## Follow-up: full-run grad-clip default miss (caught + corrected, same day)

The overfit recipe (`lr 1e-3 + grad-clip 0.1`) was set as `train_one_seed`'s **default**
and propagated into the first full head-to-head run. That was an error: clip 0.1 is an
overfit *stabiliser* for 16 memorised images; on the 5 000-image run it clips nearly every
mini-batch gradient to ~0, so **both sizes sat dead-flat from epoch 1** (tiny mAP 0.041→0.037
over 40 ep; standard loss flat at 4.97 from epoch 6). That is a strangled run, **not** a
DETR verdict — reporting 0.037 as "DETR's number" would have rebuilt the exact strawman the
overfit guard exists to prevent. The run was **stopped** (§4, no value in flat epochs).

Grad-clip sweep at 2 000 img / 8 ep / max_digits 3 (`reports/detr_clipsweep_20260616.jsonl`):

| grad-clip | loss 1→8 | mAP 1→8 | verdict |
|---|---|---|---|
| 0.1 | 5.86 → 5.36 (flat) | 0.043 → 0.040 (flat) | strangled — the bug |
| 1.0 | 5.87 → 4.59 | 0.037 → 0.0485 | learns (stable) |
| 0 (none) | 5.89 → 4.11 (steep) | 0.038 → 0.056 (climbing) | learns fastest |

**Correction:** `train_one_seed` / CLI `--grad-clip` default `0.1 → 1.0` (stable for an
unattended 40-ep run *and* learns; `0` is marginally faster but less safe). The overfit
**guard keeps its inline clip 0.1** — different regime, commented so it is not "fixed". The
corrected head-to-head (`reports/detr_headtohead_v2_20260616.jsonl`, clip 1.0, 40 ep) is the
number that will be reported; the strangled v1 jsonl is retained only as the bug artifact.
**Lesson (logged):** a recipe tuned on a toy overfit must be re-validated at production scale
*before* it becomes a default — exactly CLAUDE.md §3's production-scale-smoke rule, which I
skipped on the first launch.

## Head-to-head result (the number the whole exercise was for)

Corrected run (`reports/detr_headtohead_v2_20260616.jsonl`, clip 1.0, lr 1e-3,
l1giou, 5000 img / 40 ep / seed 0, same ClutteredMNIST + metric as RicciStim):

| model | params | final mAP50_proxy | status |
|---|---|---|---|
| **RicciStim** (upgraded, cached) | **5 896** | **0.228** | tuned (config F, 40 ep) |
| **DETR-tiny** (corrected) | 10 815 | **0.376** | clean monotone learning (loss 5.60→2.19) |
| DETR-standard | 1 243 791 | 0.041 | **COLLAPSED — not a valid number** (see below) |

**Metric parity confirmed:** RicciStim's `evaluate_map50` and DETR's
`evaluate_detr_map50` call the *identical* `match_f1_at_iou50` core (same 0.05
score threshold, same per-image-F1@IoU0.5-averaged scheme). The only post-proc
difference is NMS (RicciStim applies greedy NMS over dense anchors; DETR omits it
— correct, set-prediction detectors are designed not to need it). Same data,
budget, seed, metric core. The comparison is fair.

**Honest reading.** A competent, param-matched DETR-tiny (10.8k params) reaches
**0.376 > RicciStim's 0.228** at ~1.8× the parameters. The param-efficiency
*win* claim does **not** survive a fairly-trained baseline: RicciStim uses fewer
learned params but is **not Pareto-optimal** against a properly-trained tiny
transformer — DETR-tiny is both larger *and* substantially more accurate. The
earlier strangled run (0.037) would have falsely said "DETR loses"; the fair run
reverses it. This is the fairness guard doing exactly its job, against us.

**DETR-standard collapsed (do not cite 0.041).** The 1.24M-param model stayed flat
(loss 5.27→5.17, mAP ~0.04) under the same lr 1e-3 / clip 1.0 that the tiny model
thrived on — classic large-transformer fragility (a 3+3-layer d=128 transformer
from scratch needs lr warmup / lower lr; the tiny 1-layer d=16 model is robust to
the plain recipe). Ironically this *demonstrates* the "transformers are finicky to
tune" point — but it cut against RicciStim, because the easy-to-train tiny DETR
already beat it. The standard number is not reported as a result.

## Open / follow-up (updated)

1. **DETR-standard re-run with lr warmup** (or lr 1e-4 + longer) to get a valid
   large-model number — currently collapsed, not a verdict.
2. **Multi-seed** for DETR-tiny's 0.376 and RicciStim's 0.228 before any published
   comparison (both single-seed).
3. The headline for the TPAMI efficiency story must be rewritten: *not* "structure
   beats DETR at fewer params" (falsified here) but, at most, "structure reaches
   non-trivial detection with very few learned params" — and even that is
   below a fair param-matched transformer. The defensible line moves to
   robustness/accountability, not accuracy-per-param.

## Provenance

- Git SHA `2b80cab` (working tree dirty: the 4 files above + the pre-existing
  `docs/seminar/HyMeKo_Seminar.with_refs.pptx`).
- Env: torch 2.12.0+cu132 (CORE pin), scipy 1.17.1, numpy 2.4.6; Python 3.12.13; Win32;
  RTX 3070, CUDA 13.2.
- Seed 0 throughout. ClutteredMNIST fixture (canvas 64, max_digits 1 for the guard,
  3 for the smoke), deterministic from seed.
