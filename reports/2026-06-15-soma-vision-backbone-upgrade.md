# Soma-vision backbone upgrade — learned mixer + highway + cross-scale pyramid

**Date:** 2026-06-15 · **Plan:** `docs/plans/2026-06-15-soma-vision-backbone-upgrade/`
· **Persona:** Aiko

## Summary

`RicciStimBackbone` combined its three Bochner-wrapped branches (walk / polygon /
triangle) by a **bare sum** — the weakest possible aggregator, and the exact
configuration that was falsified for vision (RicciStim 0.14 mAP on
Cluttered-MNIST; walk-only base-Soma 0.52 vs 0.91 linear on MNIST). This change
ports the signed-link Gömb side's learned-αₖ mixer + highway routing to the
vision lane and adds a top-down cross-scale pyramid, behind opt-in flags so the
falsified baseline stays bit-reproducible for a fair A/B.

Three small `nn.Module`s in `soma/vision/aggregators.py`:
- **LearnedBranchMixer** — `Σ softmax(α)_k · b_k` (α init 0 ⇒ uniform mean, not a
  bare sum); replaces the sum.
- **HighwayGate** — `T·H(m) + (1−T)·s`, `T = σ(W[m;s])`, gate bias init −1 so the
  net starts near the (working) patch-encoder skip and *learns* to route through
  the hypergraph branches only if they help.
- **CrossScalePyramid** — top-down parent→child fusion over the quadtree's
  `parent_indices`, processed coarse→fine so a child sees its parent's *updated*
  feature.

## A/B verdict (the headline)

MNIST, 3 paired seeds × 4 epochs, n_train=1500, n_test=500, CPU. Flags **off**
(baseline bare sum) vs **on** (mixer+highway+pyramid).

| | params | seed0 | seed1 | seed2 | mean | pstd |
|---|---|---|---|---|---|---|
| baseline `ricci_stim` | 4 736 | 0.228 | 0.284 | 0.304 | **0.2720** | 0.0322 |
| upgraded `ricci_stim_up` | 5 811 | 0.296 | 0.304 | 0.310 | **0.3033** | 0.0057 |
| paired Δ | +1 075 | +0.068 | +0.020 | +0.006 | **+0.0313** | — |

**Two honest conclusions:**

1. **The upgrade is a real, consistent win over the bare sum.** Every paired seed
   improves (Δ all positive); mean +0.031 (≈ +11 % relative) and variance
   collapses (pstd 0.032 → 0.006). The strong aggregator does what it was meant
   to: it beats the thing it replaced, monotonically. Cost is +1 075 params
   (+22.7 %) and ≈ 0 extra wall time (baseline 312–374 s/seed vs upgraded
   325–360 s/seed — the three modules are negligible vs the conv branches, as the
   plan predicted).

2. **The vision falsification still stands.** Both sit at ≈ 0.27–0.30 on MNIST
   against ~0.91 for a trivial `Linear(784,10)`. Upgrading the aggregator does
   **not** rescue hypergraph-vision for MNIST classification — it tests the
   *strong* aggregator and the approach is still well below a linear baseline.
   This is the negative half of a genuine experiment, and the plan's risk note
   anticipated it. The un-falsified axis remains brain-predictivity (Cichy-92),
   not accuracy.

## Detection A/B (directional smoke — added after the classification verdict)

The falsification *headline* was detection (RicciStim 0.14 mAP on Cluttered-MNIST),
but the upgrade flags reached only the classifier. They are now threaded through
`RicciStimDetector` too; the runner gained `--upgrade`. Directional smoke (1 seed,
n_train=400, n_eval=150, 8 epochs, canvas 64, max_digits 2, config F, CPU):

| | params | final mAP50_proxy | epoch-1 loss → final | wall |
|---|---|---|---|---|
| baseline (bare sum) | 4 821 | **0.0216** | 1.24 → 0.130 | 162.5 s |
| upgraded (mixer+highway+pyramid) | 5 896 | **0.0855** | 0.52 → 0.089 | 171.0 s |

The upgrade gives **~4× the proxy mAP** here, converges markedly faster (epoch-1
train loss 0.52 vs 1.24 — the highway skip starts near the working patch encoder),
and adds +5 % wall / +22 % params. The upgraded curve is still climbing at epoch 8
(0.070 → 0.076 → 0.085), so 8 epochs *understates* it; the baseline is flat/noisy
near 0.02.

**Caveats (honest):** single seed, reduced scale. Both are well below the prior
config-F headline (~0.174 at n_train=5000, 20 epochs), so this does **not**
overturn the vision falsification — it confirms, consistently with the
classification A/B, that the strong aggregator helps detection too, directionally.
A publishable number needs a multi-seed (3–5) run at n_train=5000 / 20 epochs,
baseline vs upgraded — a multi-hour job (≈ 0.4 s/img-epoch ⇒ ~6 h for the full
matrix), to be sized and checkpointed before launch.

## Files touched

| File | Note |
|------|------|
| `signedkan_wip/src/hymeko_gomb/soma/vision/aggregators.py` | new — 3 modules |
| `…/vision/ricci_stim_backbone.py` | flags `use_arity_mixer/use_highway/use_pyramid`; combine replaces bare sum |
| `…/vision/ricci_stim_classifier.py` | threads the flags through |
| `…/vision/ricci_stim_detector.py` | threads the flags through (detection path) |
| `…/vision/train_mnist.py` | `ricci_stim` / `ricci_stim_up` model types + argparse choices |
| `experiments/run_ricci_stim_cluttered_mnist.py` | `--upgrade` flag; records `upgrade` in JSONL |
| `signedkan_wip/tests/test_soma_aggregators.py` | new — 9 unit/smoke tests |
| `…/tests/test_gomb_soma_vision_ricci_stim_detector.py` | +1 regression test (upgrade reaches detector) |
| `reports/ricci_stim_upgrade_ab_20260615.jsonl` | classification A/B raw rows (6) |
| `reports/ricci_stim_detect_ab_smoke_20260615.jsonl` | detection A/B smoke rows (2) |

CORE.YAML: none (`signedkan_wip` is non-core). No new dependency (torch already
CORE-pinned).

## Test results

- `pytest -p no:randomly signedkan_wip/tests/test_soma_aggregators.py` — **9
  passed**: mixer simplex weights + uniform-init = mean (not sum) + wrong-branch-
  count rejection; highway carries the skip when `T→0` + differentiable; pyramid
  fuses parent→child, true 2-level cascade, no-op when parentless, differentiable;
  full upgraded backbone forward+backward smoke with all three flags on.
- Production-scale smoke: the upgraded backbone ran the full MNIST A/B (above)
  end-to-end, 3 seeds, no NaN/crash — satisfies the §3 "production-scale smoke
  before queuing" rule.

## Performance

- Wall: comparable to baseline (≈ 5–6 min/seed CPU); the upgrade adds < 5 %
  wall, within noise.
- Memory: three `O(n_anchors·d)` modules + one scatter over the tree; RSS ≪ 16 GB.
- No criterion/regression-grade timing claimed — the metric of record here is
  test accuracy, not latency.

## Experiment provenance

- Git SHA `9684f09`; **working tree dirty** (this is an active dev tree — the
  `hymeko_hive` generators work and many `signedkan_wip` edits are uncommitted;
  the A/B used the in-tree `aggregators.py` + `ricci_stim_*` as listed above).
- Device: CPU. Seeds: 0, 1, 2. n_train=1500, n_test=500, n_epochs=4, batch_size
  default, Adam lr 3e-3.
- Raw rows: `reports/ricci_stim_upgrade_ab_20260615.jsonl` (6 rows).
- Two torch UserWarnings (sparse invariant checks disabled; sparse-CSR beta) —
  benign, from `stim_graph.py` / `hodge.py`, not introduced here.

## §6.5 anti-patterns

Parametric difference → config flags on the *same* class (not class-per-variant) —
correct per #8 (the difference is parametric: which aggregator modules are
instantiated, not a structural rewrite of forward). Mixer is a module, not a
string-typed branch (#7). No new globals. Defaults off ⇒ baseline reproducible.

## Open issues / follow-up

- **Cluttered-MNIST detection A/B** — directional smoke done (above): upgraded
  ~4× baseline proxy mAP, no regression. **Full multi-seed run at n_train=5000 /
  20 epochs not yet run** — needed for a publishable detection number; size +
  checkpoint before launch (~6 h CPU for the baseline-vs-upgraded matrix).
- Cortical Brain-Score (#2) remains the un-falsified axis; deferred per user until
  real Cichy-92 fMRI is sourced.
- Backlog: this closes the "do all three upgrades" item; the falsification line in
  BACKLOG.md Soma-vision section is unchanged (still falsified for accuracy).
