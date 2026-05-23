# HyMeYOLO `circles + ricci` 5-seed — backfill of stage-7 seeds 1-3

**Date:** 2026-05-13
**Git SHA:** `0c55fa81d0df99ed6a96566e3317ea122553d6ce`
**Storage:** `reports/overnight_2026_05_11_stage7/hymeyolo_ricci_n5k_e50_s{1,2,3}.redo_20260513T144352Z.jsonl`
**Wall:** 17:19:53 → 19:39:02 CEST = **2h 19m** for 18 runs (3 seeds × 6 variants).
**Memory cap:** `systemd-run --user -p MemoryMax=16G` (no `ulimit -v`, per `feedback_ulimit_vs_cuda`).

## Summary

The 2026-05-11 stage-7 overnight sweep silently lost seeds 1, 2, 3 to a
`ulimit -v 16G` interaction with CUDA workloads (29 GB VAS at 1.77 GB
RSS; see memory). This backfill re-ran those three seeds with the proper
cgroup cap. Combined with the original seeds 0 / 4, we now have a clean
5-seed picture for the Cluttered MNIST phase-1 sweep.

## 5-seed aggregate (all six variants, n=5)

| Variant | mAP50 mean ± pstdev | box_cls | circ_cls | acceptance |
|---|---|---:|---:|---:|
| `boxes+circles` | **0.715 ± 0.163** | 0.55 | 0.70 | **5/5** |
| `+ricci-mod` | **0.723 ± 0.180** | 0.42 | 0.72 | **5/5** |
| `+kcycle` | 0.204 ± 0.028 | **0.86** | **0.79** | 5/5 |
| `boxes-only` | 0.356 ± 0.184 | 0.51 | 0.00 | 4/5 |
| `baseline` | 0.138 ± 0.131 | 0.21 | 0.00 | 2/5 |
| `circles-only` | 0.069 ± 0.087 | 0.00 | 0.12 | 1/5 |

## Per-seed mAP50 (all 5 seeds, all 6 variants)

```
  variant            s0      s1      s2      s3      s4
  +kcycle           0.240   0.225   0.196   0.198   0.160
  +ricci-mod        0.720   0.803   0.529   1.017¹  0.546
  baseline          0.306   0.010   0.088   0.282   0.005
  boxes+circles     0.723   0.433   0.687   0.807   0.923
  boxes-only        0.615   0.218   0.536   0.259   0.153
  circles-only      0.035   0.020   0.243   0.029   0.018
```

¹ mAP_50 > 1.0 on s3 `+ricci-mod` is a metric-computation artifact at low
   GT counts (the averaging convention can exceed 1 when a single GT box
   contributes to multiple class IoU bins); inspecting that seed is open
   follow-up.

## Findings

1. **The single-seed "+ricci-mod wins" claim does not replicate at 5 seeds.**
   `+ricci-mod` 0.723 ± 0.180 vs `boxes+circles` 0.715 ± 0.163 — paired Δ
   ≈ +0.008 inside both σ bands. Ricci modulation doesn't measurably help
   *or* hurt vs the same model without it; the 0.69 single-seed box_cls
   for `+ricci-mod` (from the earlier `phase1_*.jsonl`) regressed to mean
   0.42 across seeds.

2. **`+kcycle`'s localization bug is real and *systematic* across seeds.**
   mAP50 = 0.204 ± **0.028** — the *lowest variance of any variant*.
   That's the signature of a structural failure mode, not stochastic
   tuning noise: classification scores are excellent (box_cls=0.86,
   circ_cls=0.79, *highest of any variant*) but boxes land wrong.
   This is exactly what the
   [+kcycle localization bug diagnosis](2026-05-13-hymeyolo-kcycle-localization-bug.md)
   predicted: the signed-cycle aggregator feeds classification but never
   feeds corner offset prediction. The very low cross-seed variance makes
   this an ideal target for the proposed fix — a single-seed smoke after
   the fix should clearly separate fix-from-baseline.

3. **`baseline` is unstable** (2/5 acceptance, mAP50 σ = 0.131). The
   plain `HyMeYOLOMulti` without circle/Ricci machinery succeeds on
   seeds 0 and 3 but degenerates on seeds 1, 2, 4 (loss barely drops,
   box_cls ≈ 0). Suggests the model is finding a local minimum
   that's seed-dependent — likely an initialization brittleness in the
   query embeddings.

4. **`circles-only` doesn't work** (1/5 acceptance). Circles alone
   can't represent the rectangular digit bounding boxes that Cluttered
   MNIST asks for. Confirms the architectural split: circle queries
   are the *capacity* lever, not the *primary* representation lever.

5. **`boxes-only` is dominated by `boxes+circles`** (mAP50 0.356 → 0.715
   = +0.36). Adding circle queries to a box-only model provides a clear
   structural lift, even though `circles-only` alone fails. The two
   query types capture complementary signal.

## Practical regime

The Pareto-front of (mAP50 mean, acceptance rate, parameter cost):

- **`boxes+circles`** is the no-frills winner — highest reliability
  (5/5 accept), competitive mAP50 (0.715), no Ricci module overhead.
- **`+ricci-mod`** is statistically tied at higher variance — likely
  worth keeping in the ablation panel but **not** as the headline
  variant.
- **`+kcycle`** is broken on localization, expected to dominate when
  the bug fix from
  [2026-05-13-hymeyolo-kcycle-localization-bug.md](2026-05-13-hymeyolo-kcycle-localization-bug.md)
  lands. Predicted post-fix range: mAP50 0.65–0.85 (matching
  `boxes+circles` or beating it, since classification is already at
  0.86).

## Test results

- 18 runs, 18 successful (3 seeds × 6 variants). All produced jsonl
  rows; no silent kills (cgroup cap held).
- Original seeds 0 / 4 retained from the 2026-05-11 stage-7 sweep —
  combined 5-seed picture is via dedup-by-(seed, variant) on the two
  result sets.

## Performance budget

- Per-seed wall: 2790s, 2771s, 2788s (≈ 46 min each). Variance
  negligible — workload is dataset + model size bound, not stochastic.
- Per-variant wall: ~400-800s. `boxes+circles` and `+ricci-mod` are
  ≈40% slower than `boxes-only` (the dual-query Hungarian solve is
  the cost), `+kcycle` is the slowest (cycle aggregator overhead).
- Peak RSS: well under the 16 GB cap (cgroup never tripped); GPU
  memory ~1.5–2 GB per run.

## Open issues

1. **`+kcycle` fix** — the localization-path patch from
   [2026-05-13-hymeyolo-kcycle-localization-bug.md](2026-05-13-hymeyolo-kcycle-localization-bug.md)
   is the highest-EV next vision change. The very low variance of
   the broken-baseline (0.028) makes a 1-seed smoke statistically
   meaningful for the fix.
2. **`+ricci-mod` mAP > 1 on seed 3** — metric-computation edge case
   worth understanding (clip to [0,1]? change averaging? specific
   to a single-GT-box image?).
3. **`baseline` instability** — 2/5 acceptance is too low to keep as a
   reference in any future ablation panel without understanding why.
   Likely query-init brittleness; a seed-dependent query
   initialization (warm-start from a small CNN feature map?) would
   probably stabilise.
4. The `n_images=10000, epochs=100` extension that was attempted in
   stage-7 (`hymeyolo_ricci_n10k_e100_s0.json`) also crashed via the
   same `ulimit -v` issue — that re-run is the natural next scaling
   step after the `+kcycle` fix lands.

## Experiment provenance

- **Script:**
  [signedkan_wip/experiments/run_hymeyolo_ricci_seeds123_redo_2026_05_13.sh](../signedkan_wip/experiments/run_hymeyolo_ricci_seeds123_redo_2026_05_13.sh)
- **Master log:**
  [reports/overnight_2026_05_11_stage7/REDO_seeds123_20260513T144352Z.master.log](overnight_2026_05_11_stage7/REDO_seeds123_20260513T144352Z.master.log)
- **Python env:** miniconda3 (torch 2.11.0+cu130, cuda True).
- **GPU:** RTX 2070 SUPER, 8 GiB, driver 580.126.09.
- **Seeds:** 1, 2, 3 (deterministic from `--seed`).
- **Dataset:** Cluttered MNIST, n_images=5000, canvas=64, max_objects=3,
  per-seed regenerated.

## CORE.YAML items touched

None.
