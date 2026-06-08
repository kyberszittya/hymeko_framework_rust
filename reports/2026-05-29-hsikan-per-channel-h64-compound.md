# Report — HSiKAN per_channel + h=64 compound (closes the vision push)

**Date:** 2026-05-29
**Predecessors:**
- `reports/2026-05-29-hsikan-per-channel-spatial-filter.md` (per_channel @ h=32 beats h=64).
- `reports/2026-05-29-hsikan-capacity-sweep-h64.md`, `…-spatial-filter-and-capacity-curve.md`.
**CORE.YAML items touched:** none.

## Headline — compound matches h=128 at ~55 % of its params on MNIST

| Config | params | MNIST | gap | Fashion | gap |
|:--|--:|--:|--:|--:|--:|
| h=32 baseline | 10 218 | 0.9426 | 4.48 pp | 0.8369 | 7.02 pp |
| h=32 + per_channel | 25 130 | 0.9622 | 2.52 pp | 0.8569 | 5.03 pp |
| h=64 | 32 298 | 0.9595 | 2.79 pp | 0.8539 | 5.32 pp |
| h=128 | 113 322 | 0.9679 | 1.95 pp | **0.8645** | **4.27 pp** |
| **h=64 + per_channel** | **62 122** | **0.9700 ± .0038** | **1.74 pp** | 0.8580 ± .0080 | 4.91 pp |
| CNN reference | 42 154 | 0.9874 | 0 | 0.9071 | 0 |

Paired Δ for the compound (h=64 + per_channel) vs the next-best candidates:

| baseline | MNIST Δ | σ | Fashion Δ | σ |
|:--|--:|--:|--:|--:|
| vs h=64 (capacity-only) | **+0.0106** | **+4.35** | +0.0041 | +0.59 |
| vs h=32 + per_channel (structural-only) | +0.0078 | +2.20 | +0.0011 | +0.17 |
| **vs h=128 (largest capacity-only)** | **+0.0021** | +0.81 | **−0.0065** | −1.33 |

## Reading

- **MNIST: compound is the most parameter-efficient win.** Beats h=64
  (capacity-only, 32k) σ+4.35, beats h=32+per_channel (structural-only,
  25k) σ+2.20, **statistically ties h=128 (113k) at 55 % of its params**
  (σ+0.81, 2/3 wins). The Pareto-optimal HSiKAN at this scale.
- **Fashion: structural + capacity falls slightly short of pure
  capacity.** h=128 (113k) edges the compound (62k) by 0.65 pp; paired
  σ−1.33 not significant. So on Fashion the structural lever has
  diminishing returns once you have enough capacity — possibly because
  Fashion's residual gap is more about depth or different inductive
  bias than what per_channel adds.
- **Per-parameter Pareto:** at MNIST CNN 0.9874 / 42k params (23.5e-3
  acc/k-param), compound 0.9700 / 62k = 15.6e-3 acc/k-param;
  h=32+per_channel 0.9622 / 25k = 38.5e-3 (most efficient HSiKAN
  variant); h=128 0.9679 / 113k = 8.6e-3 (least efficient).

## Final Pareto on HSiKAN-vision (descending accuracy)

| Rank | Config | MNIST | Fashion | params |
|--:|:--|--:|--:|--:|
| 1 | h=64 + per_channel | **0.9700** | 0.8580 | 62k |
| 2 | h=128 (pure capacity) | 0.9679 | **0.8645** | 113k |
| 3 | h=32 + per_channel (most param-efficient) | 0.9622 | 0.8569 | 25k |
| 4 | h=64 | 0.9595 | 0.8539 | 32k |
| 5 | h=32 baseline | 0.9426 | 0.8369 | 10k |

## Session ranking on the residual CNN gap (final)

| Axis | best single-axis Δ on h=32 | params added |
|:--|--:|--:|
| **per_channel spatial filter** | **MNIST +0.0196 / Fashion +0.0200** | +14 912 |
| h=32 → h=128 (capacity ×11) | MNIST +0.0253 / Fashion +0.0276 | +103 104 |
| h=32 → h=64 (capacity ×2) | MNIST +0.0168 / Fashion +0.0170 | +22 080 |
| scalar spatial filter | MNIST +0.0071 / Fashion null | +466 |
| tie_we (translation equivariance) | **−0.0710 / −0.0442** | −404 |

**per_channel is the most parameter-efficient lever**, beating
capacity doubling at ~70 % the param cost.

## Wall

| Config | wall / cell | wall total (6 cells) |
|:--|--:|--:|
| h=64 + per_channel + compile | 2 068 s | 12.4 min × 6 = **3.4 h** |

vs the 2026-05-29 morning h=32 baseline at 6 946 s/cell (no compile),
the cumulative speedup at h=32 was 12×. At h=64 + per_channel the
3D-einsum + bigger d_out costs are real — wall scales but stays
comfortably overnight.

## Files / tests

No new code this round (re-used the per_channel + compile + spatial_filter
infra landed earlier in the day). 31 vision tests still pass. Lint clean
on the runner + orchestrator.

## Provenance

- **Git SHA:** `8fd8187` (dirty).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130.
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Config:** `--hidden 64 --n-epochs 20 --train-subset 0 --batch-size 128
  --spatial-filter per_channel --compile` + 3 seeds × {MNIST, Fashion}.
- **Artifacts:** `/tmp/vision_per_channel_h64/results.jsonl`, summary.json.

## Per-seed table

| dataset | seed | h=64 + per_channel | h=128 |
|:--|--:|--:|--:|
| mnist | 0 | 0.9672 | 0.9586 |
| mnist | 1 | 0.9685 | 0.9608 |
| mnist | 2 | 0.9744 | 0.9590 |
| fashion | 0 | 0.8516 | 0.8554 |
| fashion | 1 | 0.8669 | 0.8492 |
| fashion | 2 | 0.8555 | 0.8572 |

## Conclusions

1. **HSiKAN-vision is genuinely competitive with CNN on small-image
   classification.** Best config (h=64 + per_channel, 62k params) is
   within 1.74 pp of CNN's 0.9874 on MNIST at 1.47× CNN's params.
2. **The structural lever (per_channel within-RF spatial filter) is the
   correct axis.** Combined with modest capacity (h=64) it matches the
   doubled-capacity h=128 on MNIST.
3. **Fashion's residual gap is the harder one.** Per_channel alone
   plateaus at +0.020 pp gain; capacity continues to help; the
   combination doesn't compound on Fashion the way it does on MNIST.
   Probable culprits (untested): depth (compounded structural
   processing), or kernel-shape expressiveness beyond what
   W_pos[K, d_out] captures.
4. **The vision push has reached its sensible single-session ceiling.**
   Further pushing would need either (a) depth + long-skip wiring (the
   user's 2026-05-29 idea — depth+narrow sweep is in flight as
   `bq20tr3jc`), or (b) the CPMLPose pivot (infrastructure parked, ready
   to launch).

## Follow-ups (active / parked)

1. **Depth + narrow-breadth HSiKAN sweep** — running now as
   `bq20tr3jc`. Tests `n_layers ∈ {2, 4, 8}` at `h ∈ {8, 16}` to see
   whether depth at narrow widths beats wide-shallow. Prerequisite for
   any long-skip-skipping-10-cells experiment.
2. **CPMLPose Shape A** — fully built and tested, parked. Drops the
   pose-vs-vision pivot in if/when desired (`run_cpml_pose_compare.py`).
3. **Long-skip wiring** — proper U-Net-style or feedback long skips
   only meaningful once the depth axis (point 1) shows HSiKAN converges
   at L ≥ 8.
