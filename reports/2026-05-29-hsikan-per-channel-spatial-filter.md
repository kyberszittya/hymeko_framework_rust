# Report — Per-channel within-RF spatial filter beats capacity doubling

**Date:** 2026-05-29
**Predecessors:**
- `reports/2026-05-29-hsikan-spatial-filter-and-capacity-curve.md` (scalar W_pos[K]: MNIST partial win, Fashion null).
- `reports/2026-05-29-hsikan-capacity-sweep-h64.md` (h=64: closes 24–38 % of gap).
**CORE.YAML items touched:** none.

## Headline — **structural > capacity at this scale**

| | h=32 baseline | h=32 + scalar | **h=32 + per_channel** | h=64 | h=128 | CNN |
|:--|--:|--:|--:|--:|--:|--:|
| **MNIST** | 0.9426 | 0.9497 | **0.9622 ± .0023** | 0.9595 | 0.9679 | 0.9874 |
| **Fashion** | 0.8369 | 0.8311 *(null)* | **0.8569 ± .0048** | 0.8539 | 0.8645 | 0.9071 |
| **params** | 10 218 | 10 684 | **25 130** | 32 298 | 113 322 | 42 154 |

**Paired Δ on the same seeds (3 / 3 wins everywhere):**

| baseline | MNIST Δ | σ | Fashion Δ | σ |
|:--|--:|--:|--:|--:|
| vs h=32 | **+0.0196** | **+15.5** | **+0.0200** | **+4.82** |
| vs h=32 + scalar | +0.0125 | +12.0 | **+0.0258** | +3.93 |
| **vs h=64 (capacity)** | **+0.0028** | **+2.02** | **+0.0029** | **+2.65** |

**Per-channel at h=32 beats capacity-doubled h=64 on BOTH datasets**, at
**25 k params vs 32 k** — smaller AND better. **Structural > capacity at
this scale.**

## What this resolves

The 2026-05-29 night-3 report's prediction was: *Fashion's null at
scalar would lift once W_pos becomes per-output-channel*. Exactly that:

- **Fashion scalar:** Δ −0.0058 σ−1.09 (null/slightly negative).
- **Fashion per_channel:** Δ +0.0200 σ+4.82 (decisive).

A 32× expansion of W_pos (per arity: K × 1 → K × d_out=32) catches a
structural signal a channel-invariant per-position weight could not.
This is consistent with: textures depend on *which* channel responds at
each position (edges in one channel, color/intensity in another); a
single scalar per position can't represent that.

MNIST also gained from per_channel beyond what scalar reached
(0.9497 → 0.9622, Δ +0.0125 σ+12.0) — the per-channel formulation is
strictly more expressive and the model exploits it.

## Updated ranking of axes on the residual CNN gap

| Axis | MNIST Δ | Fashion Δ | Cost (params) |
|:--|--:|--:|--:|
| **per_channel spatial filter** | **+0.0196 σ+15.5** | **+0.0200 σ+4.82** | +14 912 |
| capacity (h=32 → h=128) | +0.0253 (cumulative) | +0.0276 (cumulative) | +103 104 |
| capacity (h=32 → h=64) | +0.0168 | +0.0170 | +22 080 |
| scalar spatial filter | +0.0071 σ+3.22 | −0.0058 σ−1.09 | +466 |
| tie_we (equivariance) | **−0.071** σ−18.5 | **−0.044** σ−16.8 | −404 |

**Per-channel structural lever now leads all single-axis changes
tested**, at modest parameter cost (~6× the scalar's cost; far cheaper
than h=64 or h=128 capacity bumps).

## Remaining CNN gap

| Best HSiKAN-vision | MNIST | Fashion |
|:--|--:|--:|
| h=32 + per_channel (25 k params) | 2.52 pp from CNN | 5.03 pp |
| h=128 baseline (113 k params) | 1.95 pp | 4.27 pp |

h=128 still nominally wins, but with **4.5× the parameters**. The
natural combination experiment — **per_channel + h=64** (or +h=128) —
should close more of the remaining gap. Combination is the next sweep.

## Engineering bonus

Wall: per_channel @ h=32 + compile = **1 089 s/cell** (~2× scalar's
557 s, expected since the einsum is now 3D). Total 6-cell sweep wall:
1.8 h. Cumulative speedup vs the original code per cell:
6 946 → 1 089 = **6.4×** at h=32 + per_channel.

## Files / tests / runs

| | |
|:--|:--|
| Code changes | `spatial_filter: bool` → `str ∈ {"none","scalar","per_channel"}` enum refactor across `SignedBranchConv`, `HSiKANVisionLayer`, `HSiKANVisionClassifier`, runner CLI, orchestrator CLI. `per_channel` branch in `SignedBranchConv.forward` uses 3D `weighted_inc` and dispatched einsums (`"ved,bvd->bed"` / `"ved,bed->bvd"`). |
| Parity tests | **5 new** for per_channel (shape, param count, W_pos=ones bit-equality with baseline, trains-without-NaN, error on invalid enum value). All 31 vision tests pass. |
| Cells | 6 / 6, 0 failures. |
| Wall per cell | 1 089 s (h=32 + per_channel + compile + expandable_segments). |
| Peak GPU per cell | ~3 GiB (per_channel materialises 3D `weighted_inc` per einsum). |

## Provenance

- **Git SHA:** `8fd8187` (dirty).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130.
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Seeds:** {0,1,2}; n_epochs 20; train_subset 0; batch_size 128;
  --hidden 32; --spatial-filter per_channel --compile;
  expandable_segments.
- **Artifacts:** `/tmp/vision_per_channel/results.jsonl`, summary.json.

## Per-seed table

| dataset | seed | h=32 base | h=32 scalar | **h=32 per_channel** | h=64 |
|:--|--:|--:|--:|--:|--:|
| mnist | 0 | 0.9419 | 0.9463 | **0.9648** | 0.9586 |
| mnist | 1 | 0.9441 | 0.9531 | **0.9613** | 0.9608 |
| mnist | 2 | 0.9419 | 0.9497 | **0.9606** | 0.9590 |
| fashion | 0 | 0.8426 | 0.8311 | **0.8590** | 0.8554 |
| fashion | 1 | 0.8340 | 0.8166 | **0.8521** | 0.8492 |
| fashion | 2 | 0.8341 | 0.8456 | **0.8596** | 0.8572 |

## Conclusions

1. **The residual CNN gap is partly per-channel within-RF spatial
   structure** — confirmed decisively. The per-channel formulation was
   the correct refinement of the scalar variant's partial confirmation.
2. **Structural beats capacity at this scale.** A 25 k-param HSiKAN
   with per-channel spatial filter beats a 32 k-param HSiKAN with
   doubled hidden, on both datasets.
3. **HSiKAN-vision is now within 2.5 / 5.0 pp of CNN at 60 % of CNN's
   parameters** (25 k vs 42 k). This is a competitive vision operator.
4. **Open**: how much more does per_channel + h=64 (or h=128) close?
   That combination experiment is the next launch.

## Follow-ups (ordered)

1. **per_channel + h=64** (launching next): tests whether structural +
   capacity together fully close the gap. Predict: another 0.5–1 pp on
   each dataset; possibly within 1 pp of CNN on MNIST.
2. **per_channel + n_layers=4**: deeper structural may compound.
3. **Boundary D_v fix** (the remaining equivariance break): replace
   per-vertex degree normalisation with a constant. Should add another
   small lift on edges of the image, especially Fashion.
4. CPML+pose scoping note ready at
   `docs/scoping/2026-05-29-cpml-pose-detection-tracking.md` — distinct
   research direction; awaiting greenlight.
