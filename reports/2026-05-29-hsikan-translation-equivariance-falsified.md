# Report — HSiKAN translation-equivariance hypothesis FALSIFIED (tie_we hurts)

**Date:** 2026-05-29
**Predecessor:** `reports/2026-05-29-vision-rebench-strong-correction.md` (HSiKAN viable; 4.5 pp / 7.0 pp residual gap to CNN on MNIST / Fashion).
**Hypothesis tested:** the residual CNN gap is the per-edge weight `W_e` breaking translation equivariance. Tying `W_e` to a single scalar (the minimum equivariance-restoring change) should close most of it.
**CORE.YAML items touched:** none.

## Headline — **hypothesis falsified, 0/6 seeds favor tied; both σ ≈ -17**

| dataset | seed | untied (baseline) | tied (tie_we) | Δ (tied − untied) |
|:--|--:|--:|--:|--:|
| mnist | 0 | 0.9419 | 0.8767 | −0.0652 |
| mnist | 1 | 0.9441 | 0.8658 | −0.0783 |
| mnist | 2 | 0.9419 | 0.8723 | −0.0696 |
| **mnist MEAN** | | **0.9426** | **0.8716** | **−0.0710  σ−18.5  wins 0/3** |
| fashion | 0 | 0.8426 | 0.7937 | −0.0489 |
| fashion | 1 | 0.8340 | 0.7900 | −0.0440 |
| fashion | 2 | 0.8341 | 0.7943 | −0.0398 |
| **fashion MEAN** | | **0.8369** | **0.7927** | **−0.0442  σ−16.8  wins 0/3** |

Tying widens the CNN gap on both datasets:

| | gap to CNN (untied) | gap to CNN (tied) |
|:--|--:|--:|
| MNIST | 4.5 pp | **11.6 pp** (worse by 7.1) |
| Fashion | 7.0 pp | **11.4 pp** (worse by 4.4) |

## Interpretation

The simplest reading: **the per-edge weight `W_e` was useful capacity, not an equivariance-break.** The model was successfully using per-position weighting — possibly to attend to image regions differently (corner vs center, foreground vs background) — and forcing strict translation equivariance removes that lever.

So the residual CNN gap is **not** translation equivariance. The most plausible remaining suspects, in order:

1. **Missing within-RF spatial filter structure.** CNN's 5×5 filter has 25 distinct learnable weights per (in, out) channel pair, learning *spatial patterns* (edges, textures) *within* each RF. HSiKAN's RF aggregation is a uniform mean over the RF — 0 learned weights per position-within-RF. This is the biggest expressive asymmetry between the two operators; the natural follow-up is to add a learnable kernel-shaped weight inside each arity's incidence (`weighted_inc[v, e, pos_in_rf]` instead of binary `inc[v, e]`).
2. **Depth / capacity.** HSiKAN at n_layers=2, h=32 vs CNN at 2 conv layers, h=32. Same nominal capacity but the CNN's parameter budget is concentrated in 5×5 kernels (where the expressive bandwidth lives). Scaling HSiKAN may not close the gap if (1) is true.
3. **Boundary effects in `D_v_inv_sqrt`.** Corner pixels are in fewer RFs → different normalization. The tie_we test left this untouched; could still contribute a residual on top of (1).

## Side observation — wall

| | mean wall / cell |
|:--|--:|
| untied baseline (pre-CR-fix code) | 6 946 s |
| tied (post-CR-fix + post-cat-fix + smaller W_e) | 4 246 s |

Speedup 1.64×. Most of this is the **CR-activation fix** (1.42× from 2026-05-29 wall report); a small additional gain from tie_we's parameter reduction (10 218 → 9 814) and the cat-elimination follow-up. The cat-fix component is benchmarked separately in the next report.

## Test / quality

- 6 / 6 cells, 0 failures.
- 13 parity tests still pass (5 tie_we + 9 CRActivation).
- ruff clean on changed files (the 7 ruff errors in `hsikan_vision.py` are pre-existing unused imports).

## Provenance

- **Git SHA:** `8fd8187` (dirty: includes tie_we plumbing + cat-elimination fix).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130 (CORE pin 2.12, user-approved drift).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Config:** `--n-epochs 20 --train-subset 0 --batch-size 128 --hidden 32 --lr 1e-3 --tie-we`, seeds {0,1,2}, MNIST + Fashion. Baseline at `/tmp/vision_strong/results.jsonl` (overnight 2026-05-28 chain).
- **Artifacts:** `/tmp/vision_tie_we/results.jsonl` (6 rows), `summary.json`.

## Conclusion

The 4.5 / 7.0 pp residual CNN gap is **not** translation equivariance. The per-edge `W_e` is doing real work. The natural next experiment is to address the missing **within-RF spatial filter** — add a learnable position-within-RF weight (shared across RF positions like a CNN kernel). That's a richer change than `tie_we`: it modifies the incidence representation to have a per-position weight inside each RF.

If/when that experiment is run, the prediction is:
- If within-RF filtering closes most of the remaining gap → the gap was CNN-style spatial kernels, and a "translation-equivariant CNN-kernel HSiKAN" is the right vision-specific operator.
- If it doesn't close it → the gap is depth/capacity (3) or the boundary effect, or some combination.
