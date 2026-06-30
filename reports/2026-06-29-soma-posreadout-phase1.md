# Position-Aware Readout Program — Phase 1 (+1.5): Pooling on Cluttered-MNIST

**Date:** 2026-06-29
**Plan:** [docs/plans/2026-06-29-soma-position-aware-readout-program/](../docs/plans/2026-06-29-soma-position-aware-readout-program/)
**Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Summary

Phase 1 of the gated 3-phase program. The 2026-06-29 cell×readout sweep showed
the Gömb-Soma MNIST ceiling was *pooling-bound* — a position-preserving flatten
readout beat the linear control. But flatten (i) does not scale
(`n_patches·d` head) and (ii) was tested only on *centred* MNIST. Phase 1 builds
a scale-free **attention pool** (`out_dim = d`, content-weighted) and a
**single-digit Cluttered-MNIST classification** adapter (digit at a *random*
position), and A/Bs the readouts on both.

## Results (5-seed mean ± pstd, 5 epochs, Adam 3e-3, CPU)

**Cluttered-MNIST (random-position digit, 48×48):**

| arm | readout | params | acc |
|---|---|---:|---|
| linear control | — | 23050 | 0.2126 ± 0.0057 |
| Soma | mean-pool | 2010 | 0.3136 ± 0.0230 |
| Soma | attention (scale-free) | 2027 | 0.4662 ± 0.0302 |
| Soma | pos-attention (what+where) | 4331 | 0.4464 ± 0.0529 |
| Soma | **flatten (full spatial map)** | 24890 | **0.6166 ± 0.0157** |

**Centred MNIST (for contrast):** Soma + attention 0.5364 ± 0.0147 (vs the prior
mean-pool 0.519, flatten 0.945, linear 0.906).

Figure: `reports/figures/soma_posreadout_cluttered_20260629.png`.

## G1 gate decision

Plan G1 was *"attention ≳ flatten on centred AND attention > {mean, flatten} on
cluttered → proceed."* **The literal condition fails:** attention < flatten on
both centred (0.54 vs 0.95) and cluttered (0.47 vs 0.62). But the result is
informative, not null:

1. **Position-keeping generalizes to random position.** On cluttered, *both*
   position-aware readouts beat mean-pool: attention +0.15, flatten +0.30. The
   walk-conv features carry real signal — flatten 0.62 ≫ linear 0.21.
2. **My earlier guess was wrong** (recorded honestly): I expected absolute-position
   flatten to *collapse* on random placement. It did not — flatten preserves the
   full *spatial feature map*, and a large head learns position-conditional
   detectors that work wherever the digit lands.
3. **Pure content-attention is insufficient.** A permutation-invariant,
   content-weighted set pool captures *where the content is* but discards the
   *spatial layout* that flatten keeps — exactly the 0.15 gap. So the cheap
   scale-free pool is a partial fix, not the answer.

## Phase 1.5 — positional attention (the correction)

Hypothesis: content-attention loses because it drops position; add a learned
per-patch positional embedding (`Readout.POS_ATTENTION`, pooled vector encodes
*what + where*, 4331 params) and it should close to flatten.

**Result: it did not** — pos-attention **0.4464 ± 0.0529**, statistically the
same as plain attention (0.4662) and still far below flatten (0.6166). Adding
position to the pool did not help.

**Sharpened conclusion.** The bottleneck is *not* "lacks position" — it is the
**single-vector compression**. Both attention variants collapse 144 patches to
one 16-d vector; flatten hands the head the full 144×16 = 2304-d spatial map.
The position-keeping that matters lives in the readout's **spatial resolution**,
not in a single (even position-aware) attended vector. On bounded grids flatten
*is* the working readout; the genuinely scalable equivalent would need to keep
*multiple* pooled vectors — a **multi-query attention pool** (K queries → K·d,
K ≪ n_patches) or a coarse spatial pool — not a single query.

Measured / inferred / open: *measured* — five cluttered means + centred
attention; *inferred* — the readout bottleneck is single-vector compression, not
absence of position (pos-attention ≈ attention ≪ flatten); *open* — whether a
multi-query pool recovers flatten's accuracy at ≪ flatten's params (a separate
investigation, not blocking Phase 2).

## Decision for Phase 2

The readout sub-question converged: **flatten is the readout that works** on
these bounded grids; single-vector pools plateau at ~0.45. So Phase 2 carries
**flatten** into the full RicciStim backbone (the proven readout) to test the
actual structural question — does the 3-branch Walk/Polygon/Triangle structure +
a fair readout beat the encoder-only control on cluttered? Multi-query attention
is logged as the scalable-readout follow-up.

## Files touched

| file | change |
|---|---|
| `signedkan_wip/src/hymeko_gomb/soma/vision/walk_conv_classifier.py` | `Readout.ATTENTION`/`POS_ATTENTION` + `_AttentionReadout`/`_PosAttentionReadout` |
| `signedkan_wip/src/vision/cluttered_classification.py` | new: single-digit Cluttered-MNIST classification adapter (wraps `ClutteredMNIST`) |
| `signedkan_wip/src/hymeko_gomb/soma/vision/train_mnist.py` | `gomb_soma_attn` arm; `--dataset {mnist,cluttered}`, `--canvas`; image-size parametrized |
| `signedkan_wip/experiments/runs/soma_holonomy_ab_plot.py` | attention arm in ARM maps; `title`/`caption` params |
| `signedkan_wip/tests/test_soma_position_aware_readout.py` | new: 10 attention/adapter tests |

## CORE.YAML items touched
None. All in `signedkan_wip/`. Reuses `ClutteredMNIST`; no new dependency.

## Test results
- 10 new tests (attention softmax/scale-free/convex/perm-invariant/trainable;
  cluttered adapter shape/determinism/canvas-guard) + 9 cell/readout regression +
  4 plot tests — all pass; ruff clean.

## Performance / provenance
- Cluttered canvas 48 (≈144 patches) ≈ 70–90 s/seed/epoch (≈2.5× centred MNIST —
  the larger grid, expected; §11 reconciled at smoke). Peak RSS < 1 GB.
- One run anomaly: the first cluttered-attention arm died at exit 127 (external
  kill, mid seed-0 epoch-3, no traceback) — re-run standalone, completed clean
  (0.419/0.465/0.514/0.472/0.461). The other 19 records were unaffected.
- JSONL: `reports/soma_posreadout_phase1_20260629.jsonl` (25 records);
  cluttered-only figure subset `reports/soma_posreadout_cluttered_figure_20260629.jsonl`.
- Seeds 0–4; git SHA 9aea4f6 (tree dirty — feature diff). torch 2.12.0, CPU.

## Next (gated Phase 2/3)
- **Phase 1.5: DONE** — pos-attention did not close the gap (single-vector
  compression is the bottleneck, not position). Readout sub-question converged on
  flatten for bounded grids.
- **Phase 2:** wire **flatten** into `RicciStimClassifier` (full 3-branch) and
  A/B vs the encoder-only control (`ablate_structural_branches`) on cluttered —
  the structural question. Gate G2: full > encoder-only by >1σ.
- **Phase 3:** stacked walk-conv depth.
- **Follow-up (non-blocking):** multi-query attention pool (K·d, K≪n_patches) —
  the scalable readout that might match flatten without its `n_patches·d` head.
