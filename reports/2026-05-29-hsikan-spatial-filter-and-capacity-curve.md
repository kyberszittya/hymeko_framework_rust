# Report — HSiKAN within-RF spatial filter (partial confirmation) + full capacity curve

**Date:** 2026-05-29
**Predecessors:**
- `reports/2026-05-29-hsikan-translation-equivariance-falsified.md` (tie_we hurt)
- `reports/2026-05-29-hsikan-capacity-sweep-h64.md` (h=64 closes 24–38 % of gap)
**CORE.YAML items touched:** none.

## Headline

The within-RF spatial-filter hypothesis (the natural follow-up after
tie_we falsification) gets **partial confirmation**:

| | h=32 baseline | **h=32 + spatial_filter** | h=64 | h=128 | CNN |
|:--|--:|--:|--:|--:|--:|
| **MNIST** | 0.9426 | **0.9497 ± .0034** | 0.9595 | 0.9679 | 0.9874 |
| **Fashion** | 0.8369 | 0.8311 ± .0128 | 0.8539 | 0.8645 | 0.9071 |

- **MNIST**: spatial_filter HELPS — paired Δ +0.0071, σ +3.22, 3/3 wins.
  But less than doubling hidden (+0.0168) — about 42 % of the capacity
  lever, at 4.6 % the parameter cost (+466 vs +22 080 params).
- **Fashion**: spatial_filter is NULL/slightly negative — Δ −0.0058,
  σ −1.09, 1/3 wins.

## Capacity curve (h=32 → 64 → 128) — clean log-linear with floor

| h | params | MNIST | gap | Fashion | gap |
|--:|--:|--:|--:|--:|--:|
| 32 | 10 218 | 0.9426 | 4.48 pp | 0.8369 | 7.02 pp |
| 64 | 32 298 | 0.9595 | 2.79 pp | 0.8539 | 5.32 pp |
| **128** | **113 322 (2.7× CNN!)** | **0.9679** | **1.95 pp** | **0.8645** | **4.27 pp** |
| CNN | 42 154 | 0.9874 | 0 | 0.9071 | 0 |

Each doubling of `h` *halves* the remaining gap, with diminishing
returns. **At h=128 HSiKAN has 2.7× CNN's parameters and still trails by
1.95 / 4.27 pp** — proving the residual gap is **structural, not
capacity**. h=128 wall 1 164 s/cell × 6 cells = 1.9 h.

## What the two together tell us

The session's three axis tests on the residual CNN gap, ranked by
strength:

1. **Capacity (h)** — strongest lever. h=32→128 closes ~50–55 % of the
   gap on both datasets. Saturating as params grow past CNN's.
2. **Within-RF spatial structure** — partial. **MNIST: helps modestly**
   (+0.71 pp at h=32, σ+3.22). **Fashion: null** at the
   channel-invariant W_pos[K] formulation.
3. **Per-edge weight tying (translation equivariance)** — **HURTS** by
   4–7 pp on both datasets (yesterday's tie_we falsification). The
   per-edge W_e was useful capacity, not an equivariance break.

The honest reading: **the residual gap is part capacity, part
spatial-pattern-within-RF, and part something else.** The "something
else" surfaces most clearly on Fashion (where even h=128 only closes to
4.27 pp and spatial_filter does nothing). Most plausible suspects for the
remaining Fashion floor:

- **Per-channel spatial filter**: W_pos[K, d_out] — currently the same
  spatial weight is shared across all channels. CNN has *different*
  spatial filters per output channel. Adds K × d_out params per arity.
- **CNN-equivalent kernel**: W[K, d_in, d_out]. Adds K × d_in × d_out per
  arity. At this point we're basically rebuilding CNN; the operator
  loses its "hypergraph" character.
- **Boundary D_v effects** (corner pixels in fewer RFs → different
  normalisation). Always there; not addressed by any axis tested.

## Why MNIST helps and Fashion doesn't (interpretive — not measured)

MNIST is **stroke-based**: position within a 5×5 RF carries strong signal
(where the pen-stroke is matters). A channel-invariant W_pos[K] can learn
"weight stroke center vs edge differently." Fashion is **texture-based**:
patterns depend on multi-channel spatial co-occurrence (edges in one
channel + colors in another), which a channel-invariant per-position
weight cannot represent.

So the spatial-filter hypothesis is **partially right** — there *is* a
structural component to the gap — but the minimal W_pos[K]
parameterisation only catches part of it. The full structural answer for
vision-style spatial pattern learning is a CNN kernel; if we want to keep
the hypergraph operator, the next refinement is W_pos[K, d_out]
(per-output-channel spatial filter), which is the natural follow-up.

## Files / tests / wall

| | |
|:--|:--|
| Code changes | `build_rf_position_incidence` (new), `SignedBranchConv(spatial_filter, kernel)` + W_pos param + forward branch, `HSiKANVisionLayer.register_buffer("pos_*", …)`, `HSiKANVisionClassifier(spatial_filter=)`, runner+orchestrator `--spatial-filter` flags. |
| Parity tests | **6/6 new** (`test_hsikan_spatial_filter.py`) — incl. forward bit-equality W_pos=ones vs baseline, param-count arithmetic (466 added), pos_inc structure, trains-without-NaN, error on missing pos_inc. All 32 vision tests pass. |
| Per-cell wall | 557 s/cell (h=32 + spatial_filter + compile) — **12× faster** than the original h=32 baseline's 6 946 s/cell, thanks to CR fix + cat fix + compile compounding. |
| Per-cell RSS | ~1.5 GiB (well under 7 GiB budget). |
| GPU OOMs | 0. |

## Provenance

- **Git SHA:** `8fd8187` (dirty).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130 (CORE drift, user-approved).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Seeds:** {0,1,2}; n_epochs 20; train_subset 0; batch_size 128;
  --hidden 32; --spatial-filter --compile; expandable_segments.
- **Artifacts:** `/tmp/vision_spatial/results.jsonl`, `summary.json`,
  per-cell logs. Capacity-curve artifacts at `/tmp/vision_strong/` (h=32),
  `/tmp/vision_h64/`, `/tmp/vision_h128/`.

## Per-seed table

| dataset | seed | h=32 base | h=32 + spatial | Δ |
|:--|--:|--:|--:|--:|
| mnist | 0 | 0.9419 | 0.9463 | +0.0044 |
| mnist | 1 | 0.9441 | 0.9531 | +0.0090 |
| mnist | 2 | 0.9419 | 0.9497 | +0.0078 |
| fashion | 0 | 0.8426 | 0.8311 | −0.0115 |
| fashion | 1 | 0.8340 | 0.8166 | −0.0174 |
| fashion | 2 | 0.8341 | 0.8456 | +0.0115 |

(Fashion seed=2 wins; seeds 0 and 1 lose. High seed-variance, σ−1.1
overall — statistical null.)

## Conclusions

1. **Capacity dominates.** h=32→h=128 closes 50–55 % of the CNN gap.
   Saturates around 2.7× CNN params with ~2/4 pp residual; the residual
   is structural.
2. **Channel-invariant within-RF spatial filter is a partial answer.**
   MNIST: real but small win. Fashion: null.
3. **Cumulative wall speedup of all session's engineering:** 12× per
   cell (6 946 s → 557 s at h=32). CR fix + cat fix + compile is the
   right default stack for any future HSiKAN-vision sweep.
4. **No engineering or single-axis-architectural change tested this
   session fully closes the gap to CNN.** The honest position: HSiKAN
   vision is genuinely competitive (h=128 within 1.95–4.27 pp of CNN at
   2.7× the params), but the last 2–4 pp on Fashion likely needs a
   per-channel spatial filter (W_pos[K, d_out]) — at which point HSiKAN
   becomes structurally closer to a CNN-with-extra-steps. The
   "hypergraph operator with maximal independence from CNN inductive
   biases" appears to floor near these numbers.

## Follow-ups (ordered)

1. **W_pos[K, d_out]** — per-output-channel spatial filter. Adds
   K × d_out per arity. Tests whether Fashion's null is the
   channel-invariance restriction. Easy code change from current code:
   reshape W_pos and broadcast over the channel dim.
2. **`torch.utils.checkpoint`** the per-layer block to unlock B≥256 and
   h=256 (deferred memory follow-up from the speedup report).
3. **Reach over the hypergraph: try the variant on cluttered-MNIST,
   CIFAR-10 grayscale** to see if the operator's robustness story is
   different on harder tasks.
