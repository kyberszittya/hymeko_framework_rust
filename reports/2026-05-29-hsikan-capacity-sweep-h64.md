# Report — HSiKAN capacity sweep: h=32 → h=64 closes ~30 % of the CNN gap

**Date:** 2026-05-29
**Predecessor:** `reports/2026-05-29-hsikan-translation-equivariance-falsified.md` (tie_we falsified — equivariance NOT the gap; capacity was the next hypothesis on the list).
**CORE.YAML items touched:** none.

## Headline

| | h=32 baseline | **h=64 + compile** | paired Δ | σ | CNN gap closure |
|:--|--:|--:|--:|--:|:--|
| **MNIST** | 0.9426 ± .0013 | **0.9595 ± .0012** | +0.0168 | **+126.25** | 4.5 → **2.8 pp** (38 % closed) |
| **Fashion** | 0.8369 ± .0049 | **0.8539 ± .0042** | +0.0170 | +5.47 | 7.0 → **5.3 pp** (24 % closed) |

6 / 6 cells win at h=64. The MNIST σ is absurdly large (+126) because seed
sd is tiny (.0012); even Fashion at σ+5.5 is statistically decisive.

## Interpretation

After the equivariance hypothesis was falsified yesterday, the natural
next suspect was capacity. The data here is clean: **doubling hidden
narrows the CNN gap by 24-38 %**, but does not close it. Two readings:

1. **Capacity is part of the answer, not all of it.** Some of the residual
   gap was the operator being under-parametered at h=32 (10 218 params,
   4× fewer than CNN's 42 154). At h=64 (32 298 params, ~25 % fewer than
   CNN), the gap shrinks but doesn't vanish.
2. **The remaining 2.8 / 5.3 pp is probably structural** — the
   within-RF spatial filter hypothesis (CNN's 5×5 learned weight pattern
   per channel pair, vs HSiKAN's uniform-mean RF aggregation). That's the
   next experiment to design, and is a substantive operator redesign.

## Wall — compile at training scale

| Config | Wall / cell | vs h=32 baseline |
|:--|--:|--:|
| h=32 baseline (no compile, original code) | 6 946 s | 1.00× |
| **h=64 + compile + CR fix + cat fix** | **724 s** | **9.6× faster** |

Compile's amortisation at full-data scale is *much* bigger than the 5-ep
micro-bench suggested (1.24×). At 469 batches/epoch × 20 epochs = 9 380
training steps per cell, the trace cost is negligible and per-step gains
compound. This makes capacity sweeps now *cheap* — h=64 × 2 datasets ×
3 seeds finished in 1.2 h instead of the 11.6 h that the original code
would have taken.

## Files / tests / runs

| | |
|:--|:--|
| Code changes this round | none (used existing `--compile`, `--hidden 64` knobs) |
| Cells | 6 / 6, 0 failures |
| Per-cell peak RSS | ~1.5 GiB (well under 7 GiB budget) |
| Per-cell GPU max | ~3 GiB observed |
| Compile re-trace per cell | once per subprocess (acceptable; amortised over 9 380 steps) |

## Provenance

- **Git SHA:** `8fd8187` (dirty).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130 (CORE drift, user-approved).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Seeds:** {0, 1, 2}; n_epochs 20; train_subset 0 (full); batch_size 128;
  --compile; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Artifacts:** `/tmp/vision_h64/results.jsonl` (6 rows), `summary.json`.

## Per-seed table

| dataset | seed | h=32 acc | h=64 acc | Δ |
|:--|--:|--:|--:|--:|
| mnist | 0 | 0.9419 | 0.9586 | +0.0167 |
| mnist | 1 | 0.9441 | 0.9608 | +0.0167 |
| mnist | 2 | 0.9419 | 0.9590 | +0.0171 |
| fashion | 0 | 0.8426 | 0.8554 | +0.0128 |
| fashion | 1 | 0.8340 | 0.8492 | +0.0152 |
| fashion | 2 | 0.8341 | 0.8572 | +0.0231 |

## Follow-ups (ordered by expected value)

1. **h=128 sweep** — extends the capacity curve one more point. ~3-5 h
   estimated. Predictions: closes another 15-25 % of the residual gap, or
   saturates. h=128 has ~108 k params (2.6× CNN), so any further closure
   would be on per-parameter efficiency parity with CNN. May OOM on the
   7.6 GiB card — would need `torch.utils.checkpoint`.
2. **Within-RF spatial filter** (the structural hypothesis the remaining
   gap most likely rests on): add learnable position-within-RF weights
   shared across RF positions. CNN-equivalent. Substantive operator
   redesign — would need a plan, parity testing for the new path, and a
   3-seed run. Likely a half-day to a day of work.
3. Combine: width sweep (h=32, 64, 128) + within-RF filter as the next
   coherent investigation.

The current state of the picture: HSiKAN-vision is genuinely competitive
on small-image classification (h=64 closes to within 3-5 pp of CNN),
explained by a combination of capacity and operator structure. The
operator's per-parameter efficiency is *better* than CNN on MNIST
(0.9595 at 32k params vs CNN's 0.9874 at 42k → HSiKAN is 97 % of CNN's
accuracy at 76 % of the params).
