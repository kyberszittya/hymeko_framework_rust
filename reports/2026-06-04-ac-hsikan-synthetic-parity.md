# AC-HSiKAN vs Transformer — two-task smoke

Date: 2026-06-04
Plan: `docs/plans/2026-06-04-ac-hsikan/plan.{md,tex,pdf,tikz,mmd}`
Code: `hymeko_neuro/models/ac_hsikan/` (~440 LOC, 11/11 pytest GREEN)
Smoke harnesses:
  - `hymeko_neuro/experiments/ac_hsikan_synthetic_smoke.py` (T1)
  - `hymeko_neuro/experiments/ac_hsikan_imdb_smoke.py` (T2)

## TL;DR

| task | AC-HSiKAN | Transformer | Δ | verdict |
|---|---|---|---|---|
| Signed parity L=4 | **1.000 ± .000** | 0.732 ± .236 | **+0.268** | AC wins clean |
| Signed parity L=8 | **0.849 ± .109** | 0.512 ± .005 | **+0.337** | AC wins clean |
| IMDB sentiment (smoke) | 0.571 ± .021 | **0.721 ± .011** | **−0.151** | Transformer wins clean |

**Claim is two-sided:**
- AC-HSiKAN is competitive *and* dominant where the task structure is
  multiplicative-sign (parity, balance, sign-propagation).
- Transformer wins where the task is distributional-semantic
  (general sentiment, topic). The signed-cycle pool is the WRONG
  inductive bias there.
- **NOT a general transformer replacement.** Niche-task winner.

This is the honest paper-claim shape. See "What this is not" below.


## Headline

| L | model | val_acc (3-seed) | n_params | wall/seed |
|---|---|---|---|---|
| 4 | AC-HSiKAN | **1.0000 ± 0.0000** | **3,404** | 5.6 s |
| 4 | Transformer (iso-arch) | 0.732 ± 0.236 | 6,626 | 5.6 s |
| | **Δ** | **+0.268** | **0.51× param** | iso |
| 8 | AC-HSiKAN | **0.849 ± 0.109** | **3,404** | 18.2 s |
| 8 | Transformer (iso-arch) | 0.512 ± 0.005 | 6,626 | 16.8 s |
| | **Δ** | **+0.337** | **0.51× param** | iso |

**Both L cases PASS the plan's `≥10%` margin gate. L=8 is the harder
case (k_max=5 < L=8 forces compositional learning); AC-HSiKAN still
wins by +34% at half the parameters.**

## Task

Signed-parity classification:
- Tokens ∈ {0, 1} mapped to {−1, +1}.
- Label = `1` iff `prod(symbols) > 0` (i.e. even number of −1 symbols).
- Length L ∈ {4, 8}; CPU; 3 seeds; AdamW lr=3e-3; 30 epochs.

The task is fundamentally multiplicative-sign-aware. Transformer's
softmax attention has no inductive bias for sign product; AC-HSiKAN's
signed-cycle pool literally computes `prod(signs)` over each cycle.

## Setup

- AC-HSiKAN: `d_model=16, n_positions=L, arities=(2,3,4,5),
  top_k_per_position=min(8, L-1), n_layers=2, dropout=0`.
- Transformer: `IMDBTransformerBaseline(vocab_size=2, d_model=16,
  n_heads=2, dim_ff=64, n_layers=2)` (existing baseline at
  `hymeko_neuro/experiments/sequence/iso_param_transformer.py`).
- Both seeded identically per-seed (torch + data); paired.

## Per-seed breakdown

### L = 4 (full sequence fits in k=4 arity)

| seed | AC-HSiKAN | Transformer |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 1 | 1.000 | 0.556 (random) |
| 2 | 1.000 | 0.639 |

Transformer seed 0 fluked into learning it; seeds 1+2 stuck at
random. AC-HSiKAN deterministic 1.000 across seeds.

### L = 8 (k_max=5 < L; compositional)

| seed | AC-HSiKAN | Transformer |
|---|---|---|
| 0 | 0.965 | 0.515 |
| 1 | 0.836 | 0.515 |
| 2 | 0.748 | 0.506 |

Transformer all-3 random (0.51 ± 0.005). AC-HSiKAN well above
random in all 3, σ=0.109 reflects compositional difficulty but the
mean (0.849) clearly demonstrates learning.

## αₖ regime compass

Read at end of training (per-layer softmax over arities (2,3,4,5)):

| L | seed | layer 0 αₖ | layer 1 αₖ |
|---|---|---|---|
| 4 | 0 | [.23, .23, .29, .25] | [.20, .28, .24, .27] |
| 4 | 1 | [.25, .23, .26, .26] | [.29, .24, .22, .24] |
| 4 | 2 | [.25, .25, .28, .22] | [.17, .30, .29, .25] |
| 8 | 0 | [.23, .29, .22, .26] | [.17, .22, .34, .27] |
| 8 | 1 | [.19, .30, .17, .34] | [.33, .22, .17, .28] |
| 8 | 2 | [.18, .27, .20, .34] | [.30, .15, .16, .39] |

Notes:
- L=4: αₖ stays near-uniform — parity at L=4 can be computed via
  multiple arities (k=4 directly, or composition of k=2 pairs), so
  no strong preference emerges.
- L=8: slight tilt toward k=5 (the max arity, closest to L=8) in
  layer 0 — consistent with the hypothesis that the model wants
  the longest-range arity available.
- The regime compass is **less polarized than on signed-graph tasks**
  (e.g. Bitcoin Alpha shows clear k=4/5 dominance in the existing
  HSiKAN runs). This is a real finding: signed-parity at small L is
  arity-ambiguous; the strong-arity-preference story requires
  natural-data signal structure to surface.

## What this commit ships

- New package `hymeko_neuro/models/ac_hsikan/`: `config.py`, `sign_head.py`,
  `layer.py`, `model.py`, `__init__.py` (~440 LOC total).
- Smoke harness `hymeko_neuro/experiments/ac_hsikan_synthetic_smoke.py`.
- Unit tests `hymeko_neuro/tests/test_ac_hsikan.py` (11 tests, all
  green: SignHead shape/range/grad/STE; layer shape/residual/alpha/grad;
  classifier forward shape, alpha list, grad flow; iso-param vs
  IMDBTransformerBaseline within 30%).
- 4-format plan at `docs/plans/2026-06-04-ac-hsikan/plan.{md,tex,pdf,
  tikz,mmd}`.

## T2 — IMDB sentiment (CPU smoke)

| seed | AC-HSiKAN | Transformer | epoch trajectory (AC / TR) |
|---|---|---|---|
| 0 | 0.585 | 0.729 | [.49 .49 .59] / [.50 .61 .73] |
| 1 | 0.556 | 0.713 | [.54 .56 .54] / [.52 .60 .71] |
| **mean** | **0.571 ± 0.021** | **0.721 ± 0.011** | |
| n_params | 83,372 | 86,594 | iso-param |
| wall total | 26 s | 8 s | AC 3.3× slower |

**Setup:** sub-sampled IMDB (3000 train, 1000 val from the 25k+25k
split), L_max=64, vocab=5000, 3 epochs, lr=3e-3, CPU.

**Interpretation:**

- Transformer clean win (+15%), iso-param.
- AC-HSiKAN epoch-trajectory shows it is *still climbing* at
  epoch 3 (0.49 → 0.59) while the transformer is converging
  (0.50 → 0.73). With 10+ epochs on full 25k train AC-HSiKAN
  might recover some gap, but the wall-cost penalty (3.3× per
  epoch from the L²-scan bilinear SignHead) makes it
  uncompetitive even on best-case extrapolation.
- IMDB sentiment is not a multiplicative-sign task; the
  signed-cycle pool is the **wrong inductive bias** here.
  Word polarity composes distributionally, not via XOR-like
  parity.

This confirms the niche-task hypothesis: AC-HSiKAN wins on tasks
where the label is a multiplicative-sign function of features
(parity, balance theory, sign propagation through chains), loses
on general distributional-semantic tasks. The right follow-up
task is **negation detection / contradiction detection**, where
signed semantics matters and the transformer's distributional
bias is partially mis-aligned.

## What this commit does NOT do

- **Full IMDB (25k train, 10 epochs, GPU)**: deferred. The CPU
  smoke (above) is decisive enough — AC-HSiKAN loses by 15
  percentage points after 3 epochs of identical training; full
  training will not flip the verdict. The wall-cost (3.3×) makes
  AC-HSiKAN unattractive on this task regardless of final accuracy.
- **Per-seed reproducibility**: 3 seeds is smoke-grade; paper claim
  would require 5+ seeds.
- **L > 8 / longer-range**: not tested; the signed-parity task at
  L=16 may need larger top_k_per_position.
- **Real signed-semantic NLP task** (negation detection, contradiction):
  the synthetic parity is a proof of concept; the full claim
  ("HSiKAN-Sequence competes with transformers on signed-semantic
  tasks") needs a natural-data benchmark.

## Risk + caveats

1. **Synthetic task asymmetry**. AC-HSiKAN's `prod(signs)` pool
   computation is mathematically aligned with the signed-parity
   label. This is an intentional inductive-bias demo, not a fair
   general comparison. For language tasks where the label isn't
   sign-multiplicative, the result will be different.
2. **L=4 saturates fast**. 16 possible inputs; 4000 train samples =
   250× repetition. Both models can memorize; the differentiator
   is whether they generalize to the (also-from-same-distribution)
   val set. AC-HSiKAN does so deterministically; transformer
   stochastically.
3. **Sign-head is global (not multi-head)**. v1 design choice.
   Multi-head sign attention could change the per-seed variance.
4. **Iso-param matching is at the layer level**; FLOPS not matched.
   AC-HSiKAN's cycle pool is O(B·L·K·k·d), transformer attention is
   O(B·L²·d). At L=8 transformer has fewer FLOPS per token.

## Reproduction

```bash
PYTHONPATH=. python -m hymeko_neuro.experiments.ac_hsikan_synthetic_smoke \
    --seq-len 4 --n-train 4000 --n-val 1000 --n-epochs 30 \
    --batch-size 256 --seeds 0 1 2 --device cpu \
    --out /tmp/ac_parity_L4.json

PYTHONPATH=. python -m hymeko_neuro.experiments.ac_hsikan_synthetic_smoke \
    --seq-len 8 --n-train 8000 --n-val 2000 --n-epochs 30 \
    --batch-size 128 --seeds 0 1 2 --device cpu \
    --out /tmp/ac_parity_L8.json
```

Total wall: ~2 min combined on a laptop CPU.

## Next steps

1. **IMDB sentiment benchmark** once GPU frees (~30 min/model x 3
   seeds x 2 models ≈ 3 h). Acceptance: AC-HSiKAN within ±2% of
   transformer accuracy.
2. **Negation-flip subset of SST-2 / Stanford Sentiment Treebank**:
   real signed-semantic task. Should be the strongest niche-task
   demonstration.
3. **YAML config integration**: write
   `hymeko_neuro/experiments/configs/ac_hsikan_imdb.yaml` and
   `transformer_imdb_isoparam.yaml` for the runner framework
   built earlier today. Then both benchmarks become
   `python -m hymeko_neuro.experiments.run --config X.yaml`.
4. **Paper note**: this synthetic result is publishable as a
   "minimum demonstrator" figure in a follow-up paper on
   HSiKAN-Sequence. Not in scope for the Nature submission (which
   stays narrowly on signed-link prediction).
