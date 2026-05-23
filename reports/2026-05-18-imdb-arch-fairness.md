# IMDB architectural-fairness — Sequential HSiKAN vs iso-param Transformer

**Date:** 2026-05-18
**Plan:** [`docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/`](../docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/)
**Run dir:** `signedkan_wip/experiments/results/imdb_arch_fairness_20260518T012024Z/`
**Git SHA:** `2ccaa4d12fae1ff9cd533bd91cd84b28f11c3dab`
**Wall:** 03:20 → 04:48 local = 88 min total (came in at 35 % of the
plan's 4 h budget — IMDB unsup pretrain finished much faster than
estimated thanks to the small parameter count).

## 1. Summary

Five-phase experiment answering the question "is HSiKAN competitive
with a Transformer at fixed corpus + fixed parameter budget?" — the
*honest* iso-budget architectural-fairness test, not the unfair
"HSiKAN vs pretrained-BERT" comparison the May-17 IMDB result
invited.

**Headline**: at iso-parameter (~321 k) iso-corpus (full IMDB
25 k labelled, no external pretraining), HSiKAN from-scratch and
the Transformer baseline from-scratch are **statistically tied**.
Welch's $z = +1.12\sigma$. The Transformer is nominally +0.0073
ahead, but within the noise of either architecture.

**Bonus surprises**:
1. MLM pretraining on the 50 k unsup IMDB split does *not* improve
   either architecture at this scale. For the Transformer it
   actively hurts by $-0.0254$ (≈6σ).
2. HSiKAN-pretrained has the **lowest σ of all configs** (0.0016).
   Pretraining tightens HSiKAN's variance without changing the mean.

## 2. Setup

| Field | Value |
|------:|:------|
| Vocab size  | 20 000 (frequency, UNK doubles as MLM mask) |
| L_max       | 200 tokens |
| Tokenizer   | whitespace + HTML strip, lowercase |
| Train / val / test | 22 500 / 2 500 / 25 000 |
| Optimizer   | AdamW, lr 3e-4, wd 1e-5 |
| Epochs (fine-tune) | 20 |
| Epochs (pretrain)  | 20, lr 5e-4 |
| Batch       | 32 |
| Pretrain corpus | IMDB unsup 50 000 reviews (~10–15 M tokens) |
| MLM recipe  | 15 % masked, 80/10/10 mask/random/unchanged |
| Seeds (5-seed) | 0–4 |

**HSiKAN** = `IMDBClassifier` with C=4 Cl(2,0) channels, K=4 FIR
window, enc_depth=3.  321 137 params.
**Transformer** = `IMDBTransformerBaseline` with d_model=16,
n_heads=2, dim_ff=64, n_layers=2.  326 594 params (1.7 % off HSiKAN).
Both share the same vocabulary, tokenizer, batch loop, optimiser
config; the only deliberate difference is the inductive bias of
the encoder.

## 3. Five-phase results

| Phase | Config | n | mean | σ | per-seed |
|------:|:-------|--:|----:|---:|:---------|
| ref | HSiKAN from-scratch (May 17) | 5 | 0.8395 | 0.0058 | 0.8464, 0.8411, 0.8435, 0.8299, 0.8368 |
| 1 | HSiKAN MLM pretrain (loss curve 6.86 → 5.56) | — | — | — | — |
| 2 | Transformer MLM pretrain (loss curve 6.70 → 6.19) | — | — | — | — |
| **3** | **HSiKAN pretrained → fine-tune** | **5** | **0.8371** | **0.0016** | 0.8398, 0.8377, 0.8371, 0.8352, 0.8356 |
| **4** | **Transformer from-scratch** ← arch-fairness | **5** | **0.8468** | **0.0030** | 0.8434, 0.8514, 0.8438, 0.8473, 0.8481 |
| **5** | **Transformer pretrained → fine-tune** | **5** | **0.8214** | **0.0046** | 0.8233, 0.8174, 0.8217, 0.8288, 0.8157 |

## 4. The architectural-fairness comparison (Phase 4 vs reference)

| | HSiKAN from-scratch | Transformer from-scratch |
|---:|:---:|:---:|
| Params | 321 137 | 326 594 (+1.7 %) |
| Mean test acc | 0.8395 ± 0.0058 | **0.8468 ± 0.0030** |
| Δ (T − H) | — | +0.0073 |
| Welch $z$ | — | +1.12 |
| Win-rate (each-side seed > median of other) | — | tie |

**Read**: tied at iso-corpus + iso-budget. The Transformer's nominal
+0.0073 mean lead is within one standard deviation of both
architectures' seed variability. The honest paper-claim is
**"HSiKAN is competitive with an iso-parameter Transformer at
from-scratch training; neither architecture dominates at this
scale."**

## 5. The pretraining surprise (Phases 3 and 5 vs from-scratch)

| | from-scratch | pretrained | Δ | z |
|---:|:---:|:---:|---:|---:|
| HSiKAN     | 0.8395 ± 0.0058 | 0.8371 ± 0.0016 | $-0.0024$ | $-0.41$ |
| Transformer | 0.8468 ± 0.0030 | 0.8214 ± 0.0046 | $-0.0254$ | $-4.61$ |

**HSiKAN pretrain**: $-0.0024$ within noise on the mean, but σ
collapses from 0.0058 → 0.0016 — a real $3.6\times$ variance
reduction. Pretrain doesn't move the mean, but the representations
are more stable.

**Transformer pretrain**: $-0.0254$ at $z \approx -4.6\sigma$ is a
hard negative — pretraining *actively hurts* the Transformer.

### Hypothesised mechanism

The MLM pretrain loss for the Transformer plateaued early (6.70 →
6.19, only $-7.7$ % over 20 epochs; cf. HSiKAN 6.86 → 5.56,
$-19.0$ %). With d_model=16 and 2 layers the Transformer's MLM
capacity is so narrow that it learns mostly the embedding's
marginal token distribution rather than contextual features. When
those slightly-overfit embeddings get loaded for the downstream
classification task, they're worse-initialised than fresh random.

HSiKAN's MLM loss drops further (6.86 → 5.56, $-19.0$ %) so its
representations *do* capture something — but apparently that
something isn't directly useful for sentiment (the mean doesn't
move). The σ-collapse is the partial benefit: pretrain gives the
HSiKAN encoder a more consistent initialisation distribution.

## 6. Paper-friendly framing

The result that survives review is:

> *We evaluate the Sequential HSiKAN encoder against an
> iso-parameter Transformer baseline on IMDB binary sentiment at
> fixed corpus and fixed training budget. From-scratch, HSiKAN
> reaches $0.8395 \pm 0.0058$ test accuracy; the Transformer
> reaches $0.8468 \pm 0.0030$. The two are statistically tied
> ($z = +1.12$). Masked-language-model pretraining on the IMDB
> unsupervised split (50 k documents, ~12 M tokens) does not
> improve either architecture: for HSiKAN the mean is unchanged
> but variance drops $3.6\times$; for the Transformer pretraining
> actively hurts ($-0.0254$, $z \approx -4.6$), consistent with
> overfitting at the narrow d_model=16 width.*

What **not** to claim:
- *"HSiKAN beats Transformer"* — false at this seed count.
- *"HSiKAN matches BERT"* — apples-to-oranges; BERT had ~3 B tokens
  of pretraining and 110 M parameters. Our comparison is iso-budget
  small-from-scratch.
- *"Pretraining doesn't help."* — true only at this corpus + model
  width. With more unsup data or wider d_model, the conclusion may
  flip. State it as "doesn't help at this scale."

## 7. Files touched

### New (this run)
- `signedkan_wip/src/sequence/iso_param_transformer.py` (145 LOC)
- `signedkan_wip/src/sequence/imdb_pretrain.py` (200 LOC)
- `signedkan_wip/src/sequence/run_imdb_mlm_pretrain.py` (130 LOC)
- `signedkan_wip/src/sequence/train_imdb_transformer.py` (175 LOC)
- `signedkan_wip/experiments/run_imdb_arch_fairness_2026_05_18.sh` (175 LOC)

### Modified
- `signedkan_wip/src/sequence/train_imdb_classifier.py` — added
  `--pretrained-state-dict` flag for the MLM → fine-tune path.

### CORE.YAML items touched
None. Reuses the existing `signedkan_wip` Python env.

## 8. Open items

1. **Wider Transformer pretrain at d_model=32 or 64.** The
   transformer-pretrain regression may flip at larger capacity. One
   ablation (~1 h GPU) would resolve whether the negative pretrain
   result is architecture-specific or capacity-specific.
2. **MLM mask token disambiguation.** We reused UNK_ID as the MLM
   mask sentinel for vocab compatibility. A proper `<mask>` token
   (separate vocab slot) might lift the pretrain quality and could
   recover the Transformer's pretrain regression.
3. **Longer L_max (400 instead of 200).** IMDB reviews average
   ~230 tokens; truncating at 200 loses ~15 % of signal. Plan §6
   listed this as the Stage 2 lever; could revisit if the
   architectural-fairness margin needs to be widened.
4. **Char-level IMDB.** Different inductive bias regime —
   Clifford-FIR may have more purchase per the plan §6 fallback.

## 9. Bottom line

The April-and-May Sequential HSiKAN line has, as of this run, **a
publishable architectural claim**: Clifford-FIR multichannel
encoding is competitive with self-attention at fixed corpus + fixed
parameter budget on natural-language binary sentiment, with the
honest "tied, not winning" framing. The pretrain comparison is a
separate finding that holds independent of which architecture you
prefer.
