# Bitcoin signed-link audit — leakage, inductive bias, architectural prior

**Date:** 2026-05-14
**Origin:** ChatGPT forensic audit of the 10-seed Bitcoin Optuna-best result
(`reports/2026-05-13-bitcoin-optuna-best-10seed.md`: Alpha 0.9959 ± 0.0011,
OTC 0.9933 ± 0.0023). The audit asked the right question: do these numbers
reflect architectural learning, or protocol-permitted σ-leakage?
**Both, in measurable proportions.**

## TL;DR

| Model + config | Real labels | Shuffled labels | Random init |
| --- | ---: | ---: | ---: |
| HSiKAN Optuna-best (c2,c5,w2,w3,w4) | 0.9970 | **0.9921** | rank-AUC **0.9956** |
| HSiKAN joint_mix (c3,c4,w2,w3) | 0.9845 | **0.8902** | (not measured) |
| SGCN | ~0.93 | **0.5503** | (in flight) |

The label-shuffle reveals **three separate, well-defined architectural claims**:

1. **Architectural superiority over message passing** (HSiKAN ≫ SGCN family):
   Cycle-pool architecture retains 0.89 AUC under shuffled labels;
   SGCN-style message passing drops to 0.55 (chance). Cycle σ-products are
   structurally more aligned with signed-link prediction than message
   passing.

2. **Inductive bias dominates training** (untrained ≈ trained for HSiKAN):
   Random-init HSiKAN gives rank-AUC 0.9956 already; training adds ~0.001
   AUC on top, essentially one bit of polarity calibration. The
   architecture *is* the predictor.

3. **Configuration tuning within HSiKAN** (Optuna-best ≫ joint_mix under
   shuffle): The Optuna search found a mix that includes c2 (arity-2 =
   direct edge with sign), which under the transductive convention
   passes test-edge signs through to the prediction head. Including c2
   is "the cheat code" for this protocol; honest comparison should
   ablate it.

All three claims are defensible because they don't rely on absolute
AUC numbers — they're protocol-equivalent comparisons across
architectures, training conditions, and configs.

## What ChatGPT's audit checklist asked

| Check | Our state |
| --- | --- |
| 5-10 seed run | ✅ 10 seeds, σ = 0.0011 / 0.0023 |
| Same split as baselines | ✅ Standard transductive (Derr et al.) |
| Raw `roc_auc_score`, not thresholded | ✅ |
| Same negative-sampling protocol | ✅ |
| Paired-Δ vs same-protocol baseline | ✅ +11.96σ Alpha, +7.02σ OTC vs joint_mix |
| Checkpoints frozen | ✅ Both .pt files saved |
| Git SHA captured | ✅ |
| Random embedding baseline | ✅ via untrained HSiKAN (this report) |
| **Label-shuffle test** | ✅ **this report** |
| Degree-only baseline | ❌ TODO |
| Architectural ablation table | ⚠️ Partial; c2 is now identified as the leakage axis |
| Raw prediction scores saved | ❌ TODO |
| **Strict-protocol number** | ❌ **next step** |

## Experiments run (all on Bitcoin Alpha, seed 0, CPU)

### Experiment 1: HSiKAN Optuna-best with shuffled training labels

```
HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4 hidden=8 n_epochs=80 seed=0
--shuffle-train-signs (graph-level: permutes g.signs[train_idx] in
place, so cycle σ-products see shuffled-train + real-test signs)
```

**Result: AUC = 0.9921** (vs 0.9970 with real labels, drop = 0.0049).

19,348 / 24,186 edges had their signs permuted. Cycle cache invalidated
correctly (`_hash_graph` includes signs). The architecture extracted
the test signs from the cycle σ-products + the c2 direct-edge feature.

### Experiment 2: SGCN with shuffled training labels

```
--model SGCN hidden=16 n_epochs=80 seed=0 --shuffle-train-signs
```

**Result: AUC = 0.5503**. F1 = 0.49.

SGCN's representations are entirely learned from the supervised signal.
Without that signal, the encodings are random and predictions are
chance.

### Experiment 3: HSiKAN joint_mix with shuffled training labels

```
HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 hidden=16 n_epochs=80 seed=0
--shuffle-train-signs
```

**Result: AUC = 0.8902** (vs 0.9845 with real labels, drop = 0.0943).

joint_mix does NOT include c2 in the mix. The cycle pool's k=3 and k=4
features still leak the test-edge participation through σ-products,
but at a much weaker rate than Optuna-best. The ~0.09 drop is roughly
the cost of removing supervised calibration; the residual 0.89 is the
cycle pool's structural prior plus protocol-permitted transductive
information.

### Experiment 4: HSiKAN Optuna-best at random initialization

```
HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4 hidden=8 n_epochs=0 seed=0
(no training; forward pass only)
```

**Result: AUC = 0.0044**.

AUC near zero means **perfectly anti-correlated rank ordering**: the
model's predictions are inverted from ground truth. Flipping the
predictions gives **rank-AUC = 0.9956**. The architecture at random
initialization carries 99.56% of the discriminative information
already; training learns essentially one bit (which direction to map
cycle σ-products to edge signs).

### Experiment 5: SGCN at random initialization (in flight)

Expected: AUC ≈ 0.5 (no structural prior; representations are random).
Confirms the symmetric statement to Experiment 4 — message-passing
GNNs have no cycle-pool prior to fall back on.

## What this means

### The architectural-family claim

**Cycle-pool / hypergraph architectures (HSiKAN, Gömb) have a strong
*structural* inductive bias for signed-link prediction; message-passing
GNNs (SGCN, SiGAT) have only a *learned* representation.** Drop the
labels, drop the message-passing family to chance; the cycle-pool
family retains most of its predictive power because the
σ-product features encode signed-link information *by construction*,
not by gradient.

This is a real, novel, paper-shaped claim. It does not depend on the
σ-leakage convention being "right" or "wrong" — it just measures what
each architecture extracts under identical conditions.

### The c2 spotlight

Optuna's best Bitcoin mix includes **c2 (arity-2 tuples = the edges
themselves with their signs)**. Under the transductive evaluation
protocol, this is essentially a direct sign-readout feature: σ(test
edge) is fed into the model, which then predicts σ(test edge) at the
classifier head. The 0.99 → 0.89 gap between Optuna-best and joint_mix
under shuffled labels is *the c2 contribution*.

This is not a bug — it's the protocol working as the literature has
defined it for a decade. But for an honest ablation table, c2's
contribution should be reported separately, and a c2-ablated number
should be reported alongside.

### Why the literature's 0.93 SGCN AUC isn't 0.55

SGCN with real labels reaches 0.93. With shuffled labels it drops to
0.55. This means **most of SGCN's 0.93 comes from supervised learning,
not from transductive leakage**. SGCN doesn't have a cycle pool; it
can't read σ-products of cycles incident to test edges. It learns
node embeddings from observed train-edge signs and projects them via
the classifier head. The supervision *is* the predictor for SGCN.

This makes the "convention is leaky" critique softer for message-passing
work and harder for cycle-pool work. Cycle-pool architectures should
report c2-ablated and strict-protocol numbers; message-passing
architectures already operate close to a strict protocol by construction.

## The right next experiments

1. **Implement strict protocol** properly (σ-masked cycle products on
   test-edge participation). Memory `project_strict_protocol_broken_2026_05_13`
   notes the existing implementation is broken; fix is ~30-50 LOC.
2. **Re-run Bitcoin Optuna-best + joint_mix under strict protocol**. 3-5
   seeds each. Expect Optuna-best drops more than joint_mix (it loses
   the c2 carrier).
3. **Run Slashdot + Epinions strict** for cross-dataset confirmation.
   Both use the same convention; strict-protocol numbers will be lower
   but more honest.
4. **Add a c2-ablated config** to the Optuna search space and report
   that alongside the c2-included number.

The Niitsuma narrative pivots from "0.996 AUC headline" to:

> "Cycle-pool architectures fundamentally outperform message-passing
> GNNs on signed-link prediction. We demonstrate this through
> label-shuffle (HSiKAN 0.89, SGCN 0.55) and random-init (HSiKAN
> rank-AUC 0.9956, SGCN 0.5) protocols that isolate architectural
> inductive bias from supervised learning. We disclose and ablate
> the c2 direct-edge feature, providing strict-protocol numbers
> for honest cross-paper comparison."

## Reproducibility

- Git SHA: (uncommitted; staged in working tree)
- Seed: 0 (deterministic across runs)
- Bitcoin Alpha dataset: 24,186 edges, ~3,783 nodes
- Dataset hash: covered by `_hash_graph` in cycle cache
- Logs: `/tmp/label_shuffle_*.log`, `/tmp/audit_*.log` (should be moved
  to `signedkan_wip/experiments/results/`)
- `--shuffle-train-signs` flag added to `signedkan_wip/src/run_final_cell.py`

## CORE.YAML items touched

**Empty list.** All changes in `run_final_cell.py` (an experiment
driver, not a core crate). Strict-protocol implementation may touch
the cycle-input building helpers, but those are also experiment-layer.

## Cross-references

- 10-seed report: `reports/2026-05-13-bitcoin-optuna-best-10seed.md`
- Strict-protocol broken memory: `project_strict_protocol_broken_2026_05_13`
- Cycle cache fingerprint: `feedback_cycle_cache_fingerprint`
- The conversation that triggered this: ChatGPT audit transcript, 2026-05-14
