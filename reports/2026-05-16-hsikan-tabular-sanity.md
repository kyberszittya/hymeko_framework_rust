# HSiKAN tabular sanity — does the signed-cycle bias generalise off-graph?

**Date:** 2026-05-16
**Verdict:** **Negative on the only honest test** (tabular regression: HSiKAN RMSE 69.9 vs LinearRegression 54.3 on Diabetes). Tabular classification numbers look near-perfect but the protocol leaks labels into the edge signs; the unsupervised protocol (P2) produces single-sign graphs and the AUC is undefined. **Consistent with the 2026-05-14 vision-corner negative**: σ-cycle inductive bias works where data is *natively* signed; not where signs are derived from continuous features or labels.

## 1. Why this test exists

User ask (2026-05-16): "go with Gömb and HSIKAN on other graph datasets as well and a sanity check on simple datasets (regression classification) to see how it can generalize not only on graph data but other data as well."

The HSiKAN signed-cycle architecture has produced strong results on
Bitcoin Alpha / OTC / Slashdot / Epinions — datasets where the
edge sign is a *real signal* (trust/distrust votes between users).
The 2026-05-14 vision-corner negative bounded this: σ-products
work where data is *natively* signed; vision features (continuous
positions) gave a 5-seed mAP 0.20 — three-and-a-half times worse
than the equally-simple non-signed baseline.

**Tabular regression and classification are the canonical
"non-graph, non-signed" tests.** sklearn's Iris/Wine/Breast Cancer
(classification) and Diabetes (regression) are 4-30 dimensional
feature vectors; any signed-graph structure is *constructed*, not
intrinsic. The right question: does HSiKAN's σ-cycle bias add
value when applied to k-NN-derived graphs on tabular features?

The tabular harness was already in tree
([`signedkan_wip/src/run_tabular_smoke.py`](../signedkan_wip/src/run_tabular_smoke.py),
[`signedkan_wip/src/run_tabular_regression.py`](../signedkan_wip/src/run_tabular_regression.py))
from a 2026-05-09 plan
(`docs/plans_hsikan_tabular_benchmarks_2026_05_09.md`) but the
import paths were stale after the 2026-05-11 `mixed_arity_signedkan`
refactor. Fixed both imports as part of this report; no other
source changes.

## 2. Files touched

| File | Change |
|------|--------|
| [`signedkan_wip/src/run_tabular_smoke.py`](../signedkan_wip/src/run_tabular_smoke.py) | Stale import fix: `MultiLayerSignedKANConfig` moved to `.signedkan` in the 2026-05-11 refactor; the tabular harness still imported from `.mixed_arity_signedkan`. Net change: one import block. |
| [`signedkan_wip/src/run_tabular_regression.py`](../signedkan_wip/src/run_tabular_regression.py) | Same fix. |

**CORE.YAML items touched:** none.

## 3. Classification results

Protocol: P1 (class-agreement edge sign — `+1` if both endpoints
share a class label, `-1` otherwise) over a k-NN=5 graph. Train
the SignedKAN encoder + per-edge MLP classifier on 80% of edges,
test on 20%. 1-2k parameters total.

| Dataset | n | features | n_edges | pos_frac | AUC | F1-macro | params | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Iris          | 150 |  4 |  493 | 0.93 | **1.000** | 1.000 | 1420 | 3.2 s |
| Wine          | 178 | 13 |  634 | (similar) | **0.979** | 0.940 | 1644 | 4.7 s |
| Breast Cancer | 569 | 30 | 2168 | (similar) | **0.987** | 0.900 | 4772 | 4.8 s |

**Looks great** — but **the P1 protocol leaks labels into the
edge signs.** The task "predict edge sign" becomes "predict whether
two nodes have the same class", which a 1k-parameter model can
solve on Iris / Wine / Breast Cancer by reading the *graph
structure itself* (which is built from class labels). The high
AUCs *do not* show that HSiKAN's σ-cycle bias generalises; they
show the task is trivial once the graph is constructed.

### 3.1 P2 (unsupervised) — controls for label leakage

Protocol P2 uses *feature-correlation sign* instead of
class-agreement. No label leakage in graph construction. The
task is then: "predict the correlation sign between two nodes'
feature vectors". This is a fair test of HSiKAN's ability to
recover non-trivial sign structure.

| Dataset       | AUC  | F1-macro | Note |
|---------------|-----:|---------:|------|
| Iris          | NaN  | 1.000    | test set is single-sign (all +1) — no AUC defined |
| Wine          | NaN  | 1.000    | same |
| Breast Cancer | NaN  | 1.000    | same |

P2 produces near-100%-positive edges on these datasets (Euclidean
k-NN by similar features → features positively correlated → all
edges positive). The "AUC = NaN" rows are the signature of a
**degenerate test**, not a 100% model. The unsupervised protocol
*does not produce sign diversity* on these datasets, so HSiKAN's
σ-cycle aggregator has no signal to learn from.

**Net classification conclusion:** the available protocols are
either label-leaky (P1) or sign-degenerate (P2). No meaningful
classification claim can be made from this panel.

## 4. Regression result (the honest test)

Diabetes dataset (442 samples × 10 features), 5-fold cross-
validation, 200 epochs, k-NN=5, P2 (correlation-sign)
protocol. Per-vertex regression head trained on top of HSiKAN
node embeddings. Baselines from sklearn at default settings.

| Method | RMSE | params (approx) |
|---|---:|---:|
| LinearRegression | **54.3** | 11 |
| GradientBoostingRegressor | 57.0 | (boosted trees, default 100) |
| RandomForestRegressor | 58.0 | (100 trees) |
| **HSiKAN** | **69.9 ± 5.2** | 3 836 |
| MLPRegressor (sklearn default) | 70.9 | (varies) |

**HSiKAN's RMSE 69.9 is 29 % worse than LinearRegression's 54.3**
on a dataset that a 11-parameter linear model handles cleanly.
HSiKAN is tied with sklearn's stock MLP, but the MLP is the
weakest baseline; this is not a strong tie.

The y-range of Diabetes is [25, 346]; RMSE 69.9 corresponds to
R² ≈ 0.149. A 14.9% R² is barely above zero. **HSiKAN is not
extracting tabular regression signal at this protocol.**

## 5. Interpretation

The σ-cycle inductive bias requires *real signed structure* in
the data — the way Bitcoin Alpha's trust votes or Wikipedia
editor relations are natively signed. When the signs are
*constructed* from continuous features (P2 correlation) or from
labels (P1 class agreement), one of two failure modes occurs:

1. **The signing protocol is trivial / leaks labels** (P1
   classification): the task degenerates to "recover the protocol
   used to build the graph", which any model can do.
2. **The signing protocol is degenerate** (P2 correlation): the
   resulting graph has nearly-uniform sign, no signal for the
   σ-cycle aggregator to capture.

For tabular regression specifically (the only fully-honest test in
this panel), HSiKAN's signed-cycle bias adds no value over a
linear model. The expected mechanism — σ-product over k-cycles
encodes "structural multiplicative resonance" — has nothing to
encode when there is no real multiplicative sign structure.

**This is a confirming negative** for the bounded-domain framing
of the σ-cycle inductive bias:

* `hymeyolo_kcycle_negative_2026_05_14` (vision corner detection):
  σ-cycle 0.20 mAP_50 vs boxes+circles baseline 0.72 — same
  no-signal failure mode, derived signs from continuous corner
  positions.
* This report (tabular regression): HSiKAN 69.9 vs LinearRegression
  54.3 — same failure mode, derived signs from feature
  correlations.
* HSiKAN at-home (signed-graph link prediction): 0.99+ AUROC on
  Bitcoin Alpha/OTC at protocol; 0.91 on Wikipedia editor signed
  graphs (SOTA_RESULTS §7.5, surfaced this morning). Native
  signed data, where the bias is fit-for-purpose.

The negative results are not failures of the implementation; they
are the **correct empirical bound** on where the architecture
applies. The next steps for HSiKAN should be:

* **Stay within natively-signed-graph domains** for headline
  benchmarks (Bitcoin, Wikipedia, Slashdot, Epinions).
* **Use the framework's other primitives** (Forman κ, Hodge
  spectrum, AdaptiveQuadtree) for continuous-signal domains
  *without* the σ-cycle expectation. The GömbSoma vision and
  cortical-benchmark paths fit this constraint.
* **Avoid claiming generalisation** on tabular benchmarks; the
  data shows it doesn't.

## 6. Compute

* Iris/Wine/Breast Cancer (P1, P2): each ~5 s on CPU (miniconda3,
  torch 2.11).
* Diabetes regression 5-fold: 26 s on CPU (HSiKAN trains
  5 times for 200 epochs; sklearn baselines are sub-second).
* No GPU contention; ran in parallel with the Stage A-2
  cosine-LR smoke on a separate process.

## 7. Acceptance

- [x] Stale import fixed in both tabular scripts.
- [x] HSiKAN runs on Iris / Wine / Breast Cancer (classification)
      and Diabetes (regression).
- [x] P1 classification AUCs documented + flagged as label-leaky.
- [x] P2 unsupervised classification documented + flagged as
      sign-degenerate.
- [x] Diabetes regression numbers compared head-to-head with
      sklearn baselines.
- [x] **Honest negative conclusion** drawn without padding.
- [x] Linked to existing vision-corner negative; consistent
      bounded-domain framing.

## 8. Follow-ups (low priority)

* If a future application motivates tabular HSiKAN seriously, a
  *third* signing protocol (e.g., a feature-direction sign in a
  PCA-rotated basis) might break the P2 degeneracy. Not currently
  motivated.
* Run HSiKAN against `MLPRegressor` at *iso-parameter* count on
  Diabetes — current MLP defaults are weakly-tuned. Would tighten
  the negative claim by removing the "MLP is also weak" caveat.

---

*End of HSiKAN tabular sanity report. The σ-cycle bias does not
generalise off-graph at this protocol. The HSiKAN family stays in
its natively-signed-graph lane.*
