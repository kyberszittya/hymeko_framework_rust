# HSiKAN on basic ML benchmarks (tabular regression / classification / clustering) — 2026-05-09

User's curiosity question: *what happens when you apply HSiKAN to standard, low-dimensional tabular tasks like Iris, Boston housing, MNIST clustering?*  HSiKAN is architecturally a **signed-graph** model — to use it on tabular data, you have to construct a graph first.  This plan tests both whether the architecture generalises *at all* to non-graph data, and whether the signed-cycle bias adds anything over standard tabular baselines (logistic regression, random forest, MLP) when the graph is constructed correctly.

The novelty isn't beating XGBoost on Iris — it's whether the same architecture that hits SOTA on Slashdot also produces *interpretable* representations on tabular problems via the same α_κ routing readout.

## Goal

Apply HSiKAN to four canonical tabular tasks and answer:
1. **Does it train at all?**  Catmull-Rom splines + σ-masked aggregation + α-mixer over a tabular-derived graph — does the loss decrease?
2. **Does it match basic baselines?**  Logistic regression / RF / MLP at matched parameter budget.
3. **Does the α_κ routing reveal anything?**  Per-task routing pattern across cycle / walk / direct-edge slots.
4. **Does the signed structure help?**  Compared against an unsigned variant (all signs = +1).

## Tabular-to-signed-graph construction

The fundamental design question: how do you turn $n$ data points each described by a feature vector $x_i \in \mathbb{R}^d$ into a signed graph?

Three candidate protocols, in order of "expected to actually work":

### Protocol P1 — k-NN graph + class-agreement signs (supervised)

For each pair $(i, j)$:
- $i \sim j$ (edge exists) iff $j$ is among the $k$ nearest neighbours of $i$ by Euclidean distance in standardised feature space ($k$ ∈ {3, 5, 10})
- $\mathrm{sign}(i, j) = +1$ if $y_i = y_j$ (class agreement), $-1$ otherwise

Cycles are then closed paths through the k-NN structure; balance flag (Cartwright–Harary) tells whether the cycle stays within one class or crosses class boundaries.  HSiKAN's natural prediction task: per-edge sign prediction → equivalent to "do these neighbours share a class?"

**Strengths**: directly leverages the signed-cycle bias.
**Weaknesses**: requires labels for graph construction → only works in supervised settings.  Test labels leak into edge construction unless we carefully exclude the test edges.

### Protocol P2 — k-NN graph + correlation-sign signs (unsupervised)

For each pair $(i, j)$:
- $i \sim j$ (edge) iff k-NN as above
- $\mathrm{sign}(i, j) = \mathrm{sgn}(\mathrm{cosine}(x_i - \mu, x_j - \mu))$ — i.e., sign of the centred-feature inner product

**Strengths**: unsupervised; no label leakage.
**Weaknesses**: signs may not carry the structural-balance signal HSiKAN was designed for.  May reduce to "HSiKAN on an unsigned k-NN graph" in practice.

### Protocol P3 — bipartite (sample × feature) signed by feature value sign

For each pair $(i, f)$ where $f$ is a feature:
- $i \sim f$ (edge) iff $|x_{i,f}| > \tau$ (above-threshold feature value)
- $\mathrm{sign}(i, f) = \mathrm{sgn}(x_{i,f} - \mu_f)$ — above (+) or below (−) feature mean

**Strengths**: feature-attribute structure exposed directly; no k-NN choice.
**Weaknesses**: bipartite graphs have no closed cycles of length $k = 3$; limits HSiKAN to $k \geq 4$ even-arity slots.

## Datasets and tasks

| dataset | task | $n$ samples | $d$ features | classes | how |
|---|---|---|---|---|---|
| Iris | 3-class classification | 150 | 4 | 3 | k-NN (k=5) + class-sign |
| Wine | 3-class classification | 178 | 13 | 3 | same |
| Breast Cancer (Wisconsin) | 2-class classification | 569 | 30 | 2 | same |
| Boston / Ames housing | regression | 506 / ~3000 | 13 / ~80 | — | k-NN + correlation-sign; per-vertex regression |
| MNIST (subset) | clustering / classification | 1000-5000 | 784 (or PCA-32) | 10 | k-NN + sign by image similarity; cluster on encoder outputs |
| Wine Quality | regression (1-10 score) | 4898 | 11 | — | as housing |

All are standard sklearn / UCI datasets; no new infrastructure for loading.

## Experiments

### E1. Iris — does it train at all?

- Construct k-NN graph (k=5) over standardised features
- Class-agreement signs (P1)
- Build per-arity tuples for $\mathcal{K} = \{c_3, c_4\}$
- HSiKAN with h=8, 80 epochs, BCE on sign prediction
- 5-fold CV: train edges = within-fold, test edges = between-fold (avoid label leakage in graph construction)

**Acceptance**: 5-fold mean AUC ≥ 0.90 (Iris is easy; this is a sanity check that HSiKAN trains at all on tabular).

### E2. Iris — node classification

- Same k-NN graph
- Replace per-edge head with per-vertex k-class softmax (ce loss)
- Predict held-out 20% test samples' class labels
- Compare against: logistic regression, RF, MLP at matched param count

**Acceptance**: node-classification accuracy ≥ MLP baseline.  Win condition: ≥ RF baseline.  Crushing condition: outperforms by reading α_κ routing.

### E3. Wine + Breast Cancer — generalisation of the protocol

- Same as E2 with different feature dimensions (13, 30)
- Test whether the routing pattern transfers: α_κ on Iris vs Wine

**Acceptance**: classification accuracy ≥ MLP.  Look for: does the routing α concentrate on the same arity slots across the three datasets?

### E4. Boston / Ames housing — regression

- k-NN (k=5) graph + correlation-sign (P2) on standardised features
- Per-vertex regression: encoder produces vertex embeddings → MLP → scalar
- Train on 80% / test on 20% with 5-fold
- Compare: linear regression, RF regressor, gradient-boosted trees, MLP

**Acceptance**: RMSE ≤ MLP at iso-param.

### E5. MNIST — clustering via cycle-contrastive

- k-NN graph (k=10) over PCA-32 embeddings of MNIST (1000 sample subset)
- Sign by feature-correlation (P2)
- Train HSiKAN encoder via cycle-contrastive: positive pairs = cycles incident to same vertex, negative = random cycles
- Cluster the resulting per-vertex embeddings via K-means (k=10)
- Evaluate: ARI (adjusted Rand index) vs ground-truth digit labels

**Acceptance**: ARI ≥ K-means on raw PCA-32 baseline.  Crushing condition: ARI ≥ deep-clustering SOTA on MNIST.

### E6. Sign-ablation control across all of (E1–E5)

Re-run with all signs forced to $+1$ ("unsigned HSiKAN").  Measures whether the *signed* structure adds anything specifically.

**Acceptance**: at least one task shows signed > unsigned by > 1σ at 5-fold CV.  If unsigned matches signed everywhere, HSiKAN's tabular use is just "KAN on a k-NN graph" and the signed-cycle bias is dormant.

## Implementation notes

- New file `signedkan_wip/src/tabular_signed_graph.py` (~150 LOC): protocols P1, P2, P3; `build_signed_graph_from_tabular(X, y=None, k=5, protocol="p1")` returns a `SignedGraph` consumable by the existing pipeline.
- New cell in `run_final_cell.py`: `cell_tabular(dataset_name, task_kind, ...)` that calls into the tabular constructor and routes to BCE / CE / MSE per task.
- New runner `signedkan_wip/src/run_tabular_bench.py` (~80 LOC) that loads sklearn datasets, applies the protocol, calls `cell_tabular`, compares to baselines, dumps to JSONL.
- Total: ~300 LOC new code, sklearn already available.

## Acceptance for the plan as a whole

- E1 trains end-to-end on Iris: minimum bar.
- E2 reaches MLP-comparable accuracy on Iris/Wine/Breast Cancer: HSiKAN generalises.
- E5 ARI ≥ K-means on MNIST: unsupervised use case validated.
- E6 shows at least one regime where signs help: confirms signed-cycle bias has tabular use.

If 0/4 of these succeed, HSiKAN is a graph-specific architecture and tabular use is forced; document as a negative result.

## Risk register

| risk | probability | mitigation |
|---|---|---|
| k-NN graph is too sparse for cycle enumeration | medium | bump k to 10 or 15; use complete-graph sign-thresholded variant |
| Standard baselines (RF, GBM) crush HSiKAN at iso-param | high | the win condition was never to beat them; the framing is "same architecture transfers" |
| Sign protocol P1 leaks labels into the test set | high | careful train-only k-NN graph; held-out edges constructed from (train, test) pairs but signs unseen |
| Encoder is overparameterised for $n=150$ Iris | medium | reduce h to 4; can drop to <100 params total |
| MNIST clustering gives 10-cluster embedding that doesn't match digits | medium | report ARI honestly; this is a hard task |

## Order of operations

1. Implement `tabular_signed_graph.py` with protocols P1, P2 — half day
2. Implement `cell_tabular` in `run_final_cell.py` for classification + regression — half day
3. Iris E1 + E2 (sanity) — same day, ~1 hr GPU
4. Wine + Breast Cancer (E3) — same day
5. Boston / Ames (E4) — half day
6. MNIST clustering (E5) — 1 day, includes contrastive training loop
7. Sign-ablation (E6) — 1 day, just rerun with `--no-sign`

Total: ~4-5 days for the full sweep including write-up.

## What this plan does NOT do

- Doesn't beat XGBoost / LightGBM on tabular benchmarks.  Those are deeply specialised; HSiKAN's win condition is architectural transferability, not per-task SOTA.
- Doesn't propose using HSiKAN in production for tabular pipelines.  This is a research-curiosity plan testing the architecture's expressive class.
- Doesn't replace the SMC / journal paper threads.  This is a side-experiment that could become a workshop paper or a §V Future Work bullet.

## Connection to the structural-Kolmogorov-Arnold framing

Tabular data is the *easiest* test of the structural-KA hypothesis.  Classical KART operates on $[0,1]^n$ (tabular!).  If the structural-KA analogue is real — i.e., $f(x) = \sum_\kappa \alpha_\kappa \Phi_\kappa(x)$ where $\Phi_\kappa$ are signed-graph operators on the k-NN graph of $x$ — then on Iris-class tasks where classical KART is known to be sufficient, HSiKAN should match the classical KAN baseline.  E1 + E2 are the empirical sanity check for the structural-KA paper (`docs/plans_structural_ka_theorem_2026_05_09.md`).

If HSiKAN matches a classical KAN on Iris, that's an *empirical anchor* for the structural-KA theory: the graph-theoretic construction recovers the classical-KART expressive class on tabular data.
