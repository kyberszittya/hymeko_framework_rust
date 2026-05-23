# HSiKAN beyond signed-link prediction — general-graph extensions (2026-05-09)

A consolidated plan for the smaller-scope extensions of HSiKAN to standard graph-learning tasks: **node classification**, **cycle-contrastive pretraining**, **self-supervised masked-cycle prediction**, and **bipartite graph matching**.  Each is a half-day to two-day experiment that reuses the existing encoder with only a head / loss change.  The umbrella claim: HSiKAN is a *task-universal* signed-graph encoder, with the same α_κ routing serving as the per-task structural readout.

These four extensions are **not** the SMC paper, **not** the journal paper.  They are workshop-grade or §V Future Work demonstrations that compound into a stronger universality argument.

## Goal

For each of four generic graph-learning tasks, demonstrate that:
1. The HSiKAN encoder trains end-to-end without architectural changes.
2. The α_κ routing pattern reveals task-specific structural primitives (cycles vs walks vs direct edges).
3. The architecture is competitive with task-specific baselines at iso-param.

The four extensions are independent; each can be written up as a workshop note or absorbed into the journal paper as an "Applications" section.

## Extension 1 — Node classification on signed networks

### Task
Classify vertices in a signed graph into communities (e.g., Slashdot user groups, Bitcoin trader cohorts).

### Datasets
- Slashdot communities (extracted via signed-graph community detection on the existing dataset)
- Cora-signed (synthetic: take the citation graph, add ±1 sign by category-agreement)
- Bitcoin Alpha trader-cohort labels (if available; otherwise unsupervised cluster labels via SBM)

### What changes from current code
- Replace per-edge head `nn.Linear(d_jk, 1)` with per-vertex head `nn.Linear(d_jk_vertex, n_classes)`
- Replace BCE with cross-entropy over per-vertex predictions
- The encoder produces per-vertex embeddings as a side-product of cycle aggregation (h_v after layers); use those directly

### Acceptance
- Multi-class accuracy ≥ MLP-on-features baseline at iso-param (~330K params)
- Win condition: ≥ Signed-GCN at same param budget on Slashdot communities
- Routing readout: α_κ on node-classification differs measurably from α_κ on edge-sign prediction → confirms task-specific routing

### Cost
~30 LOC code change + 1 day for experiments

## Extension 2 — Cycle-contrastive pretraining

### Task
Self-supervised representation learning on signed graphs without labels.  Pretrain the HSiKAN encoder to produce embeddings where cycles incident to the same vertex are close (positive pairs) and random cycles are far (negative pairs).

### Method
Standard SimCLR / InfoNCE objective on cycle embeddings $\mathbf{H}^{(\kappa)}$:

$$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(h_t, h_{t'}) / \tau)}{\sum_{t'' \in \mathcal{N}^-} \exp(\mathrm{sim}(h_t, h_{t''}) / \tau)}$$

where $t, t'$ share at least one vertex (positive pair), $t''$ is sampled from the random-cycle pool (negative).

### Datasets
Same as the SMC paper: BA, OTC, Slashdot, Epinions.

### Evaluation
- Pretrain HSiKAN on cycle-contrastive
- Linear probe: train a logistic regression on edge-sign prediction using frozen encoder features
- Compare against:
  - End-to-end supervised HSiKAN (the validated 5-seed numbers)
  - Random-init encoder + linear probe
  - DeepWalk / Node2Vec embeddings

### Acceptance
- Linear-probe AUC reaches ≥ 80% of fully-supervised HSiKAN on each dataset
- Confirms that cycle structure carries label-free signal

### Cost
~150 LOC for the contrastive training loop + 2 days for experiments

## Extension 3 — Self-supervised masked-cycle prediction

### Task
Mask one vertex (or one edge) per cycle; train the encoder to predict the masked element.  Analogous to BERT's masked-language modelling but on signed cycles.

### Method
- For each cycle in $\mathcal{T}$, randomly mask one vertex (or edge sign) with probability $p_{\rm mask}$
- Encoder sees the masked cycle, predicts the masked element via a small head
- Loss: classification (vertex ID prediction) or BCE (edge-sign prediction)

### Acceptance
- Masked-edge sign prediction reaches ≥ 0.7 accuracy (vs random 0.5)
- Self-supervised pretraining + supervised fine-tune ≥ supervised-from-scratch by ≥ 1σ on at least one dataset

### Cost
~200 LOC for the masking + prediction head + 2 days for experiments

## Extension 4 — Bipartite graph matching

### Task
Match vertices between two signed graphs $G_A, G_B$ (correspondence problem on signed graphs).  Examples: matching trader profiles between Bitcoin Alpha and Bitcoin OTC; aligning user IDs across two snapshots of Slashdot.

### Method
- Encode $G_A$ → embeddings $Z_A \in \mathbb{R}^{V_A \times d}$
- Encode $G_B$ → $Z_B \in \mathbb{R}^{V_B \times d}$
- Pairwise cost matrix $C_{ij} = \|z_{A,i} - z_{B,j}\|^2$
- Sinkhorn iterations → soft permutation matrix $P$
- Loss: BCE on ground-truth correspondence pairs (synthetic) or geodesic-distance loss

### Datasets
- Synthetic: take a single signed graph, perturb (rewire 5%, flip 1% of signs), match the original to the perturbation
- Cross-time Slashdot snapshots if available

### Acceptance
- Top-1 match accuracy > 70% on the perturbed-synthetic case
- > 50% on cross-time real-data case

### Cost
~300 LOC including Sinkhorn (standard ~40 LOC) + dual-encoder wiring + 2-3 days for experiments

## Order of operations

1. Extension 1 (node classification) — 1 day; smallest commitment, immediate paper-bullet
2. Extension 4 (bipartite matching) — 2-3 days; cleanest stand-alone story
3. Extension 2 (cycle-contrastive) — 2 days; pretraining angle
4. Extension 3 (masked-cycle) — 2 days; BERT-like SSL

Total: ~7-9 days for all four.  Each is independently publishable as a workshop note; together they form an "Applications" section in the journal paper.

## Acceptance for the plan as a whole

- ≥ 2 of the four extensions land their acceptance criteria
- The α_κ routing pattern differs measurably across tasks → universality + interpretability
- Workshop-tier writeup possible if results land strongly

## Risk register

| risk | probability | mitigation |
|---|---|---|
| Node classification: signed graph community labels aren't readily available | medium | use synthetic SBM communities or extract from Bitcoin user-cohort data |
| Cycle-contrastive: positive-pair definition (shared vertex) too lenient or too strict | medium | sweep over positive-pair definitions (shared vertex, shared cycle, shared edge) |
| Bipartite matching: synthetic perturbation is too easy | medium | benchmark against cross-time real data |
| Each extension is too small for a full paper | high | the framing is consolidated: one "HSiKAN universality" workshop / journal-section paper covering all four |

## What this plan does NOT do

- Doesn't compete with task-specific SOTA on any single benchmark.  Win condition is *competitive at iso-param + interpretable routing*.
- Doesn't extend to mesh matching (separate plan) or tabular benchmarks (separate plan).
- Doesn't propose new HSiKAN primitives.

## Files

- Each extension adds one runner: `signedkan_wip/src/run_node_class.py`, `run_contrastive.py`, `run_masked_cycle.py`, `run_bipartite_match.py`
- Shared head modules: `signedkan_wip/src/heads.py` (~100 LOC for all four)
- Eval harnesses per task: ~50 LOC each
- Total: ~700 LOC + 4 task runners

## Connection to other plans

- **Tabular benchmarks** (`plans_hsikan_tabular_benchmarks_2026_05_09.md`) — Extension 1 (node classification) on tabular-induced graphs IS the tabular plan's E2 experiment.  Cross-pollinate.
- **Mesh matching** (`plans_mesh_matching_2026_05_09.md`) — Extension 4 (bipartite matching) is a special case of mesh matching with simpler structure.  Mesh matching is the "Cadillac" version; this is the "Iris-class" warm-up.
- **Structural-KA theorem** (`plans_structural_ka_theorem_2026_05_09.md`) — task-universality across these four extensions is empirical anchor for the structural-KA conjecture.
- **Predictive coding** (`plans_predictive_coding_signedgraph_2026_05_09.md`) — Extension 1 (node classification) is the cleanest small task to test PC on; smaller than the SMC link-prediction task.
