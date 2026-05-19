# SignedKAN — Future Directions

A consolidated index of research directions identified during the WiP-track
development that are scoped beyond this paper. Each section sketches
the idea, its relation to what is already in the codebase, and an
expected-impact / effort estimate.

The goal of this file is to keep the WiP-track narrative tight (four-part
contribution: gap-closing recipe, prunability, sinusoidal distillation,
fast training) while preserving the substantial menu of research
extensions the saga surfaced.

## 1. Density-aware hypergraph regularisers

### R1 — Arity penalty (structural)

$$
\mathcal{L}_{R_1} = \sum_e |e|^\beta \sum_{v \in e} \|\mathbf{c}_{v, e}\|_2^2
$$

Penalises spline coefficients on large hyperedges more aggressively
($\beta > 1$ is the knob).

**Status here**: in the current $3$-uniform triad construction
$|e|$ is constant, so $R_1$ reduces to a global $L_2$ penalty on
spline coefficients. A separate L1-on-coefficients sweep
(`run_l1_sparsity.py`) showed that any $\lambda \geq 10^{-4}$
collapses macro-$F_1$ to ~0.30 — global magnitude shrinkage is too
aggressive in this architecture. **$R_1$ becomes a distinct lever
only with variable-arity hyperedges (see §3 below)**.

### R2 — Participation penalty (vertex-side)

$$
\mathcal{L}_{R_2} = \frac{1}{|V|} \sum_v \deg_H(v)^2 \, \|\mathbf{h}_v\|_2^2
$$

Penalises high-triad-degree vertex embeddings (hub vertices that
dominate the aggregation).

**Status here**: implemented (`participation_reg.py`,
`ParticipationRegulariser`). Single-seed smoke test at $\lambda = 10^{-3}$
was effectively neutral (the max-degree normalisation pushes the
weighting into a small range). **A multi-$\lambda$ sweep is the
natural extension** — the candidate range is $\lambda \in
\{10^{-2}, 10^{-1}, 10^{0}\}$ because of the normalisation. Effort:
~30 LOC for a launcher; ~10 minutes compute.

### R3 — Entropy-gated hyperedge selection

$$
\mathcal{L}_{R_3} = \lambda_1 \sum_e |e| \cdot g_e + \lambda_2 \cdot H(\mathbf{g})
$$

Learnable per-hyperedge gate $g_e \in [0, 1]$, with the first term
a structural prior penalising reliance on dense hyperedges and the
second an entropy regulariser preventing degenerate collapse.

**Status here**: the gate-and-entropy pair is essentially what
`SignedTriadAttention` + `attention_entropy_loss` already implements
(tanh-scored attention with per-edge entropy term). The genuinely
new piece in $R_3$ is the structural prior $|e| \cdot g_e$, which
again only kicks in once we have variable arity. **Path: combine the
existing attention module with a $|e|$-weighted prior once N-edges
land**.

## 2. LSTM-style gated hyperedge blocks

Generalise the residual / highway / JK family already in the
codebase to a proper LSTM block over the multi-layer SignedKAN
forward path:

$$
\begin{aligned}
\mathbf{f}_v^{(\ell)} &= \sigma(\mathbf{W}_f \mathbf{h}_v^{(\ell)} + \mathbf{b}_f) \\
\mathbf{i}_v^{(\ell)} &= \sigma(\mathbf{W}_i \mathbf{h}_v^{(\ell)} + \mathbf{b}_i) \\
\mathbf{o}_v^{(\ell)} &= \sigma(\mathbf{W}_o \mathbf{h}_v^{(\ell)} + \mathbf{b}_o) \\
\tilde{\mathbf{h}}_v^{(\ell)} &= \tanh(\mathbf{M}_{vt}\,\text{SignedKANLayer}_\ell(\mathbf{h}_v^{(\ell)})) \\
\mathbf{c}_v^{(\ell+1)} &= \mathbf{f}_v^{(\ell)} \odot \mathbf{c}_v^{(\ell)} + \mathbf{i}_v^{(\ell)} \odot \tilde{\mathbf{h}}_v^{(\ell)} \\
\mathbf{h}_v^{(\ell+1)} &= \mathbf{o}_v^{(\ell)} \odot \tanh(\mathbf{c}_v^{(\ell+1)})
\end{aligned}
$$

**Why this is more than highway**: independent forget-and-input
gates allow purely shrinking the cell, purely accumulating it, or
any mix; the cell state $\mathbf{c}$ is *separate* from the output
$\mathbf{h}$ so long-range memory across layers does not contaminate
each layer's spline input; an output gate $\mathbf{o}$ controls
exposure.

**"Automatic and learnable organisation of forget cells"**:
per-channel $\mathbf{f}$ values can be entropy-regularised so they
do not collapse to $\equiv 0$ or $\equiv 1$. The optimiser then
self-organises which channels are short-term (low $\mathbf{f}$,
high $\mathbf{i}$) vs long-term (high $\mathbf{f}$, low $\mathbf{i}$).

**Status here**: residual ($\mathbf{f} \equiv 1$) and tied-gate
highway are implemented; per-position skip placement (heterogeneous)
is implemented and was a small free win on OTC. Independent forget
+ input + output gates are not. Effort: ~120 LOC for `LSTMHyperBlock`,
new launcher, ablation; ~15 minutes compute. Story value: paper-shaped
on its own ("the first KAN with LSTM-style gated memory cells").

## 3. Variable-arity hyperedges (N-edges, clique clusters)

Generalise from $3$-uniform triads to $k$-uniform hyperedges using
Davis 1967 weakly-balanced $k$-cycles. Unlocks $R_1$, the structural
prior in $R_3$, and a richer hyperedge construction over arbitrary
signed cliques.

**Status here**: SignedKANLayer's sub-aggregation loop is already
arity-agnostic. Only the hyperedge-construction step (Phase 1.x)
and the loss (triad balance theory generalised to $k$-cycle balance)
need extending. Effort: ~200 LOC plus careful balance-theory
derivation; ~30 minutes compute for a dataset-level sweep.

## 4. Adaptive / hierarchical knot placement

Move from uniform B-spline knot grids (current default $G\!=\!5$
on $[-1, 1]$) to adaptive knots — Forsey--Bartels-style hierarchical
B-splines, or NURBS with learnable knot positions. Concentrates
basis resolution where the activation curve varies fast.

**Status here**: not tested. The pruning result suggests our
uniform grid is already over-parameterised (78% of the 7-basis
cubic B-spline can be zeroed without accuracy loss), so the natural
question is whether *adaptive* placement could deliver the same
expressivity at $G\!=\!3$ where uniform pruning hurt accuracy
(the failed ECG row of Table~\ref{tab:gap}). Effort: substantial
($\sim\!5\times$ implementation complexity over uniform splines)
without a clear empirical motivation; **defer until a fixture
demonstrates capacity is misallocated rather than oversupplied**.

## 5. Input-conditional path routing in KAN

At inference, score each (branch, channel) spline against the input
and activate only the top-$k$. Per-input adaptive sparsity (vs the
global threshold-pruning we have now). Different inputs trigger
different sinusoidal "modes" — a per-input Fourier decomposition.

**Status here**: not tested. Likely unhelpful for AUC (active-set
search is fragile with respect to the loss landscape) but
interpretability-rich: each input's prediction depends on its own
small subset of sinusoids. Effort: ~80 LOC. **Story value is in
the visualisation, not the numbers** — defer to a journal extension
that has space for the additional figure.

## 6. Pre-trained / structural-prior node embeddings

Initialise `node_embed.weight` from random-walk embeddings
(node2vec / DeepWalk / SiNE-style spectral pre-training) instead
of $\mathcal{N}(0, 0.1^2)$. Closes part of the gap to engineered
baselines that *do* pre-train.

**Status here**: a pure-eigenvector spectral init was tried
(`signed_laplacian.py`, `make_spectral_init`) and was effectively
neutral after multi-seed sweep; the structural prior gets washed
out by training. **Pre-training a separate random-walk embedding
network and using its output as initialisation is a different beast
and untested**. Effort: 100--200 LOC plus random-walk dataloader;
moderate compute. The trained-from-scratch claim does not survive
this addition, so pre-trained init lives in a journal-extension
bucket where matched-protocol comparison to engineered baselines
is the explicit goal.

## 7. Matched-protocol re-implementation of engineered baselines

Re-run SGCN, SDGNN, DSHGNN under our exact split protocol
(80/10/10 random edge split, seed 42) so the AUC comparison is
strict apples-to-apples rather than informative-but-not-definitive.

**Status here**: deliberately deferred. The current paper reports
SignedKAN's regime ("trained from scratch, no pre-training, no
auxiliary supervision") and quotes published baseline numbers
honestly as informative comparators. A matched-protocol re-run is
the load-bearing experiment for any "we beat SGCN" claim;
without it, the AUC story is qualified.

## 8. Total-model parameter compression beyond splines

Pruning compresses spline coefficients (~0.8% of total parameters);
the remaining 99% is the node-embedding table $\mathbb{R}^{|V|
\times h}$. Orthogonal compression interventions on the embedding:

- product quantisation of the embedding table
- top-$k$ sparse embedding selection (only retain embeddings for
  vertices that participate in many triads)
- quantisation-aware training (8-bit / 4-bit weights)

**Status here**: not tested. Composes naturally with our
spline-pruning result for an end-to-end small-and-fast model.
Effort: 200--400 LOC depending on the chosen technique.

## N. Inference-latency floor reduction (added 2026-05-03)

After the cycle-enum + compile + CSR + cudagraph optimisation pass
(see `project_hsikan_inference_speedup_2026_05_03.md` in memory), the
SOTA-config HSiKAN forward bottoms out at 6 ms on Bitcoin and 28 ms on
Slashdot (RTX 2070 SUPER, single-process steady state). The remaining
gap to SGCN is **structural**: per-arity × per-layer × spline-eval
launches more kernels than SGCN's two sparse mat-muls. Three named
avenues:

### N1 — Fused CUDA kernel for `spline + sign-mask + pool`

Collapse the per-layer kernel sequence (Catmull-Rom eval → sign-mask
gather → per-sign agg → diagonal outer spline → sum-over-signs) into a
single hand-written CUDA kernel. Each spline call currently fires ~10
small launches; the cudagraph capture amortises *launch* overhead but
not the per-op kernel work. A fused kernel skips the intermediate
materialisations entirely.

**Expected impact**: 2-3× more reduction (28 → 10 ms Slashdot, 6 → 2-3
ms Bitcoin), bringing the SGCN ratio to ~2× across all datasets.
**Effort**: 1-2 weeks (warp-cooperative design, dtype/grid
parameterisation, correctness vs the eager Python path).

### N2 — Knowledge distillation of the *h*=16 SOTA teacher into an *h*=8 student

Pareto sweep on Bitcoin Alpha cuda showed *h*=8 hits 3.5 ms (4.8×
SGCN) and *n*_layers=1 hits 3.5 ms; combined gives 1.84 ms (2.5×
SGCN). Both points cost an estimated -0.02 to -0.04 AUC vs the SOTA
teacher (full evaluation deferred). Distillation aims to recover most
of that accuracy at the fast-variant latency.

**Expected impact**: 1.7-3× over the SOTA-config baseline, with
≤0.01 AUC loss vs SOTA after a successful distillation run.
**Effort**: ~1 week — distillation losses on the spline-coef level
are non-standard and may need a §IV.{distill} treatment.

### N3 — Ampere+ hardware levers (TF32, fp16 tensor cores)

Not measured here because the development GPU is sm_75 (Turing).
TF32 (sm_80+) typically gives 1.3-1.5× free on matmul-heavy paths;
fp16 tensor cores reward larger matmuls (h ≥ 32) and would compose
with N1's fused kernel. Worth re-running the bench on an A100 / H100
for a "platform" column in any extended-version table.

**Expected impact**: 1.3-2× free on Ampere+, likely composable with N1.
**Effort**: hours (env var + dtype cast); pending hardware access.

## Cross-references

- Pruning + symbolic distillation: §V of the paper, also
  `project_signedkan_pruning_2026_04_30.md` in memory.
- Heterogeneous skip placement: §V.4 of the paper,
  `run_skip_heterogeneous.py`.
- Multi-layer SignedKAN with JK + sum-pool: §IV.{multilayer} of the paper,
  `MultiLayerSignedKAN` in `signedkan.py`.
- Hypergraph triad loss: `triad_loss.py` and §IV.{triad} of the paper.
- Signed-triad attention + attention entropy: `attention.py`.
- Cross-branch information regulariser: `cross_branch_reg.py`.
- Participation regulariser ($R_2$): `participation_reg.py`.
- Iterative pruning + retraining: `run_iter_prune.py` (negative result).
- L1 sparsity during training: `run_l1_sparsity.py` (negative result).
