# HSiKAN on time-series via sequence-induced hypergraphs + frequency attention — 2026-05-09

A time-series $x_1, \ldots, x_T$ admits a natural hypergraph construction where each temporal window of length $k$ is a $k$-cycle.  HSiKAN's α-mixer over arity slots becomes a **learned wavelet-band decomposition** — different $k$-arities correspond to different frequency bands.  A frequency-domain attention head, replacing the embedding-space Hamilton inner product with a Fourier inner product, gives per-window per-frequency relevance scoring.  This plan tests whether the architecture transfers to standard time-series forecasting and classification benchmarks.

The novelty thread: *most time-series work is either RNN/Transformer (token-level attention) or dilated convolution (kernel-level locality).  Sequence-induced hypergraph attention is a third mode — global structural primitives at multiple scales, routed by an α-mixer, scored by frequency-domain attention.*

## Goal

Establish HSiKAN as a competitive time-series architecture on at least one of:
- ETT (electricity transformer temperature) — standard forecasting benchmark
- M4 / M5 forecasting — competition data
- UCR/UEA time-series classification — small-data classification

with the following three claims:
1. **Sequence-induced hypergraph construction** is feasible and trains end-to-end.
2. **α-routing emerges as an empirical wavelet decomposition** — different α weights on different $k$-arities correspond to different dominant frequencies in the data.
3. **Frequency-domain attention** lifts forecasting accuracy over the uniform-pool baseline at iso-param.

## Sequence → hypergraph construction

For a time-series $x_1, \ldots, x_T$ where $x_t \in \mathbb{R}^d$:

1. **Vertex set**: $V = \{1, \ldots, T\}$ — one vertex per time-point.
2. **Vertex features**: $h_v(t) = x_t$ — the raw time-series value (passed via `vertex_feat_dim`).
3. **Edge set**: temporal-distance edges. $(t, t+1)$ for adjacent pairs, plus optionally $(t, t+\Delta)$ for fixed dilations $\Delta \in \{2, 4, 8, \ldots\}$ (multi-scale).
4. **Edge signs**: $s(t, t') = \mathrm{sgn}(x_{t'} - x_t)$ — sign of local trend (rising/falling). For multivariate, take the sign of a chosen feature or a learnable projection.
5. **k-cycles**: temporal windows of length $k$. The cycle $(t, t+1, \ldots, t+k-1, t)$ is a closed temporal pattern at scale $k$.
6. **σ parity per cycle vertex**: standard Cartwright–Harary computation from the local trend signs.

This gives a hypergraph where:
- $k=2$ slots are adjacent-pair "sign-of-trend" edges
- $k=3$ slots are 3-window patterns (3-step trends)
- $k=4, 5, 6$ slots are longer windows (medium-frequency patterns)
- $k$ very large captures long-range / low-frequency content

The α-mixer learns *which window scale carries the prediction signal per dataset*.  This is structurally analogous to the WaveNet / TimesNet dilated-conv hierarchy but with **learnable scale weights** instead of fixed dilation factors.

## Frequency-domain attention

Currently in HSiKAN, the cycle-edge attention scores compatibility in *embedding space*:

$$a_{e,t} = \mathrm{softmax}\!\left(\langle W_q h_e, W_k h_t \rangle_{\rm Hamilton}\right)$$

For time-series, replace the inner product with a **frequency-domain Hamilton inner product**:

$$a^{\rm freq}_{e,t} = \mathrm{softmax}\!\left(\sum_\omega \tilde{q}_e(\omega) \cdot \tilde{k}_t^*(\omega)\right)$$

where $\tilde{q}_e, \tilde{k}_t$ are FFTs of the cycle's signed-edge sequence (a 1D signal of length $k$).  This generalises FEDformer's frequency attention from token-pair scoring to cycle-pair scoring.

Equivalently in matrix form:

$$Q^{\rm freq} = \mathrm{FFT}(W_q H_v[\mathrm{cycle}]),\quad K^{\rm freq} = \mathrm{FFT}(W_k H_t)$$
$$\mathrm{score} = \mathrm{Re}\bigl(Q^{\rm freq} \cdot \overline{K^{\rm freq}}\bigr)$$

The "Hamilton" structure can be preserved by treating $\tilde{q}, \tilde{k}$ as quaternions (4 frequency-block channels) and applying the existing Hamilton-product real-part scoring.

## Experiments

### TS1 — Sanity smoke on a tiny synthetic series

- Generate synthetic AR(2) series of length 1000
- Build sequence-induced hypergraph with $\mathcal{K} = \{c_3, c_4\}$
- Train HSiKAN to predict $x_t$ from $\{x_{t-1}, \ldots, x_{t-w}\}$ (1-step forecast)
- Compare to AR(2) regression and a 1-layer LSTM at iso-param

**Acceptance**: HSiKAN MSE within 50% of AR(2) (the optimal model). Confirms the architecture trains on time-series.

### TS2 — UCR univariate classification

- Pick 5 UCR datasets (chosen for diverse class structure: GunPoint, ECG200, Coffee, ItalyPowerDemand, Two_Patterns)
- Each: train HSiKAN as a classifier over the time-series labels
- Compare to: 1-NN-DTW, ResNet-1D, Rocket, MiniRocket

**Acceptance**: HSiKAN within top-3 on at least 1 of 5 datasets at matched param budget.

### TS3 — ETT forecasting

- ETTh1 (hourly electricity transformer temperature) — standard benchmark
- Forecast horizon: 24, 48, 96, 720 steps
- Multi-variate (7 features)
- Compare to: Informer, Autoformer, FEDformer, PatchTST, DLinear

**Acceptance**: ETT-h MSE within 10% of FEDformer on the 96-step horizon at iso-param.

### TS4 — Frequency-attention vs embedding-attention ablation

- TS1+TS3 with the standard embedding-space Hamilton attention
- Same with the new frequency-domain Hamilton attention head
- Compare paired-Δ AUC / MSE on each benchmark

**Acceptance**: paired Δ > 1σ on at least one benchmark in favour of frequency attention.

### TS5 — α-routing as wavelet decomposition

- Train HSiKAN with $\mathcal{K} = \{c_2, c_3, c_4, c_8, c_{16}, c_{32}\}$ on multiple time-series of known frequency content (sinusoid + noise, multi-frequency mixture, AR processes)
- Plot α distribution per dataset
- Verify that α concentrates on $k$-arities matching the dominant frequencies

**Acceptance**: visual + correlation evidence that α routes to $k$-arities matching the data's spectral peaks.  Strong confirmation: a high-frequency dataset has α concentrated at small $k$, low-frequency at large $k$.

## Datasets

| dataset | type | n samples | length T | features |
|---|---|---|---|---|
| AR(2) synth | regression | 100 series × 1000 steps | 1000 | 1 |
| GunPoint | classification | 200 | 150 | 1 |
| ECG200 | classification | 200 | 96 | 1 |
| ItalyPowerDemand | classification | 1096 | 24 | 1 |
| ETTh1 | forecasting | ~14k samples | 17K hours | 7 |
| ETTh2 | forecasting | ~14k samples | 17K hours | 7 |
| ETTm1 | forecasting | ~70k samples | 70K mins | 7 |

All standard, all available via UCR archive / [Zhou et al. ETT GitHub](https://github.com/zhouhaoyi/ETDataset).

## Implementation notes

- New `signedkan_wip/src/sequence_signed_graph.py` (~200 LOC):
  - `build_sequence_signed_graph(x, dilations=[1, 2, 4, 8])` → constructs the temporal-edge graph
  - Multi-scale dilated edge addition
  - Per-edge sign from local trend
- New `signedkan_wip/src/frequency_attention.py` (~150 LOC):
  - `_FrequencyAttentionM_e` — replaces embedding-space attention with FFT-based scoring
  - Compatible with existing Hamilton-product / Highway gate machinery
  - Cycle-batched (memory-bounded forward)
- New runner `signedkan_wip/src/run_time_series.py` (~250 LOC):
  - UCR / ETT loader
  - Forecasting / classification loop
  - Comparison against baselines via `aeon` or hand-coded
- Optional dependency: `pyts` or `aeon` for UCR loading; otherwise hand-roll
- Total: ~600 LOC new code + benchmark infrastructure

## Cost

| experiment | wall time | seeds |
|---|---|---|
| TS1 | ~10 min | 3 |
| TS2 (5 datasets) | ~30 min | 5 |
| TS3 ETT-h | ~2-4 hr | 3 |
| TS4 ablation (TS1 + TS3 paired) | ~3-5 hr | 5 |
| TS5 α-routing | ~30 min | 3 |

Total: ~6-10 hours of compute for the full sweep.  Code: ~3-5 days for implementation + writeup.

## Risk register

| risk | probability | mitigation |
|---|---|---|
| Sequence-induced hypergraph has too many cycles for long series ($T \gg 1000$) | high | cap max_k4 aggressively; use sliding-window subsequences instead of global enumeration |
| Frequency attention's FFT step is non-differentiable in practice on small batches | low | torch supports differentiable FFT (`torch.fft.fft`) |
| α-routing pattern doesn't match expected wavelet decomposition | medium | the empirical pattern itself is the result; "α didn't decompose by frequency" is a paper finding either way |
| ETT benchmark is dominated by linear models (DLinear) — hard to beat | high | scope acceptance to "competitive with FEDformer" not "beat DLinear"; the contribution is architectural fit, not absolute SOTA |
| Cycle structure on sequence is degenerate (all 1D dilations are linear chains) | medium | add multi-feature edges (between sensor channels), giving genuinely 2D structure |

## Acceptance for the plan as a whole

- TS1 trains end-to-end, MSE within 50% of AR(2): minimum
- At least one of TS2 (UCR) and TS3 (ETT) hits the per-experiment acceptance: real claim
- TS4 frequency-attention ablation lifts at least one benchmark: validates the frequency contribution
- TS5 α-routing shows wavelet-like pattern: the interpretability angle

If TS1 fails, the sequence-induced construction is wrong and the plan terminates.

## Order of operations

1. `sequence_signed_graph.py` (~1 day)
2. TS1 sanity smoke (~half day)
3. TS2 UCR classification (~1 day)
4. `frequency_attention.py` (~1-2 days)
5. TS3 ETT forecasting + TS4 ablation (~2-3 days)
6. TS5 α-routing visualisation + writeup (~1 day)

Total: ~7-10 days for full execution + paper draft.

## What this plan does NOT do

- Doesn't compete with FEDformer / PatchTST on absolute SOTA.  Win condition is competitive at iso-param with a *different* inductive bias.
- Doesn't extend to streaming / online time-series.  All forecasting is batch-mode.
- Doesn't propose new wavelet bases.  α-mixer + temporal-window cycles is the wavelet basis HSiKAN provides; we just *measure* what it picks per dataset.
- Doesn't tackle multi-modal time-series (text + audio + sensor).  Single-modality only.

## Connection to other plans

- **Tabular benchmarks** (`plans_hsikan_tabular_benchmarks_2026_05_09.md`) — time-series is "tabular with strong sequential structure"; the success of vertex-feature passing on Iris (E2 lifted from 0.92 to 0.94) suggests the same pattern will work on time-series.
- **Mesh matching** (`plans_mesh_matching_2026_05_09.md`) — both target a non-graph application of HSiKAN; together they argue the architecture is task-universal.
- **Structural-KA theorem** (`plans_structural_ka_theorem_2026_05_09.md`) — TS5 (α as wavelet decomposition) is empirical anchor for the *signal-decomposition* implication of structural-KA: if HSiKAN representations on time-series naturally decompose by frequency, the structural-KA framing extends from graphs to sequences.
- **General-graph extensions** (`plans_general_graph_extensions_2026_05_09.md`) — masked-cycle pretraining on time-series is the analogue of masked-token pretraining; could land in this plan as TS6.
- **Predictive coding** (`plans_predictive_coding_signedgraph_2026_05_09.md`) — PC on time-series is itself a research direction; mention but defer.

## Why this is venue-grade novel

Time-series transformers (Informer, Autoformer, FEDformer, PatchTST) have a year-long body of work on Fourier-domain attention.  Hypergraph time-series methods exist but use static hypergraph structure (e.g. spatio-temporal graphs).  *Sequence-induced* hypergraphs with **multi-scale α-routing + frequency attention** is, to our knowledge, the first architecture that:
- Treats temporal windows as native graph-theoretic primitives
- Routes signal across scales via a learnable α-mixer
- Scores cycles via frequency-domain attention

The cleanest paper claim: *"For time-series, sequence-induced hypergraph attention is the natural generalisation of dilated convolutions, with α-routing as a learned wavelet basis."*  Standalone NeurIPS / ICML submission.
