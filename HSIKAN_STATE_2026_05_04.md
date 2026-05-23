# What HSiKAN is now — state as of 2026-05-04

A snapshot of the project at end-of-day 2026-05-04, after the
SMC camera-ready push, the SGT baseline, HyMeKo→star pipeline,
chicken-aggression scaffold, and the Cartwright-Harary unsupervised
scorer all landed.

---

## One-line description

**HSiKAN is a mixed-arity, signed-cycle Kolmogorov-Arnold network
for signed-graph link prediction and per-vertex classification,
with two-regime applicability (cycle-rich vs walk-rich datasets),
unsupervised aggressor identification grounded in 1956 structural
balance theory, and a self-referential consumer path through its
parent IR.**

In other words: a signed-graph model that reads off, with no labels,
which vertex is the centre of hostile structure — implemented as a
spline-activation hypergraph network, eight decades of pure
mathematics composed end-to-end.

---

## Architecture (one paragraph)

For each signed graph, HSiKAN enumerates simple $k$-cycles for
$k \in \{2, 3, 4, 5\}$, computes a per-vertex sign
$\sigma_i(c) = (-1)^{n_i^{-}(c)}$ (Cartwright-Harary cycle-product
specialised to vertex $i$), and processes each $(\text{cycle},
\sigma)$ tuple through a SignedKAN layer: per-sign inner spline
$\varphi_v^\sigma$, per-sign $\sigma$-mask aggregation across the
$k$ cycle vertices, per-sign outer spline $\varphi_e^\sigma$,
sum across signs to get a cycle embedding.  The per-arity cycle
embeddings are pooled to edge embeddings via sparse
signed-incidence multiply $\mathbf{M}_e^{(k)}$, and a
softmax-normalised arity weight $\alpha_k$ mixes them
$\mathbf{H}^{\text{edge}} = \sum_k \mathrm{softmax}(\alpha)_k \cdot
\mathbf{M}_e^{(k)} \mathbf{H}^{(e, k)}$.  A linear classifier on the
edge embedding predicts the sign.  Splines are uniform Catmull-Rom
(C¹-continuous, interpolatory, 4-point gather, constant basis matrix
folded by `torch.compile`).  Training is full-batch BCE with
class-balanced positive weight.

---

## Empirical state (5 datasets, multi-seed)

### Headline AUC vs baselines (5-seed mean ± std on Bitcoin / SBM, 3-seed on Slashdot / Epinions)

| Dataset       | HSiKAN          | SGCN          | SiGAT  | SGT (NEW)       | Δ best  |
|---------------|-----------------|---------------|--------|-----------------|---------|
| Bitcoin Alpha | **0.939 ± .011** | 0.874 ± .006 | 0.899  | 0.898 ± .001    | $+0.040$ |
| Bitcoin OTC   | **0.930 ± .008** | 0.906 ± .006 | 0.934  | 0.915 ± .010    | $-0.004$ |
| Slashdot      | 0.861 ± .002    | 0.883 ± .002 | —      | **0.897 ± .002**| $-0.036$ |
| SBM $n{=}200$ | **0.911 ± .028** | 0.504 ± .065 | —      | 0.563 ± .104    | $+0.349$ |
| SBM $n{=}400$ | **0.962 ± .009** | 0.677 ± .070 | —      | 0.690 ± .025    | $+0.273$ |
| Epinions      | $0.606$ (h=16, 1 seed, chunked) | 0.931 ± .003 | — | **0.941 ± .003** | $-0.335$ |

### Two-regime story (the headline finding)

- **Cycle-rich domains (SBM):** HSiKAN dominates by $+0.27$ to $+0.35$
  AUC over the best baseline.  Both attention (SGT) and
  message-passing (SGCN) collapse to near-random.  Signed-cycle
  structure is what carries the SBM signal.
- **Walk-rich domains (Slashdot, Epinions):** SGT takes the lead
  via attention bias.  Dense walks carry the signal; cycle bias
  has less to grip.
- **Mixed (Bitcoin):** within seed-noise margin between HSiKAN /
  SGT / SiGAT.  HSiKAN slightly above on Alpha, slightly below
  on OTC.

$\alpha_k$ is the **quantitative compass** reading off which regime
the dataset sits in — Bitcoin favours $k{=}3$ triads, Slashdot
favours $k{=}5$ ($46\%$) with $k{=}2$ as direct-edge channel, SBMs
favour $k{=}4$ matching their 4-community structure.

### Pruning Pareto + symbolic distillation

- L2-norm pruning: $\sim 78\%$ of spline activations zeroable
  without AUC loss.
- Pruned $h{=}4$ Slashdot variant ($0.861 \pm .002$) outperforms
  $h{=}16$ counterpart ($0.849 \pm .003$) by $+0.012$ AUC across
  5 seeds — strongest realisation of regularisation-as-pruning
  observed.
- Symbolic distillation of survivors: $\sim 91\%$ fit best as
  $a\sin(\omega x{+}\varphi){+}c$ — Fourier-style decomposition
  confirmed against three null baselines (untrained, random
  spline coefs, Gaussian-process draws — all hover at $\sim 50\%$
  sinusoidal, gap $+35$ to $+40$\,pp).

### ph18c entropy follow-up

- Multi-term entropy_lyapunov regulariser has TWO confirmed
  positives now (was one): highway-10 at $L{=}10$
  ($\Delta = +0.092$\,pp at 10-seed, $p{<}0.05$), and resmlp-40 at
  $L{=}40$ ($\Delta = +0.213$\,pp at $\lambda{=}0.5$,
  $p{<}0.05$).  Path I's calibration law $\lambda_{\text{multi}}
  \sim \lambda_{\text{scalar}} / L$ predicted the resmlp-40
  recovery exactly.  Highway-20 confirmed depth-fragile across the
  full $\lambda$-grid.

---

## Infrastructure shipped (around the model)

**Cycle enumeration (Rust, PyO3-bound):**
- `hymeko.enumerate_k_cycles_rs` — closed simple cycles, DFS +
  bitset visited + BFS-distance pruning + rayon parallel + atomic
  early-stop.
- `hymeko.enumerate_k_cycles_color_coded_rs` — Alon-Yuster-Zwick
  unbiased sampler.
- `hymeko.enumerate_k_cycles_path_closure_rs` — alternative
  enumeration.
- `hymeko.enumerate_k_walks_rs` — open length-$L$ simple walks
  (Walk-HSiKAN primitive, NEW today, 12-case verification suite).
- Reservoir sampling, EarlyStop, Full-enumeration sink modes.

**HyMeKo IR pipeline:**
- `data/nn/hsikan_mixed.hymeko` — HSiKAN topology in HyMeKo
  source form (4 arities × `signedkan_layer` + `arity_mixer` +
  `signed_classifier`).
- `data/nn/walk_hsikan.hymeko` — Walk-HSiKAN companion source
  (mirror).
- `data/nn/meta_nn.hymeko` — extended with Tier-3 layer types
  (`signedkan_layer`, `walk_layer`, `arity_mixer`,
  `signed_classifier`).
- `transforms/torch_dataflow/queries.hymeko` + `template.py` —
  emitter handles the Tier-3 types; new `bind:+:all_csv` template
  directive for multi-source fan-in.
- `python/ehk_torch_stub` — stub helper classes (`SignedKANLayer`,
  `WalkLayer`, `ArityMixer`, `SignedClassifier`).
- `hymeko emit ... --format torch_dataflow` produces a runnable
  PyTorch module: forward + backward + 5-step SGD all green.
- Permanent test: `scripts/verify_hsikan_emit.py`.

**HyMeKo → star expansion → HSiKAN:**
- `scripts/hymeko_to_signed_graph.py` — walks any HyMeKo source,
  emits the (`edges_u`, `edges_v`, `signs`) format HSiKAN
  consumes.  Zero new Rust on the data path.
- Cycle-count probe over 5 nets: pure feedforward → 0 cycles at
  any $k$; fan-in topologies → exactly $\binom{K}{2}$ cycles at
  $k{=}6$.  HSiKAN-on-HyMeKo defaults to $k{=}2$ for sequential
  nets, gravitates to $k{=}6$ for fan-in.

**Memory-bound scaling (NEW today):**
- `HSIKAN_CHUNK_T` env-var splits T-cycles dimension during
  forward, processes chunks sequentially.  Lets the full
  $\mathcal{K}{=}\{2,3,4\}$ recipe run at $h{=}16$ on Epinions
  on $8$\,GB GPUs (would OOM otherwise).  Verified equivalent
  on Bitcoin Alpha within seed noise.
- `HSIKAN_MAX_K2 / HSIKAN_MAX_K3` env-vars for fine-grained
  cycle-budget control per arity.

**Komondor HPC:**
- `scripts/slurm/run_hsikan_epinions.sbatch` — generic SBATCH
  template (account / partition placeholders to fill in).  On
  a 40\,GB A100 the published recipe should run uncrowded.

**Baselines:**
- SGT (Signed Graph Transformer) — `signedkan_wip/src/baselines/sgt.py`
  (NEW today): pre-LayerNorm encoder + sign-aware sparse
  self-attention + position-wise FFN.  3-seed sweep across 6
  datasets.
- Existing: SGCN, SiGAT, SiGAT-attn, MLP+GCN, VanillaKAN.

---

## Application domains shipped

**Standard signed-graph link prediction:**
- Bitcoin Alpha, Bitcoin OTC, Slashdot, Epinions (SNAP)
- SBM $n{=}200, n{=}400$ (synthetic, 4-community)
- Synthetic kinematic / pose fixtures

**HyMeKo IRs as signed graphs (NEW today):**
- Star-expand any neural-network description into the format
  HSiKAN consumes.  Self-referential consumer.
- Tested on `mnist_resmlp_3`, `mnist_highway_10`, `disjoint_net`,
  `hsikan_mixed`, `walk_hsikan`.

**Animal anatomy + behaviour (NEW today, scaffold only — needs
real data):**
- `data/anatomy/chicken_anatomy.hymeko` — 12-keypoint chicken
  skeleton with rigid bones (sign $+1$) + flexible joints
  (sign $-1$) + 4 kinematic chains.  47 star-edges; 47 cycles
  at $k{=}6$ matching the 4 anatomical regions.
- `signedkan_wip/src/chicken/` — flock simulator + trajectory →
  signed-graph converter + supervised aggressor classifier +
  unsupervised aggressor scoring (negative-out-degree baseline,
  Cartwright-Harary cycle-balance, HSiKAN self-supervised,
  rank-ensemble of all three).
- 8-seed unsupervised AUC: baseline $0.638 \pm .164$, CH
  $0.586 \pm .107$, HSiKAN $0.653 \pm .155$, **ensemble $0.693
  \pm .120$**.  Cartwright-Harary scorer has lowest single-scorer
  variance because pure topology carries no training noise.

**Cross-thread anchor:** the same Cartwright-Harary 1956 structural
balance theorem that defines HSiKAN's $\sigma$-rule (cycle-product
parity) **also defines aggressor identity in flocks**: aggressors
generate balanced $(-, -, +)$ triads with their victim cliques.
Same theorem, two scales.

---

## Mathematical foundation

Eight layers, eight decades of theory.  Detailed PDF at
`reports/hsikan_math_foundation.pdf` (4 pages).  Ladder:

| # | Layer                           | Source date  | Reference                                |
|---|---------------------------------|--------------|------------------------------------------|
| 1 | Signed-graph theory             | 1946-1967    | Heider; Cartwright-Harary; Davis         |
| 2 | Hypergraph reductions           | 1973         | Berge; Thesis 2 (this dissertation)      |
| 3 | Kolmogorov-Arnold representation| 1957 / 2024  | Kolmogorov; Liu et al. (KAN)             |
| 4 | Spline approximation            | 1974-1984    | Catmull-Rom; de Boor; Kochanek-Bartels   |
| 5 | Mixed-arity softmax             | this work    | $\alpha_k$ readout                       |
| 6 | Symbolic distillation           | this work    | Fourier-style decomposition              |
| 7 | Cycle enumeration (algorithmic) | 1995         | Alon-Yuster-Zwick color-coding           |
| 8 | Information geometry / regs     | this work    | Path I calibration; Lyapunov wrapping    |

---

## What's NOT yet shipped

**Open / external-data-blocked:**
- Real chicken video validation — Éva has raw video, no
  annotations yet.  YOLOv8+ByteTrack bootstrap recommended;
  pipeline ready to plug in.
- Komondor SLURM run — script written, account/partition not yet
  filled in.
- Real-AUC HSiKAN-from-HyMeKo emit (Item #4 final) — week-long;
  needs real `signedkan_wip` Tier-3 helpers in
  `ehk_torch_stub` instead of stubs.

**Architectural extensions queued:**
- Walk-HSiKAN as a trained model (enumerator + HyMeKo source +
  emitter all exist; ML model not yet trained).
- HyMeKo factor-view PDFs auto-generated from the IR (instead of
  hand-drawn TikZ figures in the SMC paper).
- Chunked-incidence forward (the durable fix for $|V|{\gtrsim}10^5$
  scaling — currently we have cycle-batch chunking but the
  M_vt buffer is still single-tensor).

**Paper-side TODO:**
- SMC 2026 WIP camera-ready (Table I + III + IV refreshed today
  with 5-seed numbers; SGT column added; §III.G control-baseline
  prose; §V self-referential-consumer bullet; §IV.B Epinions
  scaling note).
- Journal extension (T-SMC-S branched, paper/arxiv_v1/) — would
  carry Walk-HSiKAN, HyMeKo round-trip, MJCF binary-graph
  ablation, and the chicken-aggression application as full
  contributions.

---

## Files of record

- `paper/smc2026_hsikan_wip/main.tex` — SMC 2026 WIP paper
  (5 pages, builds clean).
- `reports/hsikan_math_foundation.pdf` — 4-page mathematical
  foundation document (NEW today).
- `HSIKAN_ROADMAP_2026_05_04.md` — day-by-day deliverable log
  with all timestamps.
- `HSIKAN_STATE_2026_05_04.md` — this file.
- `signedkan_wip/src/chicken/README.md` — chicken-aggression
  pipeline walkthrough.
- `data/anatomy/chicken_anatomy.hymeko` — 12-keypoint anatomical
  hypergraph.
- `signedkan_wip/src/baselines/sgt.py` — Signed Graph Transformer
  baseline.
- `signedkan_wip/src/run_sgt_sweep.py` — multi-seed runner.
- `signedkan_wip/src/run_sinusoid_controls.py` — symbolic-control
  sweep.
- `scripts/hymeko_to_signed_graph.py` — HyMeKo → star expansion.
- `scripts/verify_hsikan_emit.py`, `scripts/verify_walks.py` —
  regression suites.
- `hymeko_py/src/cycles.rs` — Rust cycle + walk enumerator.

Memory entries in
`~/.claude/projects/-home-kyberszittya-hakiko-ws-hymeko-hymeko-framework-rust/memory/`:
6 new today (`project_signedkan_sinusoid_controls_2026_05_04.md`,
`project_hsikan_hymeko_emit_2026_05_04.md`,
`project_sgt_baseline_2026_05_04.md`,
`project_hymeko_to_signed_graph_2026_05_04.md`,
`project_chicken_aggression_2026_05_04.md`,
plus the ph18c update to `project_phase18_architectural_parity.md`).

---

## The aesthetic point — once more, for the file

The whole stack composes from
**1946 (Heider)** $\to$
**1956 (Cartwright-Harary)** $\to$
**1957 (Kolmogorov)** $\to$
**1973 (Berge)** $\to$
**1974 (Catmull-Rom)** $\to$
**1995 (Alon-Yuster-Zwick)** $\to$
**2024 (Liu KAN)** $\to$
**2026 (HSiKAN)**.

Eighty years of pure mathematics, quietly waiting to read off
which chicken is the aggressor.
