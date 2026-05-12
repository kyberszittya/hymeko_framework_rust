# Fractal maps for cycle generation and evaluation in HSiKAN — 2026-05-09

The Rust k-cycle enumerator is fast (per `project_rust_cycle_enum_2026_05_02`) and unblocks $k{=}4$ on Slashdot in $\sim 4.6$~s.  Going further --- $k{\geq}6$ on dense graphs, full Epinions, real-time mesh cycles --- runs into the combinatorial wall: $O(\bar d^{\,k}/k!)$ cycles per arity grows fast enough that even an idealised parallel enumerator hits memory pressure before it hits time pressure.  This plan proposes a different direction: **iterated function systems (IFS) as both a structured cycle generator and a fractal-dimension-based readout for $\alpha_\kappa$ routing**.

The goal is twofold:
1. **Generation.**  Replace exhaustive cycle enumeration on dense graphs with a fractal/IFS sampler whose attractor has a structure that matches the graph's multi-scale community structure.  Produce cycle samples that span the same primitive space as enumerated cycles but at orders-of-magnitude smaller cardinality.
2. **Evaluation.**  Use fractal-dimension estimates of cycle distributions and node-embedding fields to characterise what the trained $\alpha_\kappa$ routing has discovered.  Test the conjecture that $\alpha$ correlates with the graph's intrinsic fractal scaling.

If both land, fractal maps become a third mode of HSiKAN cycle handling: alongside exhaustive enumeration (current) and top-K heuristic pruning (current), an IFS-driven sampler that's **scale-aware, generative, and parameter-light**.

## Why this is novel

Signed-graph methods today either (a) enumerate cycles up to a fixed arity (HSiKAN), (b) use spectral / Laplacian operators (SGT), or (c) propagate through walks (SGCN, SiGAT).  None of them treat the cycle distribution as a *fractal measure* on the graph, even though real social signed graphs (Slashdot, Epinions, Wikipedia) have well-documented multi-scale community structure with non-integer fractal dimension.  Fractal-aware sampling and evaluation of cycle distributions is, to our knowledge, unexplored.

Two specific theoretical anchors:

- **Box-counting / correlation dimension** of cycle distributions on real signed graphs is a measurable scalar that varies across datasets.  If $\alpha_\kappa$ at convergence correlates with this dimension, we have a quantitative claim about what the routing has learned.
- **Cycle attractors** under repeated contractive composition are well-defined for IFS in metric spaces.  Defining the right metric on graph cycles is the technical challenge --- but solved properly, it gives a generative model of cycles parameterised by a handful of contractive maps.

## Generation: IFS-driven cycle sampling

### Construction

For each arity $k$, define an IFS over the space $\mathcal{T}_k(G)$ of $k$-cycles in graph $G$.  An IFS consists of $M$ contractive maps $f_1, \ldots, f_M : \mathcal{T}_k \to \mathcal{T}_k$ where each $f_m$ is *graph-aware*: it takes a current cycle $c$ and outputs a perturbed cycle $f_m(c)$ that is close to $c$ in some graph-theoretic metric.

Three candidate maps:

- **$f_1$ — single-vertex substitution**: replace one cycle vertex with a randomly-chosen neighbour.
- **$f_2$ — single-edge rotation**: pick an edge of the cycle, pivot the cycle around it, replacing one vertex.
- **$f_3$ — local-sign flip + structural reroute**: replace the cycle with another $k$-cycle through a chosen edge whose sign-product matches a target balance constraint.

Each $f_m$ has a small probability $p_m$ of being applied.  Iterating from a random seed cycle for $T$ steps produces samples from the IFS attractor.  By the standard ergodic argument of fractal sampling, the distribution converges to the unique invariant measure on $\mathcal{T}_k$ supported on the IFS attractor.

### How this differs from random-walk-on-cycles

Random-walk samplers (e.g., simple Metropolis on cycles) are equivalent to a degenerate IFS with one map.  An IFS with $M{>}1$ maps and varied contraction ratios produces multi-scale samples --- the attractor has Hausdorff dimension $\log M / \log(1/r)$ for uniform contraction ratio $r$.  This is what *fractal* about it: the cycle samples have structure across scales, mirroring the graph's multi-scale community structure.

### Compressed cycle representation

Once the IFS is trained / specified, we no longer need to store the millions of enumerated cycles.  We store:
- The IFS parameters: $M$ maps + their probabilities + their parameters
- A sampling budget $T$
- A seed-cycle distribution

At forward time, generate the cycle batch on-the-fly via $T$ IFS iterations.  This trades enumeration memory for sampling time.  For Slashdot at $k{=}4$, $200\,000$ enumerated cycles ($\sim 16$~MB integer storage) might compress to an IFS with $\sim 100$ parameters + a constant-time sampler.

### Acceptance for the generation track

- **G1**: IFS sampler produces cycle distributions whose top-1000 most-frequent cycles overlap $\geq 80\%$ with the top-1000 enumerated cycles on Bitcoin Alpha.
- **G2**: Training HSiKAN on IFS-sampled cycles instead of enumerated cycles gives 5-seed AUC within 0.005 of the enumerated-cycle baseline on Bitcoin Alpha.
- **G3**: On Slashdot, IFS-sampled cycles at $T{=}50\,000$ steps reach AUC within 0.01 of the enumerated $200\,000$-cycle baseline at $\sim 4\times$ less cycle storage.

## Evaluation: fractal dimension as $\alpha$-routing readout

### Box-counting / correlation dimension on cycle distributions

For each arity $\kappa$, compute the correlation dimension $D_\kappa$ of the cycle distribution:

$$C_\kappa(r) = \frac{1}{T_\kappa^2} \#\{(t, t') : d_{\rm graph}(t, t') < r\},\quad D_\kappa = \lim_{r \to 0} \frac{\log C_\kappa(r)}{\log r}$$

where $d_{\rm graph}$ is a cycle--cycle distance (e.g., minimum Hausdorff distance over vertex sets, or shortest-path between cycle centres).  Empirically estimate $D_\kappa$ via box-counting on the sampled cycle distribution.

### Hypothesis: $\alpha$ correlates with $D_\kappa$

The conjecture: HSiKAN's $\alpha_\kappa$ at convergence is positively correlated with $D_\kappa$ across $\kappa$.  Cycle arities whose distribution is denser (higher correlation dimension) carry more representational capacity, and the optimiser puts more $\alpha$ mass on them.

This is testable directly: train HSiKAN, measure $\alpha_\kappa$ at convergence, measure $D_\kappa$ on the training cycle distributions, plot.

### Acceptance for the evaluation track

- **E1**: $D_\kappa$ is computable in $O(T_\kappa^2)$ time on Bitcoin Alpha for $k \in \{2,3,4,5\}$.  (Standard box-counting; baseline cost.)
- **E2**: Across the four 5-seed-validated datasets (BA, OTC, Slashdot, SBM), Pearson $\rho(\alpha_\kappa, D_\kappa) > 0.5$.  Confirms the routing is fractal-dimension-aware.
- **E3**: Strong claim --- on synthetic graphs generated with controlled fractal dimension, $\alpha_\kappa$ shifts monotonically with the dimension knob.

## Synthetic graphs with controlled fractal dimension

To make E3 testable, generate signed graphs with tunable fractal dimension.  Three options:

- **R-MAT (Recursive MATrix)**: standard fractal graph generator; tune the corner probabilities $\{a, b, c, d\}$ to produce graphs with chosen power-law slope (related to fractal dimension).
- **Hierarchical SBM**: nested stochastic blocks with self-similar structure across scales; tune the depth and branching factor.
- **Random Sierpinski signed graph**: take the Sierpinski triangle's recursive construction, randomise edge signs by structural-balance constraints; produces a graph with explicit fractal Hausdorff dimension.

For each generator, sweep the dimension parameter (e.g., R-MAT $a{\in}[0.4, 0.9]$), train HSiKAN, measure $\alpha_\kappa$.  If $\alpha$ shifts with dimension, the readout is real.

## Experiments

| | what | accept |
|---|---|---|
| G1 | IFS sampler overlap with enumerated cycles | top-1000 overlap $\geq 80\%$ on BA |
| G2 | Train on IFS samples; compare 5-seed AUC | within 0.005 of enumerated baseline on BA |
| G3 | Same on Slashdot at $4\times$ smaller storage | within 0.01 |
| E1 | Box-count $D_\kappa$ on real signed graphs | implementation works in $O(T^2)$ |
| E2 | $\rho(\alpha, D)$ across BA/OTC/Slashdot/SBM | $\rho > 0.5$ |
| E3 | $\alpha$ vs synthetic dimension | monotone shift |

## Implementation notes

- New `signedkan_wip/src/cycle_ifs.py` (~250 LOC):
  - Three IFS maps $f_1, f_2, f_3$ for signed-cycle perturbation
  - Iterator with seed-cycle pool + per-step contractive selection
  - `sample_cycles_ifs(g, k, n_samples)` returns a `list[SignedNTuple]` consumable by the HSiKAN encoder
- New `signedkan_wip/src/cycle_fractal_dim.py` (~150 LOC):
  - Box-counting / correlation-dimension estimator over cycle--cycle Hausdorff distance
  - Wrapper that takes a `list[SignedNTuple]` and returns $D$
- New `signedkan_wip/src/synth_fractal_graph.py` (~200 LOC):
  - R-MAT generator with controllable fractal slope
  - Hierarchical SBM with depth/branching knobs
  - Sierpinski signed-graph generator
- Runner `signedkan_wip/src/run_fractal_eval.py` (~150 LOC):
  - Trains HSiKAN over (real, synthetic) datasets
  - Logs $\alpha_\kappa$, $D_\kappa$, and AUC
  - Produces the regression $\alpha \sim D$ for E2
- Total: ~750 LOC new code

## Cost

| experiment | wall time | seeds |
|---|---|---|
| G1 sampler overlap | ~1 hr | n/a |
| G2 IFS-vs-enum BA | ~3 hr | 5 |
| G3 IFS-vs-enum Slashdot | ~5 hr | 5 |
| E1 dimension estimation | ~30 min | n/a |
| E2 $\alpha$-$D$ correlation | ~1 hr | reuse 5-seed runs |
| E3 synthetic sweep | ~5 hr | 3-seed × 5 dimensions |

Total: ~1-2 weeks for full sweep + write-up.

## Risk register

| risk | probability | mitigation |
|---|---|---|
| IFS maps don't preserve cycle balance / sign structure | high | test with $f_3$ (balance-preserving rerouting); reject samples that violate constraints |
| Cycle--cycle Hausdorff distance is expensive ($O(k^2)$ per pair) | medium | cache distances; subsample to $T_\kappa{=}5\,000$ for box-counting |
| Correlation dimension $D_\kappa$ doesn't separate datasets well | medium | try alternative metrics (information dimension, multifractal spectrum) |
| IFS sampler converges to a degenerate attractor (single cycle) | medium | $f_1, f_2, f_3$ designed with fixed contractive mass to ensure expanding measure on $\mathcal{T}_k$ |
| Synthetic fractal-graph generators don't have HSiKAN-relevant signed structure | medium | Sierpinski signed-graph generator constructs balanced/unbalanced cycles by design |

## Order of operations

1. **Box-counting dimension estimator** (E1) --- standalone, minimal-risk.  Compute $D_\kappa$ on the existing 5-seed BA / OTC / Slashdot runs.  ~1 day.
2. **$\alpha$-$D$ correlation analysis** (E2) --- reuse existing checkpoints.  ~half day.  If $\rho > 0.5$, a real readout claim lands; otherwise the evaluation track is null.
3. **IFS sampler implementation** (G1) --- ~3 days for the three contractive maps + sample-quality test against enumerated cycles.
4. **G2 BA paired comparison** --- ~1 day GPU.
5. **G3 Slashdot paired comparison** --- ~2 days GPU.
6. **Synthetic fractal-graph generators** (E3) --- ~3 days for generators + sweep + plot.
7. **Paper draft** (4-6 pp NeurIPS workshop format) --- ~1 week.

Total: ~3 weeks for the full plan.

## Acceptance for the plan as a whole

- **Tier 1** (workshop): E1 + E2 land --- HSiKAN's $\alpha$ correlates with cycle fractal dimension on real graphs.  Submit to NeurIPS workshop on graph learning / structured prediction.
- **Tier 2** (NeurIPS): G2 + E2 + E3 land --- IFS sampler is a viable cycle-storage compression *and* the fractal-dimension readout is empirically supported.
- **Tier 3** (top venue): G3 lands --- IFS sampler scales to Slashdot at $4\times$ memory savings.

## What this plan does NOT do

- Doesn't replace exhaustive enumeration in the SMC paper.  All 5-seed validated numbers stand on enumerated cycles.
- Doesn't propose a new HSiKAN architecture.  The cycle sampler / fractal-dimension readout are *upstream* of the architecture --- they change which cycles get fed in, not how they're processed.
- Doesn't claim universality.  Restricted to signed graphs with sufficient cycle structure (BA, OTC, Slashdot, SBM, possibly Epinions).
- Doesn't extend to mesh / time-series / tabular --- those have their own plans where graph structure is induced differently.

## Connection to other plans

- **k-Cycle vision detection** (`docs/plans_kcycle_vision_2026_05_07.md`) --- IFS-driven cycle generation could replace SLIC-superpixel cycle enumeration on images, giving a fractal-dimension-aware face detector.
- **Mesh matching** (`plans_mesh_matching_2026_05_09.md`) --- mesh triangulations are well-known to have fractal dimensions matching the underlying surface; IFS sampling of triangle-pair correspondences could be a faster alternative to Sinkhorn matching at scale.
- **Time-series** (`plans_hsikan_time_series_2026_05_09.md`) --- temporal-window cycles have natural fractal structure (self-similarity across timescales).  The fractal-dimension readout would land cleanly on time-series too --- $\alpha$-routing as a wavelet/fractal decomposition.
- **Structural-KA theorem** (`plans_structural_ka_theorem_2026_05_09.md`) --- fractal dimension of the cycle distribution is a candidate measure for the *operator-class richness* needed in the density-conjecture proof.  Could anchor the constructive direction.
- **Streaming top-K cycle enumerator** (`docs/plans_streaming_enumerator_2026_05_07.md`) --- IFS sampler is a *generative* alternative to streaming top-K; fits the same use case (memory-bounded cycle handling at scale).
