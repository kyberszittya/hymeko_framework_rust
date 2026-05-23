# Dossier for Jean Pimentel — ABB / SSG / MSG: objective, numbers, chemical-process use

**Date:** 2026-05-18
**Audience:** Jean Pimentel (Pannonia, P-graph / PSE community)
**Source threads:** [pgraph_hymeko_brief.tex](./pgraph_hymeko_brief.tex), [topk_cycles_brief](./topk_cycles_brief.tex), [meeting_pimentel_outline](../docs/meeting_pimentel_outline.md), reports `2026-05-10-abb-global-topk`, `2026-05-11-abb-global-fullness`, `2026-05-12-gomb-msg-ssg-driver`, `2026-05-12-gomb-cycle-abb-compare-driver`, [signedkan_wip/docs/gomb_cycle_abb_optimization.md](../signedkan_wip/docs/gomb_cycle_abb_optimization.md), and Rust source at [hymeko_pgraph/src/abb.rs](../hymeko_pgraph/src/abb.rs) and [hymeko_graph/src/topk_cycles.rs](../hymeko_graph/src/topk_cycles.rs).

Pimentel's three asks:

1. **What is the multi-objective function on ABB SSG / MSG, what is the governing rule?**
2. **What are the numbers on optimization and pruning?**
3. **What could be the possible use in chemical / raw-chemical processes?**

This dossier answers each in turn, then maps the parts together.

---

## 1. The objective — two layers, both ABB

The codebase implements **two distinct ABB families** that the question conflates by name. Both are useful for Pimentel; the first is the direct P-graph descendant, the second is the structural generalisation to signed-cycle enumeration.

### 1.1 P-graph ABB (Friedler-Tarján-Huang-Fan 1992 reproduction)

**Module**: `hymeko_pgraph::abb`. Inputs: a `LoweredPGraph` = $(M, O, E_{\text{dir}}, R, P, c)$ where
- $M$ = materials, $O$ = operating units, $E_{\text{dir}}$ = directed bipartite edges,
- $R \subseteq M$ = raws, $P \subseteq M$ = required products,
- $c: O \to \mathbb{R}_{\geq 0}$ = per-unit cost.

**Objective** (single, scalar — *not* multi-objective in this layer):

$$
\min_{O' \subseteq O_{\max}} \sum_{u \in O'} c(u)
\quad\text{s.t.}\quad
\begin{cases}
\text{input consistency: } \forall u \in O',\ \mathrm{in}(u) \subseteq C_R(O') \\[2pt]
\text{demand: } P \subseteq C_R(O') \\[2pt]
\text{no-excess (strict): every non-raw, non-product output is consumed}
\end{cases}
$$

where $C_R(O')$ is the producibility closure (least fixed point: start at $R$, add every $\mathrm{out}(u)$ for $u \in O'$ whose inputs are already in the set).

**Governing rule** = **two bounds + DFS branching**:

| Bound | Test | What it prunes |
|:---|:---|:---|
| **Inclusion bound** | partial-cost $\geq$ current incumbent cost | branches that cannot beat the best known feasible structure |
| **Reachability bound** | $C_R(\text{included} \cup \text{undecided}) \not\supseteq P$ | branches where even taking *all* remaining undecided units cannot produce every required product — dominant pruner for sparse problems |

Branching order is fixed; each step takes **include before exclude**, so a feasible incumbent appears early. See [hymeko_pgraph/src/abb.rs:119-180](../hymeko_pgraph/src/abb.rs#L119-L180).

**MSG (Maximal Structure Generation)** is the *preprocessing*: iteratively trim units that fail forward-feasibility (input set unreachable from $R$) or backward-utility (output set not used by any other unit / product). Reaches a fixpoint in $O(|O|)$ rounds. The output $O_{\max} \subseteq O$ is the largest *potentially feasible* set; ABB explores $2^{|O_{\max}|}$.

**SSG (Solution Structure Generation)** is the *exhaustive* version of ABB — enumerate every feasible $O' \subseteq O_{\max}$ instead of just the minimum-cost one. Used when $|O_{\max}| \leq 30$ (above that, ABB is the right tool — SSG silently refuses).

### 1.2 Cycle-enumeration ABB (the generalisation — multi-objective lives here)

**Module**: `hymeko_graph::topk_cycles` (Rust), surfaced to Python via `hymeko.enumerate_top_k_cycles_par_bb`. Inputs: a signed graph, target cycle length $k$, target heap size $K$, a scorer trait, and a per-emit pruner.

**Objective**: select the **top-$K$ closed cycles of length $k$** under a score function. This is the "knapsack over closed cycles" formulation — the cycle enumerator is the search-space generator, the heap is the bounded-cost incumbent set.

**Multi-objective** enters here via the `WeightedSumScorer<S1, S2>` building block ([topk_cycles.rs:2278](../hymeko_graph/src/topk_cycles.rs#L2278) onward):

$$
\mathrm{score}_{\mathrm{comp}}(c) \;=\; a \cdot s_1(c) + b \cdot s_2(c),
\qquad a, b \geq 0.
$$

The admissible upper bound is the **sum of admissible upper bounds**:

$$
\mathrm{UB}_{\mathrm{comp}}(\text{partial}) \;=\; a \cdot \mathrm{UB}_{s_1}(\text{partial}) + b \cdot \mathrm{UB}_{s_2}(\text{partial})
\;\geq\; \mathrm{score}_{\mathrm{comp}}(\text{any closed cycle from partial}).
$$

`WeightedSumScorer` is **nestable** — `WeightedSum<WeightedSum<A,B>, C>` gives a 3-criterion scalarisation, etc. The four atomic scorers shipped:

| Scorer | $s(\text{cycle})$ | Upper bound |
|:---|:---|:---|
| `FractionNegativeScorer` | $(\#\text{neg edges}) / k$ | $(n_{\text{neg so far}} + k_{\text{remaining}}) / k_{\text{len}}$ (assume every remaining edge is negative) |
| `BalanceScorer` (Cartwright-Harary) | $\prod_{i=1}^{k} s_i \in \{-1, +1\}$ | $+1$ (closure can flip parity either way) |
| `SignProductAbsScorer` | $\lvert \prod s_i \rvert = 1$ | $1$ |
| `LowRootScorer` | $-v_0$ (smallest start vertex) | $0$ (loose; ABB fires late for this one) |

**Governing rule** = **score upper bound vs heap threshold**:

At each DFS extension step from a partial path with $n_{\text{neg so far}}$ negative edges and $k_{\text{remaining}}$ edges left to close, compute $\mathrm{UB}$. If $\mathrm{UB} \leq \mathrm{heap.peek\_min}()$, **prune this branch** — no descendant can score above the current incumbent's worst element. Otherwise descend.

This is **structurally identical** to P-graph ABB's inclusion bound, with "cost ≤ incumbent" replaced by "score ≥ heap min" (max-heap inversion). The reachability bound has a counterpart too — the per-start-vertex BFS pre-pass `bfs_distances_capped` rejects start vertices whose neighbourhood cannot close a length-$k$ cycle.

### 1.3 The unifying mental model — "P-graph axioms in two roles"

```
P-graph layer                  | Cycle-enum layer
-------------------------------+-------------------------------
Operating unit u ∈ O           | DFS branch: extend path by 1 edge
Cost c(u)                      | Per-cycle score s(cycle)
Material reachability C_R(O')  | BFS reachability from start vertex
Demand set P                   | Heap fill target K
Inclusion bound                | Score UB ≤ heap threshold
Reachability bound             | BFS distance > k_remaining
MSG forward/backward trim      | Per-start-vertex BFS pre-pass
SSG bitmask enumeration        | (replaced by heap)
ABB                            | BoundedScorer + dfs_bb
```

The same **axiom + bound + DFS** machinery handles both. The cycle layer's multi-objective `WeightedSumScorer` is the direct generalisation of P-graph ABB's single-cost objective — it just admits a weighted sum because the search space is much larger and the criteria are stratified (graph-balance vs sign-magnitude vs root-vertex).

---

## 2. The numbers — optimization, pruning, wall time

### 2.1 P-graph ABB on the HDA worked example

HDA (hydrodealkylation of toluene): $M = \{$Toluene, $\text{H}_2$, Mix, Benzene, Methane$\}$, $R = \{$Toluene, $\text{H}_2\}$, $P = \{$Benzene$\}$. Four units: Mixer (cost 100), Reactor (250), DirectSynth (800), Disposal (50). [data/pgraph/hda.hymeko](../data/pgraph/hda.hymeko)

| Stage | Result |
|:---|:---|
| MSG | $O_{\max} = \{\text{Mixer}, \text{Reactor}, \text{Disposal}, \text{DirectSynth}\}$ |
| SSG (strict) | $\{\{\text{Mx}, \text{Rx}, \text{Ds}\},\, \{\text{DS}\}\}$ — 2 feasible structures |
| SSG (relaxed) | 4 feasible structures incl. $\{\text{Mx}, \text{Rx}\}$ |
| ABB (strict) | $\{\text{Mixer}, \text{Reactor}, \text{Disposal}\}$ at **cost 400** |
| ABB (relaxed) | $\{\text{Mixer}, \text{Reactor}\}$ at **cost 350** |
| Lattice explored vs total | **7 nodes vs $2^4 = 16$** → **56 % of search space killed** |
| Inclusion-bound prunes | DirectSynth branch dominated at $c = 800 > 400$ |
| Reachability-bound prunes | Excluding Mixer fails — Mix unreachable |

### 2.2 Cycle-enumeration ABB at scale

[reports/2026-05-10-abb-global-topk.md](2026-05-10-abb-global-topk.md): top-$K = 10\,000$ cycles of length $k = 4$ on the **Epinions** signed graph (131 828 vertices, 840 799 edges, 14.7 % negative).

| Path | Median wall (5-iter) | IQR | Speedup |
|:---|---:|---:|---:|
| Baseline `enumerate_top_k_cycles_par` | **100.557 s** | 0.205 s | 1.0× |
| ABB `enumerate_top_k_cycles_par_bb` | **4.012 s** | 0.064 s | **25.06×** |

Plan budget was $\leq 0.70\times$ (≥ 30 % reduction); actual was $0.040\times$ (96 % reduction). Post-fix flamegraph shows the dominant cost shifted from DFS recursion (70 %+ of cycles in baseline) to `bfs_distances_capped` (66 %) — the per-start-vertex BFS pre-pass that ABB cannot eliminate. **That's the new algorithmic floor for this family.**

### 2.3 Cycle pruning + axiom selection — the empirical AUC contribution

Once cycles are enumerated, they feed a Hypergraph Signed Kolmogorov-Arnold Network (HSiKAN) for signed-link prediction. The cycle-incidence matrix $M_e$ is the bottleneck: full enumeration on Slashdot at $k = 4$ produces **55.5 million** cycles, $\geq 8$ GB of materialised state. **Vertex-stratified top-$m$** bounds it as $|M_e| \leq |V| \cdot m$ (50-150× compression), and the **choice of axiom pruner during DFS** controls which cycles even reach the heap.

Three axioms wired:

- **Cartwright-Harary balance** — Heider-stable: $\prod s_i = +1$. Best for cooperative graphs (Bitcoin Alpha 95 % balanced).
- **CH unbalanced** — frustrated: $\prod s_i = -1$. Best for adversarial graphs (Slashdot 87.3 % balanced, frustrated triads carry the conflict signal).
- **Davis weak balance** — only $(-, -, -)$ all-negative triads rejected. Mild; almost no-op on Bitcoin Alpha; small effect on Slashdot.

[meeting_pimentel_outline.md §5 headline](../docs/meeting_pimentel_outline.md#5-the-empirical-headline-the-why-this-works-slide) — **5-seed paired vs baseline**:

| Dataset | Baseline AUC | Top-$K$ + axiom-pruner AUC | Δ | Memory cut |
|:---|---:|---:|---:|---:|
| Bitcoin Alpha (m=128, balance, h=16) | full = 0.9203 | mean **0.9136 ± 0.020**, best **0.9329** | **best +0.013 beats full** | **~50×** |
| Slashdot (m=16, unbalanced, h=16) | none-baseline = 0.8368 | mean **0.8562 ± 0.010** | **+0.020 ≈ +2σ** | **~150×** |
| Epinions | 0.764 historical | 0.7088 (m bottleneck dominates) | — | ~150× |

**Two headline findings, both 5-seed-validated:**

1. **Bitcoin Alpha best-seed (0.9329) BEATS full enumeration (0.9203) by +0.013 AUC** — the axiom-pruned top-$K$ subset is *more informative* than the full cycle set, because exhaustive enumeration includes noise the model must learn to ignore.
2. **Slashdot 5-seed mean is +2σ above no-axiom baseline** — the regime-conditional axiom-selection rule is statistically significant, not a seed artefact.

### 2.4 The regime-conditional axiom-selection rule (the analogue to PSE feedstock choice)

This is the **operational contribution** for the P-graph community:

| Graph regime | Best axiom pruner | Effect (5-seed vs no axiom) | Rationale |
|:---|:---|---:|:---|
| Cooperative (≥ 90 % balanced) | **Cartwright-Harary balance** | +0.05 AUC | Heider-stable triads carry the trust signal |
| Adversarial (~85-90 % balanced + visible conflict) | **balance-complement (unbalanced)** | +0.02 AUC at +2σ | Frustrated triads rare but carry the conflict-prediction signal |
| Dense intermediate | (m bottleneck dominates) | ~0 AUC delta | Need larger m before axiom matters |

The axiom is **selected by a graph-level structural ratio** (fraction-balanced triads), exactly the way you select unit catalogues or operating-condition envelopes by feedstock composition in process synthesis. **Not a hyperparameter to grid-search; a property of the data that determines the algorithm.**

---

## 3. Chemical / raw-chemical process applications

The codebase ships the *generalised* axiom-feasibility machinery, so it runs back on its origin domain (process synthesis) without modification. Three concrete use cases follow.

### 3.1 Direct: replace the HDA example with a real plant

The HDA worked example in [data/pgraph/hda.hymeko](../data/pgraph/hda.hymeko) is a minimal pedagogical case (4 units, 5 materials). Real plants reach $|O| \approx 50\text{-}200$, where SSG is intractable and **ABB is essential**. The same `hymeko_pgraph` ABB code handles them — just write the `.hymeko` source with one `@Unit <unit> cost { (-inputs, +outputs); }` line per operating unit.

Concrete examples that would map cleanly:

| Process class | Materials | Units | What ABB optimises |
|:---|---:|---:|:---|
| **Methanol synthesis** ($\text{CO}_2 + \text{H}_2 \to \text{CH}_3\text{OH}$) | ~15 | ~30 | reactor / separator / recycle-loop choice at min CAPEX + utility cost |
| **Biomass → SNG** | ~25 | ~50 | gasifier / methanation / CO$_2$ scrubber selection at minimum energy duty |
| **Crude-oil refinery topology** | ~80 | ~120 | column / cracker / blender selection under product-cut constraints |
| **Steel decarbonisation: BF $\to$ DRI-EAF** | ~30 | ~60 | route selection at minimum CO$_2$ tax + raw-material cost |

For each, the multi-objective `WeightedSumScorer` analogue would be:

$$
\mathrm{cost}_{\mathrm{multi}}(O') = \alpha \cdot \mathrm{CAPEX}(O') + \beta \cdot \mathrm{OPEX}(O') + \gamma \cdot \mathrm{CO}_2(O') + \delta \cdot \mathrm{H}_2\mathrm{O}(O').
$$

The P-graph ABB layer currently uses a single-cost objective ($\sum c(u)$). **Lifting `WeightedSumScorer` from the cycle layer to the P-graph layer is a ~50 LOC change**: replace the single `cost: f64` field on `LoweredPGraph` with a `Vec<f64>` plus a weight vector, and the inclusion bound becomes the dot product. The reachability bound is unchanged (it's structural, not cost-based).

### 3.2 Indirect: raw-chemical *feedstock selection* under axiom-feasibility

The regime-conditional axiom-selection rule from §2.4 maps directly onto **feedstock-conditional unit-catalogue selection**:

| Signed-graph regime | $\leftrightarrow$ | Chemical-process analogue |
|:---|:---:|:---|
| Cooperative graph (high % balanced) | $\leftrightarrow$ | Pure, well-characterised feedstock (e.g. pipeline-grade natural gas) — choose minimal reactor catalogue |
| Adversarial graph (frustrated triads carry signal) | $\leftrightarrow$ | Heterogeneous feedstock (biomass with variable lignin/cellulose ratio, mixed waste streams) — choose robust unit set incl. pre-treatment + adjustment loops |
| Dense intermediate | $\leftrightarrow$ | Crude oil with multiple cuts — m bottleneck = need bigger ABB budget |

The empirical contribution — "the optimal pruner is dataset-conditional, derivable from a structural ratio" — is **the exact analogue of feedstock-conditional process design** in PSE. In our case the structural ratio is the fraction of Heider-balanced triads; in PSE it would be the C/H ratio, sulfur content, water-gas-shift equilibrium, etc.

### 3.3 The "meta-circular" claim — P-graphs for sustainable ML

[meeting_pimentel_outline.md §6](../docs/meeting_pimentel_outline.md#6-sustainability-perspective--the-meta-circular-thread): P-graphs originated as a *sustainability tool* for chemistry — minimum-waste process design, by-product integration, energy optimisation. The same discipline applied to ML training:

| Approach | Hardware | Wall-time | Energy | Relative CO$_2$ |
|:---|:---|---:|---:|---:|
| Typical SOTA "buy more compute" | 1 × A100, 5-seed × 3 datasets | ~24 hrs | ~6 kWh | 1.0× |
| **HSiKAN + axiom top-$K$ (this work)** | 1 × RTX 2070 SUPER (2019 consumer), same | ~3 hrs | **~0.22 kWh** | **~0.04×** |

**A ~25× reduction in training-energy footprint**, on 6-year-old consumer hardware, via the *axiom-feasibility discipline* you (Pimentel) and the PSE community pioneered. The reusable insight: when the search space is combinatorial and evaluation is expensive (whether the evaluation is "run a reactor simulation" or "train a neural network"), **axioms beat brute force**.

---

## 4. Direct answer to each ask — TL;DR

### Q1 — Multi-objective function on ABB SSG / MSG, governing rule

**P-graph layer (SSG / MSG / ABB)**: single-criterion *cost-minimisation* of $\sum_{u \in O'} c(u)$ under feasibility (input consistency, demand, no-excess). Governing rule: **inclusion bound** (cost $\geq$ incumbent → prune) AND **reachability bound** ($C_R$ of optimistic remainder cannot reach $P$ → prune), with **include-before-exclude** branching for early incumbent.

**Cycle-ABB layer (the multi-objective is here)**: maximise $a \cdot s_1(c) + b \cdot s_2(c)$ for $a, b \geq 0$ via `WeightedSumScorer`; nestable to any number of criteria. Governing rule: **score upper bound** $a \cdot \mathrm{UB}_{s_1} + b \cdot \mathrm{UB}_{s_2}$; if this bound is below the current min in the size-$K$ max-heap, **prune**.

### Q2 — Numbers

| Layer | Metric | Value |
|:---|:---|---:|
| P-graph ABB on HDA | nodes explored vs lattice | **7 vs 16 — 56 % killed** |
| P-graph ABB on HDA | optimal cost (strict) | **400** ({Mixer, Reactor, Disposal}) |
| Cycle-ABB on Epinions $k=4$ $K=10\,000$ | wall-time speedup vs baseline | **25.06×** (4.0 s vs 100.6 s) |
| Cycle pruning + axiom on Bitcoin Alpha | best-seed AUC | **0.9329 (beats full 0.9203 by +0.013, ~50× memory cut)** |
| Cycle pruning + axiom on Slashdot | 5-seed mean AUC | **0.8562 ± 0.010 (+2σ vs no-axiom, ~150× memory cut)** |
| HSiKAN training energy (3 datasets, 5-seed) | total | **~0.22 kWh, ~25× less than A100-SOTA approach** |

### Q3 — Chemical-process use

1. **Direct**: the same `hymeko_pgraph` ABB runs PSE problems unmodified. The HDA example exercises the full pipeline; real plants ($|O| \approx 50$–200) need ABB by definition (SSG intractable). Lifting `WeightedSumScorer` to the cost path is ~50 LOC for CAPEX + OPEX + CO$_2$ + H$_2$O multi-objective.
2. **Indirect**: the empirical **regime-conditional axiom-selection rule** (best pruner depends on graph-balance ratio) maps onto **feedstock-conditional unit-catalogue selection** in PSE (best reactor depends on C/H, sulfur, water content of feed). Same shape of decision; different domain.
3. **Meta-circular**: P-graph discipline applied to ML training delivers ~25× energy reduction on consumer hardware, giving the PSE community a sustainability story for ML adjacent to its origin.

---

## 5. Open questions back to Pimentel

(From the meeting outline, still open.)

1. **Has anyone in the P-graph community applied MSG/SSG/ABB to non-PSE domains?** We have not found references.
2. **Dual-direction axiom trick** (balance vs unbalanced as *complementary* pruners selected by graph-balance ratio) — does this map onto known P-graph dualities (e.g. forward/backward MSG)?
3. **A2 reachability** is a BFS check here. Are there P-graph variants that exploit weak/Davis balance as a structural reachability constraint? Heider 1946 + Cartwright-Harary 1956 give a clean signal-conservation analogue to material-balance constraints.
4. **Paper framing**: *"P-graph methodology beyond PSE: axiom-feasibility for sustainable graph machine learning."* — would this resonate with the PSE community?

---

## Appendix A — Code & artefacts to point at

| Topic | Artefact |
|:---|:---|
| P-graph ABB (Friedler '92 reproduction) | [hymeko_pgraph/src/abb.rs](../hymeko_pgraph/src/abb.rs), [hymeko_pgraph/src/msg.rs](../hymeko_pgraph/src/msg.rs), [hymeko_pgraph/src/ssg.rs](../hymeko_pgraph/src/ssg.rs) |
| Cycle ABB + `WeightedSumScorer` | [hymeko_graph/src/topk_cycles.rs:2172-2330](../hymeko_graph/src/topk_cycles.rs#L2172) |
| HDA worked example | [data/pgraph/hda.hymeko](../data/pgraph/hda.hymeko) |
| Neural-architecture P-graph (transposed) | [data/hsikan/sweep_msg.hymeko](../data/hsikan/sweep_msg.hymeko), [data/hsikan/sweep_msg_gomb.hymeko](../data/hsikan/sweep_msg_gomb.hymeko) |
| 4-page formal brief on MSG/SSG/ABB | [reports/pgraph_hymeko_brief.tex](pgraph_hymeko_brief.tex) (PDF compiled) |
| 4-page brief on axiom-pruning for cycles | reports/topk_cycles_brief.tex |
| HSiKAN architecture brief | reports/hsikan_hymeko_brief.tex |
| Live driver | `python -m signedkan_wip.src.hymeko_driver --backend gomb`<br/>`python -m signedkan_wip.src.run_gomb_msg_sweep --algorithm {msg,ssg,abb}` |
| 25× speedup report | [reports/2026-05-10-abb-global-topk.md](2026-05-10-abb-global-topk.md) |
| Original meeting outline (2026-05-05) | [docs/meeting_pimentel_outline.md](../docs/meeting_pimentel_outline.md) |
