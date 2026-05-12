# Experiment Design: HyMeKo vs. GNN on Hypergraph-Native Tasks

---

## 1. Central Hypothesis

GNNs bounded by the 1-WL test cannot distinguish certain non-isomorphic hypergraph structures. HyMeKo's query engine operates natively on the incidence representation without projection loss. The experiment should expose this gap empirically, not just theoretically.

Two sub-hypotheses:

- **H1 (Expressiveness):** HyMeKo queries correctly classify structural properties that 1-WL–equivalent GNNs conflate.
- **H2 (Efficiency):** For sparse, structured hypergraphs (e.g., robotic kinematic chains, FSMs), HyMeKo query latency is competitive with or better than GNN inference under equivalent input sizes.

---

## 2. Task Definition

**Primary task:** Binary structural property prediction on hypergraphs.

Choose properties with known WL-hardness:

| Property | WL-hard? | Notes |
|---|---|---|
| k-regularity of hyperedges | Yes (for k≥3) | Distinguishes clique vs. true hyperedge |
| Presence of a specific signed sub-hypergraph | Yes | Direct HyMeKo query |
| Component count after hyperedge removal | Yes | Connectivity in projection is lossy |
| Arity distribution of hyperedges | No | Baseline sanity check — both should solve |

Use at least one WL-hard and one WL-easy property to validate that baselines work where they should.

---

## 3. Data Generation Pipeline

```
HyMeKo DSL (.hko) → hymeko_core HIR → serialization → {GNN input, HyMeKo input}
```

### 3.1 Generator

Write a Rust generator in `hymeko_core` (or a Python wrapper over the compiled HIR) that produces synthetic `.hko` files with controlled parameters:

```
N_v  ∈ {16, 32, 64, 128}        # vertex count
N_e  ∈ {N_v/2, N_v, 2*N_v}      # hyperedge count
arity k ~ Uniform(2, K_max)      # K_max ∈ {3, 5, 8}
signed ∈ {true, false}           # G-SPHF incidence signs
label = f(structure)             # deterministic ground truth
```

For each (N_v, N_e, K_max) triple, generate 1000 samples. Total: ~27,000 samples across 27 configurations. 80/10/10 split.

Ground-truth labels are computed analytically at generation time — no approximation.

### 3.2 HyMeKo Branch

Directly load HIR; execute a compiled `.hko` query pattern. The query IS the classifier. No learned parameters.

### 3.3 GNN Branch — Two Projections

**Clique expansion** (standard): hyperedge $e = \{v_1,...,v_k\}$ → complete graph $K_k$ on same vertices. Edge weight = 1/k to normalize arity.

**Star expansion** (bipartite): introduce one auxiliary node per hyperedge. Edges: $v_i \leftrightarrow e_j$ for all $v_i \in e_j$. Node features carry arity.

Both projections discard multi-membership structure to varying degrees. This is the structural information loss the experiment is designed to expose.

---

## 4. Baselines

| Model | Input | Notes |
|---|---|---|
| GCN (2-layer) | clique expansion | Kipf & Welling |
| GAT (2-layer) | clique expansion | attention doesn't recover arity |
| GCN | star expansion | bipartite, arity preserved as feature |
| HGNN | incidence matrix B directly | Feng et al. 2019 — fairer baseline |
| AllSetTransformer | set-based hyperedge encoding | strongest hypergraph GNN baseline |
| HyMeKo query | HIR | rule-based, zero parameters |

AllSetTransformer is the critical baseline — it is the strongest current hypergraph NN. If HyMeKo beats it on WL-hard properties at zero parameters, that is the core publishable claim.

---

## 5. Input Feature Construction from HyMeKo HIR

Extract the following from HIR for all GNN baselines:

```python
# From hymeko_core HIR serialization
B       # incidence matrix (N_v x N_e), signed if G-SPHF
X_v     # vertex features: [degree, signed_degree, vertex_type_onehot]
X_e     # hyperedge features: [arity, weight, edge_type_onehot]
labels  # ground truth vector
```

For clique/star expansion, construct adjacency from B programmatically. All baselines receive identical information content — the difference is representation, not raw data.

---

## 6. Metrics

| Metric | Purpose |
|---|---|
| Accuracy / F1 | Primary task performance |
| Accuracy stratified by K_max | Does arity matter? GNNs degrade at high arity |
| Accuracy on WL-hard vs WL-easy properties | Expressiveness ablation |
| Inference latency (ms) vs N_v | Scalability |
| Parameter count | Model complexity |
| Projection information loss (ΔH) | Shannon entropy of hyperedge arity pre/post projection |

For the latency comparison: HyMeKo query time includes compilation. Measure separately — compile time (one-shot, amortized) and query time (per-instance). GNN: training time is separate from inference.

---

## 7. Ablations

1. **Signature ablation**: disable signs in G-SPHF incidence; measure accuracy drop. Establishes value of signed structure.
2. **Query complexity ablation**: simple path query vs. structural pattern query — shows HyMeKo scales O(match complexity), GNN scales O(layers × edges).
3. **Arity sweep**: fix N_v=64, vary K_max from 2 to 8. GNNs should degrade; HyMeKo should not.

---

## 8. Expected Outcome Map

```
Property type    │  GCN  │  HGNN  │  AllSet  │  HyMeKo
─────────────────┼───────┼────────┼──────────┼─────────
WL-easy          │  ✓    │  ✓     │  ✓       │  ✓
WL-hard, low k   │  ~    │  ~     │  ✓       │  ✓
WL-hard, high k  │  ✗    │  ~     │  ~       │  ✓
Signed structure │  ✗    │  ✗     │  ✗       │  ✓  (G-SPHF only)
```

The signed structure row is unique to the G-SPHF framework — no existing hypergraph GNN encodes oriented incidence signs natively.

---

## 9. Paper Mapping

| Venue | Scope |
|---|---|
| **SISY / AD&I** | Lightweight version — single WL-hard task, 3 baselines, efficiency table. Fits applied-systems framing. |
| **SMC Regular / MDPI Actuators** | Full version with signed ablation and AllSetTransformer comparison. G-SPHF signed row is the novel contribution hook. |

---

## 10. Implementation Roadmap

The tightest path to runnable code:

1. Extend `hymeko_core` with a `generate::synthetic` module (Rust) — outputs `.hko` + label JSON.
2. Serialize HIR to `networkx`-compatible format via a thin Python FFI or dump incidence/adjacency as `.npz`.
3. Use **PyTorch Geometric** for all GNN baselines — `HypergraphConv` is already in PyG for HGNN; AllSetTransformer has a reference implementation.
4. HyMeKo query classifier: deterministic Rust binary returning 0/1.
5. Benchmark harness in Python — `subprocess` call for HyMeKo, standard PyG eval loop for GNNs.

The only genuinely new code is the synthetic generator and the HIR→npz serializer. Everything else reuses existing infrastructure.

---

## Appendix: Notation Reference

| Symbol | Meaning |
|---|---|
| B | Incidence matrix, shape (N_v × N_e) |
| Bσ | Signed incidence matrix (G-SPHF) |
| N_v | Number of vertices |
| N_e | Number of hyperedges |
| k | Hyperedge arity |
| K_max | Maximum arity in generator |
| HIR | HyMeKo Intermediate Representation |
| WL | Weisfeiler–Leman graph isomorphism test |
| ΔH | Entropy difference (projection information loss) |
