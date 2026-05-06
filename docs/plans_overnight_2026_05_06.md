# Overnight schedule — 2026-05-06

Three new research directions to schedule, in addition to the GPU
queue already running.

## Already in flight

| order | PID | what | cells | wall remaining |
|---|---|---|---:|---:|
| 1 (current) | 626616 | Slashdot direct-msg + entropy + walk-cycle 12-cell | 8 | ~120 min |
| 2 (queued) | 635532 | Phase A community-pruner 5-seed × 3 datasets | 15 | ~150 min |

## New scheduled work

### 3. Hypergraph convolution on visual datasets (Neocognitron-style)

`signedkan_wip/src/vision/neocog_hgnn.py` (built tonight, 200 lines).

- Architecture: 2-layer HGNN classifier
  - S₁: 5×5 receptive fields, stride 2 → first hyperedge layer
  - C₁: 8×8 pooled patches, stride 4 → second hyperedge layer
  - Real Feng et al. 2019 HGNN convolution: `D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ`
- Datasets: MNIST, FashionMNIST, optional CIFAR-10 (grayscale)
- Comparisons: HGNN vs MLP vs TinyCNN, all at hidden=32
- Goal: validate that hypergraph convolution with hand-engineered
  Neocognitron clustering can compete with CNNs on small image
  classification. **NOT** to beat CNNs on absolute accuracy — this
  is a methodological demonstration that the same hypergraph
  apparatus we used for signed graphs transfers to vision when the
  hyperedge structure encodes domain priors (here: spatial locality).

- Connection to today's findings: same MSG/SSG/ABB framework could
  pick the receptive-field structure (S-cell sizes, strides, layer
  depth) as a P-graph axiom-feasibility problem.

Smoke test: 926 params, forward OK on 2×1×28×28 → 2×10. Logits.

### 4. Per-layer Kochanek-Bartels parameter sweep

User's idea (2026-05-06 02:00): use different K-B (tension t,
continuity c, bias b) parameters at different depths in HSiKAN.

**Background**: K-B splines generalize Catmull-Rom by adding 3 free
parameters per control point:

  - **t** (tension): how much the curve is "stretched" toward
    control points; t=0 → Catmull-Rom, t=1 → linear (no overshoot)
  - **c** (continuity): C¹ continuity at control points; c=0 → smooth,
    c=1 → corner (kink at the point)
  - **b** (bias): direction the curve "leans"; b=0 → balanced

In HSiKAN, every layer's `KochanekBartelsActivation` gets a
learnable `tcb` parameter. **Currently all layers init to (0, 0, 0)**
i.e. behave identically to Catmull-Rom at start. The user's idea:
**initialize different depths with different (t, c, b)** so each
layer has a different inductive bias.

- Layer 0 (input → hidden): smooth, t=0, c=0 (= CR baseline)
- Layer 1 (hidden → hidden): higher tension t=0.3 (sharper curves)
- Layer 2 (hidden → output): higher continuity c=0.3 (preserves
  corners — useful for classification boundaries)

**Implementation**: needs an `init_tcb` argument added to
`KochanekBartelsActivation`. Then expose via env var:
`HSIKAN_KB_INIT_PER_LAYER="0,0,0;0.3,0,0;0,0.3,0"` (semicolon-
separated per-layer triples).

**Hypothesis**: a depth-varying spline geometry encodes a richer
function class than uniform Catmull-Rom. Different layers learn
different curve regimes; the optimizer doesn't have to flatten
their differences.

**Cost**: small, 3 extra params per (channel, knot) per layer. The
parameter count stays in the same order of magnitude.

**Schedule**: Phase 0 — add `init_tcb` arg to
`KochanekBartelsActivation`. Phase 1 — sweep on Bitcoin Alpha (cheap)
with single + per-layer init combos. Phase 2 — best config on
Slashdot/Epinions.

### 5. Triton kernels — candidate list

**Goal**: identify the cycle-aware aggregation hot loops where a
fused Triton kernel would meaningfully beat PyTorch's eager
execution. **Do NOT write Triton tonight** — this is a planning
document that lists the candidates.

| candidate | location | bottleneck? | est. lift |
|---|---|---|---:|
| Sparse `M_e` × edge embedding aggregation | `mixed_arity_signedkan._scatter_softmax` and friends | likely (used in attention path) | 2-5× |
| Catmull-Rom batched eval (`_kb_eval` / `_catmull_rom_eval`) | `splines.py` | secondary (already torch.compile-friendly) | 1.5-2× |
| Cycle DFS itself | `hymeko_graph::topk_cycles` | already rayon-parallel; Triton wouldn't help (CPU-bound) | n/a |
| Hypergraph convolution forward (S₁/C₁ in Neocog HGNN) | `vision/neocog_hgnn.py` | yes for large incidence matrices | 2-3× |
| Per-vertex top-m heap merge in cycle enum | Rust path | already optimal | n/a |

**Recommended first kernel**: the M_e attention-softmax path, because
that's where today's GPU OOMs happen. Triton would let attention
fit at m≥32 on 8 GB.

**Status**: documentation only tonight. Implementation is a 2-3 day
task per kernel.

## Schedule order

These run sequentially after the in-flight + already-queued
experiments. Each new direction has its own bash script:

```
626616  Slashdot direct-msg push (in flight)
635532  Phase A community-pruner test (queued)
[NEW]   Hypergraph + visual datasets (MNIST/FashionMNIST × {hgnn,mlp,cnn})
[NEW]   K-B per-layer init sweep (Bitcoin Alpha first, then Slashdot)
[FUTURE] Triton kernel implementation (multi-day effort, not tonight)
```

ETA all overnight done: ~6:00 AM, well before any meeting tomorrow.

## What we'll know by morning

1. Whether HSiKAN-Slashdot+entropy holds at 3+ seeds (multi-seed
   confirmation of the entropy reg lift).
2. Whether community-conditional axiom rescues Epinions where global
   axiom doesn't (the heterogeneous-graph test).
3. Whether HGNN with Neocognitron-style receptive fields beats MLP
   baseline on MNIST (vision transferability test).
4. Whether per-layer K-B init affects HSiKAN's expressiveness on the
   classic signed graphs.

By morning we have **either four positive results to add to the
paper, or four honest negative results that sharpen the story.**
