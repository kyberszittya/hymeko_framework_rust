# Structural entropy over hierarchical HyMeKo hypergraphs

**Status:** design note, 2026-04-21. Step 1 of the 5-step entropy hot-swap plan
(spec: `input/entropy_hypergraph_pytorch_spec.md` §5.4; surface rewrite:
`docs/torch_backend_views.md`).

**Scope.** Defines the IR-side structural-entropy metric that the
`hymeko_query::entropy` module computes as a pure walk over the canonical
hierarchical hypergraph IR. This is distinct from — and a sibling of — the
PyTorch-side `h_struct_v1..v10` family in `ehk_torch/entropy/structural.py`
(§2.3 of the spec). The two are not interchangeable: the Python variants
consume runtime sparse incidence tensors `(B, σ)` and must match Csaba's
existing SISY-2026 parity set; the IR metric consumes the compiled
hierarchical IR and is the signal we run *before* we ever build a `B`.

---

## What we are measuring (and why)

The hot-swap capability (§5.4) rebuilds a network when structural entropy
signals that the current architecture is over- or under-specified for the
task. The trigger runs on the HyMeKo IR because:

1. **Pre-codegen.** We want to inspect candidate architectures cheaply —
   before emitting `.py`, before instantiating `nn.Parameter`, before any
   tensor allocation. The IR is where the structure lives.
2. **Per-scope.** The hierarchical IR (`docs/torch_backend_views.md`) gives
   us per-layer sub-hypergraphs as first-class entities. A layer whose
   internal factor graph is near-uniform in arity is a different refactor
   target from a dataflow whose top-level skeleton is near-uniform.
   Entropy computed *per hypervertex scope* gives the rewrite proposer
   (step 3 of the plan) a place to cut.
3. **Determinism.** The IR is content-addressable (Proposition 2 of the SMC
   paper). A deterministic entropy walk gives us a hash-addressable
   structural fingerprint — same IR, same hash, same entropy, every time.

---

## Definition

Given a scope `S` (a `DeclId` whose body is walked; the module root for the
global metric), let

- `V(S)` = `{d ∈ children*(S) : kind(d) = Node}` — hypervertices in scope
- `E(S)` = `{d ∈ children*(S) : kind(d) = Edge}` — hyperedges in scope
- `I(e)` = multiset of `(target, sign)` pairs across all `HyperArc` children
  of `e` — the signed incidence of edge `e`

`children*` walks the decl tree *without descending through hypervertex
bodies*: a layer hypervertex `layer_0` is in `V(module_root)`, but
`layer_0`'s own internal neurons and factors are not — they belong to
`V(layer_0)`. This is the per-scope locality that makes compositional
entropy well-defined.

We define three components, each a standard Shannon entropy in nats over a
discrete distribution derived from the structure:

### H_arity — hyperedge-arity entropy

```
P_arity(k) = |{e ∈ E(S) : |I(e)| = k}| / |E(S)|
H_arity(S) = - Σ_k P_arity(k) · ln P_arity(k)
```

Zero when every hyperedge has the same arity (a regular graph pattern); max
`ln |E(S)|` when every hyperedge has a distinct arity. Reading:
heterogeneity of the hyperedge shapes in this scope.

### H_sign — mean per-edge sign entropy

For each `e ∈ E(S)` with `|I(e)| > 0`, let `p_+(e), p_-(e), p_0(e)` be the
fraction of `+`, `−`, `~` incidences in `I(e)`. Then

```
H_sign(e) = - Σ_s∈{+,-,0} p_s(e) · ln p_s(e)       (0 · ln 0 := 0)
H_sign(S) = mean_{e ∈ E(S)} H_sign(e)
```

Zero when every hyperedge uses a single sign (a pure factor or a pure
dataflow-through edge); positive when incidences mix roles (a factor with
a neutral port-witness incidence alongside `+` sources and `−` sinks — the
lynchpin shape from `docs/torch_backend_views.md`). Reading: average
role-mixing of incidences in this scope.

### H_degree — node-degree entropy

For each `v ∈ V(S)`, let `deg(v)` count incidences whose `target` is `v`,
across all `I(e)` for `e ∈ E(S)`. Let

```
P_deg(d) = |{v ∈ V(S) : deg(v) = d}| / |V(S)|
H_degree(S) = - Σ_d P_deg(d) · ln P_deg(d)
```

Zero when every vertex participates in the same number of hyperedges;
positive when connectivity is heterogeneous. Reading: irregularity of
vertex usage in this scope.

### H_struct — the aggregate

```
H_struct(S) = (H_arity(S) + H_sign(S) + H_degree(S)) / 3
```

Equal weights by default. The three components are returned separately so
downstream consumers can reweight without recomputing; the aggregate is for
quick ranking.

### Edge cases (required for numerical stability)

- `|V(S)| = 0` or `|E(S)| = 0` → every component is `0.0`. No NaN, no Inf.
- A hyperedge `e` with `|I(e)| = 0` (degenerate) is counted in the arity
  distribution at `k = 0`, contributes `0` to `H_sign`, and contributes
  nothing to any `deg(v)`.
- `0 · ln 0` is taken as `0` everywhere (limit convention).

These match the §8 "Numerical stability" acceptance criterion of the spec
(empty hypergraph, single hyperedge, disconnected components must not
return NaN or Inf).

---

## Worked example — `data/nn/simple_net.hymeko`

Running `compute_entropy_hierarchical` on the compiled IR of
`simple_net.hymeko` produces three scopes of interest.

### Module scope (outer dataflow)

`V` = `{x, h, y, layer_0, layer_1}` (5 hypervertices at module scope —
three tensors plus two layer hypervertices).
`E` = `{flow_0, flow_1}` (two dataflow hyperedges).

- `I(flow_0) = {(+ x), (~ layer_0), (− h)}` — arity 3, signs `{+, ~, −}`.
- `I(flow_1) = {(+ h), (~ layer_1), (− y)}` — arity 3, signs `{+, ~, −}`.

- `P_arity = {3: 1.0}` → `H_arity = 0` (both edges arity 3).
- Every edge has `p_+ = p_- = p_0 = 1/3` → `H_sign(e) = ln 3 ≈ 1.0986`;
  `H_sign = ln 3`.
- `deg(x) = 1`, `deg(h) = 2`, `deg(y) = 1`, `deg(layer_0) = 1`,
  `deg(layer_1) = 1`. `P_deg = {1: 0.8, 2: 0.2}` →
  `H_degree = -(0.8 ln 0.8 + 0.2 ln 0.2) ≈ 0.5004`.

Aggregate: `H_struct ≈ (0 + 1.0986 + 0.5004) / 3 ≈ 0.533` nats.

### `layer_0` scope (the 3-5 hidden block)

`V` = 3 input ports + 5 hidden neurons + 5 output ports + 3 attribute
decls (`kernel: + <isa> ggk.bspline { degree 3; n_knots 8; }` — the
inline-isa spec and its two `field value` attribute statements each lower
to Node decls) = 16.
`E` = 5 inner factors, each arity 5 with signs `(+, +, +, −, ~)`.

- `H_arity = 0` (all 5 factors arity 5).
- Per-edge sign distribution: `p_+ = 3/5`, `p_- = 1/5`, `p_0 = 1/5` →
  `H_sign(e) = -(0.6 ln 0.6 + 0.2 ln 0.2 + 0.2 ln 0.2) ≈ 0.9503`;
  `H_sign = 0.9503`.
- Every input port has `deg = 5` (appears in every factor); every hidden
  neuron and output port has `deg = 1`; the 3 attribute decls have
  `deg = 0` (not referenced by any incidence). `P_deg = {0: 3/16,
  1: 10/16, 5: 3/16}` → `H_degree ≈ 0.9215`.

Aggregate: `H_struct ≈ (0 + 0.9503 + 0.9215) / 3 ≈ 0.624` nats.

### `layer_1` scope (the 5-2 output block)

Similar shape: `V` = 5 input ports + 2 output neurons + 2 output ports
+ 3 attribute decls (`kernel: + <isa> ggk.rbf { n_centres 16; }` — the
isa spec and one attribute) = 12.
`E` = 2 factors, each arity 7 with signs `(+, +, +, +, +, −, ~)`.

- `H_arity = 0`.
- Per-edge: `p_+ = 5/7`, `p_- = 1/7`, `p_0 = 1/7` →
  `H_sign(e) ≈ 0.7963`; `H_sign ≈ 0.7963`.
- Every input port has `deg = 2`; every inner neuron and output port
  has `deg = 1`; the 3 attribute decls have `deg = 0`. Plugging into the
  histogram gives `H_degree ≈ 1.078`.

Aggregate: `H_struct ≈ (0 + 0.7963 + 1.078) / 3 ≈ 0.625` nats.

### Reading

Both layer scopes land near `H_struct ≈ 0.62`, but the mix of components
differs: `layer_0` carries more sign-mixing (wider factor fan-in relative
to the inner-neuron alphabet), `layer_1` carries more degree-irregularity
(ports shared across more factors). The outer dataflow scope is the
noisiest in sign — `H_sign = ln 3 ≈ 1.099`, the theoretical max for the
three-sign alphabet — but has smaller `H_degree` (regular participation:
one use per tensor, one role per layer). That is exactly the signature of
a fresh dataflow skeleton; as the network grows the `~`-role becomes the
dominant sign and `H_sign` drops.

The integration test
`hymeko_query/tests/test_entropy.rs::outer_scope_matches_design_note`
pins these numbers against the real compiled IR — if the lowering ever
changes (e.g., attribute decls stop being Node-typed), both the numbers
above and the test assertion must move together.

---

## Contract for the Rust implementation

### API shape

```rust
// hymeko_query/src/entropy.rs

pub struct StructuralEntropy {
    pub h_arity:  f64,
    pub h_sign:   f64,
    pub h_degree: f64,
    pub h_total:  f64,   // (h_arity + h_sign + h_degree) / 3.0
    pub n_vertices: usize,
    pub n_edges:    usize,
}

pub fn compute_entropy(ir: &Ir, scope: DeclId) -> StructuralEntropy;

pub fn compute_entropy_hierarchical(
    ir: &Ir,
) -> Vec<(DeclId, StructuralEntropy)>;
```

- `compute_entropy` walks only *direct* decl children of `scope`, not
  descending into hypervertex bodies. Pass `DeclId::NONE` (or the module
  root) for the outermost scope.
- `compute_entropy_hierarchical` returns one entry per scope that has at
  least one `Edge` child — i.e., the module root plus every hypervertex
  whose body contains inner hyperedges. Order: pre-order by `DeclId`.

### Guarantees

- **Pure.** No mutation of the `Ir`, no allocation beyond the return
  value. Safe to call from any read-path (rewrite proposer, `hymeko query
  --entropy`, a future `reinfer_structure_and_rebuild`).
- **Deterministic.** Same `Ir` → bit-identical `StructuralEntropy`.
  Iteration order follows `DeclId`; hash maps used only for the
  degree-count accumulator and keyed by `DeclId` (stable).
- **Numerically stable.** Per §"Edge cases" above. Unit tests cover
  `|V|=0`, `|E|=0`, a single hyperedge, and a two-layer fixture.
- **Complexity.** `O(|V(S)| + |E(S)| + Σ_e |I(e)|)` per scope —
  single-pass arity + sign accumulator, one pass for degree counts. For
  the hierarchical variant: `O(|V| + |E| + Σ_e |I(e)|)` over the whole IR.

### What we are *not* doing here

- **Not** a PyTorch entropy estimator. This walks the IR; `ehk_torch` does
  not enter.
- **Not** the ten-variant `h_struct_v1..v10` family. That ships on the
  Python side and must match Csaba's SISY-2026 parity set. The Rust
  metric is the *IR-side sibling*: it measures the compiled structure,
  not the runtime tensors. When the hot-swap fires, both fire: the Rust
  side picks a scope to rewrite (coarse, cheap), the Python side scores
  the resulting `B` (fine, reference-matched).
- **Not** the rewrite proposer. That is step 3 of the plan; it *consumes*
  this metric.

---

## How the 5-step plan consumes this

From `project_pytorch_backend.md`:

1. **Entropy metric in `hymeko_query` as a pure IR walk** — *this note is
   the spec for step 1.*
2. **Compute on IR** — CLI + library exposure. `hymeko query --entropy
   <file.hymeko>` emits per-scope `StructuralEntropy` as JSON; library
   consumers call `compute_entropy_hierarchical` directly.
3. **Split-layer rewrite proposer** — reads the per-scope metric; the
   scope with the highest `h_sign` (or a user-chosen component) is the
   candidate for refactoring. K-means (k=2) on incidence-row signatures
   proposes the split.
4. **Regen via template** — proposer emits a rewritten IR; `torch_*`
   transforms re-emit `.py` unchanged.
5. **Python-side weight transfer** — `from_hmk(path, recompile=True)` +
   the compatible-subset rule from the spec.

The metric defined here is therefore the *signal* on which the hot-swap
operates. Everything downstream consumes a `Vec<(DeclId,
StructuralEntropy)>` and acts on it.
