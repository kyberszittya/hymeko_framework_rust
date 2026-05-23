# Concept: The IR

The HyMeKo IR is a **signed hierarchical hypergraph**. Three primary entity kinds:

| kind | what it is |
|---|---|
| **Node** (decl_kind = `Node`) | a vertex / typed entity (e.g. a robot link, a tensor, a cell, a chemical species) |
| **Edge** (decl_kind = `Edge`) | a hyperedge — groups multiple nodes / edges via signed arc-refs (e.g. a joint connecting two links, a dataflow op) |
| **HyperArc** (decl_kind = `HyperArc`) | a signed reference *from* an edge *to* a node or another edge (the `+target1` / `-target2` / `~op` refs in the source syntax) |

## In Rust

```rust
pub struct Ir {
    pub decl_nodes: Vec<DeclNode>,    // every named decl
    pub nodes:      Vec<NodeRecord>,  // subset that are pure nodes
    pub edges:      Vec<EdgeRecord>,  // subset that are edges (have arc refs)
    pub arcs:       Vec<ArcRecord>,   // signed arc-refs (one per `+target` / `-target` / `~op`)
}

pub struct DeclNode {
    pub name: SymId,        // interned in StringTable
    pub kind: DeclKind,     // Node | Edge | HyperArc
    pub parent: DeclId,     // None or scoping decl
    pub anno: Annotation,   // tags + comments
}

pub struct EdgeRecord {
    pub decl: DeclId,
    pub bases: Vec<SignedRef>,  // inheritance: `: lyr.signedkan_layer`
    pub arcs:  Vec<ArcId>,      // signed refs to other decls
}

pub struct ArcRecord {
    pub refs: Vec<SignedRefR>,   // each carries (sign: i8, target: DeclId)
}
```

## Why signed

A hyperedge's arc-refs carry signs `+1`, `-1`, or `~0` (rare). This lets one structure express:
- **Polarity** in signed graphs (trust / distrust)
- **Cycle orientation** in signed-cycle KAN aggregation
- **Producer / consumer** in P-graph processes (`-input` consumes, `+output` produces)
- **Dataflow direction** in NN architectures (`+x` flows in, `-y` flows out, `~op` is the operation)

The same signed-incidence matrix `M_e^{(k)}` shows up in all of these — that's the unifying abstraction.

## Why hierarchical

Decls have a `parent` field. A `description_name { context { ... } }` block creates a scope; nested scopes refine the type system locally without polluting the global namespace. Inheritance walks (`base.target()`) cross scope boundaries via the resolver.

## Building one programmatically

Most of the time you parse from a `.hymeko` file. To build directly in Rust (rare; tests and benchmarks):

```rust
let mut ir = Ir::default();
let mut strings = StringTable::default();
let foo_id = strings.intern("foo");
ir.decl_nodes.push(DeclNode {
    name: foo_id, kind: DeclKind::Node, parent: DeclId::none(),
    anno: Annotation::default(),
});
// ... etc
```

Prefer the parser. The IR is meant to be a translation target, not a hand-authored format.

## Tensor projections of the IR

For numerical / GPU work, the IR projects to:

- **2D incidence** (vertex × edge): `compile_clique_expansion` (in `hymeko_hre::engine::compile_clique_expansion_core`)
- **3D signed-incidence** (vertex × vertex × hyperedge): `compile_clique_tensor_expansion` (the same operator in 3D)
- **k-cycle structure tensor**: `enumerate_k_cycles_rs` (vertices `[n_cycles, k]`, signs `[n_cycles, k]`)

These are the inputs the HSiKAN forward passes consume.

## See also

- [Tensor decomposition](./tensor-decomposition.md) — what HOSVD does to these projections
- [Quickstart: Parse](../quickstart/01-parse.md) — first contact with an IR
