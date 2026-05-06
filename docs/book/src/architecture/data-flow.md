# Data flow: parse → IR → query → emit

Every HyMeKo invocation — CLI, Python, WASM, library — follows the same pipeline:

```
.hymeko source
     │
     ▼
┌──────────────┐
│  parser      │  lex + parse → syntax tree (parser/)
└──────────────┘
     │
     ▼
┌──────────────────────┐
│  resolution          │  name lookup, inheritance walks (hymeko_core::resolution)
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  IR builder          │  Ir { decl_nodes, nodes, edges, arcs } (hymeko_core::ir)
└──────────────────────┘
     │
     ├──── pred / query ────► QueryEngine matches (hymeko_query)
     │                                │
     │                                ▼
     │                         NamedQuery results
     │
     ├──── codegen ──────────► transforms/<name>/queries.hymeko + template.<ext>
     │                                │
     │                                ▼
     │                         emitted artefact (URDF / Python / DOT / …)
     │
     └──── snapshot ────────► SnapshotDto / DOT graph (hymeko_formats::snapshot)
```

## Stage 1: parse

`parser/src/lib.rs` defines a hand-rolled parser. It accepts:
- Description blocks: `name { const VAR = value; using path as alias; }`
- Context blocks: `context { ... declarations ... }`
- Node decls: `name: base { tag1; tag2; child_field value; }`
- Edge decls: `@name <kind1, kind2> { (+target1, ~op, -result); body; }`
- Constants: `const NAME = expression;` (in description header)
- Comments: `//` line comments

Returns a `parser::ast::Description` syntax tree.

## Stage 2: resolution

`hymeko_core::resolution::name_resolver::NameResolver` walks the syntax tree and:
- Resolves `using` imports — `using nn.tensors as ten;` makes `ten.t_input` reachable
- Resolves base references — `: lyr.signedkan_layer` becomes a `DeclId`
- Resolves arc-ref targets — `+x` becomes a signed pointer to the `x` decl
- Resolves cross-file imports — `@"meta_nn.hymeko";` pulls in the meta description

The `StringTable` interns all identifiers; `DeclId` is an index into the resolved decl array.

## Stage 3: IR build

`hymeko_core::ir::ir::Ir` is a struct of Vec arrays:

```rust
pub struct Ir {
    pub decl_nodes: Vec<DeclNode>,    // every named decl (Node | Edge | HyperArc)
    pub nodes:      Vec<NodeRecord>,  // just nodes (subset)
    pub edges:      Vec<EdgeRecord>,  // just edges (subset)
    pub arcs:       Vec<ArcRecord>,   // signed arc-refs
    pub strings:    StringTable,      // ID → string
}
```

This is the canonical representation. Everything downstream walks this struct.

## Stage 4 (option A): query

`hymeko_query::QueryEngine` evaluates predicates against `decl_nodes`:

```rust
let engine = QueryEngine::new(&ir, &resolver);
let pred = Predicate::node().and(Predicate::inherits("link"));
for did in engine.matches(&pred) {
    println!("{}", resolver.resolve(ir.decl_nodes[did.0].name));
}
```

The string-form predicate evaluator (`KIND(...) AND INHERITS(...)`) lives in `hymeko_query::predicate_expr` (single source of truth, used by Python wheel + WASM).

## Stage 4 (option B): codegen

`hymeko_formats::codegen::generate_description(ir, resolver, name, format)`:

1. Look up the format's transform in `TransformRegistry`
2. Run the queries declared in `transforms/<format>/queries.hymeko`
3. Bind each query's match results to template variables
4. Render `transforms/<format>/template.<ext>` with the binding context

Result: an emitted string. Same machinery for URDF, SDF, MJCF, DOT, torch_dataflow.

## Stage 4 (option C): snapshot

`hymeko_formats::snapshot::snapshot(ir, st)` produces a `SnapshotDto` (owned-string JSON-shaped struct) for visualization / debugging. `snapshot_json()` serializes to JSON; `emit_dot_graph()` produces Graphviz DOT.

## What about the model views

For URDF / SDF / MJCF / Gazebo, the codegen also pivots through a `KinematicModel` extracted by `hymeko_query::kinematics::extract_kinematic_model`. This is a typed projection of the IR: `Vec<Link>`, `Vec<Joint>` with parsed origin / axis / limits. After the May 2026 cleanup, both the legacy `generate_urdf()` and the template-driven path call `generate_urdf_from_model(&model)` — single emission step.

## Adding a new stage

Want to insert (e.g.) a structural-entropy pass between resolution and codegen? Add a step that consumes `&Ir` and produces metadata:

```rust
let entropy = compute_entropy_hierarchical(&ir, &resolver);
config.insert_metadata("entropy", entropy);
let urdf = generate_description(&ir, &resolver, name, OutputFormat::Urdf)?;
```

Most analyses are non-destructive on the IR. Mutations should produce a new IR rather than mutating in place — match the existing pattern.

## Next

- [Extension points](./extension-points.md) — where to plug new formats / queries / layer kinds
- [Layered architecture](./layers.md) — abstract view of the same pipeline
