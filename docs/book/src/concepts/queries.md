# Concept: Queries

The query layer exposes two predicate surfaces:

## 1. String-form predicates

Used from CLI / Python / WASM. Grammar in `hymeko_query::predicate_expr`:

```
expression := atom (AND atom)*
atom       := KIND(name)                 // first inherited base
            | INHERITS(name)             // transitive inheritance
            | SCOPEDIN(name)             // ancestor inherits name
            | HASARCREF(±sign, atom)     // edge has signed arc-ref
            | ANY                        // always true
```

```bash
target/release/hymeko query foo.hymeko 'INHERITS(link) AND SCOPEDIN(my_robot)'
```

## 2. Typed predicates

Used from Rust. `hymeko_query::predicate::Predicate`:

```rust
let pred = Predicate::node()
    .and(Predicate::inherits("link"))
    .and(Predicate::has_tag("upper_arm"));
let engine = QueryEngine::new(&ir, &resolver);
for did in engine.matches(&pred) {
    /* ... */
}
```

The format emitters use the typed form (faster, type-safe). The user-facing surfaces use the string form (more flexible, less ceremony).

## Named queries

A `NamedQuery { label, predicate }` bundles a name + predicate; multiple named queries form a query set that the codegen template iterates.

```rust
pub fn urdf_queries() -> Vec<NamedQuery> {
    vec![
        NamedQuery { label: "links".into(),
                     predicate: Predicate::node().and(Predicate::inherits("link")) },
        // ...
    ]
}
```

In `transforms/<name>/queries.hymeko` this same structure is declared in HyMeKo syntax — the engine compiles it to `Vec<NamedQuery>` automatically.

## See also

- [Quickstart: Query an IR](../quickstart/07-query.md)
- [Add a new query](../recipes/add-a-query.md)
