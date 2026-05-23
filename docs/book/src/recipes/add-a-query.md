# Recipe: Add a new query

Goal: a new typed predicate combinator (e.g. `Predicate::has_field("mass")`) reusable across format emitters.

## Where queries live

- `hymeko_query/src/predicate.rs` — typed `Predicate` enum + builders + value comparisons
- `hymeko_query/src/predicate_expr.rs` — string-form evaluator (KIND/INHERITS/SCOPEDIN/HASARCREF)
- `hymeko_query/src/engine.rs` — `QueryEngine::matches`, walks the IR

## Add a typed atom

1. Extend the `Predicate` enum in `predicate.rs`:

```rust
pub enum Predicate {
    Kind(DeclKind),
    InheritsFrom(String),
    HasTag(String),
    HasField(String),                         // <-- new
    NumericField { name: String, cmp: ValuePredicate },  // <-- e.g. mass > 1.0
    And(Box<Predicate>, Box<Predicate>),
    // ...
}
```

2. Add builders:

```rust
impl Predicate {
    pub fn has_field(field: &str) -> Self {
        Self::HasField(field.to_string())
    }
    pub fn numeric_field(field: &str, cmp: ValuePredicate) -> Self {
        Self::NumericField { name: field.to_string(), cmp }
    }
}
```

3. Implement evaluation in `engine.rs::Predicate::matches`:

```rust
impl Predicate {
    pub fn matches(&self, did: DeclId, ir: &Ir, st: &StringTable) -> bool {
        match self {
            // ... existing arms ...
            Self::HasField(name) => {
                let decl = &ir.decl_nodes[did.0];
                decl.body.iter().any(|c| st.resolve(c.name) == name)
            }
            Self::NumericField { name, cmp } => {
                /* find child by name, parse value, compare */
            }
        }
    }
}
```

## Wire into the string evaluator (optional)

If you also want it accessible from CLI / Python:

```rust
// hymeko_query/src/predicate_expr.rs
pub fn match_atom(atom: &str, did: DeclId, ir: &Ir, st: &StringTable) -> bool {
    if let Some(rest) = atom.strip_prefix("HASFIELD(") {
        let name = rest.trim_end_matches(')');
        return decl_has_field(did, name, ir, st);
    }
    // ... existing arms ...
}
```

Add a unit test in `hymeko_query/tests/`.

## Use it

```rust
let pred = Predicate::node()
    .and(Predicate::inherits("link"))
    .and(Predicate::has_field("mass"));
let engine = QueryEngine::new(&ir, &resolver);
let heavy_links = engine.matches_predicate(&pred);
```

Or from a `transforms/<name>/queries.hymeko`:

```hymeko
massive_links: link, hasfield("mass") {}
```

(if you've added the syntax extension in the queries-DSL parser).

## See also

- [Concepts: Queries](../concepts/queries.md) — surface overview
- [Quickstart: Query an IR](../quickstart/07-query.md) — using existing predicates
