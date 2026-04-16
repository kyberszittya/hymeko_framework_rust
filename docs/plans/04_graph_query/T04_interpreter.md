# T04 — AST-to-Predicate Interpreter

**Status:** ✅ DONE  
**File:** `hymeko_core/src/query/interpret.rs` (250 lines)

---

## Purpose

This is the bridge that makes "query-as-description" work. It takes a parsed `.hymeko` AST (the same `Description` type your compiler produces) and converts each element into a `Predicate` tree. The parser is completely untouched — interpretation happens after parsing.

## How It Works

```
query_file.hymeko
       │
       ▼
  LALRPOP Parser (unchanged)
       │
       ▼
  AST: Description<'_, &str>
       │
       ▼
  interpret_as_queries(ast)     ← THIS MODULE
       │
       ▼
  Vec<NamedQuery>
       │
       ▼
  QueryEngine::query_all()
```

## Mapping Rules

| AST Construct | Interpretation |
|---------------|---------------|
| `NodeDecl { name: "_" }` | Skip `Named` predicate → wildcard |
| `NodeDecl { name: "x" }` | `Named("x")` |
| `NodeDecl { bases: [link] }` | `InheritsFrom("link")` |
| `NodeDecl { anno.tags: ["isa"] }` | `HasTag("isa")` |
| `NodeDecl { anno.tags: ["gt"], value: Num(10) }` | `HasValue(NumGt(10.0))` |
| `NodeDecl { body: Some([children]) }` | `HasChild(...)` for each child |
| `EdgeDecl { ... }` | `Kind(Edge)` + same rules |
| `HyperArc { refs: [+x, -y] }` inside edge | `HasPlusRef(...)`, `HasMinusRef(...)` |

## Tag-Encoded Comparisons

Since `>=` and `<=` conflict with `<`/`>` angle brackets in the grammar, comparisons use annotation tags:

```
mass <gt> 10.0;     →  HasValue(NumGt(10.0))
mass <lte> 5.0;     →  HasValue(NumLte(5.0))
mass <eq> 25.0;     →  HasValue(NumEq(25.0))
```

The interpreter partitions tags into comparison operators (`gt`, `lt`, `gte`, `lte`, `ne`, `eq`) and normal annotation tags. Comparison tags consume the value; normal tags become `HasTag` predicates.

## Entry Point

```rust
pub fn interpret_as_queries(ast: &Description<'_, &str>) -> Vec<NamedQuery>
```

Each top-level item in the description becomes one `NamedQuery`. Items named `_` get auto-generated labels (`node_query_0`, `edge_query_1`, etc.).

## Example: URDF Query Set

```
urdf_queries {
    _ : link {}                              → Kind(Node) ∧ InheritsFrom("link")
    @_ : conti_joint { (+_ : link, -_ : link) ; }  → Kind(Edge) ∧ InheritsFrom("conti_joint")
                                                        ∧ HasPlusRef(Any) ∧ HasMinusRef(Any)
}
```

## Known Limitations

1. **Ref-level inheritance not interpreted yet** — `+ _ : link` in a query puts `link` in the ref atom's base list, but the interpreter currently treats ref targets as name-only matches. For v1 this is handled by the programmatic API (`Predicate::HasPlusRef(Box::new(Predicate::inherits("link")))`) rather than through the interpreter.

2. **No variable binding** — `?var` syntax is designed but not parsed (needs `?` token in LALRPOP, see T10).

3. **No deep containment** — `..` for "match at any nesting depth" is not yet supported.
