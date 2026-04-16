# T02–T03 — Query Engine Core (Predicate + Engine)

**Status:** ✅ DONE  
**Files:** `hymeko_core/src/query/predicate.rs` (141 lines), `hymeko_core/src/query/engine.rs` (250 lines)

---

## T02 — Predicate Algebra (`predicate.rs`)

### Predicate Variants

| Variant | Matches | Example |
|---------|---------|---------|
| `Any` | Everything | Wildcard `_` |
| `Kind(DeclKind)` | By declaration type | `node()`, `edge()` |
| `Named(String)` | Exact resolved name | `named("base_link")` |
| `NamePrefix(String)` | Name starts with | `name_prefix("wheel")` |
| `InheritsFrom(String)` | Transitive base chain | `inherits("link")` |
| `HasTag(String)` | Annotation tag present | `has_tag("isa")` |
| `HasChild(Box<Pred>)` | At least one child matches | Containment `{ ... }` |
| `HasParent(Box<Pred>)` | Parent matches | Upward navigation |
| `HasValue(ValuePred)` | Value on the decl itself | `mass 25.0;` |
| `ChildValue(name, vp)` | Named child's value | `ChildValue("mass", NumGt(10.0))` |
| `HasPlusRef(Box<Pred>)` | Edge has +ref to target | `+ base_link` |
| `HasMinusRef(Box<Pred>)` | Edge has -ref to target | `- wheel_fr` |
| `HasNeutralRef(Box<Pred>)` | Edge has ~ref to target | `~ something` |
| `HasRef(Box<Pred>)` | Edge has any-sign ref | Any ref |
| `And(Vec<Pred>)` | All must match | `.and()` builder |
| `Or(Vec<Pred>)` | Any must match | `.or()` builder |
| `Not(Box<Pred>)` | Must not match | `.not()` builder |

### ValuePredicate

```
NumEq(f64)   NumGt(f64)   NumLt(f64)   NumGte(f64)   NumLte(f64)
StrEq(String)   Any
```

### Builder Pattern

```rust
let heavy_links = Predicate::node()
    .and(Predicate::inherits("link"))
    .and(Predicate::ChildValue("mass".into(), ValuePredicate::NumGt(10.0)));
```

Builders flatten nested `And`/`Or` nodes automatically for efficiency.

---

## T03 — Query Engine (`engine.rs`)

### Architecture

```
QueryEngine<'a, R: NameResolver>
├── ir: &'a Ir
├── resolver: &'a R           // Interner or StringTable
├── config: QueryConfig        // max_inherit_depth = 8
│
├── query(&Predicate) → QueryResult
├── query_all(&[NamedQuery]) → Vec<(String, QueryResult)>
├── matches(DeclId, &Predicate) → bool
│
├── check_inherits(did, base_name, depth) → bool    [private]
├── get_bases(did) → &[SignedRefR]                   [private]
└── check_arc_ref(did, sign, target_pred) → bool     [private]
```

### NameResolver Trait

Generic over the two name-storage strategies:

```rust
pub trait NameResolver {
    fn resolve(&self, id: SymId) -> &str;
}

impl NameResolver for Interner { ... }      // daemon, tests
impl NameResolver for StringTable { ... }   // Python bindings
```

### Key Design Decisions

1. **Full linear scan per query** — iterates all `decl_nodes`. For 57-node robot models this is ~microseconds. Index-based acceleration is unnecessary until models exceed 10K+ nodes.

2. **Depth-bounded inheritance** — `check_inherits` caps at 8 levels to prevent infinite loops on cyclic inheritance. Your deepest chain is 3 (`joint_fr → conti_joint → joint`).

3. **Arc ref matching navigates `EdgeRec.arcs`** — each arc's `SignedRefR` list is checked against the sign filter and target predicate.

4. **`QueryResult` carries `(DeclId, String)`** — domain transforms use the `DeclId` to navigate the IR directly for value extraction.
