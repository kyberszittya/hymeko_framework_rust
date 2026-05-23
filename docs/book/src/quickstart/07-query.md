# Quickstart: Query an IR

The query language is a small DSL for finding nodes / edges that match structural predicates. Everything in the codegen pipeline goes through it.

## The predicate atoms

| atom | what it matches |
|---|---|
| `KIND(name)` | decl whose first inherited base equals `name` |
| `INHERITS(name)` | decl transitively inheriting `name` |
| `SCOPEDIN(name)` | decl with an ancestor inheriting `name` |
| `HASARCREF(sign, inner)` | edge with a signed arc-ref of `sign` (+1/-1) pointing at a decl matching `inner` |
| `<a> AND <b>` | conjunction |
| `ANY` | always true |

## From the CLI

```bash
target/release/hymeko query data/robotics_imported/wam/wam.hymeko 'INHERITS(link)'
# Produces a list of all link names
# base_link
# shoulder_link
# upper_arm_link
# ...
```

```bash
target/release/hymeko query data/robotics_imported/wam/wam.hymeko 'INHERITS(rev_joint)'
# All revolute joints
```

```bash
target/release/hymeko query data/nn/hsikan_mixed.hymeko \
    'INHERITS(signedkan_layer) AND SCOPEDIN(hsikan_mixed)'
# All signedkan_layer nodes inside the hsikan_mixed description
```

## From Python

```python
import hymeko

src = open("data/robotics_imported/wam/wam.hymeko").read()
doc = hymeko.compile_description(src)

# Run a single query
revolute_joints = doc.query("INHERITS(rev_joint)")
print(revolute_joints)
# ['shoulder_pan', 'shoulder_lift', 'elbow', ...]

# Count without enumerating
n_links = doc.query_count("INHERITS(link)")
print(f"robot has {n_links} links")
```

## How the engine works

`hymeko_query::QueryEngine` walks the IR's `decl_nodes` array and evaluates the predicate against each. The string-form evaluator lives in `hymeko_query/src/predicate_expr.rs` (single source of truth — both the Python wheel and the wasm demo go through this module).

Beneath the string predicate is a typed predicate combinator system — `Predicate::node().and(Predicate::inherits("link"))` is the typed form of the string `KIND(node) AND INHERITS(link)`. The format emitters use the typed form (see `hymeko_formats/src/urdf.rs::urdf_queries()`).

## Predicate-driven extraction

The URDF emitter is built on queries:

```rust
// hymeko_formats/src/urdf.rs
pub fn urdf_queries() -> Vec<NamedQuery> {
    vec![
        NamedQuery { label: "links".into(),
            predicate: Predicate::node().and(Predicate::inherits("link")) },
        NamedQuery { label: "revolute_joints".into(),
            predicate: Predicate::edge().and(Predicate::inherits("rev_joint")) },
        ...
    ]
}
```

The codegen template iterates each named query's results and emits the corresponding XML / Python / DOT.

## Write your own query

In a `transforms/<name>/queries.hymeko`:

```hymeko
my_format {
    my_links: link {}
    my_motors: motor_actuator {}
    my_revolute: rev_joint {}
}
```

Then in your template:

```text
{{#each my_links}}<link name="{{name}}"/>{{/each}}
```

See [Add a new format](../recipes/add-a-format.md) for the full template machinery.

## Next

- [Add a new query](../recipes/add-a-query.md) — typed `Predicate` API
- [Concepts: Queries](../concepts/queries.md) — deeper dive
