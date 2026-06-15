# hymeko_hive_query

`hymeko_hive_query` is the LALRPOP parser for HIVE's small association-query
language. It parses query text into typed `hymeko_hive::AssociationQuery`
values. HIVE itself executes the query; this crate only owns syntax and
lowering.

The core idea is:

```text
subject -- association --> object
```

So query constructs such as `KIND`, `HASARCREF`, and attribute lookup are not
separate mechanisms. They are all association patterns.

## Association Syntax

```text
assoc <subject> <association> <object>
```

Examples:

```text
assoc node kind type(trait)
assoc relation kind type(commits_to)
assoc node attr attr(weight)
assoc relation endpoint(-) node.kind(behavior)
assoc relation(rule_finish_with_evidence) endpoint(plus) node(epistemic_care)
```

These lower to `AssociationQuery` structures such as:

```rust
AssociationQuery {
    subject: EntitySelector::AnyRelation,
    association: AssociationKind::Endpoint {
        sign: Some(Sign::Minus),
    },
    object: AssociationObject::NodeType("behavior".into()),
}
```

## Subjects

```text
node
relation
node(<id>)
relation(<id>)
```

`node` and `relation` mean "any node" and "any relation". The parenthesized
forms select a specific entity.

## Associations

```text
kind
attr
attribute
endpoint
endpoint(+)
endpoint(-)
endpoint(~)
endpoint(plus)
endpoint(minus)
endpoint(neutral)
```

`kind` matches entity type. `attr` / `attribute` matches attached attribute keys.
`endpoint` matches relation endpoints, optionally filtered by signed incidence.

## Objects

```text
any
type(<type>)
attr(<key>)
attribute(<key>)
node(<id>)
node.kind(<type>)
```

`node.kind(<type>)` is the association-query equivalent of asking for endpoints
whose incident node has a specific type.

## Legacy Aliases

The crate keeps two compatibility aliases for the older predicate style:

```text
KIND(trait)
HASARCREF(-, KIND(behavior))
```

They lower to:

```text
assoc node kind type(trait)
assoc relation endpoint(-) node.kind(behavior)
```

This keeps existing vocabulary recognizable while making the generalized
association model the real substrate.

## LLM Usage

For direct LLM-to-HIVE interaction, there are two intended levels:

```text
Minimal descriptor:
relation.endpoint_type:-:behavior

LALRPOP query subset:
assoc relation endpoint(-) node.kind(behavior)
```

The minimal descriptor is useful for strict machine output. This crate's query
language is better for readable prompts, examples, logs, and human-authored
query files.

## Worked Example: Core Robotics Vocabulary

`hymeko_core` already tests `../data/robotics/meta_kinematics.hymeko` in
`hymeko_core/tests/domain_transformations/parse_files.rs`. That fixture is the
robotics vocabulary used by the full robot models: links, frames, controllers,
sensors, axes, and joint relation kinds.

```text
controllers {
    meta_controller {
        @state_interface {}
        @command_interface {}
    }
    joint_trajectory_controller: + <isa> meta_controller { ... }
    diff_drive_controller: + <isa> meta_controller { ... }
    force_torque_sensor_controller: + <isa> meta_controller { ... }
    forward_position_controller: + <isa> meta_controller { ... }
    forward_velocity_controller: + <isa> meta_controller { ... }
}

axes {
    axis_definition {}
    AXIS_X: + <isa> axis_definition { ... }
    AXIS_Y: + <isa> axis_definition { ... }
    AXIS_Z: + <isa> axis_definition { ... }
    AXIS_M_Z: + <isa> axis_definition { ... }
}

@fixed_joint:    + <isa> elements.joint {}
@rev_joint:      + <isa> elements.joint { ... }
@conti_joint:    + <isa> elements.joint {}
@prismatic_joint:+ <isa> elements.joint { ... }
```

The `hymeko_core` test pins this fixture at 57 nodes and 12 edges, and checks
that:

- five controllers inherit from `meta_controller`;
- four axes inherit from `axis_definition`;
- `rgb_camera` and `laser_scanner` inherit from `sensor`;
- `joint_state_broadcaster` remains a passive sensor node without inheritance.

Projected into HIVE, those checks are association queries.

Find controller nodes:

```text
assoc node kind type(meta_controller)
```

Find nodes that inherit from the controller base once inheritance is represented
as an association in HIVE:

```text
assoc node inherits type(meta_controller)
```

`inherits` is not executed by `hymeko_hive` yet; the query syntax is the target
shape for the core check above. Today, equivalent direct-kind queries can still
ask for specific controller nodes after lowering has materialized them:

```text
assoc node kind type(joint_trajectory_controller)
assoc node kind type(diff_drive_controller)
assoc node kind type(force_torque_sensor_controller)
assoc node kind type(forward_position_controller)
assoc node kind type(forward_velocity_controller)
```

Find axis-definition nodes:

```text
assoc node kind type(axis_definition)
```

Find concrete axes:

```text
assoc node kind type(AXIS_X)
assoc node kind type(AXIS_Y)
assoc node kind type(AXIS_Z)
assoc node kind type(AXIS_M_Z)
```

Find sensor vocabulary:

```text
assoc node kind type(sensor)
assoc node kind type(rgb_camera)
assoc node kind type(laser_scanner)
assoc node kind type(joint_state_broadcaster)
```

Find relation kinds for robot wiring:

```text
assoc relation kind type(control_plugin)
assoc relation kind type(sim_plugin)
assoc relation kind type(fixed_joint)
assoc relation kind type(rev_joint)
assoc relation kind type(conti_joint)
assoc relation kind type(prismatic_joint)
```

When a 6DOF robot instance such as the anthropomorphic arm is lowered through
this vocabulary, the same association language asks instance-level questions:

```text
assoc relation endpoint(-) node(AXIS_Z)
assoc relation endpoint(-) node.kind(link)
```

Those instance queries are intentionally the same shape as the vocabulary
queries; only the HIVE state changes.

Legacy-style direct kind lookup remains compatibility sugar:

```text
KIND(rev_joint)
```

## Worked Example: Fano Graph

`hymeko_core` tests `../data/typical_graphs/fano_graph.hymeko` through
`hymeko_core/tests/typical_graphs/*`. The constants in
`hymeko_core/tests/typical_graphs/fano/constants.rs` pin the structure:

- block name: `fano`;
- point nodes: `n0` through `n6`;
- edge relations: `e0` through `e6`;
- every edge has three neutral arc refs;
- every point has degree three.

```text
n0 {}
n1 {}
...
n6 {}

@e0 { (~n0, ~n1, ~n3); }
@e1 { (~n0, ~n2, ~n6); }
...
@e6 { (~n1, ~n5, ~n6); }
```

The expected edge targets are:

```text
e0: n0, n1, n3
e1: n0, n2, n6
e2: n0, n4, n5
e3: n1, n2, n4
e4: n2, n3, n5
e5: n3, n4, n6
e6: n1, n5, n6
```

After lowering into HIVE, one natural projection is to type the seven vertices
as `point` and the seven ternary relations as `line`.

Find all Fano points:

```text
assoc node kind type(point)
```

Find all Fano lines:

```text
assoc relation kind type(line)
```

Find all lines incident to `n0`:

```text
assoc relation endpoint(~) node(n0)
```

Find all neutral-incidence lines over points:

```text
assoc relation endpoint(~) node.kind(point)
```

The last query intentionally returns association matches, not just relation ids:
each Fano line has three neutral endpoints, so a fully materialized result has
21 endpoint associations. If the caller wants unique relation ids, it can
deduplicate the `subject` field of the returned `AssociationMatch` values.

For an LLM, this gives a compact and auditable question:

```text
User intent: "Which Fano lines touch n0?"
LLM query:   assoc relation endpoint(~) node(n0)
HIVE result: e0, e1, e2
```

## Generated Query Fixtures

The Rust HIVE layer also provides deterministic generators in
`hymeko_hive::generators` for query/database tests:

```rust
use hymeko_hive::generators::{steiner_triple_system, sunflower};

let fano = steiner_triple_system(7).unwrap();
let sts9 = steiner_triple_system(9).unwrap();
let sf = sunflower(3, 2, 2).unwrap();
```

Generated objects can be committed directly:

```rust
let mut store = HiveStore::new();
let generated = steiner_triple_system(7).unwrap();
let tx = generated.transaction("sts-7", store.state_hash(), "generator", 0);
store.commit(tx).unwrap();
```

Then the same query language can test combinatorial invariants:

```text
assoc node kind type(point)
assoc relation kind type(line)
assoc relation endpoint(~) node(n0)
assoc relation endpoint(~) node.kind(point)
```

For a sunflower:

```text
assoc node kind type(core_point)
assoc node kind type(petal_point)
assoc relation kind type(petal)
assoc relation endpoint(~) node.kind(core_point)
```

This gives HIVE query performance and correctness tests scalable synthetic
families, while staying aligned with the web editor's generator idea.

## Testing

Run:

```text
cargo test -p hymeko_hive_query
cargo clippy -p hymeko_hive_query --all-targets -- -D warnings
```

The parser tests cover association syntax, specific entity selectors, and
legacy aliases.
