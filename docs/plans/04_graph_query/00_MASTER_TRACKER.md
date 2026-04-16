# HyMeKo Paper 2 — Master Task Tracker

**Last updated:** 2026-03-21  
**Deadline:** IEEE SMC Regular — March 22

---

## Summary

| Category | Done | Remaining | Lines Written |
|----------|------|-----------|---------------|
| Code review & optimization | 3/3 | 0 | ~25 |
| Query engine core | 3/3 | 0 | 391 |
| AST-to-Predicate interpreter | 1/1 | 0 | 250 |
| Kinematic extraction | 1/1 | 0 | 390 |
| Domain transforms (URDF) | 1/1 | 0 | 189 |
| Domain transforms (SDF) | 1/1 | 0 | 114 |
| Domain transform trait | 1/1 | 0 | 29 |
| LALRPOP extensions | 0/1 | 1 | 0 |
| Domain transforms (Gazebo) | 0/1 | 1 | 0 |
| Domain transforms (Isaac Sim) | 0/1 | 1 | 0 |
| Tests | 0/4 | 4 | 0 |
| Paper 2 text updates | 0/5 | 5 | 0 |
| **Total** | **12/23** | **11** | **1,388** |

---

## Completed Tasks

### T01 — Code Review & SignedRefR Optimization
- **File:** `hymeko_core/src/ir/ir.rs`
- **What:** Added `impl SignedRefR` with `atom()`, `target()`, `sign()` inherent methods
- **Why:** Eliminates the triple-match pattern (`Plus(a) | Minus(a) | Neutral(a) => a`) scattered across the codebase. The free functions in `ir/common.rs` still work but new code uses the methods directly.
- **Lines:** +25

### T02 — Predicate Algebra
- **File:** `hymeko_core/src/query/predicate.rs` (141 lines, new)
- **What:** `Predicate` enum with 17 variants, `ValuePredicate` enum, `NamedQuery` struct, builder methods
- **Covers:** Kind, Named, NamePrefix, InheritsFrom, HasTag, HasChild, HasParent, HasValue, ChildValue, HasPlusRef, HasMinusRef, HasNeutralRef, HasRef, And, Or, Not, Any

### T03 — Query Engine
- **File:** `hymeko_core/src/query/engine.rs` (250 lines, new)
- **What:** `QueryEngine<R: NameResolver>` with `query()`, `query_all()`, `matches()`
- **Key internals:** `check_inherits` (transitive, depth-bounded), `get_bases` (works for both nodes and edges), `check_arc_ref` (iterates edge's arc records)
- **Generic over:** `Interner` (tests/daemon) and `StringTable` (Python bindings)

### T04 — AST-to-Predicate Interpreter
- **File:** `hymeko_core/src/query/interpret.rs` (250 lines, new)
- **What:** `interpret_as_queries()` converts a parsed `.hymeko` AST into `Vec<NamedQuery>`
- **Handles:** `_` wildcard, `: base` inheritance, `<tag>` annotations, `<gt>`/`<lt>` tag-encoded comparisons, `{ children }` containment, `@edge { +x -y }` arc ref patterns

### T05 — Kinematic Model Extraction
- **File:** `hymeko_core/src/query/kinematic.rs` (390 lines, new)
- **What:** `KinematicModel`, `LinkInfo`, `JointInfo`, `GeometryInfo` structs + `extract_kinematic_model()` function
- **Handles:** Mass, geometry (box/cylinder/sphere), origin, color via ref-following, joint parent/child disambiguation (link vs axis by inheritance check), weight annotation extraction for `[[x,y,z],[r,p,y]]` origins

### T06 — URDF Generation
- **File:** `hymeko_core/src/query/urdf.rs` (189 lines, new)
- **What:** `generate_urdf()`, `urdf_queries()`, `validate_robot_schema()`
- **Output:** Complete URDF XML with `<robot>`, `<link>`, `<joint>`, `<inertial>`, `<visual>`, `<collision>`, `<material>`, `<origin>`, `<axis>`, `<limit>`
- **Conversions:** Degree-to-radian for RPY, xml_escape for attribute values

### T07 — SDF Generation
- **File:** `hymeko_core/src/query/sdf.rs` (114 lines, new)
- **What:** `generate_sdf()` producing SDF 1.7 XML
- **Differences from URDF:** `<pose>` instead of `<origin>`, `<inertia>` matrix, no `continuous` joint type (mapped to revolute with unbounded limits), `relative_to` attribute on pose

### T08 — Domain Transform Trait
- **File:** `hymeko_core/src/query/transform.rs` (29 lines, new)
- **What:** `DomainTransform` trait with `queries()` and `generate()` methods
- **Purpose:** Makes the paper's "domain-agnostic engine" claim concrete in code

### T09 — Module Integration
- **File:** `hymeko_core/src/query/mod.rs` (7 lines, new)
- **File:** `hymeko_core/src/lib.rs` (1 line edit: added `pub mod query;`)

---

## Remaining Tasks

### T10 — LALRPOP `?` Token Extension
- **Files:** `parser/src/lexer/token.rs`, `parser/src/lexer/common.rs`, `parser/src/hymeko.lalrpop`
- **Effort:** 3 one-line changes
- **Risk:** Low — `?` is unused in the current grammar
- **Priority:** Nice-to-have for paper (can mention as "designed" without it)

### T11 — Gazebo World Configuration
- **File:** `hymeko_core/src/query/gazebo.rs` (estimated ~120 lines)
- **Needs:** Plugin extraction from `sim_plugin`/`control_plugin` edges in robot_4wh
- **Priority:** Future work for paper

### T12 — Isaac Sim USD Export
- **File:** `hymeko_core/src/query/isaac.rs` (estimated ~100 lines)
- **Needs:** USD Prim hierarchy with `UsdPhysics` articulation schema
- **Priority:** Future work for paper

### T13 — Test: Meta-Kinematics Schema Queries
- **File:** `hymeko_core/tests/query/test_query_meta.rs`
- **Covers:** All nodes, all edges, inheritance (axes, controllers, sensors, joints, meta_element), tag queries, Not/Or predicates
- **Priority:** CRITICAL for paper Table II

### T14 — Test: Robot Cross-Import Queries
- **File:** `hymeko_core/tests/query/test_query_robot.rs`
- **Covers:** Link inheritance across imports, joint type queries, arc ref matching (+link, -link, -axis), heavy links (ChildValue), name prefix, schema validation, batch query_all
- **Priority:** CRITICAL for paper Table II

### T15 — Test: URDF Generation
- **File:** `hymeko_core/tests/query/test_urdf.rs`
- **Covers:** XML structure, link/joint counts, parent/child topology, origin degree→radian, axis presence/absence (fixed vs continuous), geometry types, well-formedness
- **Priority:** CRITICAL for paper validation section

### T16 — Test: SDF Generation
- **File:** `hymeko_core/tests/query/test_sdf.rs`
- **Covers:** SDF structure, joint type mapping, pose format
- **Priority:** Medium

### T17–T21 — Paper 2 Text Updates
- Verify Table II match counts from test output
- Add URDF generation line count
- Verify "57 nodes, 12 edges" from IR dump
- Add timing numbers (optional)
- List SDF/Gazebo/Isaac Sim as future work

---

## File Map

```
hymeko_core/src/query/
├── mod.rs            7 lines   ✅ T09
├── predicate.rs    141 lines   ✅ T02
├── engine.rs       250 lines   ✅ T03
├── interpret.rs    250 lines   ✅ T04
├── kinematic.rs    390 lines   ✅ T05
├── transform.rs     29 lines   ✅ T08
├── urdf.rs         189 lines   ✅ T06
├── sdf.rs          114 lines   ✅ T07
├── gazebo.rs         — lines   ❌ T11
└── isaac.rs          — lines   ❌ T12

hymeko_core/src/ir/ir.rs        +25 lines  ✅ T01 (SignedRefR impl)
hymeko_core/src/lib.rs           +1 line   ✅ T09 (pub mod query)

hymeko_core/tests/query/
├── mod.rs                       ❌ T13-T16
├── test_query_meta.rs           ❌ T13
├── test_query_robot.rs          ❌ T14
├── test_urdf.rs                 ❌ T15
└── test_sdf.rs                  ❌ T16
```
