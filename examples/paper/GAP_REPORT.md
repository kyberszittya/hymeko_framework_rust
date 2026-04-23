# Canonical Example — Gap Report

**File:** `examples/paper/hymeko_robot.hymeko`
**Reference:** Listing A.6.1 of the MDPI Technologies manuscript.

This note records every deviation between the paper's printed Listing A.6.1 and
the actual `.hymeko` source compiled for the experiments. The measured
structural counts are compared against the paper's §4.7.1 claims, and each
adaptation is flagged so the author can decide whether to update the listing,
update the prose, or both.

## Structural comparison

| Quantity | Paper (§4.7.1) | Measured | Δ | Cause |
|---|---:|---:|---:|---|
| Contextual hyperedges \|E\| | 10 | **10** | 0 | ✓ match |
| Arity multiset | (2,2,3,3,3,3,3,3,5,5) | **(2,2,3,3,3,3,3,3,5,5)** | — | ✓ match |
| Participating vertices \|V\| | 21 | **23** | +2 | Adaptation — see §A.1 |
| nnz(B) | 34 | **32** | −2 | Adaptation — see §A.2 |

The arity multiset — the most structurally meaningful invariant — matches
exactly. The two deltas are explainable by the adaptations below.

## Adaptations applied

### §A.1 `mode_parallel` and `operating_mode` as distinct vertices (+2 in |V|)

The paper's listing declares:

```hymeko
mode: grasp_mode.parallel {}       // grasping_context
operating_mode: mode <collaborative> {}   // safety_context
```

The first uses a dotted-path type (`grasp_mode.parallel`) and the second uses
a `mode` type with a tag. In the current HyMeKo v0.1 surface, dotted-path type
access (`grasp_mode.parallel`) is not yet supported by the inheritance
resolver. The adaptation replaces them with two concrete distinct vertices:

```hymeko
mode_parallel:  + <isa> grasp_mode {}
operating_mode: + <isa> mode {}
```

Both `grasp_mode` and `mode` are declared as bare-type parent nodes at the
top of the `hymeko_robot` scope. This preserves the structural role (two
different mode-kind vertices referenced by two different contextual
hyperedges) but counts as 2 additional vertices vs. the paper's implicit
aliasing.

**Recommended paper action:** either update the listing to use two distinct
mode vertices (matching the measured structure), or state in §4.7.1 that
the mode entries are aliased; current text is ambiguous.

### §A.2 `reference` indirection flattened (−2 in nnz(B))

The paper's listing introduces two `reference`-typed nodes inside
`safety_context`:

```hymeko
health_input: reference { + maintenance_context.component_health }
brake_input:  reference { + maintenance_context.brake_response }
```

These are then used inside `@braking_capability` as:

```hymeko
@braking_capability: aggregation {
    + health_input, + brake_input - braking_estimate
}
```

The `reference` nodes each carry one signed incidence to the maintenance-side
vertex, contributing 2 entries to nnz(B). In the adaptation, `@braking_capability`
references the maintenance-side vertices directly via qualified paths:

```hymeko
@braking_capability: + <isa> aggregation {
    (+ maintenance_context.component_health, + maintenance_context.brake_response, - braking_estimate);
}
```

This preserves the cross-context dependency semantically (the braking
capability edge has +arcs targeting vertices inside the maintenance context
scope, which is the structural property the paper argues is higher-order)
but replaces the two `reference` indirections with direct qualified-name
refs, removing those two auxiliary arcs from nnz(B).

**Recommended paper action:** if the author wants `nnz(B) = 34` to remain
the canonical figure, either restore the `reference` indirection in the
current HyMeKo surface (requires resolver support for `reference`-typed
pass-through nodes, which is not yet implemented) or update the text to
say `nnz(B) = 32` on the adapted listing. The second option is cleaner
because the paper's structural argument does not depend on the 32-vs-34
count — it depends on the arity multiset, which matches exactly.

### §A.3 Robot kinematic structure flattened

The paper's listing nests the 5 kinematic links + 4 joints + 3 sensors
inside `robot: component { ... }`. The adaptation places them at the
top level of the `hymeko_robot` scope. This has no effect on the
contextual-subset counts (joints and sensors are not contextual
hyperedges), but means the full-IR count of decls differs from a
hierarchically-nested version. No effect on the paper's claimed numbers.

### §A.4 `AXIS_Z` and `AXIS_Y` declared as bare vertices

The paper's joint declarations reference `- AXIS_Z` and `- AXIS_Y` without
declaring them. The adaptation adds two top-level bare-vertex declarations
(`AXIS_Z {}`, `AXIS_Y {}`). These are not counted in the contextual-subset
|V| because they are not referenced by any contextual hyperedge.

### §A.5 Base types declared inline

The paper's listing uses `link {}`, `joint {}`, etc. as type declarations
without explicit `<isa>` relations. These are preserved verbatim and
function as base-type parents that other decls inherit from via `+ <isa>`.
No structural change from the paper's intended semantics.

## Summary

The adaptation preserves the paper's structural headline claim (10
contextual hyperedges with the exact arity multiset) and the five paper
predicates P1–P5 (all match their expected counts; see
`hymeko_bench/results/query_latency.csv`). The two +2/−2 deltas on
vertex and nnz counts are accounted for by syntactic adaptations that
the current HyMeKo surface forces and do not affect the paper's
higher-order-representation argument.
