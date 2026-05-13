//! Maximal Structure Generation (MSG) — Friedler et al. 1992.
//!
//! Given a P-graph $(M, O, E)$ with a raw set $R \subseteq M$ and a
//! product set $P \subseteq M$, the **maximal structure** is the
//! sub-P-graph that contains exactly those operating units that are
//! *both* forward-feasible (every input is raw or produced by some
//! surviving unit) *and* backward-feasible (every output reaches a
//! product through some surviving unit), iterated to a fixpoint.
//!
//! Every combinatorially feasible solution structure is a subgraph of
//! the maximal structure, so MSG is the obligatory pre-pass for SSG /
//! ABB and prunes operating units that cannot participate in any
//! feasible solution.

use std::collections::BTreeSet;

use hymeko::common::ids::DeclId;

use crate::lowering::LoweredPGraph;

/// Output of MSG: the surviving units and the materials they produce
/// or consume.
#[derive(Debug, Clone)]
pub struct MaximalStructure {
    /// Operating units in the maximal structure.
    pub units: BTreeSet<DeclId>,
    /// Materials touched by the maximal structure (raw, produced, or
    /// consumed by surviving units, plus the required products).
    pub materials: BTreeSet<DeclId>,
}

/// Compute the maximal structure by alternating forward / backward
/// trimming to a fixpoint.
///
/// This is the canonical Friedler-Tarján-Huang-Fan 1992 MSG: a unit
/// $u$ survives iff *every* input is raw or producible by some
/// surviving unit, AND *every* output is either a required product
/// or consumed by some surviving unit. The "every output is useful"
/// rule is what keeps legitimate consumer / disposal units in MSG;
/// switching to "some output reaches a product" is a stricter (and
/// non-standard) variant that drops sink-only units.
pub fn maximal_structure(p: &LoweredPGraph) -> MaximalStructure {
    let mut units: BTreeSet<DeclId> = p.units.clone();

    loop {
        let before = units.len();

        // Forward pass: every input must be raw or producible by some
        // surviving unit.
        let producible: BTreeSet<DeclId> = close_producible(p, &units, &p.raws);
        units.retain(|u| {
            let inputs = p.unit_inputs.get(u).cloned().unwrap_or_default();
            inputs.iter().all(|m| producible.contains(m))
        });

        // Backward pass: every output must be a required product or
        // consumed by some surviving unit.  An output that ends up
        // being neither is a dead-end byproduct that prevents the
        // unit from belonging to any feasible strict P-graph
        // solution.
        let consumed_by_surviving: BTreeSet<DeclId> = units
            .iter()
            .flat_map(|u| {
                p.unit_inputs
                    .get(u)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
            })
            .collect();
        let useful: BTreeSet<DeclId> = consumed_by_surviving.union(&p.products).copied().collect();
        units.retain(|u| {
            let outputs = p.unit_outputs.get(u).cloned().unwrap_or_default();
            outputs.iter().all(|m| useful.contains(m))
        });

        if units.len() == before {
            break;
        }
    }

    // Materials touched by the surviving units, plus raws that are
    // actually consumed and products.
    let mut materials: BTreeSet<DeclId> = BTreeSet::new();
    for u in &units {
        if let Some(ins) = p.unit_inputs.get(u) {
            materials.extend(ins.iter().copied());
        }
        if let Some(outs) = p.unit_outputs.get(u) {
            materials.extend(outs.iter().copied());
        }
    }
    materials.extend(p.products.iter().copied());

    MaximalStructure { units, materials }
}

/// Forward closure: smallest set $C \supseteq R$ such that whenever a
/// unit in `units` has all its inputs in $C$, its outputs are added.
pub fn close_producible(
    p: &LoweredPGraph,
    units: &BTreeSet<DeclId>,
    raws: &BTreeSet<DeclId>,
) -> BTreeSet<DeclId> {
    let mut c: BTreeSet<DeclId> = raws.clone();
    loop {
        let before = c.len();
        for u in units {
            let inputs = p.unit_inputs.get(u).cloned().unwrap_or_default();
            if inputs.iter().all(|m| c.contains(m)) {
                if let Some(outs) = p.unit_outputs.get(u) {
                    c.extend(outs.iter().copied());
                }
            }
        }
        if c.len() == before {
            break;
        }
    }
    c
}

/// Backward closure: smallest set $C \supseteq \mathit{products}$
/// such that whenever a unit in `units` has *any* output in $C$, its
/// inputs are added (they are needed to produce something useful).
pub fn close_consumable(
    p: &LoweredPGraph,
    units: &BTreeSet<DeclId>,
    products: &BTreeSet<DeclId>,
) -> BTreeSet<DeclId> {
    let mut c: BTreeSet<DeclId> = products.clone();
    loop {
        let before = c.len();
        for u in units {
            let outputs = p.unit_outputs.get(u).cloned().unwrap_or_default();
            if outputs.iter().any(|m| c.contains(m)) {
                if let Some(ins) = p.unit_inputs.get(u) {
                    c.extend(ins.iter().copied());
                }
            }
        }
        if c.len() == before {
            break;
        }
    }
    c
}
