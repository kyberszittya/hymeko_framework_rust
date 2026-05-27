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
//!
//! # Correspondence with the canonical Friedler 1992 axioms
//!
//! (Established in the 2026-05-19 phase-2 audit; see
//! `reports/2026-05-19-pgraph-axiom-semantics-fix-phase2.md`.)
//!
//! - The **forward pass** ("every input must be raw or producible")
//!   enforces **A2/S2 forward direction** restricted to materials
//!   that are inputs of surviving units: any consumed material that
//!   is neither raw nor produced is rejected (so the surviving
//!   M-nodes that are *inputs* satisfy `no-ancestor ⟹ raw`).
//! - The **backward pass** enforces **A4/S4**: every surviving
//!   O-node has a path to some product material. The fixpoint
//!   iteration is the standard backward-reachability closure.
//! - The **`strict_no_excess` strengthener** is an *orthogonal*
//!   condition (Friedler 1992 calls this the "strict P-graph
//!   rule"): every output of every surviving unit must be product
//!   or consumed. It is **not** part of S1..S5 — it is an
//!   additional waste-free constraint.
//!
//! What MSG does **not** check (these live at the schema layer in
//! [`crate::axioms`]):
//!
//! - **A1/S1** (products are M-nodes): preserved by construction of
//!   `LoweredPGraph`.
//! - **A2/S2 reverse direction** (raws must not be produced inside):
//!   relies on the `LoweredPGraph` having no `u → r` edges for
//!   `r ∈ raws`. Verified at schema construction via
//!   [`crate::axioms::AxiomBundle`].
//! - **A3/S3** (O-nodes in catalogue): schema-layer concern.
//! - **A5/S5** (every M-node has ≥1 incident edge): schema-layer
//!   concern; MSG never reasons about stray M-nodes.

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

/// MSG knobs (2026-05-19, Stage relaxed-MSG).
///
/// `strict_no_excess = true` is the **canonical Friedler 1992** rule:
/// every output of a surviving unit must be a product *or* consumed
/// by another surviving unit. This is the right rule for a P-graph
/// solved under strict no-excess feasibility (no waste streams).
///
/// `strict_no_excess = false` is the **P-graph Studio default**:
/// at least one output is product-or-consumed (other outputs are
/// vented to disposal). This matches the textbook chapter
/// examples shipped in `data/pgraph/Chapter*/`.
#[derive(Debug, Clone, Copy)]
pub struct MaximalStructureOptions {
    /// Strict (Friedler 1992) when `true`; relaxed (P-graph Studio
    /// default) when `false`. See type-level docs for the
    /// semantic difference.
    pub strict_no_excess: bool,
}

impl Default for MaximalStructureOptions {
    fn default() -> Self {
        Self {
            strict_no_excess: true,
        }
    }
}

/// Back-compat shim: same as [`maximal_structure_with_options`] with
/// [`MaximalStructureOptions::default()`] (strict no-excess).
pub fn maximal_structure(p: &LoweredPGraph) -> MaximalStructure {
    maximal_structure_with_options(p, MaximalStructureOptions::default())
}

/// Compute the maximal structure by alternating forward / backward
/// trimming to a fixpoint, under explicit options.
///
/// Both passes iterate to a fixpoint of the combined operator. The
/// backward pass's criterion is controlled by
/// [`MaximalStructureOptions::strict_no_excess`].
pub fn maximal_structure_with_options(
    p: &LoweredPGraph,
    opts: MaximalStructureOptions,
) -> MaximalStructure {
    let mut units: BTreeSet<DeclId> = p.units.clone();

    loop {
        let before = units.len();

        // Forward pass: every input must be raw or producible by some
        // surviving unit.
        let producible: BTreeSet<DeclId> = close_producible(p, &units, &p.raws);
        units.retain(|u| p.inputs(*u).iter().all(|m| producible.contains(m)));

        // Backward pass: criterion depends on options.
        let consumed_by_surviving: BTreeSet<DeclId> = units
            .iter()
            .flat_map(|u| p.inputs(*u).iter().copied())
            .collect();
        let useful: BTreeSet<DeclId> = consumed_by_surviving.union(&p.products).copied().collect();
        units.retain(|u| {
            let outputs = p.outputs(*u);
            if opts.strict_no_excess {
                // Friedler 1992: every output must be useful.
                outputs.iter().all(|m| useful.contains(m))
            } else {
                // P-graph Studio relaxed: at least one output useful
                // (others are vented to disposal). Empty-output
                // units (sink consumers) are kept iff the empty
                // condition vacuously holds — but a Friedler P-graph
                // unit always has |outset| >= 1, so the `any` test
                // is meaningful when outputs is non-empty. When
                // outputs is empty, the unit is a pure disposal sink
                // and is kept (no constraint).
                outputs.is_empty() || outputs.iter().any(|m| useful.contains(m))
            }
        });

        if units.len() == before {
            break;
        }
    }

    // Materials touched by the surviving units, plus raws that are
    // actually consumed and products.
    let mut materials: BTreeSet<DeclId> = BTreeSet::new();
    for u in &units {
        materials.extend(p.inputs(*u).iter().copied());
        materials.extend(p.outputs(*u).iter().copied());
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
            if p.inputs(*u).iter().all(|m| c.contains(m)) {
                c.extend(p.outputs(*u).iter().copied());
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
            if p.outputs(*u).iter().any(|m| c.contains(m)) {
                c.extend(p.inputs(*u).iter().copied());
            }
        }
        if c.len() == before {
            break;
        }
    }
    c
}
