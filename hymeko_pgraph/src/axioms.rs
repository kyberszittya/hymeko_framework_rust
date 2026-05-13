//! Phase 1 — axiom checker (scaffold, full implementation pending).
//!
//! The five P-graph axioms (Friedler et al. 1992):
//!
//! - **A1.** Every final-product M-node is in the graph.
//! - **A2.** Every M-node has a path to a product node.
//! - **A3.** Every O-node corresponds to a real operating unit.
//! - **A4.** Every O-node has at least one input M-node and one output
//!   M-node.
//! - **A5.** If an M-node is consumed, the consuming edge exists.
//!
//! This module currently scaffolds the [`AxiomViolation`] type and the
//! [`AxiomBundle`] entry point. A4 (degree constraint) is implemented
//! end-to-end since it is a pure schema property; the others are
//! parked as `todo!()` arms pending Phase 1 of the plan.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use hymeko::common::ids::DeclId;

use crate::schema::PGraphSchema;

/// One axiom check's outcome with its wall-clock duration.
#[derive(Debug, Clone)]
pub struct AxiomTrace {
    /// Short name (`"A1"` … `"A5"`).
    pub name: &'static str,
    /// Wall-clock time spent evaluating this axiom.
    pub duration: Duration,
    /// `None` if the axiom passed, `Some(violation)` if not.
    pub outcome: Option<AxiomViolation>,
}

/// One concrete violation of a P-graph axiom.
///
/// Each variant names the offending declarations so the consumer can
/// surface a specific error rather than a yes/no verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxiomViolation {
    /// A1 — required final-product M-nodes are absent from the IR.
    MissingProducts {
        /// Required products that were not present.
        missing: Vec<DeclId>,
    },
    /// A2 — at least one M-node has no path to any final-product node.
    UnreachableNodes {
        /// M-nodes with no path to any product.
        unreachable: Vec<DeclId>,
    },
    /// A3 — at least one O-node does not correspond to a registered
    /// operating unit (its declaration is missing required tags).
    InvalidUnits {
        /// O-nodes failing the unit-validation check.
        invalid: Vec<DeclId>,
    },
    /// A4 — at least one O-node has zero input M-nodes or zero output
    /// M-nodes.
    DegreeViolations {
        /// O-nodes with `in_degree == 0` or `out_degree == 0`.
        offenders: Vec<DeclId>,
    },
    /// A5 — at least one M-node is consumed but its consuming edge
    /// does not exist in the schema.
    MissingEdges {
        /// M-nodes whose consumption edge is missing.
        missing: Vec<DeclId>,
    },
}

/// Bundle of axiom checks applicable to a [`PGraphSchema`].
///
/// Constructed via [`AxiomBundle::default`]; the optional
/// `valid_units` whitelist powers A3 (units in the master
/// catalogue). When the whitelist is empty A3 is disabled.
#[derive(Debug, Default)]
pub struct AxiomBundle {
    /// A3 — master catalogue of valid operating units. If non-empty,
    /// every O-node must be in this set. If empty, A3 is skipped.
    pub valid_units: BTreeSet<DeclId>,
    /// Materials in the raw set ($R$). Used by A2's reachability
    /// check (a node must be reachable from $R$ *and* must reach a
    /// product) and by A5 (raw materials need not be produced by
    /// any unit).
    pub raws: BTreeSet<DeclId>,
}

impl AxiomBundle {
    /// Convenience constructor — supply raws and the unit whitelist
    /// in one call. Pass empty sets to disable A3 / A2.
    pub fn new(
        raws: impl IntoIterator<Item = DeclId>,
        valid_units: impl IntoIterator<Item = DeclId>,
    ) -> Self {
        Self {
            raws: raws.into_iter().collect(),
            valid_units: valid_units.into_iter().collect(),
        }
    }

    /// Run every axiom check, returning the (axiom-name, duration,
    /// outcome) triples in evaluation order. The outcome is `None`
    /// when the axiom passed and `Some(violation)` when it didn't.
    /// All five axioms run unconditionally — caller decides whether
    /// `Ok` or `Err` is the right summary.
    pub fn validate_timed(
        &self,
        schema: &PGraphSchema,
        required_products: &BTreeSet<DeclId>,
    ) -> Vec<AxiomTrace> {
        let mut out = Vec::with_capacity(5);

        let t0 = Instant::now();
        let v = check_a1(schema, required_products);
        out.push(AxiomTrace {
            name: "A1",
            duration: t0.elapsed(),
            outcome: v,
        });

        let t0 = Instant::now();
        let v = check_a2(schema, required_products);
        out.push(AxiomTrace {
            name: "A2",
            duration: t0.elapsed(),
            outcome: v,
        });

        let t0 = Instant::now();
        let v = check_a3(schema, &self.valid_units);
        out.push(AxiomTrace {
            name: "A3",
            duration: t0.elapsed(),
            outcome: v,
        });

        let t0 = Instant::now();
        let v = check_a4(schema);
        out.push(AxiomTrace {
            name: "A4",
            duration: t0.elapsed(),
            outcome: v,
        });

        let t0 = Instant::now();
        let v = check_a5(schema, &self.raws);
        out.push(AxiomTrace {
            name: "A5",
            duration: t0.elapsed(),
            outcome: v,
        });

        out
    }

    /// Run every axiom check and return the list of violations
    /// encountered. An empty `Ok(())` means the schema satisfies all
    /// implemented axioms.
    pub fn validate(
        &self,
        schema: &PGraphSchema,
        required_products: &BTreeSet<DeclId>,
    ) -> Result<(), Vec<AxiomViolation>> {
        let mut violations: Vec<AxiomViolation> = Vec::new();

        // A1 — required products must exist as M-nodes in the schema.
        let mut missing_products = Vec::new();
        for prod in required_products {
            match schema.kind(*prod) {
                Some(crate::schema::PNodeKind::Material) => {}
                _ => missing_products.push(*prod),
            }
        }
        if !missing_products.is_empty() {
            violations.push(AxiomViolation::MissingProducts {
                missing: missing_products,
            });
        }

        // A2 — every M-node has a path through the schema's directed
        // edges to some required product.
        //
        // Implementation: BFS forward from every M-node along
        // schema.edges() and check whether any required-product is
        // reached. A node that doesn't reach any product is a dead
        // material.
        if !required_products.is_empty() {
            let mut unreachable = Vec::new();
            for m in schema.m_nodes() {
                if required_products.contains(&m) {
                    continue; // products trivially "reach" themselves
                }
                if !reaches_any(schema, m, required_products) {
                    unreachable.push(m);
                }
            }
            if !unreachable.is_empty() {
                violations.push(AxiomViolation::UnreachableNodes { unreachable });
            }
        }

        // A3 — every O-node is in the unit catalogue (when the
        // catalogue is non-empty).
        if !self.valid_units.is_empty() {
            let mut invalid = Vec::new();
            for o in schema.o_nodes() {
                if !self.valid_units.contains(&o) {
                    invalid.push(o);
                }
            }
            if !invalid.is_empty() {
                violations.push(AxiomViolation::InvalidUnits { invalid });
            }
        }

        // A4 — every O-node has ≥ 1 input and ≥ 1 output M-node.
        let mut offenders = Vec::new();
        for o in schema.o_nodes() {
            if schema.in_degree(o) == 0 || schema.out_degree(o) == 0 {
                offenders.push(o);
            }
        }
        if !offenders.is_empty() {
            violations.push(AxiomViolation::DegreeViolations { offenders });
        }

        // A5 — every M-node that is *consumed* (has an incoming
        // edge into some unit *that does not also produce it*) and
        // is not raw must be produced by at least one unit.
        //
        // Equivalently: for every consumed-but-not-produced M-node,
        // the M-node must be in the raw set.
        {
            let mut produced: BTreeSet<DeclId> = BTreeSet::new();
            let mut consumed: BTreeSet<DeclId> = BTreeSet::new();
            for (_, src, dst) in schema.edges() {
                match (schema.kind(src), schema.kind(dst)) {
                    (Some(crate::schema::PNodeKind::Material), _) => {
                        consumed.insert(src);
                    }
                    (_, Some(crate::schema::PNodeKind::Material)) => {
                        produced.insert(dst);
                    }
                    _ => {}
                }
            }
            let mut missing = Vec::new();
            for m in schema.m_nodes() {
                if consumed.contains(&m) && !produced.contains(&m) && !self.raws.contains(&m) {
                    missing.push(m);
                }
            }
            if !missing.is_empty() {
                violations.push(AxiomViolation::MissingEdges { missing });
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

// ─── Per-axiom helpers (used by both validate and validate_timed) ──

fn check_a1(schema: &PGraphSchema, required_products: &BTreeSet<DeclId>) -> Option<AxiomViolation> {
    let mut missing = Vec::new();
    for prod in required_products {
        match schema.kind(*prod) {
            Some(crate::schema::PNodeKind::Material) => {}
            _ => missing.push(*prod),
        }
    }
    if missing.is_empty() {
        None
    } else {
        Some(AxiomViolation::MissingProducts { missing })
    }
}

fn check_a2(schema: &PGraphSchema, required_products: &BTreeSet<DeclId>) -> Option<AxiomViolation> {
    if required_products.is_empty() {
        return None;
    }
    let mut unreachable = Vec::new();
    for m in schema.m_nodes() {
        if required_products.contains(&m) {
            continue;
        }
        if !reaches_any(schema, m, required_products) {
            unreachable.push(m);
        }
    }
    if unreachable.is_empty() {
        None
    } else {
        Some(AxiomViolation::UnreachableNodes { unreachable })
    }
}

fn check_a3(schema: &PGraphSchema, valid_units: &BTreeSet<DeclId>) -> Option<AxiomViolation> {
    if valid_units.is_empty() {
        return None;
    }
    let mut invalid = Vec::new();
    for o in schema.o_nodes() {
        if !valid_units.contains(&o) {
            invalid.push(o);
        }
    }
    if invalid.is_empty() {
        None
    } else {
        Some(AxiomViolation::InvalidUnits { invalid })
    }
}

fn check_a4(schema: &PGraphSchema) -> Option<AxiomViolation> {
    let mut offenders = Vec::new();
    for o in schema.o_nodes() {
        if schema.in_degree(o) == 0 || schema.out_degree(o) == 0 {
            offenders.push(o);
        }
    }
    if offenders.is_empty() {
        None
    } else {
        Some(AxiomViolation::DegreeViolations { offenders })
    }
}

fn check_a5(schema: &PGraphSchema, raws: &BTreeSet<DeclId>) -> Option<AxiomViolation> {
    let mut produced: BTreeSet<DeclId> = BTreeSet::new();
    let mut consumed: BTreeSet<DeclId> = BTreeSet::new();
    for (_, src, dst) in schema.edges() {
        match (schema.kind(src), schema.kind(dst)) {
            (Some(crate::schema::PNodeKind::Material), _) => {
                consumed.insert(src);
            }
            (_, Some(crate::schema::PNodeKind::Material)) => {
                produced.insert(dst);
            }
            _ => {}
        }
    }
    let mut missing = Vec::new();
    for m in schema.m_nodes() {
        if consumed.contains(&m) && !produced.contains(&m) && !raws.contains(&m) {
            missing.push(m);
        }
    }
    if missing.is_empty() {
        None
    } else {
        Some(AxiomViolation::MissingEdges { missing })
    }
}

/// BFS in the directed schema: does `from` reach any node in
/// `targets`?
fn reaches_any(schema: &PGraphSchema, from: DeclId, targets: &BTreeSet<DeclId>) -> bool {
    use std::collections::VecDeque;
    // Build a one-shot adjacency map for the BFS.  This is O(|E|)
    // per call; for the scale where the axiom checker runs (small
    // P-graphs) that's fine.  When this becomes hot we can cache a
    // CSR on the schema.
    let mut adj: BTreeMap<DeclId, Vec<DeclId>> = BTreeMap::new();
    for (_, src, dst) in schema.edges() {
        adj.entry(src).or_default().push(dst);
    }
    let mut visited: BTreeSet<DeclId> = BTreeSet::new();
    let mut q: VecDeque<DeclId> = VecDeque::new();
    q.push_back(from);
    visited.insert(from);
    while let Some(v) = q.pop_front() {
        if targets.contains(&v) && v != from {
            return true;
        }
        if let Some(ns) = adj.get(&v) {
            for &n in ns {
                if visited.insert(n) {
                    q.push_back(n);
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{PGraphSchema, PNodeKind};
    use hymeko::common::ids::EdgeId;
    use std::collections::BTreeMap;

    fn d(i: usize) -> DeclId {
        DeclId::new(i)
    }
    fn e(i: usize) -> EdgeId {
        EdgeId::new(i)
    }

    /// A4 caught: an O-node with no incoming edges.
    #[test]
    fn a4_catches_isolated_unit() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::OperatingUnit),
            (d(1), PNodeKind::Material),
        ]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1)))]); // U → M, no input
        let schema = PGraphSchema::try_new(kinds, edges).expect("bipartite OK");
        let bundle = AxiomBundle::default();
        let products = BTreeSet::from([d(1)]);
        let result = bundle.validate(&schema, &products);
        match result {
            Err(v) => {
                assert!(v.iter().any(|x| matches!(
                    x,
                    AxiomViolation::DegreeViolations { offenders } if offenders.contains(&d(0))
                )));
            }
            Ok(()) => panic!("expected DegreeViolations on the isolated O-node"),
        }
    }

    /// A1 caught: a required product is not in the schema.
    #[test]
    fn a1_catches_missing_product() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),
            (d(1), PNodeKind::OperatingUnit),
        ]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1)))]);
        let schema = PGraphSchema::try_new(kinds, edges).expect("bipartite OK");
        let bundle = AxiomBundle::default();
        // d(99) is not in the schema.
        let products = BTreeSet::from([d(99)]);
        let result = bundle.validate(&schema, &products);
        match result {
            Err(v) => {
                assert!(v.iter().any(|x| matches!(
                    x,
                    AxiomViolation::MissingProducts { missing } if missing.contains(&d(99))
                )));
            }
            Ok(()) => panic!("expected MissingProducts"),
        }
    }

    /// Worked-example: 2-input/1-output unit with the product declared
    /// passes A1 and A4.
    #[test]
    fn worked_example_passes_a1_and_a4() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),
            (d(1), PNodeKind::Material),
            (d(2), PNodeKind::OperatingUnit),
            (d(3), PNodeKind::Material),
        ]);
        let edges = BTreeMap::from([
            (e(0), (d(0), d(2))),
            (e(1), (d(1), d(2))),
            (e(2), (d(2), d(3))),
        ]);
        let schema = PGraphSchema::try_new(kinds, edges).expect("bipartite OK");
        // A2 needs a raw set + reachability; A5 needs raws too.
        let bundle = AxiomBundle::new([d(0), d(1)], []);
        let products = BTreeSet::from([d(3)]);
        bundle
            .validate(&schema, &products)
            .expect("worked example satisfies A1, A2, A4, A5");
    }

    /// A2 caught: a material that does not reach any product.
    #[test]
    fn a2_catches_dead_material() {
        // A1 → U1 → P (good)
        // A2 → U2 → Junk (Junk is not a product)
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),      // A1 (raw)
            (d(1), PNodeKind::OperatingUnit), // U1
            (d(2), PNodeKind::Material),      // P (product)
            (d(3), PNodeKind::Material),      // A2 (raw)
            (d(4), PNodeKind::OperatingUnit), // U2
            (d(5), PNodeKind::Material),      // Junk (no consumer)
        ]);
        let edges = BTreeMap::from([
            (e(0), (d(0), d(1))),
            (e(1), (d(1), d(2))),
            (e(2), (d(3), d(4))),
            (e(3), (d(4), d(5))),
        ]);
        let schema = PGraphSchema::try_new(kinds, edges).unwrap();
        let bundle = AxiomBundle::new([d(0), d(3)], []);
        let products = BTreeSet::from([d(2)]);
        let result = bundle.validate(&schema, &products);
        match result {
            Err(v) => {
                assert!(v.iter().any(|x| matches!(
                    x,
                    AxiomViolation::UnreachableNodes { unreachable }
                        if unreachable.contains(&d(5))
                )));
            }
            Ok(()) => panic!("expected UnreachableNodes for Junk"),
        }
    }

    /// A3 caught: an O-node not in the master catalogue.
    #[test]
    fn a3_catches_unwhitelisted_unit() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),
            (d(1), PNodeKind::OperatingUnit), // legit
            (d(2), PNodeKind::Material),
            (d(3), PNodeKind::OperatingUnit), // not in catalogue
        ]);
        let edges = BTreeMap::from([
            (e(0), (d(0), d(1))),
            (e(1), (d(1), d(2))),
            (e(2), (d(0), d(3))),
            (e(3), (d(3), d(2))),
        ]);
        let schema = PGraphSchema::try_new(kinds, edges).unwrap();
        // Only d(1) is in the catalogue; d(3) must be flagged.
        let bundle = AxiomBundle::new([d(0)], [d(1)]);
        let products = BTreeSet::from([d(2)]);
        let result = bundle.validate(&schema, &products);
        match result {
            Err(v) => {
                assert!(v.iter().any(|x| matches!(
                    x,
                    AxiomViolation::InvalidUnits { invalid }
                        if invalid.contains(&d(3))
                )));
            }
            Ok(()) => panic!("expected InvalidUnits"),
        }
    }

    /// A5 caught: a non-raw material that is consumed but never produced.
    #[test]
    fn a5_catches_unproduced_consumed_material() {
        // U consumes M_phantom but no edge produces it, and M_phantom
        // is not raw.
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),      // raw
            (d(1), PNodeKind::OperatingUnit), // U
            (d(2), PNodeKind::Material),      // product
            (d(3), PNodeKind::Material),      // M_phantom
        ]);
        let edges = BTreeMap::from([
            (e(0), (d(0), d(1))),
            (e(1), (d(3), d(1))), // U also consumes M_phantom
            (e(2), (d(1), d(2))),
        ]);
        let schema = PGraphSchema::try_new(kinds, edges).unwrap();
        let bundle = AxiomBundle::new([d(0)], []); // raws = {d(0)}
        let products = BTreeSet::from([d(2)]);
        let result = bundle.validate(&schema, &products);
        match result {
            Err(v) => {
                assert!(v.iter().any(|x| matches!(
                    x,
                    AxiomViolation::MissingEdges { missing }
                        if missing.contains(&d(3))
                )));
            }
            Ok(()) => panic!("expected MissingEdges for M_phantom"),
        }
    }

    /// A5 *not* fired when the consumed material is raw.
    #[test]
    fn a5_silent_when_consumed_is_raw() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),
            (d(1), PNodeKind::OperatingUnit),
            (d(2), PNodeKind::Material),
        ]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1))), (e(1), (d(1), d(2)))]);
        let schema = PGraphSchema::try_new(kinds, edges).unwrap();
        let bundle = AxiomBundle::new([d(0)], []);
        let products = BTreeSet::from([d(2)]);
        bundle
            .validate(&schema, &products)
            .expect("raw material satisfies A5 trivially");
    }
}
