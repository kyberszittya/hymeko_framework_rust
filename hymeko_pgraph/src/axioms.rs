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

use std::collections::BTreeSet;

use hymeko::common::ids::DeclId;

use crate::schema::PGraphSchema;

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
/// Constructed empty; exposes [`AxiomBundle::validate`] as the single
/// entry point. Phase 1 of the plan replaces the stubbed checks with
/// real implementations.
#[derive(Debug, Default)]
pub struct AxiomBundle;

impl AxiomBundle {
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

        // A2, A3, A5 — Phase 1 plan items, parked.
        // (A2 reuses the existing A* / D* Lite traversal once
        //  hymeko_core::traversal exposes a compatible reachability
        //  primitive; A3 needs an `is_unit` predicate on declarations;
        //  A5 needs the consumption-edge invariant from the IR side.)

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
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
        let bundle = AxiomBundle::default();
        let products = BTreeSet::from([d(3)]);
        bundle
            .validate(&schema, &products)
            .expect("worked example satisfies A1 + A4");
    }
}
