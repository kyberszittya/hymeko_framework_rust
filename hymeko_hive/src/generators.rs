//! Deterministic hypergraph generators for HIVE tests and demos.
//!
//! The canonical combinatorial *algorithm* lives in [`hymeko::generators`]
//! (`hymeko_core`); this module is a thin **adapter** that turns a
//! [`hymeko::generators::HypergraphDesign`] (index-level vertices + edges) into
//! HIVE-store types ([`HiveNode`] / [`HiveRelation`]) and transaction-ready
//! deltas. There is no second copy of the generation logic here.
//!
//! These mirror the web editor's pure generators
//! (`docs/editor/views/generators.js`), but produce HIVE deltas directly
//! instead of `.hymeko` source text.

use std::collections::BTreeMap;

use hymeko::generators::HypergraphDesign;

use crate::{
    AttributeValue, HiveDelta, HiveNode, HiveRelation, HiveTransaction, NodeId, Sign, TransactionId,
};

/// Re-export the canonical generator error so HIVE callers see one error type.
pub use hymeko::generators::GeneratorError;
/// Re-export the canonical typed generator selector.
pub use hymeko::generators::HypergraphGenerator;

/// Generated HIVE hypergraph as a transaction-ready delta bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedHypergraph {
    pub name: String,
    pub nodes: Vec<HiveNode>,
    pub relations: Vec<HiveRelation>,
}

impl GeneratedHypergraph {
    /// Convert the generated object into add-node/add-relation deltas.
    #[must_use]
    pub fn deltas(&self) -> Vec<HiveDelta> {
        self.nodes
            .iter()
            .cloned()
            .map(HiveDelta::AddNode)
            .chain(self.relations.iter().cloned().map(HiveDelta::AddRelation))
            .collect()
    }

    /// Wrap the generated deltas in a transaction using the supplied parent hash.
    #[must_use]
    pub fn transaction(
        &self,
        id: impl Into<TransactionId>,
        parent_hash: crate::HiveHash,
        actor: impl Into<String>,
        seq: u64,
    ) -> HiveTransaction {
        HiveTransaction::new(id, parent_hash, actor, seq, self.deltas())
    }
}

/// Map a core [`HypergraphDesign`]'s edges into neutral-arc HIVE relations, using
/// `node_id` to resolve each member index to its HIVE node id.
fn relations_from_design(
    design: &HypergraphDesign,
    node_id: impl Fn(usize) -> NodeId,
    relation_ty: &str,
    relation_name: impl Fn(usize) -> String,
) -> Vec<HiveRelation> {
    design
        .edges
        .iter()
        .enumerate()
        .map(|(k, edge)| {
            HiveRelation::new(
                relation_name(k),
                relation_ty,
                edge.iter().map(|&m| (Sign::Neutral, node_id(m))).collect(),
            )
        })
        .collect()
}

/// Generate a Steiner triple system S(2,3,n) as HIVE points (`point`) and lines
/// (`line`). Delegates the combinatorics to [`hymeko::generators::steiner_design`].
///
/// # Errors
/// Propagates [`GeneratorError::InvalidSteinerOrder`] for an `n` outside the
/// existence condition (`n ≡ 1 or 3 mod 6`, `n ≥ 7`).
pub fn steiner_triple_system(n: usize) -> Result<GeneratedHypergraph, GeneratorError> {
    let design = hymeko::generators::steiner_design(n)?;
    let nodes = (0..design.n_vertices)
        .map(|i| HiveNode::new(format!("n{i}"), "point"))
        .collect();
    let relations = relations_from_design(
        &design,
        |i| NodeId(format!("n{i}")),
        "line",
        |k| format!("e{k}"),
    );
    Ok(GeneratedHypergraph {
        name: format!("STS_{n}"),
        nodes,
        relations,
    })
}

/// Generate a sunflower / delta-system. Each petal relation contains the same
/// core nodes plus its own private petal nodes. Delegates the index layout to
/// [`hymeko::generators::sunflower_design`].
///
/// # Errors
/// Propagates [`GeneratorError::InvalidSunflower`] when `petals == 0` or
/// `petal == 0`.
pub fn sunflower(
    petals: usize,
    core: usize,
    petal: usize,
) -> Result<GeneratedHypergraph, GeneratorError> {
    let design = hymeko::generators::sunflower_design(petals, core, petal)?;
    // Core index layout (contract): vertices 0..core are the shared core; the
    // rest are petals, petal `p` occupying core + p*petal .. core + (p+1)*petal.
    let node_id = move |idx: usize| -> NodeId {
        if idx < core {
            NodeId(format!("core_{idx}"))
        } else {
            let off = idx - core;
            NodeId(format!("p{}v{}", off / petal, off % petal))
        }
    };
    let nodes = (0..design.n_vertices)
        .map(|idx| {
            let (ty, role) = if idx < core {
                ("core_point", "core")
            } else {
                ("petal_point", "petal")
            };
            let mut node = HiveNode::new(node_id(idx), ty);
            node.attributes.insert(
                "sunflower_role".to_string(),
                AttributeValue::Text(role.to_string()),
            );
            node
        })
        .collect();
    let relations = relations_from_design(&design, node_id, "petal", |k| format!("petal_{k}"));
    Ok(GeneratedHypergraph {
        name: format!("Sunflower_{petals}_{core}_{petal}"),
        nodes,
        relations,
    })
}

/// Generate the complete `r`-uniform hypergraph K_n^(r) as HIVE points (`point`)
/// and hyperedges (`hyperedge`). Delegates the combinatorics to
/// [`hymeko::generators::complete_uniform_design`].
///
/// # Errors
/// Propagates [`GeneratorError::InvalidComplete`] when `r` is outside `2..=n`.
pub fn complete_uniform(n: usize, r: usize) -> Result<GeneratedHypergraph, GeneratorError> {
    let design = hymeko::generators::complete_uniform_design(n, r)?;
    let nodes = (0..design.n_vertices)
        .map(|i| HiveNode::new(format!("v{i}"), "point"))
        .collect();
    let relations = relations_from_design(
        &design,
        |i| NodeId(format!("v{i}")),
        "hyperedge",
        |k| format!("e{k}"),
    );
    Ok(GeneratedHypergraph {
        name: format!("Complete_{n}_{r}"),
        nodes,
        relations,
    })
}

/// Count unordered point-pair coverage in a generated triple system.
#[must_use]
pub fn pair_coverage(relations: &[HiveRelation]) -> BTreeMap<(NodeId, NodeId), usize> {
    let mut coverage = BTreeMap::new();
    for relation in relations {
        let mut points = relation
            .endpoints
            .iter()
            .map(|(_, node)| node.clone())
            .collect::<Vec<_>>();
        points.sort();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                *coverage
                    .entry((points[i].clone(), points[j].clone()))
                    .or_insert(0) += 1;
            }
        }
    }
    coverage
}

#[cfg(test)]
mod tests {
    use crate::{AssociationObject, AssociationQuery, EntitySelector, HiveStore};

    use super::*;

    #[test]
    fn fano_generator_matches_core_fixture_shape() {
        let generated = steiner_triple_system(7).unwrap();
        assert_eq!(generated.nodes.len(), 7);
        assert_eq!(generated.relations.len(), 7);
        assert!(
            generated
                .relations
                .iter()
                .all(|rel| rel.endpoints.len() == 3)
        );

        let coverage = pair_coverage(&generated.relations);
        assert_eq!(coverage.len(), 21);
        assert!(coverage.values().all(|count| *count == 1));
    }

    #[test]
    fn bose_steiner_system_covers_every_pair_once() {
        let generated = steiner_triple_system(9).unwrap();
        assert_eq!(generated.nodes.len(), 9);
        assert_eq!(generated.relations.len(), 12);

        let coverage = pair_coverage(&generated.relations);
        assert_eq!(coverage.len(), 36);
        assert!(coverage.values().all(|count| *count == 1));
    }

    /// Assert the generated HIVE relations form a valid S(2,3,n).
    fn assert_valid_sts(n: usize, generated: &GeneratedHypergraph) {
        assert_eq!(generated.nodes.len(), n);
        assert_eq!(generated.relations.len(), n * (n - 1) / 6, "triple count");
        assert!(
            generated
                .relations
                .iter()
                .all(|rel| rel.endpoints.len() == 3)
        );
        let coverage = pair_coverage(&generated.relations);
        assert_eq!(coverage.len(), n * (n - 1) / 2, "not every pair is covered");
        assert!(coverage.values().all(|count| *count == 1));
    }

    #[test]
    fn steiner_systems_valid_for_every_offered_order() {
        // 7 (Fano), 9/15/21 (Bose), 13/19/25 (MRV backtracking) — the adapter
        // must faithfully carry every order the core can build.
        for n in [7usize, 9, 13, 15, 19, 21, 25] {
            let generated = steiner_triple_system(n).unwrap();
            assert_valid_sts(n, &generated);
        }
    }

    #[test]
    fn steiner_rejects_orders_outside_the_existence_condition() {
        for bad in [4usize, 5, 6, 8, 10, 11, 12, 14] {
            assert_eq!(
                steiner_triple_system(bad).unwrap_err(),
                GeneratorError::InvalidSteinerOrder(bad),
                "should reject n={bad}"
            );
        }
    }

    #[test]
    fn complete_uniform_has_binom_edges_each_of_arity_r() {
        for (n, r) in [(4usize, 3usize), (5, 3), (6, 2), (6, 4)] {
            let generated = complete_uniform(n, r).unwrap();
            assert_eq!(generated.nodes.len(), n);
            assert_eq!(
                generated.relations.len() as u128,
                hymeko::generators::binom(n, r)
            );
            assert!(
                generated
                    .relations
                    .iter()
                    .all(|rel| rel.endpoints.len() == r)
            );
            let keys: std::collections::BTreeSet<Vec<String>> = generated
                .relations
                .iter()
                .map(|rel| {
                    let mut ids: Vec<String> =
                        rel.endpoints.iter().map(|(_, id)| id.0.clone()).collect();
                    ids.sort();
                    ids
                })
                .collect();
            assert_eq!(keys.len(), generated.relations.len(), "duplicate hyperedge");
        }
    }

    #[test]
    fn complete_uniform_rejects_arity_outside_2_to_n() {
        assert_eq!(
            complete_uniform(3, 5).unwrap_err(),
            GeneratorError::InvalidComplete { n: 3, r: 5 }
        );
        assert_eq!(
            complete_uniform(5, 1).unwrap_err(),
            GeneratorError::InvalidComplete { n: 5, r: 1 }
        );
    }

    #[test]
    fn generated_fano_supports_hive_association_queries() {
        let mut store = HiveStore::new();
        let generated = steiner_triple_system(7).unwrap();
        let tx = generated.transaction("sts-7", store.state_hash(), "generator", 0);
        store.commit(tx).unwrap();

        let points = store.query_associations(&AssociationQuery::node_kind("point"));
        assert_eq!(points.len(), 7);

        let lines = store.query_associations(&AssociationQuery::relation_kind("line"));
        assert_eq!(lines.len(), 7);

        let incident_to_n0 = store.query_associations(&AssociationQuery {
            subject: EntitySelector::AnyRelation,
            association: crate::AssociationKind::Endpoint {
                sign: Some(Sign::Neutral),
            },
            object: AssociationObject::Node("n0".into()),
        });
        assert_eq!(incident_to_n0.len(), 3);
    }

    #[test]
    fn generated_sunflower_has_core_intersection_and_queryable_roles() {
        let generated = sunflower(3, 2, 2).unwrap();
        assert_eq!(generated.nodes.len(), 8);
        assert_eq!(generated.relations.len(), 3);

        let mut store = HiveStore::new();
        let tx = generated.transaction("sunflower", store.state_hash(), "generator", 0);
        store.commit(tx).unwrap();

        let core = store.query_associations(&AssociationQuery::node_kind("core_point"));
        assert_eq!(core.len(), 2);

        let petals = store.query_associations(&AssociationQuery::relation_kind("petal"));
        assert_eq!(petals.len(), 3);

        let core_endpoint_matches = store.query_associations(&AssociationQuery {
            subject: EntitySelector::AnyRelation,
            association: crate::AssociationKind::Endpoint {
                sign: Some(Sign::Neutral),
            },
            object: AssociationObject::NodeType("core_point".to_string()),
        });
        assert_eq!(core_endpoint_matches.len(), 6);
    }
}
