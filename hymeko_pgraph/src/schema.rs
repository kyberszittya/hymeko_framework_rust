//! Phase 0 — schema and bipartite types.
//!
//! The P-graph schema is a partition of declarations in a HyMeKo IR
//! into *Material* (M) nodes and *Operating-Unit* (O) nodes, plus a
//! directed-edge set whose endpoints are constrained to alternate
//! between the two kinds. The constraint
//! ([`PGraphSchema::is_bipartite_consistent`]) is enforced at
//! construction time, not on every query.

use std::collections::{BTreeMap, BTreeSet};

use hymeko::common::ids::{DeclId, EdgeId};
use thiserror::Error;

/// Bipartite type tag for a P-graph node.
///
/// In the process-engineering reading: **Material** nodes are
/// inputs/outputs / intermediates (chemical species, parts, tensor
/// shapes); **OperatingUnit** nodes are processes that consume some
/// materials and produce others (reactors, machines, neural-network
/// layers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PNodeKind {
    /// Material node: a passive resource type.
    Material,
    /// Operating-Unit node: an active process that consumes / produces
    /// materials.
    OperatingUnit,
}

/// Errors raised when constructing a [`PGraphSchema`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PGraphError {
    /// A declaration ID has been assigned both [`PNodeKind::Material`]
    /// and [`PNodeKind::OperatingUnit`].
    #[error("decl {decl:?} assigned conflicting PNodeKinds: {first:?} vs {second:?}")]
    ConflictingKind {
        /// The conflicting declaration.
        decl: DeclId,
        /// The first kind assigned.
        first: PNodeKind,
        /// The second (conflicting) kind.
        second: PNodeKind,
    },
    /// A directed edge endpoint is not assigned a [`PNodeKind`].
    #[error("edge {edge:?} endpoint {decl:?} has no PNodeKind")]
    UnknownEndpoint {
        /// The edge with the missing endpoint.
        edge: EdgeId,
        /// The declaration without a kind.
        decl: DeclId,
    },
    /// A directed edge connects two declarations of the same
    /// [`PNodeKind`] (M↔M or O↔O), violating the bipartite constraint.
    #[error("edge {edge:?} non-bipartite: {src:?} ({src_kind:?}) → {dst:?} ({dst_kind:?})")]
    NonBipartite {
        /// The offending edge.
        edge: EdgeId,
        /// Source declaration.
        src: DeclId,
        /// Kind of source.
        src_kind: PNodeKind,
        /// Destination declaration.
        dst: DeclId,
        /// Kind of destination.
        dst_kind: PNodeKind,
    },
}

/// Bipartite Material / Operating-Unit overlay on a HyMeKo IR.
///
/// The schema is *not* the IR itself; it is a sidecar that interprets
/// IR declarations under a P-graph reading. Edges are directed and
/// indexed by the HyMeKo IR's [`EdgeId`].
#[derive(Debug, Clone)]
pub struct PGraphSchema {
    /// Per-declaration kind assignment.
    nodes: BTreeMap<DeclId, PNodeKind>,
    /// Directed edge endpoints, indexed by HyMeKo IR `EdgeId`.
    /// `(src, dst)` — by construction `kind(src) != kind(dst)`.
    edges: BTreeMap<EdgeId, (DeclId, DeclId)>,
    /// Cache: set of declarations that are O-nodes.
    o_nodes: BTreeSet<DeclId>,
    /// Cache: set of declarations that are M-nodes.
    m_nodes: BTreeSet<DeclId>,
}

impl PGraphSchema {
    /// Construct a [`PGraphSchema`] from explicit kind assignments and
    /// directed edges. Returns the schema if every edge is bipartite
    /// (M↔O), otherwise the first encountered violation.
    pub fn try_new(
        kinds: BTreeMap<DeclId, PNodeKind>,
        edges: BTreeMap<EdgeId, (DeclId, DeclId)>,
    ) -> Result<Self, PGraphError> {
        let mut o_nodes = BTreeSet::new();
        let mut m_nodes = BTreeSet::new();
        for (d, k) in &kinds {
            match k {
                PNodeKind::Material => {
                    m_nodes.insert(*d);
                }
                PNodeKind::OperatingUnit => {
                    o_nodes.insert(*d);
                }
            }
        }
        // Bipartite check on edges.
        for (e, (src, dst)) in &edges {
            let src_kind = kinds
                .get(src)
                .copied()
                .ok_or(PGraphError::UnknownEndpoint {
                    edge: *e,
                    decl: *src,
                })?;
            let dst_kind = kinds
                .get(dst)
                .copied()
                .ok_or(PGraphError::UnknownEndpoint {
                    edge: *e,
                    decl: *dst,
                })?;
            if src_kind == dst_kind {
                return Err(PGraphError::NonBipartite {
                    edge: *e,
                    src: *src,
                    src_kind,
                    dst: *dst,
                    dst_kind,
                });
            }
        }
        Ok(Self {
            nodes: kinds,
            edges,
            o_nodes,
            m_nodes,
        })
    }

    /// Return the kind assigned to a declaration, or `None` if the
    /// declaration is not part of the schema.
    pub fn kind(&self, decl: DeclId) -> Option<PNodeKind> {
        self.nodes.get(&decl).copied()
    }

    /// All Material declarations.
    pub fn m_nodes(&self) -> impl Iterator<Item = DeclId> + '_ {
        self.m_nodes.iter().copied()
    }

    /// All Operating-Unit declarations.
    pub fn o_nodes(&self) -> impl Iterator<Item = DeclId> + '_ {
        self.o_nodes.iter().copied()
    }

    /// All directed edges, in `(EdgeId, src, dst)` form.
    pub fn edges(&self) -> impl Iterator<Item = (EdgeId, DeclId, DeclId)> + '_ {
        self.edges.iter().map(|(e, (s, d))| (*e, *s, *d))
    }

    /// In-degree of a declaration: number of edges with `dst == decl`.
    pub fn in_degree(&self, decl: DeclId) -> usize {
        self.edges.values().filter(|(_, d)| *d == decl).count()
    }

    /// Out-degree of a declaration: number of edges with `src == decl`.
    pub fn out_degree(&self, decl: DeclId) -> usize {
        self.edges.values().filter(|(s, _)| *s == decl).count()
    }

    /// True iff every edge connects declarations of opposite kind.
    /// Always `true` for a schema constructed via [`Self::try_new`];
    /// exposed for downstream tests that mutate the schema.
    pub fn is_bipartite_consistent(&self) -> bool {
        for (_, (src, dst)) in &self.edges {
            let Some(sk) = self.nodes.get(src).copied() else {
                return false;
            };
            let Some(dk) = self.nodes.get(dst).copied() else {
                return false;
            };
            if sk == dk {
                return false;
            }
        }
        true
    }

    /// Number of M-nodes.
    pub fn n_materials(&self) -> usize {
        self.m_nodes.len()
    }

    /// Number of O-nodes.
    pub fn n_operating_units(&self) -> usize {
        self.o_nodes.len()
    }

    /// Number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn d(i: usize) -> DeclId {
        DeclId::new(i)
    }
    fn e(i: usize) -> EdgeId {
        EdgeId::new(i)
    }

    /// Smallest valid schema: 1 M-node, 1 O-node, 1 edge.
    #[test]
    fn smallest_valid() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),
            (d(1), PNodeKind::OperatingUnit),
        ]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1)))]);
        let s = PGraphSchema::try_new(kinds, edges).expect("valid bipartite");
        assert_eq!(s.n_materials(), 1);
        assert_eq!(s.n_operating_units(), 1);
        assert_eq!(s.n_edges(), 1);
        assert!(s.is_bipartite_consistent());
        assert_eq!(s.kind(d(0)), Some(PNodeKind::Material));
        assert_eq!(s.kind(d(1)), Some(PNodeKind::OperatingUnit));
        assert_eq!(s.in_degree(d(1)), 1);
        assert_eq!(s.out_degree(d(0)), 1);
    }

    /// M↔M edge rejected at construction time.
    #[test]
    fn rejects_m_to_m() {
        let kinds = BTreeMap::from([(d(0), PNodeKind::Material), (d(1), PNodeKind::Material)]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1)))]);
        let err = PGraphSchema::try_new(kinds, edges).unwrap_err();
        match err {
            PGraphError::NonBipartite {
                src_kind: PNodeKind::Material,
                dst_kind: PNodeKind::Material,
                ..
            } => {}
            other => panic!("expected NonBipartite(M, M), got {:?}", other),
        }
    }

    /// O↔O edge rejected at construction time.
    #[test]
    fn rejects_o_to_o() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::OperatingUnit),
            (d(1), PNodeKind::OperatingUnit),
        ]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1)))]);
        let err = PGraphSchema::try_new(kinds, edges).unwrap_err();
        assert!(matches!(err, PGraphError::NonBipartite { .. }));
    }

    /// Edge endpoint with no kind assignment rejected.
    #[test]
    fn rejects_unknown_endpoint() {
        let kinds = BTreeMap::from([(d(0), PNodeKind::Material)]);
        let edges = BTreeMap::from([(e(0), (d(0), d(1)))]); // d(1) has no kind
        let err = PGraphSchema::try_new(kinds, edges).unwrap_err();
        assert!(matches!(err, PGraphError::UnknownEndpoint { .. }));
    }

    /// Worked-example schema: a single operating unit consuming two
    /// materials and producing one — the canonical P-graph atom.
    #[test]
    fn unit_with_two_inputs_one_output() {
        let kinds = BTreeMap::from([
            (d(0), PNodeKind::Material),      // input A
            (d(1), PNodeKind::Material),      // input B
            (d(2), PNodeKind::OperatingUnit), // unit U
            (d(3), PNodeKind::Material),      // product P
        ]);
        let edges = BTreeMap::from([
            (e(0), (d(0), d(2))), // A → U
            (e(1), (d(1), d(2))), // B → U
            (e(2), (d(2), d(3))), // U → P
        ]);
        let s = PGraphSchema::try_new(kinds, edges).expect("valid");
        assert_eq!(s.n_materials(), 3);
        assert_eq!(s.n_operating_units(), 1);
        assert_eq!(s.in_degree(d(2)), 2);
        assert_eq!(s.out_degree(d(2)), 1);
    }
}
