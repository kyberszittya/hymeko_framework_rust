use crate::common::ids::{NodeId, EdgeId, DeclId};
use crate::ir::ir::{Ir, SignedRefR};
use crate::traversal::graphview::{GraphView};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BergeState {
    Node(NodeId),
    Edge(EdgeId),
}

pub struct HyperGraphView {
    /// NodeId -> list of incident EdgeId
    pub node_to_edges: Vec<Vec<EdgeId>>,
    /// EdgeId -> list of incident NodeId
    pub edge_to_nodes: Vec<Vec<NodeId>>,

    pub node_decl: Vec<DeclId>, // NodeId -> DeclId
    pub edge_decl: Vec<DeclId>, // EdgeId -> DeclId
}

impl HyperGraphView {
    /// Constructor: default empty view (for testing)



    /// Build incidence tables from IR (single source of truth).
    pub fn from_ir(ir: &Ir) -> Self {
        // NodeId and EdgeId are dense indices into ir.nodes / ir.edges.
        let mut node_to_edges: Vec<Vec<EdgeId>> = vec![Vec::new(); ir.nodes.len()];
        let mut edge_to_nodes: Vec<Vec<NodeId>> = vec![Vec::new(); ir.edges.len()];

        let node_decl: Vec<DeclId> = ir.nodes.iter().map(|n| n.decl).collect();
        let edge_decl: Vec<DeclId> = ir.edges.iter().map(|e| e.decl).collect();


        // Helper: extract target DeclId from SignedRefR
        fn ref_target(r: &SignedRefR) -> DeclId {
            match r {
                SignedRefR::Plus(a) => a.target,
                SignedRefR::Minus(a) => a.target,
                SignedRefR::Neutral(a) => a.target,
            }
        }

        for (eid_usize, edge) in ir.edges.iter().enumerate() {
            let eid = EdgeId(eid_usize as u32);

            // collect nodes for this edge (dedup via temporary bool/HashSet)
            // use a small HashSet because arcs may repeat nodes
            let mut seen_nodes = std::collections::HashSet::<NodeId>::new();

            for &aid in &edge.arcs {
                let arc = &ir.arcs[aid.0 as usize];

                for r in &arc.refs {
                    let did = ref_target(r);

                    if let Some(nid) = ir.decl_to_node[did.0 as usize] {
                        if seen_nodes.insert(nid) {
                            edge_to_nodes[eid_usize].push(nid);
                            node_to_edges[nid.0 as usize].push(eid);
                        }
                    }
                }
            }

            // Optional: keep deterministic order
            edge_to_nodes[eid_usize].sort_by_key(|n| n.0);
            edge_to_nodes[eid_usize].dedup();
        }

        // Optional: deterministic order for node_to_edges too
        for v in &mut node_to_edges {
            v.sort_by_key(|e| e.0);
            v.dedup();
        }

        Self {
            node_to_edges,
            edge_to_nodes,
            node_decl,
            edge_decl,
        }
    }
}

pub struct BergeView<'a> {
    pub hg: &'a HyperGraphView,
}

impl<'a> GraphView for BergeView<'a> {
    type Node = BergeState;

    type NeighIter<'b> = Box<dyn Iterator<Item = BergeState> + 'b>
    where
        Self: 'b;

    fn neighbors<'b>(&'b self, s: BergeState) -> Self::NeighIter<'b> {
        match s {
            BergeState::Node(nid) => Box::new(
                self.hg.node_to_edges[nid.0 as usize]
                    .iter()
                    .copied()
                    .map(BergeState::Edge),
            ),
            BergeState::Edge(eid) => Box::new(
                self.hg.edge_to_nodes[eid.0 as usize]
                    .iter()
                    .copied()
                    .map(BergeState::Node),
            ),
        }
    }
}