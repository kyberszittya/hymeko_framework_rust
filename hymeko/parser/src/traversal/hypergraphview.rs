use crate::common::ids::{NodeId, EdgeId, DeclId};
use crate::ir::ir::{Ir, SignedRefR};
use crate::traversal::graphview::{GraphView};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BergeState {
    Node(NodeId),
    Edge(EdgeId),
}

pub enum BergeIter<'a> {
    Node(std::slice::Iter<'a, EdgeId>),
    Edge(std::slice::Iter<'a, NodeId>),
}

impl<'a> Iterator for BergeIter<'a> {
    type Item = BergeState;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BergeIter::Node(it) => it.next().map(|&e| BergeState::Edge(e)),
            BergeIter::Edge(it) => it.next().map(|&n| BergeState::Node(n)),
        }
    }
}

pub struct HyperGraphView {
    pub node_decl: Vec<DeclId>, // NodeId -> DeclId
    pub edge_decl: Vec<DeclId>, // EdgeId -> DeclId
    /// Flat array of all incident edges.
    pub flat_node_edges: Vec<EdgeId>,
    /// Offsets into flat_node_edges for each NodeId.
    pub node_offsets: Vec<u32>,
    /// Flat array of all incident nodes.
    pub flat_edge_nodes: Vec<NodeId>,
    /// Offsets into flat_edge_nodes for each EdgeId.
    pub edge_offsets: Vec<u32>,


}

impl HyperGraphView {
    /// Constructor: default empty view (for testing)



    /// Build incidence tables from IR (single source of truth).
    pub fn from_ir(ir: &Ir) -> Self {
        let num_nodes = ir.nodes.len();
        let num_edges = ir.edges.len();

        let node_decl: Vec<DeclId> = ir.nodes.iter().map(|n| n.decl).collect();
        let edge_decl: Vec<DeclId> = ir.edges.iter().map(|e| e.decl).collect();

        // Phase 1: Collect raw pairs
        let mut pairs: Vec<(EdgeId, NodeId)> = Vec::new();

        for (eid_usize, edge) in ir.edges.iter().enumerate() {
            let eid = EdgeId(eid_usize as u32);
            for &aid in &edge.arcs {
                let arc = &ir.arcs[aid.0 as usize];
                for r in &arc.refs {
                    let target = match r {
                        SignedRefR::Plus(a) | SignedRefR::Minus(a) | SignedRefR::Neutral(a) => a.target,
                    };
                    if let Some(nid) = ir.decl_to_node[target.0 as usize] {
                        pairs.push((eid, nid));
                    }
                }
            }
        }

        // Phase 2: Sort and Deduplicate (The DOD way)
        pairs.sort_unstable_by_key(|&(e, n)| (e.0, n.0));
        pairs.dedup();

        // Phase 3: Build CSR for Edge -> Nodes
        let mut flat_edge_nodes = Vec::with_capacity(pairs.len());
        let mut edge_offsets = vec![0; num_edges + 1];

        for &(eid, nid) in &pairs {
            flat_edge_nodes.push(nid);
            edge_offsets[eid.0 as usize + 1] += 1; // Count degrees
        }
        for i in 0..num_edges {
            edge_offsets[i + 1] += edge_offsets[i]; // Prefix sum
        }

        // Phase 4: Build CSR for Node -> Edges (Reverse sort)
        pairs.sort_unstable_by_key(|&(e, n)| (n.0, e.0));

        let mut flat_node_edges = Vec::with_capacity(pairs.len());
        let mut node_offsets = vec![0; num_nodes + 1];

        for &(eid, nid) in &pairs {
            flat_node_edges.push(eid);
            node_offsets[nid.0 as usize + 1] += 1;
        }
        for i in 0..num_nodes {
            node_offsets[i + 1] += node_offsets[i];
        }

        Self {
            node_decl, edge_decl,
            flat_node_edges, node_offsets,
            flat_edge_nodes, edge_offsets,
        }
    }
}

pub struct BergeView<'a> {
    pub hg: &'a HyperGraphView,
}

impl<'a> GraphView for BergeView<'a> {
    type Node = BergeState;
    type NeighIter<'b> = BergeIter<'b> where Self: 'b;

    #[inline(always)]
    fn neighbors<'b>(&'b self, s: BergeState) -> Self::NeighIter<'b> {
        match s {
            BergeState::Node(nid) => {
                let start = self.hg.node_offsets[nid.0 as usize] as usize;
                let end = self.hg.node_offsets[nid.0 as usize + 1] as usize;
                BergeIter::Node(self.hg.flat_node_edges[start..end].iter())
            },
            BergeState::Edge(eid) => {
                let start = self.hg.edge_offsets[eid.0 as usize] as usize;
                let end = self.hg.edge_offsets[eid.0 as usize + 1] as usize;
                BergeIter::Edge(self.hg.flat_edge_nodes[start..end].iter())
            },
        }
    }
}