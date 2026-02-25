use crate::common::ids::{NodeId, EdgeId, DeclId};
use crate::ir::ir::{Ir, SignedRefR, ValueR};
use crate::traversal::aggregation::{agg_sign, agg_weight, AggCfg};
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
    /// Parallel array: sign/weight info per node-edge incidence (aligned with flat_node_edges)
    pub flat_node_sign: Vec<i8>,
    /// Offsets into flat_node_edges for each NodeId.
    pub node_offsets: Vec<usize>,
    /// Flat array of all incident nodes.
    pub flat_edge_nodes: Vec<NodeId>,
    /// Parallel array: sign per edge-node incidence (aligned with flat_edge_nodes)
    pub flat_edge_sign: Vec<i8>,
    /// Offsets into flat_edge_nodes for each EdgeId.
    pub edge_offsets: Vec<usize>,
    /// Optional weights (e.g., for GNNs), aligned with the corresponding flat arrays.
    pub flat_node_w: Vec<f32>, // aligned with flat_node_edges
    pub flat_edge_w: Vec<f32>, // aligned with flat_edge_nodes
    pub edge_weight: Vec<f32>, // per EdgeId (global)


}

#[inline(always)]
fn extract_ref_weight_scalar(weights: &Option<Vec<ValueR>>) -> f32 {
    // Default: ha nincs weight, akkor 1.0
    let Some(ws) = weights else { return 1.0; };

    // Flatten 1 szintig: [Num(..)] vagy [[..]] eseteket is kezelünk
    let mut nums: Vec<f32> = Vec::new();

    for v in ws {
        match v {
            ValueR::Num(x) => nums.push(*x as f32),
            ValueR::List(xs) => {
                for vv in xs {
                    if let ValueR::Num(x) = vv {
                        nums.push(*x as f32);
                    }
                }
            }
            _ => {}
        }
    }

    if nums.is_empty() { 1.0 }
    else if nums.len() == 1 { nums[0] }
    else {
        // Többdimenziós eseteknél ez egy “placeholder” döntés.
        // Most SUM: evidenciagyűjtés jelleg (és determinisztikus).
        nums.into_iter().sum()
    }
}

impl HyperGraphView {
    #[inline(always)]
    fn sign_of(r: &SignedRefR) -> i8 {
        match r {
            SignedRefR::Plus(_) =>  1,
            SignedRefR::Minus(_) => -1,
            SignedRefR::Neutral(_) => 0,
        }
    }



    /// Deterministic aggregation when the same (edge,node) appears multiple times.
    /// Policy: prefer non-neutral; if conflicting (+ and -), collapse to 0 (neutral).
    #[inline(always)]
    fn agg_sign(a: i8, b: i8) -> i8 {
        if a == b { return a; }
        if a == 0 { return b; }
        if b == 0 { return a; }
        // conflict: (+1,-1) -> 0
        0
    }

    /// Number of nodes in the view.
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        // node_offsets has length = num_nodes + 1
        self.node_offsets.len().saturating_sub(1)
    }

    /// Number of edges in the view.
    #[inline(always)]
    pub fn num_edges(&self) -> usize {
        self.edge_offsets.len().saturating_sub(1)
    }

    /// Returns the [start, end) range for a node's incident edges.
    #[inline(always)]
    pub fn node_span(&self, nid: NodeId) -> (usize, usize) {
        let s = self.node_offsets[nid.0 as usize] as usize;
        let e = self.node_offsets[nid.0 as usize + 1] as usize;
        (s, e)
    }

    /// Returns the [start, end) range for an edge's incident nodes.
    #[inline(always)]
    pub fn edge_span(&self, eid: EdgeId) -> (usize, usize) {
        let s = self.edge_offsets[eid.0 as usize] as usize;
        let e = self.edge_offsets[eid.0 as usize + 1] as usize;
        (s, e)
    }

    /// Borrow the slice of incident edges for a given node.
    #[inline(always)]
    pub fn node_edges(&self, nid: NodeId) -> &[EdgeId] {
        let (s, e) = self.node_span(nid);
        &self.flat_node_edges[s..e]
    }

    /// Borrow the slice of incident nodes for a given edge.
    #[inline(always)]
    pub fn edge_nodes(&self, eid: EdgeId) -> &[NodeId] {
        let (s, e) = self.edge_span(eid);
        &self.flat_edge_nodes[s..e]
    }



    /// Build incidence tables from IR (single source of truth).
    pub fn from_ir(ir: &Ir, cfg: &AggCfg) -> Self {
        let num_nodes = ir.nodes.len();
        let num_edges = ir.edges.len();

        let node_decl: Vec<DeclId> = ir.nodes.iter().map(|n| n.decl).collect();
        let edge_decl: Vec<DeclId> = ir.edges.iter().map(|e| e.decl).collect();

        let mut triples: Vec<(EdgeId, NodeId, i8, f32)> = Vec::new();

        for (eid_usize, edge) in ir.edges.iter().enumerate() {
            let eid = EdgeId(eid_usize);
            for &aid in &edge.arcs {
                let arc = &ir.arcs[aid.0];

                let arc_w: f32 = 1.0;

                for r in &arc.refs {
                    let (target, _sgn) = match r {
                        SignedRefR::Plus(a) | SignedRefR::Minus(a) | SignedRefR::Neutral(a)
                        => (a.target, Self::sign_of(r)),
                    };

                    if let Some(nid) = ir.decl_to_node[target.0] {
                        let (_target, sgn, ref_w) = match r {
                            SignedRefR::Plus(a) => (a.target,  1i8, extract_ref_weight_scalar(&a.weights)),
                            SignedRefR::Minus(a) => (a.target, -1i8, extract_ref_weight_scalar(&a.weights)),
                            SignedRefR::Neutral(a) => (a.target, 0i8, extract_ref_weight_scalar(&a.weights)),
                        };


                        let w = arc_w * ref_w;
                        triples.push((eid, nid, sgn, w));
                    }
                }
            }
        }

        // sort by (eid,nid)
        triples.sort_unstable_by_key(|&(e,n,_,_)| (e.0, n.0));

        // reduce/dedup
        let mut pairs: Vec<(EdgeId, NodeId, i8, f32)> = Vec::with_capacity(triples.len());
        for (eid, nid, sgn, w) in triples {
            match pairs.last_mut() {
                Some((pe, pn, ps, pw)) if pe.0 == eid.0 && pn.0 == nid.0 => {
                    // előbb weight, aztán sign (sign agg használhat súlyt)
                    let new_w = agg_weight(cfg, *pw, w);
                    let new_s = agg_sign(cfg, *ps, sgn, *pw, w);
                    *pw = new_w;
                    *ps = new_s;
                }
                _ => pairs.push((eid, nid, sgn, w)),
            }
        }

        // CSR: Edge -> Nodes
        let mut flat_edge_nodes = Vec::with_capacity(pairs.len());
        let mut flat_edge_sign  = Vec::with_capacity(pairs.len());
        let mut flat_edge_w     = Vec::with_capacity(pairs.len());
        let mut edge_offsets = vec![0usize; num_edges + 1];

        for &(eid, nid, sgn, w) in &pairs {
            flat_edge_nodes.push(nid);
            flat_edge_sign.push(sgn);
            flat_edge_w.push(w);
            edge_offsets[eid.0 as usize + 1] += 1;
        }
        for i in 0..num_edges { edge_offsets[i+1] += edge_offsets[i]; }

        // CSR: Node -> Edges
        pairs.sort_unstable_by_key(|&(e,n,_,_)| (n.0, e.0));

        let mut flat_node_edges = Vec::with_capacity(pairs.len());
        let mut flat_node_sign  = Vec::with_capacity(pairs.len());
        let mut flat_node_w     = Vec::with_capacity(pairs.len());
        let mut node_offsets = vec![0usize; num_nodes + 1];

        for &(eid, nid, sgn, w) in &pairs {
            flat_node_edges.push(eid);
            flat_node_sign.push(sgn);
            flat_node_w.push(w);
            node_offsets[nid.0 as usize + 1] += 1;
        }
        for i in 0..num_nodes { node_offsets[i+1] += node_offsets[i]; }

        Self {
            node_decl,
            edge_decl,
            flat_node_edges,
            flat_node_sign,
            flat_node_w,
            node_offsets,
            flat_edge_nodes,
            flat_edge_sign,
            flat_edge_w,
            edge_offsets,
            edge_weight: vec![1.0; num_edges],
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

