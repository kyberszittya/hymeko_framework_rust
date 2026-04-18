//! Berge / Levi bipartite traversal — walks the Node ↔ Edge incidence
//! structure of a [`HyperGraphView`]. Each enter-node / enter-edge event
//! is reported to the supplied [`HypergraphVisitor`], so callers can attach
//! pattern matchers, tracers, or live reactive hooks without modifying the
//! loop.
//!
//! ## Naming — "Berge" vs "Levi" vs "König"
//!
//! The bipartite graph on vertex set `V ⊔ E` with an edge `{v, e}` for every
//! incidence `v ∈ e` has three names in the literature: **Levi graph**
//! (Levi, 1942), **Berge representation** (Berge, *Hypergraphs*, 1973),
//! and **König representation**. This module uses `berge_*` identifiers
//! (and `hymeko_core::traversal::hypergraphview::BergeState` / `BergeView`
//! from core) for historical reasons. Readers who prefer the Levi-graph
//! name can use the aliases at the bottom of this module
//! ([`levi_bfs`], [`levi_dfs`], and the `LeviState` / `LeviView`
//! re-exports from `hymeko_core::traversal::hypergraphview`) — they point
//! at the same code.
//!
//! Two front-ends:
//! - [`berge_bfs`] — level-order, using `VecDeque`.
//! - [`berge_dfs`] — pre-order, using a stack.
//!
//! Both consult `visitor.should_continue()` after every event so a visitor
//! can request early termination (handy for "find first match" workflows).

use std::collections::VecDeque;

use hymeko::common::ids::{EdgeId, NodeId};
use hymeko::tensor::common::Real;
use hymeko::tensor::tensor_val::{EdgeWeight, IncVal};
use hymeko::traversal::graphview::GraphView;
use hymeko::traversal::hypergraphview::{BergeState, BergeView, HyperGraphView};

use crate::visitor::HypergraphVisitor;

/// Dense visited-set keyed by `BergeState`, O(1) mark + check.
struct VisitedSet {
    nodes: Vec<bool>,
    edges: Vec<bool>,
}

impl VisitedSet {
    fn new(num_nodes: usize, num_edges: usize) -> Self {
        Self {
            nodes: vec![false; num_nodes],
            edges: vec![false; num_edges],
        }
    }

    /// Mark the state as visited and return the *previous* value — so the
    /// caller treats `true` as "already seen, skip".
    #[inline(always)]
    fn mark_and_check(&mut self, s: BergeState) -> bool {
        match s {
            BergeState::Node(n) => {
                let v = self.nodes[n.0];
                self.nodes[n.0] = true;
                v
            }
            BergeState::Edge(e) => {
                let v = self.edges[e.0];
                self.edges[e.0] = true;
                v
            }
        }
    }
}

#[inline]
fn fire_enter<V: HypergraphVisitor>(visitor: &mut V, state: BergeState, depth: usize) {
    match state {
        BergeState::Node(nid) => visitor.on_enter_node(nid, depth),
        BergeState::Edge(eid) => visitor.on_enter_edge(eid, depth),
    }
}

/// Fire `on_incidence` whenever we step from a Node to an Edge. `sign` comes
/// from the parallel `flat_node_sign` array maintained by `HyperGraphView`.
#[inline]
fn fire_incidence_on_transition<V, EW, F, Vis>(
    hg: &HyperGraphView<V, EW, F>,
    from: BergeState,
    to: BergeState,
    visitor: &mut Vis,
) where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
    Vis: HypergraphVisitor,
{
    if let (BergeState::Node(nid), BergeState::Edge(eid)) = (from, to) {
        // Look up the sign of this (nid, eid) incidence in the flat array.
        let (s, e) = (hg.node_offsets[nid.0], hg.node_offsets[nid.0 + 1]);
        for p in s..e {
            if hg.flat_node_edges[p] == eid {
                visitor.on_incidence(nid, eid, hg.flat_node_sign[p]);
                return;
            }
        }
    }
}

/// Berge BFS from `start`. Events are ordered: the start state's enter hook
/// first, then each neighbour's enter hook in level-order. Incidence hooks
/// fire for every Node→Edge transition.
pub fn berge_bfs<V, EW, F, Vis>(view: &BergeView<'_, V, EW, F>, start: BergeState, visitor: &mut Vis)
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
    Vis: HypergraphVisitor,
{
    let num_nodes = view.hg.num_nodes();
    let num_edges = view.hg.num_edges();
    let mut seen = VisitedSet::new(num_nodes, num_edges);

    let mut q: VecDeque<(BergeState, usize)> = VecDeque::new();
    seen.mark_and_check(start);
    fire_enter(visitor, start, 0);
    if !visitor.should_continue() {
        return;
    }
    q.push_back((start, 0));

    while let Some((u, depth)) = q.pop_front() {
        for v in view.neighbors(u) {
            fire_incidence_on_transition(view.hg, u, v, visitor);
            if !visitor.should_continue() {
                return;
            }
            if !seen.mark_and_check(v) {
                fire_enter(visitor, v, depth + 1);
                if !visitor.should_continue() {
                    return;
                }
                q.push_back((v, depth + 1));
            }
        }
    }
}

/// Berge DFS pre-order from `start`. Same hook contract as [`berge_bfs`];
/// traversal order is depth-first via an explicit stack.
pub fn berge_dfs<V, EW, F, Vis>(view: &BergeView<'_, V, EW, F>, start: BergeState, visitor: &mut Vis)
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
    Vis: HypergraphVisitor,
{
    let num_nodes = view.hg.num_nodes();
    let num_edges = view.hg.num_edges();
    let mut seen = VisitedSet::new(num_nodes, num_edges);

    let mut stack: Vec<(BergeState, usize)> = Vec::new();
    stack.push((start, 0));

    while let Some((u, depth)) = stack.pop() {
        if seen.mark_and_check(u) {
            continue;
        }
        fire_enter(visitor, u, depth);
        if !visitor.should_continue() {
            return;
        }

        for v in view.neighbors(u) {
            fire_incidence_on_transition(view.hg, u, v, visitor);
            if !visitor.should_continue() {
                return;
            }
            stack.push((v, depth + 1));
        }
    }
}

/// Convenience helper: BFS from a node ID.
pub fn berge_bfs_from_node<V, EW, F, Vis>(
    view: &BergeView<'_, V, EW, F>,
    nid: NodeId,
    visitor: &mut Vis,
) where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
    Vis: HypergraphVisitor,
{
    berge_bfs(view, BergeState::Node(nid), visitor);
}

/// Convenience helper: BFS from an edge ID.
pub fn berge_bfs_from_edge<V, EW, F, Vis>(
    view: &BergeView<'_, V, EW, F>,
    eid: EdgeId,
    visitor: &mut Vis,
) where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
    Vis: HypergraphVisitor,
{
    berge_bfs(view, BergeState::Edge(eid), visitor);
}

// ---------------------------------------------------------------------------
// Levi-graph aliases
// ---------------------------------------------------------------------------
//
// Same functions under the Levi-graph name. Zero cost — these are `pub use`
// aliases, not wrappers.

pub use berge_bfs as levi_bfs;
pub use berge_dfs as levi_dfs;
pub use berge_bfs_from_node as levi_bfs_from_node;
pub use berge_bfs_from_edge as levi_bfs_from_edge;
