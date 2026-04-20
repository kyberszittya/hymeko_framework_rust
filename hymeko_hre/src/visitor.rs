//! Visitor pattern for hypergraph traversal events.
//!
//! [`HypergraphVisitor`] is a synchronous trait with default no-op methods, so
//! callers only implement the hooks they care about. Traversal routines (e.g.
//! [`crate::traversal::berge`]) call these hooks as they walk the bipartite
//! incidence graph, giving consumers a place to inject logic like pattern
//! matching, tracing, or live reactive updates without modifying the
//! traversal itself.
//!
//! ## Composition
//!
//! [`ChainVisitor`] multiplexes a traversal into several visitors in order;
//! each hook is called on every child in turn. Use this to combine, say, a
//! trace logger with a pattern matcher in a single walk.
//!
//! ## Concurrent / "lively" delivery (future)
//!
//! This module ships the synchronous L1 trait. L2 ("broadcast via
//! `crossbeam_channel::Sender`") and L3 (async `Stream`) visitors are
//! alternative implementations of the same trait and will land in follow-up
//! slices when we need them — the traversal loop stays unchanged.

use std::collections::BTreeMap;

use hymeko::common::ids::{DeclId, EdgeId, NodeId};
use hymeko_hnn::traversal::hypergraphview::BergeState;

/// A captured binding produced when a [`HypergraphVisitor`] matches a
/// pattern. Keyed by the variable name the author wrote in the pattern.
pub type MatchBindings = BTreeMap<String, DeclId>;

/// Events a traversal calls into a visitor at each step. All methods have
/// defaults that do nothing, so implementers only write what they need.
///
/// `depth` is the level in the BFS/DFS frontier (0 at the start node).
pub trait HypergraphVisitor {
    /// Fired the first time a node is visited.
    fn on_enter_node(&mut self, _nid: NodeId, _depth: usize) {}

    /// Fired the first time a hyperedge is visited.
    fn on_enter_edge(&mut self, _eid: EdgeId, _depth: usize) {}

    /// Fired when an incidence is traversed from `nid` to `eid` with
    /// signed-incidence `sign` (+1, -1, or 0 for neutral).
    fn on_incidence(&mut self, _nid: NodeId, _eid: EdgeId, _sign: i8) {}

    /// Fired by a pattern-matching visitor (see [`PatternMatcherVisitor`])
    /// when the current traversal position satisfies a named pattern.
    fn on_pattern_match(&mut self, _name: &str, _bindings: &MatchBindings) {}

    /// Return `false` to request an early termination of the traversal.
    /// Default `true` means "keep going".
    fn should_continue(&self) -> bool {
        true
    }
}

/// Fan a single traversal out to multiple visitors. Hooks fire in the order
/// the visitors were pushed; `should_continue` AND-combines the children.
pub struct ChainVisitor {
    children: Vec<Box<dyn HypergraphVisitor>>,
}

impl ChainVisitor {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub fn push<V: HypergraphVisitor + 'static>(&mut self, v: V) {
        self.children.push(Box::new(v));
    }

    pub fn with<V: HypergraphVisitor + 'static>(mut self, v: V) -> Self {
        self.push(v);
        self
    }

    pub fn len(&self) -> usize {
        self.children.len()
    }

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

impl Default for ChainVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HypergraphVisitor for ChainVisitor {
    fn on_enter_node(&mut self, nid: NodeId, depth: usize) {
        for v in self.children.iter_mut() {
            v.on_enter_node(nid, depth);
        }
    }
    fn on_enter_edge(&mut self, eid: EdgeId, depth: usize) {
        for v in self.children.iter_mut() {
            v.on_enter_edge(eid, depth);
        }
    }
    fn on_incidence(&mut self, nid: NodeId, eid: EdgeId, sign: i8) {
        for v in self.children.iter_mut() {
            v.on_incidence(nid, eid, sign);
        }
    }
    fn on_pattern_match(&mut self, name: &str, bindings: &MatchBindings) {
        for v in self.children.iter_mut() {
            v.on_pattern_match(name, bindings);
        }
    }
    fn should_continue(&self) -> bool {
        self.children.iter().all(|v| v.should_continue())
    }
}

/// A tiny example visitor — counts every node and edge visit. Useful as a
/// sanity check in tests and as a template for writing richer visitors.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingVisitor {
    pub nodes_visited: usize,
    pub edges_visited: usize,
    pub incidences_seen: usize,
}

impl HypergraphVisitor for CountingVisitor {
    fn on_enter_node(&mut self, _nid: NodeId, _depth: usize) {
        self.nodes_visited += 1;
    }
    fn on_enter_edge(&mut self, _eid: EdgeId, _depth: usize) {
        self.edges_visited += 1;
    }
    fn on_incidence(&mut self, _nid: NodeId, _eid: EdgeId, _sign: i8) {
        self.incidences_seen += 1;
    }
}

/// Collects the Berge-state visit order so tests can assert traversal shape.
#[derive(Debug, Default, Clone)]
pub struct TraceVisitor {
    pub trace: Vec<BergeState>,
}

impl HypergraphVisitor for TraceVisitor {
    fn on_enter_node(&mut self, nid: NodeId, _depth: usize) {
        self.trace.push(BergeState::Node(nid));
    }
    fn on_enter_edge(&mut self, eid: EdgeId, _depth: usize) {
        self.trace.push(BergeState::Edge(eid));
    }
}

/// Simple pattern matcher: fires [`HypergraphVisitor::on_pattern_match`] via
/// `sink` whenever the traversal enters an edge whose ID is in the
/// `target_edges` set, binding the captured DeclId under `capture_name`.
///
/// This is deliberately minimal — a real pattern matcher would consult a
/// predicate tree or a template. It demonstrates the wiring so visitors can
/// inject domain-specific matching logic without touching the traversal
/// loop.
pub struct PatternMatcherVisitor<F>
where
    F: FnMut(&str, &MatchBindings),
{
    pub name: String,
    pub target_edges: Vec<EdgeId>,
    pub capture_name: String,
    pub edge_to_decl: BTreeMap<EdgeId, DeclId>,
    pub sink: F,
}

impl<F> HypergraphVisitor for PatternMatcherVisitor<F>
where
    F: FnMut(&str, &MatchBindings),
{
    fn on_enter_edge(&mut self, eid: EdgeId, _depth: usize) {
        if !self.target_edges.contains(&eid) {
            return;
        }
        let Some(&decl) = self.edge_to_decl.get(&eid) else {
            return;
        };
        let mut bindings = MatchBindings::new();
        bindings.insert(self.capture_name.clone(), decl);
        (self.sink)(&self.name, &bindings);
    }
}
