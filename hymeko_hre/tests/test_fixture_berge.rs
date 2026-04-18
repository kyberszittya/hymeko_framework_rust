//! End-to-end Berge traversal + visitor tests on real `.hymeko` fixtures.
//!
//! Loads `data/robotics/*.hymeko`, builds a `HyperGraphView` via the full
//! `ModuleStore → Ir → from_ir` pipeline, then walks it with the visitors
//! shipped in `hymeko_hre::visitor`. Real fixtures import
//! `meta_kinematics.hymeko`, which contributes many standalone type-decl
//! nodes that are NOT incident to any hyperedge — so BFS from a random
//! `NodeId` may not walk anywhere. The tests therefore start from a node
//! that is known to be incident to at least one hyperedge (the first node
//! in the first edge's incidence list).

mod common;

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use hymeko::common::ids::{DeclId, EdgeId, NodeId};
use hymeko::tensor::tensor_val::EdgeWScalar;
use hymeko::traversal::hypergraphview::{BergeState, BergeView, HyperGraphView};

use hymeko_hre::traversal::berge::{berge_bfs_from_node, berge_dfs};
use hymeko_hre::visitor::{
    ChainVisitor, CountingVisitor, HypergraphVisitor, MatchBindings, PatternMatcherVisitor,
    TraceVisitor,
};

use crate::common::{load_and_lower, view_f32};

const MINI_ARM: &str = "../data/robotics/mini_arm.hymeko";
const MOVEO_ARM: &str = "../data/robotics/anthropomorphic_arm.hymeko";

type View = HyperGraphView<f32, EdgeWScalar<f32>, f32>;

/// Pick a node guaranteed to be incident to at least one hyperedge.
/// Uses the first node of the first edge's incidence span.
fn first_connected_node(hg: &View) -> NodeId {
    assert!(hg.num_edges() > 0, "view has no edges");
    hg.flat_edge_nodes[hg.edge_offsets[0]]
}

/// Set of edges reachable (in the Berge bipartite sense) from `start`.
fn reachable_edges_from(view: &BergeView<'_, f32, EdgeWScalar<f32>, f32>, start: NodeId) -> BTreeSet<EdgeId> {
    let mut tracer = TraceVisitor::default();
    berge_bfs_from_node(view, start, &mut tracer);
    tracer
        .trace
        .into_iter()
        .filter_map(|s| match s {
            BergeState::Edge(eid) => Some(eid),
            _ => None,
        })
        .collect()
}

// ---- mini_arm: traversal reaches more than just the start state --------

#[test]
fn mini_arm_berge_bfs_from_connected_node_reaches_multiple_states() {
    let (_store, compiled) = load_and_lower(MINI_ARM).expect("mini_arm");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };

    let start = first_connected_node(&hg);
    let mut trace = TraceVisitor::default();
    berge_bfs_from_node(&view, start, &mut trace);

    // The robot subgraph `{base_link, spinner, spin_joint}` in mini_arm is a
    // connected component of at least 3 Berge states.
    assert!(
        trace.trace.len() >= 3,
        "expected at least 3 visited states in the connected component, got {}",
        trace.trace.len()
    );
    assert_eq!(trace.trace[0], BergeState::Node(start));
}

#[test]
fn mini_arm_counting_visitor_consistent_with_trace() {
    let (_store, compiled) = load_and_lower(MINI_ARM).expect("mini_arm");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };

    let start = first_connected_node(&hg);

    let mut trace = TraceVisitor::default();
    berge_bfs_from_node(&view, start, &mut trace);

    let mut counter = CountingVisitor::default();
    berge_bfs_from_node(&view, start, &mut counter);

    let trace_nodes = trace
        .trace
        .iter()
        .filter(|s| matches!(s, BergeState::Node(_)))
        .count();
    let trace_edges = trace
        .trace
        .iter()
        .filter(|s| matches!(s, BergeState::Edge(_)))
        .count();

    assert_eq!(
        counter.nodes_visited, trace_nodes,
        "counting visitor must agree with tracer on nodes"
    );
    assert_eq!(
        counter.edges_visited, trace_edges,
        "counting visitor must agree with tracer on edges"
    );
    assert!(counter.incidences_seen >= counter.edges_visited);
}

// ---- moveo: pattern matcher fires for reachable target edges -----------

#[test]
fn moveo_pattern_matcher_fires_once_per_reachable_target_edge() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };

    let start = first_connected_node(&hg);
    // Probe which edges are actually reachable from `start` in this view.
    let reachable: BTreeSet<EdgeId> = reachable_edges_from(&view, start);
    assert!(
        !reachable.is_empty(),
        "expected at least one reachable edge from the connected start node"
    );

    // Target the reachable set. The matcher must fire exactly once per
    // target edge since BFS visits each edge at most once.
    let targets: Vec<EdgeId> = reachable.iter().copied().collect();
    let edge_to_decl: std::collections::BTreeMap<_, _> = targets
        .iter()
        .enumerate()
        .map(|(k, &eid)| (eid, DeclId::new(k + 1_000)))
        .collect();

    let hits: Rc<RefCell<Vec<MatchBindings>>> = Rc::new(RefCell::new(Vec::new()));
    let sink_hits = Rc::clone(&hits);

    let matcher = PatternMatcherVisitor {
        name: "any_joint".to_string(),
        target_edges: targets.clone(),
        capture_name: "joint".to_string(),
        edge_to_decl,
        sink: move |_name: &str, b: &MatchBindings| {
            sink_hits.borrow_mut().push(b.clone());
        },
    };

    let mut chain = ChainVisitor::new().with(matcher);
    berge_bfs_from_node(&view, start, &mut chain);

    assert_eq!(
        hits.borrow().len(),
        targets.len(),
        "pattern matcher should fire once per reachable target edge"
    );
    // Every binding must carry the `joint` capture we asked for.
    for bindings in hits.borrow().iter() {
        assert!(bindings.contains_key("joint"));
    }
}

#[test]
fn moveo_pattern_matcher_ignores_unreachable_edges() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };

    // Pick a target that is definitely out-of-range (no such edge id).
    let bogus = EdgeId::new(hg.num_edges() + 10_000);
    let matcher = PatternMatcherVisitor {
        name: "never_match".to_string(),
        target_edges: vec![bogus],
        capture_name: "x".to_string(),
        edge_to_decl: std::collections::BTreeMap::new(),
        sink: |_name: &str, _b: &MatchBindings| panic!("should never fire"),
    };

    let start = first_connected_node(&hg);
    let mut chain = ChainVisitor::new().with(matcher);
    berge_bfs_from_node(&view, start, &mut chain);
    // No panic == success.
}

// ---- moveo: DFS preorder state IDs are in range -----------------------

#[test]
fn moveo_dfs_preorder_state_ids_are_in_range() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };

    let mut trace = TraceVisitor::default();
    berge_dfs(&view, BergeState::Node(first_connected_node(&hg)), &mut trace);

    for state in &trace.trace {
        match state {
            BergeState::Node(nid) => assert!(nid.0 < hg.num_nodes()),
            BergeState::Edge(eid) => assert!(eid.0 < hg.num_edges()),
        }
    }
    assert!(trace.trace.len() >= 3);
}

// ---- moveo: early-exit short-circuits the traversal --------------------

#[test]
fn moveo_visitor_can_stop_traversal_early() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };
    let start = first_connected_node(&hg);

    // Baseline: how many edges does BFS normally reach?
    let baseline_reachable = reachable_edges_from(&view, start).len();
    assert!(
        baseline_reachable >= 2,
        "need at least 2 reachable edges to exercise early-exit; got {}",
        baseline_reachable
    );

    /// Visitor that stops after the 1st enter-edge event.
    struct StopAfterOneEdge {
        seen_edges: usize,
    }
    impl HypergraphVisitor for StopAfterOneEdge {
        fn on_enter_edge(&mut self, _eid: EdgeId, _depth: usize) {
            self.seen_edges += 1;
        }
        fn should_continue(&self) -> bool {
            self.seen_edges < 1
        }
    }

    let mut v = StopAfterOneEdge { seen_edges: 0 };
    berge_bfs_from_node(&view, start, &mut v);

    assert_eq!(v.seen_edges, 1, "exactly one edge should have been visited before stop");
    assert!(
        v.seen_edges < baseline_reachable,
        "early-exit should prevent visiting every reachable edge ({} / baseline {})",
        v.seen_edges,
        baseline_reachable
    );
}

// ---- chain visitor: trace + matcher share a single walk ----------------

#[test]
fn moveo_chain_visitor_traces_and_matches_in_one_walk() {
    let (_store, compiled) = load_and_lower(MOVEO_ARM).expect("moveo");
    let hg = view_f32(&compiled);
    let view = BergeView { hg: &hg };

    let start = first_connected_node(&hg);
    let reachable: BTreeSet<EdgeId> = reachable_edges_from(&view, start);
    let target = *reachable
        .iter()
        .next()
        .expect("at least one reachable edge");

    let hits: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
    let sink_hits = Rc::clone(&hits);

    let matcher = PatternMatcherVisitor {
        name: "first_reachable".to_string(),
        target_edges: vec![target],
        capture_name: "edge".to_string(),
        edge_to_decl: std::collections::BTreeMap::from([(target, DeclId::new(7))]),
        sink: move |_name: &str, _b: &MatchBindings| {
            *sink_hits.borrow_mut() += 1;
        },
    };

    let mut chain = ChainVisitor::new()
        .with(CountingVisitor::default())
        .with(matcher);
    berge_bfs_from_node(&view, start, &mut chain);

    assert_eq!(*hits.borrow(), 1, "target edge matcher must fire exactly once");
}
