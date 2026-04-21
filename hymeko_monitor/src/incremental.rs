//! Incremental predicate evaluation.
//!
//! Extends Rath–Bergmann–Ökrös–Varró (2008) incremental pattern matching
//! to signed-incidence hypergraphs. The core idea: each predicate caches
//! its last-evaluation match set and the set of vertex/hyperedge ids
//! whose update could invalidate it. On a new state, the engine diffs
//! the new state against the last and only re-evaluates predicates
//! whose dependency set intersects the diff.
//!
//! ## Status
//!
//! **SKELETON**. v0.1 may ship with pessimistic "always re-evaluate"
//! semantics and earn incrementality in v0.2. The trait surface below
//! is what the monitor calls; the interior strategy can start naive.

use crate::predicate::HypergraphState;

/// A delta between two consecutive hypergraph states.
#[derive(Debug, Default)]
pub struct StateDelta<H: HypergraphState> {
    /// Vertices whose attributes or type changed.
    pub changed_vertices: Vec<H::VertexId>,
    /// Hyperedges whose attributes, type, or incidences changed.
    pub changed_edges: Vec<H::EdgeId>,
    /// If true, fall back to global re-evaluation (e.g. for the first
    /// observation, or when the caller cannot compute a precise delta).
    pub global: bool,
}

impl<H: HypergraphState> StateDelta<H> {
    /// A delta that forces global re-evaluation.
    pub fn global() -> Self {
        Self { changed_vertices: Vec::new(), changed_edges: Vec::new(), global: true }
    }
}

/// Compute the delta between two states.
///
/// v0.1 ships a pessimistic implementation: always return `global`. v0.2
/// should diff the vertex / edge / attribute sets properly; this is
/// where the incremental speedup lives.
pub fn diff_states<H: HypergraphState>(_prev: &H, _next: &H) -> StateDelta<H> {
    StateDelta::global()
}
