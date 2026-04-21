//! Core traits: hypergraph states and predicates over them.
//!
//! The monitor is intentionally decoupled from the concrete HyMeKo IR.
//! A caller implements [`HypergraphState`] on their IR type (or wraps
//! `hymeko_core`'s type), and the monitor consumes it through the trait.
//!
//! This is the contract surface. Everything in `formula/`, `monitor/`, and
//! `incremental.rs` is written against these traits.

use std::hash::Hash;

/// The sign carried by an incidence in a signed-incidence hypergraph.
///
/// `Plus` marks source-role participation, `Minus` marks target-role,
/// `Neutral` marks unsigned (specification-only) participation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Sign {
    /// Source role (+).
    Plus,
    /// Target role (−).
    Minus,
    /// Neutral / specification-only role (~).
    Neutral,
}

/// A read-only snapshot of a HyMeKo-style hypergraph at a point in time.
///
/// Implementations are expected to be cheap to clone by reference or to
/// live behind a borrow. The monitor never retains states across observe
/// calls — it only extracts whatever predicate verdicts it needs.
pub trait HypergraphState {
    /// Vertex identifier type (should be small and `Copy`).
    type VertexId: Copy + Eq + Hash + std::fmt::Debug;
    /// Hyperedge identifier type.
    type EdgeId:   Copy + Eq + Hash + std::fmt::Debug;
    /// Type-lattice identifier type.
    type TypeId:   Copy + Eq + std::fmt::Debug;
    /// Attribute value type (scalar or tagged value).
    type Attr;

    /// Iterate all vertex ids.
    fn vertices(&self) -> Box<dyn Iterator<Item = Self::VertexId> + '_>;

    /// Iterate all hyperedge ids.
    fn edges(&self) -> Box<dyn Iterator<Item = Self::EdgeId> + '_>;

    /// Iterate incidences of a given hyperedge: `(participant, sign)`.
    fn incidences(
        &self,
        e: Self::EdgeId,
    ) -> Box<dyn Iterator<Item = (Self::VertexId, Sign)> + '_>;

    /// Type of a vertex.
    fn vertex_type(&self, v: Self::VertexId) -> Self::TypeId;

    /// Type of a hyperedge.
    fn edge_type(&self, e: Self::EdgeId) -> Self::TypeId;

    /// Subtype query over the type lattice T: returns true iff `t <: base`.
    fn inherits(&self, t: Self::TypeId, base: Self::TypeId) -> bool;

    /// Fetch a named attribute of a vertex.
    fn vertex_attr(&self, v: Self::VertexId, key: &str) -> Option<&Self::Attr>;

    /// Fetch a named attribute of a hyperedge.
    fn edge_attr(&self, e: Self::EdgeId, key: &str) -> Option<&Self::Attr>;
}

/// A predicate evaluable over a hypergraph state.
///
/// - [`HypergraphPredicate::eval`] gives the boolean reading.
/// - [`HypergraphPredicate::robustness`] gives the quantitative reading
///   used by STL. Returns `+∞` / `−∞` for structural predicates without
///   natural numeric content; otherwise returns the signed margin.
/// - [`HypergraphPredicate::dependencies`] reports which
///   vertices/hyperedges must be re-inspected on the next state; used by
///   [`crate::incremental`] to skip re-evaluation when nothing relevant
///   changed.
pub trait HypergraphPredicate<H: HypergraphState> {
    /// Boolean evaluation.
    fn eval(&self, h: &H) -> bool;

    /// Quantitative evaluation. Default: `+∞` on true, `−∞` on false.
    fn robustness(&self, h: &H) -> f64 {
        if self.eval(h) { f64::INFINITY } else { f64::NEG_INFINITY }
    }

    /// Dependencies — the set of ids whose update could change this
    /// predicate's verdict. Default: assume global (pessimistic).
    fn dependencies(&self, _h: &H) -> Dependencies<H> {
        Dependencies::global()
    }
}

/// The dependency footprint of a predicate evaluation. Used by the
/// incremental engine to decide whether re-evaluation is required.
#[derive(Debug)]
pub struct Dependencies<H: HypergraphState> {
    /// Specific vertices the predicate reads.
    pub vertices: Vec<H::VertexId>,
    /// Specific hyperedges the predicate reads.
    pub edges: Vec<H::EdgeId>,
    /// If true, re-evaluate on every state (used by truly global
    /// predicates like `ACYCLIC` or `CONNECTED`).
    pub global: bool,
}

impl<H: HypergraphState> Dependencies<H> {
    /// Pessimistic default: re-evaluate always.
    pub fn global() -> Self {
        Self { vertices: Vec::new(), edges: Vec::new(), global: true }
    }

    /// Specific-vertex dependency.
    pub fn of_vertices(vertices: Vec<H::VertexId>) -> Self {
        Self { vertices, edges: Vec::new(), global: false }
    }

    /// Specific-edge dependency.
    pub fn of_edges(edges: Vec<H::EdgeId>) -> Self {
        Self { vertices: Vec::new(), edges, global: false }
    }
}
