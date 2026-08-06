//! Layer-1 facade: the HyMeKo gatekeeper.
//!
//! [`HymekoEngine`] wraps the CORE `parser` crate (`parser::parse_description`,
//! the `evaluateSyntax` action). It is an Adapter/Facade — it adds no algorithm
//! logic of its own; parsing stays behind the locked parser boundary. On a valid
//! refinement it commits the source as the new [`Ambience`] (the `mutateActiveState`
//! → `freezeState` transition); on a syntax error it returns structured feedback
//! for the error loop.

use std::collections::HashMap;

use parser::ast::{AstStr, HyperItem, SignedRef};

use crate::context::Ambience;
use crate::synthesize::Refinement;

/// Rendered syntax-failure feedback (`hymeko.errorFeedback`).
#[derive(Debug, Clone)]
pub struct ErrorFeedback {
    /// Human/agent-readable parser diagnostic.
    pub message: String,
}

/// Outcome of gate-keeping one refinement.
#[derive(Debug, Clone)]
pub enum EvalOutcome {
    /// Parsed and committed; carries the new ambience generation.
    Accepted { generation: u64 },
    /// Rejected by the parser; carries the feedback for self-correction.
    Rejected(ErrorFeedback),
}

/// The deterministic engine's public face to AKOIRE.
#[derive(Debug, Default)]
pub struct HymekoEngine {
    ambience: Ambience,
}

impl HymekoEngine {
    /// A fresh engine with empty ambience.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ambience: Ambience::empty(),
        }
    }

    /// Borrow the current frozen state.
    #[must_use]
    pub fn ambience(&self) -> &Ambience {
        &self.ambience
    }

    /// Gate-keep one refinement.
    ///
    /// # Postconditions
    /// - On `Accepted`, the ambience generation has increased by one and its
    ///   `source`/`edge_names` reflect `refinement`.
    /// - On `Rejected`, the ambience is unchanged and the feedback message is
    ///   non-empty.
    pub fn evaluate(&mut self, refinement: &Refinement) -> EvalOutcome {
        match parser::parse_description(&refinement.0) {
            Ok(ast) => {
                let edge_names = collect_edge_names(&ast);
                // Commit while the borrow `ast` is still alive is unnecessary —
                // names are already owned `String`s, so we clone the source.
                self.ambience.commit(refinement.0.clone(), edge_names);
                EvalOutcome::Accepted {
                    generation: self.ambience.generation(),
                }
            }
            // Token/LexError derive `Debug`, not `Display`; `{:?}` is the
            // available rendering and is sufficient for agent feedback.
            Err(e) => EvalOutcome::Rejected(ErrorFeedback {
                message: format!("{e:?}"),
            }),
        }
    }
}

/// Dry-run parse (non-mutating): `Some((name, arity))` per edge if `source` parses, `None` on a
/// syntax error. Arity is the referenced-vertex count (`EdgeInner.bases.len()`).
///
/// The search planner ([`crate::hotaru::SearchHotaru`]) uses this as its feasibility + edge oracle,
/// so a candidate HIVE-delta state is vetted through the *same* parser boundary
/// [`HymekoEngine::evaluate`] gate-keeps with — never a re-implemented check. Unlike `evaluate`,
/// it commits nothing, so a planner may probe arbitrarily many hypothetical states (and check them
/// against the `Kyosei` arity bound).
///
/// # Postconditions
/// `Some(edges)` ⇒ `parser::parse_description(source)` succeeded and `edges` are its `(name, arity)`
/// pairs; `None` ⇒ it failed. No state is mutated.
#[must_use]
pub fn preview_edges(source: &str) -> Option<Vec<(String, usize)>> {
    parser::parse_description(source)
        .ok()
        .map(|ast| collect_edges(&ast))
}

/// Dry-run parse returning just the edge names — the name projection of [`preview_edges`].
#[must_use]
pub fn preview_edge_names(source: &str) -> Option<Vec<String>> {
    preview_edges(source).map(|edges| edges.into_iter().map(|(name, _)| name).collect())
}

/// Extract every edge name from a parsed description — the name projection of [`collect_edges`].
fn collect_edge_names(ast: &AstStr<'_>) -> Vec<String> {
    collect_edges(ast)
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// Extract every edge as `(name, arity)` from a parsed description (depth-first, including edges
/// nested in node/edge bodies). Arity is the number of referenced vertices. Owned strings so the
/// result outlives the borrowed AST. The single traversal both name- and arity-consumers derive from.
fn collect_edges(ast: &AstStr<'_>) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    collect_items(&ast.items, &mut out);
    out
}

fn collect_items(items: &[HyperItem<'_, &str>], out: &mut Vec<(String, usize)>) {
    for item in items {
        match item {
            HyperItem::Edge(edge) => {
                out.push((edge.inner.name.to_string(), edge.inner.bases.len()));
                collect_items(&edge.inner.body, out);
            }
            HyperItem::Node(node) => {
                if let Some(body) = &node.inner.body {
                    collect_items(body, out);
                }
            }
            HyperItem::Arc(_) => {}
        }
    }
}

/// One hyperedge of a [`GraphView`]: its name and the nodes it connects.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// The edge's declared name.
    pub name: String,
    /// The names of the nodes the edge references (its `bases`).
    pub nodes: Vec<String>,
}

/// A structural view of a parsed HyMeKo model — the declared nodes and the hyperedges over them.
///
/// This is the *graph the source describes*: what HOTARU plans over when it produces a semantic plan
/// (operations grounded in real nodes/edges), rather than opaque source strings.
#[derive(Debug, Clone, Default)]
pub struct GraphView {
    nodes: Vec<String>,
    edges: Vec<GraphEdge>,
}

impl GraphView {
    /// The declared node names.
    #[must_use]
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// The hyperedges (name + connected nodes).
    #[must_use]
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Whether a node with this name is declared.
    #[must_use]
    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.iter().any(|n| n == name)
    }

    /// Whether an edge with this name exists.
    #[must_use]
    pub fn has_edge(&self, name: &str) -> bool {
        self.edges.iter().any(|e| e.name == name)
    }

    /// Whether `a` and `b` lie in the same connected component, treating each hyperedge as fully
    /// connecting the nodes it references (union-find over the incidence).
    ///
    /// # Postconditions
    /// `false` if either name is not a declared node; `true` when `a == b` and it is a node.
    #[must_use]
    pub fn connected(&self, a: &str, b: &str) -> bool {
        if !self.has_node(a) || !self.has_node(b) {
            return false;
        }
        if a == b {
            return true;
        }
        let index: HashMap<&str, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let mut parent: Vec<usize> = (0..self.nodes.len()).collect();
        for edge in &self.edges {
            let mut members = edge
                .nodes
                .iter()
                .filter_map(|n| index.get(n.as_str()).copied());
            if let Some(anchor) = members.next() {
                for m in members {
                    let (ra, rb) = (uf_find(&mut parent, anchor), uf_find(&mut parent, m));
                    parent[ra] = rb;
                }
            }
        }
        uf_find(&mut parent, index[a]) == uf_find(&mut parent, index[b])
    }
}

/// Iterative union-find root with path halving.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Dry-run parse into a [`GraphView`] (nodes + hyperedges), or `None` on a syntax error.
///
/// The graph-element oracle HOTARU's semantic planner ([`crate::hotaru::SearchHotaru::plan_graph`]) reads
/// to ground its candidate operations in the model's real nodes and to test topological goals.
#[must_use]
pub fn preview_graph(source: &str) -> Option<GraphView> {
    parser::parse_description(source)
        .ok()
        .map(|ast| collect_graph(&ast))
}

fn collect_graph(ast: &AstStr<'_>) -> GraphView {
    let mut nodes: Vec<String> = ast
        .header
        .iter()
        .map(|n| n.inner.name.to_string())
        .collect();
    let mut edges: Vec<GraphEdge> = Vec::new();
    collect_graph_items(&ast.items, &mut nodes, &mut edges);
    GraphView { nodes, edges }
}

fn collect_graph_items(
    items: &[HyperItem<'_, &str>],
    nodes: &mut Vec<String>,
    edges: &mut Vec<GraphEdge>,
) {
    for item in items {
        match item {
            HyperItem::Node(node) => {
                nodes.push(node.inner.name.to_string());
                if let Some(body) = &node.inner.body {
                    collect_graph_items(body, nodes, edges);
                }
            }
            HyperItem::Edge(edge) => {
                edges.push(GraphEdge {
                    name: edge.inner.name.to_string(),
                    nodes: edge.inner.bases.iter().map(ref_name).collect(),
                });
                collect_graph_items(&edge.inner.body, nodes, edges);
            }
            HyperItem::Arc(_) => {}
        }
    }
}

/// The referenced node name of a signed reference (its dotted path joined).
fn ref_name(sr: &SignedRef<'_, &str>) -> String {
    let atom = match sr {
        SignedRef::Plus(a) | SignedRef::Minus(a) | SignedRef::Neutral(a) => a,
    };
    atom.target.path.join(".")
}
