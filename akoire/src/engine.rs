//! Layer-1 facade: the HyMeKo gatekeeper.
//!
//! [`HymekoEngine`] wraps the CORE `parser` crate (`parser::parse_description`,
//! the `evaluateSyntax` action). It is an Adapter/Facade — it adds no algorithm
//! logic of its own; parsing stays behind the locked parser boundary. On a valid
//! refinement it commits the source as the new [`Ambience`] (the `mutateActiveState`
//! → `freezeState` transition); on a syntax error it returns structured feedback
//! for the error loop.

use parser::ast::{AstStr, HyperItem};

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

/// Extract every edge name from a parsed description (depth-first, including
/// edges nested in node/edge bodies). Returns owned strings so the result
/// outlives the borrowed AST.
fn collect_edge_names(ast: &AstStr<'_>) -> Vec<String> {
    let mut out = Vec::new();
    collect_items(&ast.items, &mut out);
    out
}

fn collect_items(items: &[HyperItem<'_, &str>], out: &mut Vec<String>) {
    for item in items {
        match item {
            HyperItem::Edge(edge) => {
                out.push(edge.inner.name.to_string());
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
