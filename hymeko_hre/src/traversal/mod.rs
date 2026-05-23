//! Hypergraph traversal routines owned by `hymeko_hre`.
//!
//! Today: [`berge`] BFS/DFS over the bipartite incidence representation that
//! `hymeko_core::traversal::hypergraphview::BergeView` already exposes. Each
//! routine takes an `&mut impl HypergraphVisitor` so callers can inject
//! pattern matching, tracing, or reactive hooks without touching the
//! traversal loop itself.
//!
//! Follow-ups: `tree` (decl-tree preorder/postorder), `weighted` (Dijkstra on
//! weighted incidence), and a broadcast-channel visitor for concurrent
//! subscribers.

pub mod berge;
