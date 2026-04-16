//! Query-driven graph rewriting: decouple domain transforms from the query engine.
//!
//! Instead of hardcoded URDF/SDF/MJCF generators, transforms are defined by:
//!   1. **Query definitions** (.hymeko files) — what to match
//!   2. **Templates** (.xml/.sdf/.dot templates) — how to emit
//!
//! The rewrite engine connects them: match → extract fields → render template.
//!
//! ```text
//! queries.hymeko ──→ interpret_as_queries() ──→ Vec<NamedQuery>
//!                                                      │
//!                                               QueryEngine.query_batch()
//!                                                      │
//!                                               HashMap<label, Vec<QueryMatch>>
//!                                                      │
//! template.xml ──→ parse_template() ──────────→ render(blocks, results, config)
//!                                                      │
//!                                                   output string
//! ```

pub mod match_context;
pub mod template;

pub use template::{execute_transform, TransformSpec};
