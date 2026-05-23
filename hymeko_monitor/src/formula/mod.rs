//! Formula ASTs for LTL, CTL (parse-only in v0.1), and STL.
//!
//! STL is the primary target; LTL is a thin wrapper that uses
//! `[0, +∞)`-horizon STL operators; CTL is present as an AST only
//! (model checking deferred to v0.2 when structural rewrites land).

pub mod ltl;
pub mod stl;
// pub mod ctl;   // v0.2
