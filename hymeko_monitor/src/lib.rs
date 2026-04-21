//! # hymeko_monitor
//!
//! Runtime monitoring of temporal-logic properties over signed-incidence
//! directed hypergraphs (the HyMeKo intermediate representation).
//!
//! ## Scope
//!
//! This crate implements the semantics and the online monitor construction
//! described in the companion paper (`paper/paper_outline.tex`). It targets
//! the **bounded-horizon fragment of Signal Temporal Logic (STL)** over
//! traces of hypergraph states connected by attribute-update transitions.
//!
//! ## Entry points
//!
//! - Build a formula with [`formula::stl`] combinators.
//! - Wrap it in a [`monitor::stl::StlMonitor`].
//! - Feed samples with [`Monitor::observe`]; read verdicts with
//!   [`Monitor::verdict`].
//!
//! ## Non-goals (v0.1)
//!
//! See `SPEC.md`. In short: no CTL model checking, no unbounded LTL, no
//! distributed monitoring, no shield synthesis.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod formula;
pub mod incremental;
pub mod monitor;
pub mod predicate;
pub mod robustness;
pub mod trace;
pub mod window;

pub use monitor::{Monitor, Verdict};
pub use predicate::{Dependencies, HypergraphPredicate, HypergraphState, Sign};
pub use trace::{Sample, Timestamp};

/// Error type for monitor construction and evaluation.
#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    /// A formula contains an unbounded temporal operator; the online
    /// monitor requires bounded horizons.
    #[error("formula contains unbounded temporal operator; bounded horizons required")]
    UnboundedHorizon,

    /// Sample timestamps are not strictly increasing.
    #[error("non-monotonic timestamps: previous={prev}, new={new}")]
    NonMonotonicTime { prev: Timestamp, new: Timestamp },

    /// The formula references a predicate that cannot be evaluated on the
    /// supplied hypergraph state (e.g., missing attribute key).
    #[error("predicate evaluation failed: {0}")]
    PredicateError(String),
}
