//! Hypergraph Rewriting Engine (HRE) — stateful orchestrator that compiles
//! [`hymeko_core::ir::ir::Ir`] into sparse tensor expansions and, under the
//! `ipc` feature, streams those expansions into shared memory.
//!
//! See `docs/plans/05_hre_extraction/plan.md` for the rationale behind
//! splitting this crate out of `hymeko_core`.

// IPC / raw-pointer helpers and the iceoryx subscriber scaffold predate strict
// `clippy -D warnings` defaults on every workspace feature combination.
#![allow(
    dead_code,
    mismatched_lifetime_syntaxes,
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
)]

pub mod engine;
pub mod expansion;
pub mod traversal;
pub mod visitor;

pub use engine::hypergraphengine::HypergraphEngine;
