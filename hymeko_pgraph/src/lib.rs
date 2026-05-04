//! # `hymeko_pgraph`
//!
//! P-graph framework over the canonical signed-incidence hypergraph IR
//! defined in [`hymeko_core::ir`]. P-graphs are a well-known
//! process-engineering formalism (Friedler, Tarján, Huang, Fan 1992)
//! for modelling synthesis problems as bipartite Material / Operating-Unit
//! graphs with five structural axioms (A1–A5). This crate embeds the
//! P-graph formalism into the HyMeKo IR so:
//!
//! 1. axiom-checking becomes a query on top of the existing IR
//!    (Phase 1, [`axioms`]);
//! 2. Maximal Structure Generation (MSG) and Solution Structure
//!    Generation (SSG) become constructive operations over the IR
//!    (Phase 2, [`msg`] / [`ssg`]);
//! 3. an entropy-guided Accelerated Branch-and-Bound (ABB) solves
//!    NAS / synthesis problems using the IR's signed-incidence
//!    spectral entropy (Phase 3, [`entropy`] / [`abb`]).
//!
//! ## Phase 0 (this crate, today)
//!
//! - [`schema::PNodeKind`]: bipartite type tag — Material vs.
//!   OperatingUnit.
//! - [`schema::PGraphSchema`]: per-decl `PNodeKind` map plus a directed
//!   edge set, with [`schema::PGraphSchema::is_bipartite_consistent`]
//!   gate enforced at construction.
//! - [`AxiomViolation`] enum scaffold (A1–A5) for Phase 1.
//!
//! Subsequent phases (axioms, MSG/SSG, entropy oracle, ABB) populate
//! the modules listed in the plan at
//! `docs/plans/plans_20260429/hymeko_pgraph_plan.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod axioms;
pub mod schema;

pub use axioms::AxiomViolation;
pub use schema::{PGraphError, PGraphSchema, PNodeKind};
