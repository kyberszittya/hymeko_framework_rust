//! Optional **umbrella** crate for the HyMeKo workspace.
//!
//! The workspace already shares one `[workspace]` root (`Cargo.toml` at the repo
//! root). This crate exists for consumers who want a **single dependency line**
//! and explicit feature flags instead of listing `hymeko_core`, `hymeko_graph`,
//! etc. separately.
//!
//! # Re-exports
//!
//! - [`hymeko`] — always available (package `hymeko_core`, library name `hymeko`).
//! - [`clifford`] — [`hymeko_clifford`] when feature `clifford` is enabled.
//! - [`compute`] — [`hymeko_compute`] when feature `gpu` is enabled.
//! - [`graph`] — [`hymeko_graph`] when feature `graph` is enabled.
//!
//! # Example (`Cargo.toml`)
//!
//! ```toml
//! [dependencies]
//! hymeko_kit = { path = "../hymeko_kit", features = ["graph", "clifford"] }
//! ```

#![warn(missing_docs)]

pub use hymeko;

#[cfg(feature = "clifford")]
pub use hymeko_clifford as clifford;

#[cfg(feature = "gpu")]
pub use hymeko_compute as compute;

#[cfg(feature = "graph")]
pub use hymeko_graph as graph;
