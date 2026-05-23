//! PyO3 cycle-enumeration boundary.
//!
//! Decomposed 2026-05-11 per CLAUDE.md §6.5 anti-pattern #4 (no
//! ≥300-LOC single file) and the user's "no monstrosity" rule. The
//! algorithm code is in `hymeko_graph::{unsigned_cycles, color_coding,
//! path_closure, walks_unsigned, cycle_sampler}`; this module is the
//! thin PyO3↔numpy bridge.
//!
//! Layout:
//!   * `io`         — numpy / SignedGraph / scorer / vertex-filter glue
//!   * `unsigned`   — `enumerate_unsigned_rs` + 4 legacy thin wrappers (DFS /
//!     color-coding / path-closure / walks)
//!   * `per_vertex` — `enumerate_cycles_rs` Strategy entry (8-variant collapse)
//!   * `top_k`      — `enumerate_top_k_cycles_rs` + `_entropy_rs` (4-variant collapse)
//!
//! Re-exports keep `crate::cycles::<fn>` paths unchanged so `lib.rs`'s
//! module-registration block stays as-is.

mod io;
mod per_vertex;
mod top_k;
mod unsigned;

pub use per_vertex::enumerate_cycles_rs;
pub use top_k::{enumerate_top_k_cycles_entropy_rs, enumerate_top_k_cycles_rs};
pub use unsigned::{
    enumerate_k_cycles_color_coded_rs, enumerate_k_cycles_path_closure_rs,
    enumerate_k_cycles_rs, enumerate_k_walks_rs, enumerate_unsigned_rs,
};
