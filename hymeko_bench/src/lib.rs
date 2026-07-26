//! Shared harness support for the SMC 2026 camera-ready measurements.
//!
//! Split by concern (CLAUDE.md §6.5 #4): allocation accounting, synthetic
//! corpus generation, statistics, and IR structural accounting. The binaries
//! `bench_memory` (Task 1) and `bench_scaling_sweep` (Task 2) compose these;
//! neither re-implements the pieces.

pub mod corpus;
pub mod incidence;
pub mod stats;
pub mod track_alloc;

pub use track_alloc::{TrackingAlloc, current_bytes, measure, peak_bytes, reset_peak};
