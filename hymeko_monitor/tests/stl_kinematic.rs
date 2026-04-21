//! Integration test: kinematic-chain limit monitoring.
//!
//! This test is the executable form of Case Study 1 in
//! `paper/paper_outline.tex` (§5). It does NOT compile in v0.1 — it is
//! the *target* test that should pass once the monitor implementation
//! in `src/monitor/stl.rs` is complete.
//!
//! ## Scenario
//!
//! A 7-link anthropomorphic arm. Each joint has:
//!   - attribute `"position"`   — current joint angle (radians)
//!   - attribute `"limit_min"`  — soft lower limit
//!   - attribute `"limit_max"`  — soft upper limit
//!   - tag       `"collab"`     — whether the robot is in
//!                                collaborative mode
//!
//! Property:
//!     G_{[0, 30]} ( collab → G_{[0, 0.1]} (
//!         ∀ joint. position(joint) ∈ [limit_min(joint), limit_max(joint)]
//!     ) )
//!
//! Trace: 30 s simulated motion, 100 Hz sampling. At t = 12.3 s, joint 3
//! exceeds its limit_max. The monitor must:
//!   - emit positive robustness for t < 12.2 s
//!   - emit robustness crossing zero between the sample before and the
//!     sample after the limit breach
//!   - emit negative robustness for t ≥ 12.3 s
//!
//! ## How to run (once the monitor is implemented)
//!
//! ```text
//! cargo test --release --test stl_kinematic
//! ```

// NB: this file is currently a commented-out target. Uncomment and fill
// in the fixture once `StlMonitor::observe` is implemented.

/*
use std::sync::Arc;

use hymeko_monitor::formula::stl::*;
use hymeko_monitor::formula::ltl::always;
use hymeko_monitor::monitor::{stl::StlMonitor, Monitor};
use hymeko_monitor::predicate::{HypergraphPredicate, HypergraphState};
use hymeko_monitor::trace::Sample;

// A minimal in-memory HypergraphState implementation for testing.
// The real test fixture pulls from `hymeko_core` once wired in.
mod fixture {
    // Fields: vertex ids, edge ids, attributes keyed by (id, key) -> f64,
    // tags per vertex.
    // Mutable knobs: set_attr(v, k, val), set_tag(v, tag, bool).
}

#[test]
fn detects_limit_violation_at_t_12_3() {
    // 1. Build the arm fixture (7 joints, collab=true, all limits nominal).
    // 2. Build the property phi (see module doc).
    // 3. Construct the monitor.
    // 4. For each sample in the 3000-sample trace:
    //       monitor.observe(sample);
    // 5. Assert:
    //       - monitor.verdict() has positive robustness through t=12.2
    //       - robustness crosses zero between t=12.2 and t=12.4
    //       - robustness is negative from t=12.3 onward
    todo!("fixture + trace + assertions");
}
*/
