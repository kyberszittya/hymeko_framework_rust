//! Integration test: kinematic-chain limit monitoring.
//!
//! Executable form of Case Study 1 in `paper/paper_outline.tex` (§5.1).
//!
//! ## Scenario
//!
//! A 7-link anthropomorphic arm. Each link is a vertex with attributes:
//!   - `"position"`   — current joint angle (radians)
//!   - `"limit_min"`  — soft lower limit
//!   - `"limit_max"`  — soft upper limit
//!
//! All links are tagged collaborative-mode for the duration of the
//! trace. The property the SPEC names is:
//!
//!     G_{[0, 30]} ( collab → G_{[0, 0.1]} (
//!         ∀ joint. limit_min ≤ position ≤ limit_max
//!     ) )
//!
//! For the test we use a rolling-1-second outer window
//! (`G_{[0, 1]}` instead of `G_{[0, 30]}`) so verdicts settle within
//! the trace duration; the inner safety property is unchanged. The
//! property still expresses "collaborative-mode + safety holds for
//! the next 0.1 s, evaluated continuously."
//!
//! Trace: 30 s @ 100 Hz (3000 samples). At t = 12.30 s, joint 3
//! violates `limit_max` for one sample.
//!
//! ## What this test asserts
//!
//! - Through the violation-free portion of the trace, the verdict
//!   robustness is positive.
//! - Once the verdict's settled time crosses the violation, robustness
//!   becomes negative.

use std::collections::HashMap;
use std::sync::Arc;

use hymeko_monitor::formula::stl::*;
use hymeko_monitor::monitor::stl::StlMonitor;
use hymeko_monitor::predicate::{
    Dependencies, HypergraphPredicate, HypergraphState, Sign,
};
use hymeko_monitor::trace::Sample;
use hymeko_monitor::Monitor;

// ─── Fixture: 7-link arm hypergraph state ─────────────────────────────

const N_JOINTS: u32 = 7;

#[derive(Clone)]
struct ArmState {
    /// joint id (0..N_JOINTS) → (position, limit_min, limit_max)
    joints: HashMap<u32, [f64; 3]>,
    /// whether collaborative mode is active globally
    collab: bool,
}

impl ArmState {
    fn new() -> Self {
        let mut joints = HashMap::new();
        for j in 0..N_JOINTS {
            joints.insert(j, [0.0, -3.14, 3.14]);
        }
        Self { joints, collab: true }
    }
    fn set_position(&mut self, joint: u32, pos: f64) {
        if let Some(rec) = self.joints.get_mut(&joint) {
            rec[0] = pos;
        }
    }
}

impl HypergraphState for ArmState {
    type VertexId = u32;
    type EdgeId = u32;
    type TypeId = u32;
    type Attr = f64;

    fn vertices(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(self.joints.keys().copied())
    }
    fn edges(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(std::iter::empty())
    }
    fn incidences(&self, _e: u32) -> Box<dyn Iterator<Item = (u32, Sign)> + '_> {
        Box::new(std::iter::empty())
    }
    fn vertex_type(&self, _v: u32) -> u32 { 0 }
    fn edge_type(&self, _e: u32) -> u32 { 0 }
    fn inherits(&self, _t: u32, _b: u32) -> bool { false }
    fn vertex_attr(&self, v: u32, key: &str) -> Option<&f64> {
        let rec = self.joints.get(&v)?;
        match key {
            "position" => Some(&rec[0]),
            "limit_min" => Some(&rec[1]),
            "limit_max" => Some(&rec[2]),
            _ => None,
        }
    }
    fn edge_attr(&self, _e: u32, _k: &str) -> Option<&f64> { None }
}

// ─── Predicates ───────────────────────────────────────────────────────

/// `collab` — collaborative-mode active. Robustness ±1 (structural
/// predicate; finite signed margin so `min` propagates cleanly inside
/// the implication).
struct CollabActive;
impl HypergraphPredicate<ArmState> for CollabActive {
    fn eval(&self, h: &ArmState) -> bool { h.collab }
    fn robustness(&self, h: &ArmState) -> f64 {
        if h.collab { 1.0 } else { -1.0 }
    }
    fn dependencies(&self, _h: &ArmState) -> Dependencies<ArmState> {
        Dependencies::global()
    }
}

/// `∀ joint. limit_min ≤ position ≤ limit_max`. Robustness is the
/// minimum slack across all joints — distance from the violated bound,
/// negative if any joint is outside its limits.
struct AllJointsWithinLimits;
impl HypergraphPredicate<ArmState> for AllJointsWithinLimits {
    fn eval(&self, h: &ArmState) -> bool {
        h.joints.values().all(|r| r[0] >= r[1] && r[0] <= r[2])
    }
    fn robustness(&self, h: &ArmState) -> f64 {
        h.joints.values()
            .map(|r| f64::min(r[0] - r[1], r[2] - r[0]))
            .fold(f64::INFINITY, f64::min)
    }
    fn dependencies(&self, _h: &ArmState) -> Dependencies<ArmState> {
        Dependencies::global()
    }
}

// ─── The test ─────────────────────────────────────────────────────────

#[test]
fn detects_limit_violation_at_t_12_3() {
    type Pred = dyn HypergraphPredicate<ArmState>;

    // Rolling-1s outer window so verdicts settle within the trace.
    let phi = always_bounded(
        0.0, 1.0,
        implies(
            Stl::Pred(Arc::new(CollabActive) as Arc<Pred>),
            always_bounded(0.0, 0.1, Stl::Pred(Arc::new(AllJointsWithinLimits) as Arc<Pred>)),
        ),
    );

    let mut monitor: StlMonitor<ArmState, Pred> =
        StlMonitor::new(phi).expect("bounded horizon");

    let dt = 0.01;
    let mut state = ArmState::new();
    let n_samples = 3000;
    let violation_idx = 1230; // t = 12.30 s

    // Three sample points across the trace:
    //   pre    — settled time well before the violation enters the
    //            lookahead window  → robustness should be positive
    //   during — settled time inside the window where the violation
    //            is visible to the outer Always  → robustness should
    //            be negative
    //   post   — settled time after the violation leaves the lookahead
    //            window  → robustness should rebound to positive
    //
    // Formula horizon = 1.1 s (outer 1 + inner 0.1). The single-sample
    // violation at t=12.30 is visible to outer settled times in
    // [11.20, 12.30].
    let mut pre: Option<f64> = None;
    let mut during: Option<f64> = None;
    let mut post: Option<f64> = None;

    for i in 0..n_samples {
        let t = (i as f64) * dt;
        for j in 0..N_JOINTS {
            let pos = 0.5 * (t + (j as f64) * 0.1).sin();
            state.set_position(j, pos);
        }
        if i == violation_idx {
            state.set_position(3, 4.0);
        }

        monitor.observe(Sample { state: state.clone(), t });

        if let Some(v) = monitor.verdict() {
            let t_star = v.t;
            if t_star > 0.5 && t_star < 11.0 {
                pre = Some(v.robustness); // overwrite: keep latest
            }
            if t_star > 11.5 && t_star < 12.3 && during.is_none() {
                during = Some(v.robustness); // first verdict in danger zone
            }
            if t_star > 12.5 && post.is_none() {
                post = Some(v.robustness); // first verdict past danger zone
            }
        }
    }

    let pre = pre.expect("expected a pre-violation verdict");
    let during = during.expect("expected a during-violation verdict");
    let post = post.expect("expected a post-violation verdict");

    assert!(pre > 0.0,
            "pre-violation robustness should be positive, got {pre}");
    assert!(during < 0.0,
            "during-violation robustness should be negative, got {during}");
    assert!(post > 0.0,
            "post-violation robustness should rebound to positive, got {post}");
}
