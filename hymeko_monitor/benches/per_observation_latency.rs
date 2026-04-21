//! Per-observation latency benchmark for the STL monitor.
//!
//! Measures wall-clock for `Monitor::observe` on the kinematic-arm
//! fixture from `tests/stl_kinematic.rs`, sweeping the formula's outer
//! horizon (1, 5, 10, 30 s) to characterise how per-observation cost
//! scales with sliding-window depth. This is the source of the §5.1
//! latency line for the RV paper.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use hymeko_monitor::formula::stl::*;
use hymeko_monitor::monitor::stl::StlMonitor;
use hymeko_monitor::predicate::{
    Dependencies, HypergraphPredicate, HypergraphState, Sign,
};
use hymeko_monitor::trace::Sample;
use hymeko_monitor::Monitor;

const N_JOINTS: u32 = 7;

#[derive(Clone)]
struct ArmState {
    joints: HashMap<u32, [f64; 3]>,
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
    fn set_position(&mut self, j: u32, p: f64) {
        if let Some(r) = self.joints.get_mut(&j) { r[0] = p; }
    }
}

impl HypergraphState for ArmState {
    type VertexId = u32; type EdgeId = u32; type TypeId = u32; type Attr = f64;
    fn vertices(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(self.joints.keys().copied())
    }
    fn edges(&self) -> Box<dyn Iterator<Item = u32> + '_> { Box::new(std::iter::empty()) }
    fn incidences(&self, _e: u32) -> Box<dyn Iterator<Item = (u32, Sign)> + '_> {
        Box::new(std::iter::empty())
    }
    fn vertex_type(&self, _v: u32) -> u32 { 0 }
    fn edge_type(&self, _e: u32) -> u32 { 0 }
    fn inherits(&self, _t: u32, _b: u32) -> bool { false }
    fn vertex_attr(&self, v: u32, k: &str) -> Option<&f64> {
        let r = self.joints.get(&v)?;
        match k { "position" => Some(&r[0]),
                  "limit_min" => Some(&r[1]),
                  "limit_max" => Some(&r[2]),
                  _ => None }
    }
    fn edge_attr(&self, _e: u32, _k: &str) -> Option<&f64> { None }
}

struct CollabActive;
impl HypergraphPredicate<ArmState> for CollabActive {
    fn eval(&self, h: &ArmState) -> bool { h.collab }
    fn robustness(&self, h: &ArmState) -> f64 { if h.collab { 1.0 } else { -1.0 } }
    fn dependencies(&self, _h: &ArmState) -> Dependencies<ArmState> { Dependencies::global() }
}

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
    fn dependencies(&self, _h: &ArmState) -> Dependencies<ArmState> { Dependencies::global() }
}

fn bench_horizon(outer_b: f64, n_samples: usize) {
    type Pred = dyn HypergraphPredicate<ArmState>;
    let phi = always_bounded(
        0.0, outer_b,
        implies(
            Stl::Pred(Arc::new(CollabActive) as Arc<Pred>),
            always_bounded(0.0, 0.1,
                Stl::Pred(Arc::new(AllJointsWithinLimits) as Arc<Pred>)),
        ),
    );
    let mut monitor: StlMonitor<ArmState, Pred> =
        StlMonitor::new(phi).expect("bounded horizon");

    let dt = 0.01;
    let mut state = ArmState::new();

    // Warm-up the windows (3× horizon worth of samples).
    let warmup = ((outer_b + 0.1) * 3.0 / dt) as usize + 100;
    for i in 0..warmup {
        let t = (i as f64) * dt;
        for j in 0..N_JOINTS {
            state.set_position(j, 0.5 * (t + (j as f64) * 0.1).sin());
        }
        monitor.observe(Sample { state: state.clone(), t });
    }

    // Timed phase.
    let t_start = Instant::now();
    for i in warmup..(warmup + n_samples) {
        let t = (i as f64) * dt;
        for j in 0..N_JOINTS {
            state.set_position(j, 0.5 * (t + (j as f64) * 0.1).sin());
        }
        monitor.observe(Sample { state: state.clone(), t });
    }
    let dt_total = t_start.elapsed();
    let per_obs_us = dt_total.as_micros() as f64 / n_samples as f64;
    println!(
        "horizon={outer_b:>5.1}s  samples={n_samples}  total={:>6.1}ms  per_obs={per_obs_us:>7.2}μs  rate={:>7.0} obs/s",
        dt_total.as_millis() as f64,
        1.0e6 / per_obs_us,
    );
}

fn main() {
    println!("hymeko_monitor — per-observation latency on the kinematic-arm fixture");
    println!("formula: G_{{[0,T]}}(collab → G_{{[0,0.1]}}(∀ joint. within_limits))");
    println!();
    for &outer_b in &[1.0_f64, 5.0, 10.0, 30.0] {
        bench_horizon(outer_b, 5_000);
    }
}
