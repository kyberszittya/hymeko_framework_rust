//! Per-observation latency benchmark (v0.1 stub).
//!
//! The full benchmark wires up a fixture, builds an STL monitor, and
//! sweeps window size while measuring per-`observe()` wall-clock. v0.1
//! ships this as a stub because `StlMonitor::observe()` is `todo!()`;
//! once the monitor is implemented the body fills in following the
//! pattern in `hymeko_query/tests/bench_workflow.rs`.

fn main() {
    eprintln!("hymeko_monitor::per_observation_latency — stub (v0.1)");
    eprintln!("StlMonitor::observe() is todo!() in v0.1; nothing to bench yet.");
}
