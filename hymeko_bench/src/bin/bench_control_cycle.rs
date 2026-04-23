//! Round-3 Exp-3 control-loop timing benchmark.
//!
//! Times ONE control cycle of the canonical grasping scenario across
//! three pipelines with comparable parameter counts:
//!
//!   1. hypergraph   cached IR → HGNN layer y = B W B^T x at F=8 → pool →
//!                   MLP head [F, 16, 4] → velocity command
//!   2. td3          flat state vector [15] → MLP [15, 16, 16, 4] →
//!                   velocity command (matched param count)
//!   3. pid_td3      classical PID on 4 joint-tracking errors +
//!                   TD3 actor forward pass for setpoint corrections
//!
//! Per pipeline, wall-clock is decomposed into the four stages
//! (state_assembly, forward, decode, total). N iterations per pipeline.
//!
//! **Honest framing** (per brief §3.3): this is a compute-complexity
//! comparison, not a performance comparison. None of the three pipelines
//! is trained for the task. Weights are deterministic pseudo-random;
//! outputs are meaningless. The numbers are latency only.
//!
//! Emits `hymeko_bench/results/control_cycle.csv` with columns
//!   pipeline, stage, median_us, p95_us, stddev_us, iterations
//!
//! Usage:
//!   cargo run --release -p hymeko_bench --bin bench_control_cycle

use std::{fs, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use serde::Serialize;

use hymeko::module_store::module_store::{HymekoParser, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
use parser::ast::AstStr;

use hymeko_hnn::tensor::message_passing::{gather_edges_from_nodes, scatter_nodes_from_edges};
use hymeko_hnn::traversal::hypergraphview::HyperGraphView;

// ───────────────────────────────────────────────────────────────────────────
struct LalrpopParser;

impl HymekoParser for LalrpopParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        parser::parse_description(src).map_err(|e| format!("{e:?}"))
    }
}

const AGG_CFG: AggCfg = AggCfg {
    weight: WeightAgg::Sum,
    sign: SignAgg::PreferNonNeutral,
    clamp01: false,
};

// Architecture constants.
const F_HG: usize = 8;      // hypergraph feature channels
const H_MLP: usize = 16;    // MLP hidden size (TD3 / head)
const STATE_DIM: usize = 15; // flat state vector size (~canonical observables)
const ACTION_DIM: usize = 4; // 4-DOF velocity command

// ───────────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(about = "Round-3 Exp-3 control-loop timing: hypergraph vs TD3 vs PID-TD3")]
struct Cli {
    /// Canonical grasping scenario IR.
    #[arg(long, default_value = "examples/paper/hymeko_robot.hymeko")]
    input: PathBuf,
    /// Output CSV.
    #[arg(long, default_value = "hymeko_bench/results/control_cycle.csv")]
    out: PathBuf,
    /// Iterations per pipeline.
    #[arg(long, default_value_t = 10_000)]
    iters: usize,
    /// Warm-up iterations (not recorded).
    #[arg(long, default_value_t = 100)]
    warmup: usize,
    /// Optional: dump per-iteration raw stage times for the TD3 pipeline
    /// (for 3-D latency scatter in the paper). One row per iteration with
    /// columns: iter, state_assembly_us, forward_us, decode_us, total_us.
    #[arg(long)]
    raw_td3_out: Option<PathBuf>,
}

#[derive(Serialize, Debug, Clone)]
struct Row {
    pipeline: String,
    stage: String,
    median_us: f64,
    p95_us: f64,
    stddev_us: f64,
    iterations: usize,
}

#[derive(Serialize, Debug, Clone)]
struct RawRow {
    iter: usize,
    state_assembly_us: f64,
    forward_us: f64,
    decode_us: f64,
    total_us: f64,
}

// ───────────────────────────────────────────────────────────────────────────
/// Tiny deterministic PRNG (splitmix64). Same weights every run.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Uniform in (-0.5, 0.5]
        (z as f64 / u64::MAX as f64 - 0.5) as f32
    }
    fn vec(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.next_f32()).collect() }
    fn mat(&mut self, rows: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..rows).map(|_| self.vec(cols)).collect()
    }
}

// ───────────────────────────────────────────────────────────────────────────
/// Dense forward step: y = ReLU(W x + b) if relu else y = W x + b.
#[inline(always)]
fn dense_layer(w: &[Vec<f32>], b: &[f32], x: &[f32], y: &mut [f32], relu: bool) {
    let n = w.len();
    debug_assert_eq!(y.len(), n);
    for i in 0..n {
        let row = &w[i];
        let mut s = b[i];
        for j in 0..x.len() {
            s += row[j] * x[j];
        }
        y[i] = if relu { s.max(0.0) } else { s };
    }
}

// ───────────────────────────────────────────────────────────────────────────
struct HgPipeline {
    hg: HyperGraphView<f32, EdgeWScalar<f32>, f32>,
    // HGNN: per-edge diagonal weight (trainable per-edge; fixed for bench).
    edge_w: Vec<f32>,
    // Feature-channel input embedding: expand a scalar node reading to F_HG channels.
    embed: Vec<f32>,                 // [F_HG]
    // Head MLP: F_HG -> H_MLP -> ACTION_DIM
    w1: Vec<Vec<f32>>, b1: Vec<f32>, // [H_MLP x F_HG], [H_MLP]
    w2: Vec<Vec<f32>>, b2: Vec<f32>, // [ACTION_DIM x H_MLP], [ACTION_DIM]
    // Scratch buffers.
    node_x: Vec<f32>,                // [|V|] per-channel input
    edge_buf: Vec<f32>,              // [|E|] gather result
    node_y: Vec<f32>,                // [|V|] scatter result
    pooled: Vec<f32>,                // [F_HG] mean-pooled
    hidden: Vec<f32>,                // [H_MLP]
    action: Vec<f32>,                // [ACTION_DIM]
}

impl HgPipeline {
    fn new(hg: HyperGraphView<f32, EdgeWScalar<f32>, f32>, rng: &mut Rng) -> Self {
        let n_v = hg.num_nodes();
        let n_e = hg.num_edges();
        // Edge weights near 1.0 — avoid zero-ing the HGNN forward.
        let edge_w: Vec<f32> = (0..n_e).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
        let embed = rng.vec(F_HG);
        let w1 = rng.mat(H_MLP, F_HG);
        let b1 = rng.vec(H_MLP);
        let w2 = rng.mat(ACTION_DIM, H_MLP);
        let b2 = rng.vec(ACTION_DIM);
        Self {
            hg,
            edge_w,
            embed,
            w1, b1, w2, b2,
            node_x: vec![0.0; n_v],
            edge_buf: vec![0.0; n_e],
            node_y: vec![0.0; n_v],
            pooled: vec![0.0; F_HG],
            hidden: vec![0.0; H_MLP],
            action: vec![0.0; ACTION_DIM],
        }
    }

    fn param_count(&self) -> usize {
        self.edge_w.len()
            + self.embed.len()
            + self.w1.iter().map(|r| r.len()).sum::<usize>() + self.b1.len()
            + self.w2.iter().map(|r| r.len()).sum::<usize>() + self.b2.len()
    }

    #[inline(always)]
    fn state_assembly(&mut self, sensors: &[f32]) {
        let n_v = self.node_x.len();
        for i in 0..n_v { self.node_x[i] = sensors[i % sensors.len()]; }
    }

    #[inline(always)]
    fn forward(&mut self) {
        // F_HG channels of gather→scatter, each scaled by embed[f]; mean-pool.
        for f in 0..F_HG {
            let scale = self.embed[f];
            // Temporarily scale node_x by embed[f] (restore after).
            for i in 0..self.node_x.len() { self.node_x[i] *= scale; }
            gather_edges_from_nodes(&self.hg, &self.node_x, &mut self.edge_buf, true);
            // Apply per-edge weight W (diagonal).
            for e in 0..self.edge_buf.len() { self.edge_buf[e] *= self.edge_w[e]; }
            scatter_nodes_from_edges(&self.hg, &self.edge_buf, &mut self.node_y, true);
            // Pool mean of y into channel f.
            let mut s = 0.0f32;
            for v in &self.node_y { s += *v; }
            self.pooled[f] = s / (self.node_y.len().max(1) as f32);
            // Restore node_x for next channel.
            for i in 0..self.node_x.len() { self.node_x[i] /= scale; }
        }
        // Head MLP.
        dense_layer(&self.w1, &self.b1, &self.pooled, &mut self.hidden, true);
        dense_layer(&self.w2, &self.b2, &self.hidden, &mut self.action, false);
    }

    #[inline(always)]
    fn decode(&self, vel: &mut [f32]) {
        for i in 0..vel.len() { vel[i] = self.action[i].tanh(); }
    }
}

// ───────────────────────────────────────────────────────────────────────────
struct Td3Pipeline {
    w1: Vec<Vec<f32>>, b1: Vec<f32>, // [H_MLP x STATE_DIM]
    w2: Vec<Vec<f32>>, b2: Vec<f32>, // [H_MLP x H_MLP]
    w3: Vec<Vec<f32>>, b3: Vec<f32>, // [ACTION_DIM x H_MLP]
    state: Vec<f32>, h1: Vec<f32>, h2: Vec<f32>, action: Vec<f32>,
}

impl Td3Pipeline {
    fn new(rng: &mut Rng) -> Self {
        Self {
            w1: rng.mat(H_MLP, STATE_DIM),  b1: rng.vec(H_MLP),
            w2: rng.mat(H_MLP, H_MLP),      b2: rng.vec(H_MLP),
            w3: rng.mat(ACTION_DIM, H_MLP), b3: rng.vec(ACTION_DIM),
            state: vec![0.0; STATE_DIM],
            h1: vec![0.0; H_MLP],
            h2: vec![0.0; H_MLP],
            action: vec![0.0; ACTION_DIM],
        }
    }
    fn param_count(&self) -> usize {
        self.w1.iter().map(|r| r.len()).sum::<usize>() + self.b1.len()
            + self.w2.iter().map(|r| r.len()).sum::<usize>() + self.b2.len()
            + self.w3.iter().map(|r| r.len()).sum::<usize>() + self.b3.len()
    }
    #[inline(always)]
    fn state_assembly(&mut self, sensors: &[f32]) {
        for i in 0..STATE_DIM { self.state[i] = sensors[i % sensors.len()]; }
    }
    #[inline(always)]
    fn forward(&mut self) {
        dense_layer(&self.w1, &self.b1, &self.state,  &mut self.h1, true);
        dense_layer(&self.w2, &self.b2, &self.h1,     &mut self.h2, true);
        dense_layer(&self.w3, &self.b3, &self.h2,     &mut self.action, false);
    }
    #[inline(always)]
    fn decode(&self, vel: &mut [f32]) {
        for i in 0..vel.len() { vel[i] = self.action[i].tanh(); }
    }
}

// ───────────────────────────────────────────────────────────────────────────
struct PidTd3Pipeline {
    td3: Td3Pipeline,
    kp: [f32; ACTION_DIM], ki: [f32; ACTION_DIM], kd: [f32; ACTION_DIM],
    setpoint: [f32; ACTION_DIM],
    integral: [f32; ACTION_DIM],
    prev_err: [f32; ACTION_DIM],
    err:      [f32; ACTION_DIM],
    pid_out:  [f32; ACTION_DIM],
    /// Flat state built inside state_assembly (errors + current measurement)
    td3_state: Vec<f32>,
}

impl PidTd3Pipeline {
    fn new(rng: &mut Rng) -> Self {
        Self {
            td3: Td3Pipeline::new(rng),
            kp: [0.8, 0.8, 0.6, 0.6],
            ki: [0.02; ACTION_DIM],
            kd: [0.1; ACTION_DIM],
            setpoint: [0.0; ACTION_DIM],
            integral: [0.0; ACTION_DIM],
            prev_err: [0.0; ACTION_DIM],
            err: [0.0; ACTION_DIM],
            pid_out: [0.0; ACTION_DIM],
            td3_state: vec![0.0; STATE_DIM],
        }
    }
    fn param_count(&self) -> usize { self.td3.param_count() + 3 * ACTION_DIM }
    #[inline(always)]
    fn state_assembly(&mut self, sensors: &[f32]) {
        // PID step: err = setpoint - measured; integral += err; d = err - prev.
        // The first ACTION_DIM sensor readings are treated as joint states.
        for i in 0..ACTION_DIM {
            let measured = sensors[i];
            self.err[i] = self.setpoint[i] - measured;
            self.integral[i] += self.err[i];
            let deriv = self.err[i] - self.prev_err[i];
            self.prev_err[i] = self.err[i];
            self.pid_out[i] = self.kp[i] * self.err[i]
                             + self.ki[i] * self.integral[i]
                             + self.kd[i] * deriv;
        }
        // TD3 state: concatenated errors + full sensor window.
        for i in 0..STATE_DIM { self.td3_state[i] = sensors[i % sensors.len()]; }
    }
    #[inline(always)]
    fn forward(&mut self) {
        self.td3.state.copy_from_slice(&self.td3_state);
        self.td3.forward();
    }
    #[inline(always)]
    fn decode(&self, vel: &mut [f32]) {
        for i in 0..ACTION_DIM {
            vel[i] = (self.pid_out[i] + 0.2 * self.td3.action[i]).tanh();
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
fn median_p95(xs: &mut Vec<f64>) -> (f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let med = xs[n / 2];
    let p95 = xs[((n as f64 * 0.95) as usize).min(n - 1)];
    (med, p95)
}

fn stddev(xs: &[f64]) -> f64 {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

fn summarize(mut v: Vec<f64>, iters: usize, pipeline: &str, stage: &str) -> Row {
    let (med, p95) = median_p95(&mut v);
    let sd = stddev(&v);
    Row {
        pipeline: pipeline.to_string(),
        stage: stage.to_string(),
        median_us: med,
        p95_us: p95,
        stddev_us: sd,
        iterations: iters,
    }
}

// ───────────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.input.exists() {
        anyhow::bail!("input IR not found: {}", cli.input.display());
    }
    eprintln!(
        "Control-cycle bench: {} iters/pipeline, warmup={}, F_HG={}, H_MLP={}, STATE_DIM={}",
        cli.iters, cli.warmup, F_HG, H_MLP, STATE_DIM
    );

    // Cached compile of canonical scenario (hypergraph pipeline uses this;
    // TD3 and PID-TD3 only need its shape via sensor_dim).
    let mut store = ModuleStore::new(StdFsProvider::new(), LalrpopParser);
    let compiled = store
        .compile(&cli.input)
        .map_err(|e| anyhow::anyhow!("compile canonical: {e:?}"))?;
    let ex = ScalarWeightExtractor::default();
    let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &AGG_CFG, &ex);
    let sensor_dim = hg.num_nodes().max(STATE_DIM);

    let mut rng = Rng::new(0xC0FFEE);

    // Build sensor-reading vector — single deterministic window used by all three.
    let sensors: Vec<f32> = (0..sensor_dim).map(|i| ((i as f32) * 0.13).sin()).collect();

    // Pipeline construction.
    let mut p_hg  = HgPipeline::new(hg, &mut rng);
    let mut p_td3 = Td3Pipeline::new(&mut rng);
    let mut p_pid = PidTd3Pipeline::new(&mut rng);

    eprintln!(
        "  params: hypergraph={} td3={} pid_td3={}",
        p_hg.param_count(), p_td3.param_count(), p_pid.param_count()
    );
    eprintln!("  canonical graph: |V|={} |E|={}", p_hg.hg.num_nodes(), p_hg.hg.num_edges());

    let mut vel = vec![0.0_f32; ACTION_DIM];
    let mut rows = Vec::with_capacity(4 * 3);
    let mut td3_raw: Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = None;

    // Helper to time one pipeline's stages. When `$capture_raw` is true,
    // the raw per-iteration timing vectors are cloned out into td3_raw
    // so they can be dumped to a secondary CSV for the 3-D latency plot.
    macro_rules! bench_pipeline {
        ($name:expr, $p:expr, $capture_raw:expr) => {{
            // Warm-up.
            for _ in 0..cli.warmup {
                $p.state_assembly(&sensors);
                $p.forward();
                $p.decode(&mut vel);
                std::hint::black_box(&vel);
            }
            let mut t_sa    = Vec::with_capacity(cli.iters);
            let mut t_fw    = Vec::with_capacity(cli.iters);
            let mut t_dec   = Vec::with_capacity(cli.iters);
            let mut t_total = Vec::with_capacity(cli.iters);
            for _ in 0..cli.iters {
                let t0 = Instant::now();
                $p.state_assembly(&sensors);
                let t1 = Instant::now();
                $p.forward();
                let t2 = Instant::now();
                $p.decode(&mut vel);
                let t3 = Instant::now();
                std::hint::black_box(&vel);
                t_sa.push((t1 - t0).as_secs_f64() * 1e6);
                t_fw.push((t2 - t1).as_secs_f64() * 1e6);
                t_dec.push((t3 - t2).as_secs_f64() * 1e6);
                t_total.push((t3 - t0).as_secs_f64() * 1e6);
            }
            if $capture_raw {
                td3_raw = Some((t_sa.clone(), t_fw.clone(), t_dec.clone(), t_total.clone()));
            }
            rows.push(summarize(t_sa,    cli.iters, $name, "state_assembly"));
            rows.push(summarize(t_fw,    cli.iters, $name, "forward"));
            rows.push(summarize(t_dec,   cli.iters, $name, "decode"));
            rows.push(summarize(t_total, cli.iters, $name, "total"));
        }};
    }

    bench_pipeline!("hypergraph", p_hg,  false);
    bench_pipeline!("td3",        p_td3, cli.raw_td3_out.is_some());
    bench_pipeline!("pid_td3",    p_pid, false);

    // Emit CSV.
    if let Some(parent) = cli.out.parent() { fs::create_dir_all(parent).ok(); }
    let mut w = csv::Writer::from_path(&cli.out)?;
    for r in &rows { w.serialize(r)?; }
    w.flush()?;

    // Emit raw TD3 per-iteration CSV if requested.
    if let (Some(path), Some((sa, fw, dec, tot))) = (&cli.raw_td3_out, td3_raw) {
        if let Some(parent) = path.parent() { fs::create_dir_all(parent).ok(); }
        let mut rw = csv::Writer::from_path(path)?;
        for i in 0..sa.len() {
            rw.serialize(RawRow {
                iter: i,
                state_assembly_us: sa[i],
                forward_us: fw[i],
                decode_us: dec[i],
                total_us: tot[i],
            })?;
        }
        rw.flush()?;
        eprintln!("Wrote raw TD3 per-iteration CSV: {}", path.display());
    }

    // Print a summary.
    eprintln!("\nResults (median_us, p95_us, stddev_us):");
    for r in &rows {
        eprintln!("  {:12} {:16} med={:8.3} p95={:8.3} sd={:8.3}",
                  r.pipeline, r.stage, r.median_us, r.p95_us, r.stddev_us);
    }

    // Acceptance check: hypergraph total ≤ 3× TD3 total.
    let hg_total = rows.iter().find(|r| r.pipeline=="hypergraph" && r.stage=="total").unwrap().median_us;
    let td3_total = rows.iter().find(|r| r.pipeline=="td3" && r.stage=="total").unwrap().median_us;
    let ratio = hg_total / td3_total;
    eprintln!("\nAcceptance: hypergraph/td3 total ratio = {:.2}× (bound 3×, alarm >5×)", ratio);
    if ratio > 5.0 {
        eprintln!("  WARNING: ratio > 5× — check whether IR is being reloaded per iteration.");
    }

    eprintln!("Wrote {}", cli.out.display());
    Ok(())
}
