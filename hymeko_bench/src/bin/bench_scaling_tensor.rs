//! Round-3 Exp-4 scaling sweep.
//!
//! For each fixture in `hymeko_bench/corpora/03_size_sweep/` (|E| ∈
//! {10, 100, 1000, 10000}, fixed mean arity d̄=3), times three operations:
//!
//!   op_a  compile-to-IR             (N=100 iters, fresh store each rep)
//!   op_b  star expansion            (N=100 iters, cached IR + HyperGraphView)
//!   op_c  MP forward pass F=64      (N=1000 iters, gather → scatter × F channels)
//!
//! Emits `hymeko_bench/results/scaling.csv` with columns
//!   corpus, num_hyperedges, mean_arity, operation, median_ms, p95_ms,
//!   predicted_ms, ratio
//! where predicted_ms is the asymptotic O(|E|) bound anchored at the
//! |E|=10 data point and ratio = median / predicted.
//!
//! Usage:
//!   cargo run --release -p hymeko_bench --bin bench_scaling_tensor

use std::{fs, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

use hymeko::module_store::module_store::{HymekoParser, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
use parser::ast::AstStr;

use hymeko_hnn::tensor::message_passing::{gather_edges_from_nodes, scatter_nodes_from_edges};
use hymeko_hnn::traversal::hypergraphview::HyperGraphView;
use hymeko_hre::expansion::star_expansion_coo;

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
const F_DIM: usize = 64;

// ───────────────────────────────────────────────────────────────────────────
#[derive(Deserialize, Debug, Clone)]
struct FixtureEntry {
    name: String,
    n_vertices: usize,
    n_hyperedges: usize,
    mean_arity: f64,
    #[allow(dead_code)]
    source_bytes: usize,
    path: String,
}

#[derive(Serialize, Debug, Clone)]
struct Row {
    corpus: String,
    num_hyperedges: usize,
    mean_arity: f64,
    operation: String,
    median_ms: f64,
    p95_ms: f64,
    predicted_ms: f64,
    ratio: f64,
}

#[derive(Parser, Debug)]
#[command(about = "Round-3 Exp-4 scaling sweep: compile / star-expansion / MP(F=64)")]
struct Cli {
    /// Corpus directory (must contain index.json).
    #[arg(long, default_value = "hymeko_bench/corpora/03_size_sweep")]
    fixtures: PathBuf,
    /// Output CSV.
    #[arg(long, default_value = "hymeko_bench/results/scaling.csv")]
    out: PathBuf,
    /// Iterations for compile (op_a) and star expansion (op_b).
    #[arg(long, default_value_t = 100)]
    ab_iters: usize,
    /// Iterations for MP forward pass (op_c).
    #[arg(long, default_value_t = 1000)]
    c_iters: usize,
    /// Warm-up iterations (not recorded).
    #[arg(long, default_value_t = 5)]
    warmup: usize,
}

// ───────────────────────────────────────────────────────────────────────────
fn time_ms<F: FnOnce()>(f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1000.0
}

fn median_p95(mut xs: Vec<f64>) -> (f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let med = xs[n / 2];
    let p95 = xs[((n as f64 * 0.95) as usize).min(n - 1)];
    (med, p95)
}

fn load_fixtures(root: &PathBuf) -> Result<Vec<FixtureEntry>> {
    let idx = root.join("index.json");
    let txt = fs::read_to_string(&idx)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", idx.display()))?;
    let mut v: Vec<FixtureEntry> = serde_json::from_str(&txt)?;
    v.sort_by_key(|f| f.n_hyperedges);
    Ok(v)
}

fn compile_once(path: &PathBuf) -> Result<()> {
    let mut store = ModuleStore::new(StdFsProvider::new(), LalrpopParser);
    store
        .compile(path)
        .map_err(|e| anyhow::anyhow!("compile: {e:?}"))?;
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
fn bench_fixture(cli: &Cli, f: &FixtureEntry) -> Result<(f64, f64, f64, f64, f64, f64)> {
    let path = cli.fixtures.join(&f.path);

    // op_a — compile-to-IR
    for _ in 0..cli.warmup {
        compile_once(&path)?;
    }
    let mut compile_times = Vec::with_capacity(cli.ab_iters);
    for _ in 0..cli.ab_iters {
        compile_times.push(time_ms(|| {
            compile_once(&path).unwrap();
        }));
    }
    let (a_med, a_p95) = median_p95(compile_times);

    // Prepare cached IR + HyperGraphView once for ops b and c.
    let mut store = ModuleStore::new(StdFsProvider::new(), LalrpopParser);
    let compiled = store
        .compile(&path)
        .map_err(|e| anyhow::anyhow!("compile cached: {e:?}"))?;
    let ex = ScalarWeightExtractor::default();
    let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &AGG_CFG, &ex);

    // op_b — star expansion
    for _ in 0..cli.warmup {
        let _ = std::hint::black_box(star_expansion_coo(&hg));
    }
    let mut expand_times = Vec::with_capacity(cli.ab_iters);
    for _ in 0..cli.ab_iters {
        expand_times.push(time_ms(|| {
            let _ = std::hint::black_box(star_expansion_coo(&hg));
        }));
    }
    let (b_med, b_p95) = median_p95(expand_times);

    // op_c — MP forward pass at feature dim F. Per-channel: x_e = B^T x;
    // y = B x_e. We run F channels per iteration (linear in F, matches
    // the single-layer ST-HGNN y = B W B^T x with W = I).
    let n = hg.num_nodes();
    let m = hg.num_edges();
    // Deterministic node features: x[i,f] = (i + f) as f32 * 1e-3.
    let mut node_x = vec![vec![0.0_f32; n]; F_DIM];
    for fx in 0..F_DIM {
        for i in 0..n {
            node_x[fx][i] = ((i + fx) as f32) * 1e-3;
        }
    }
    let mut edge_buf = vec![0.0_f32; m];
    let mut node_y = vec![0.0_f32; n];

    for _ in 0..cli.warmup {
        for fx in 0..F_DIM {
            gather_edges_from_nodes(&hg, &node_x[fx], &mut edge_buf, true);
            scatter_nodes_from_edges(&hg, &edge_buf, &mut node_y, true);
            std::hint::black_box(&node_y);
        }
    }
    let mut mp_times = Vec::with_capacity(cli.c_iters);
    for _ in 0..cli.c_iters {
        mp_times.push(time_ms(|| {
            for fx in 0..F_DIM {
                gather_edges_from_nodes(&hg, &node_x[fx], &mut edge_buf, true);
                scatter_nodes_from_edges(&hg, &edge_buf, &mut node_y, true);
                std::hint::black_box(&node_y);
            }
        }));
    }
    let (c_med, c_p95) = median_p95(mp_times);

    Ok((a_med, a_p95, b_med, b_p95, c_med, c_p95))
}

// ───────────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let cli = Cli::parse();
    let fixtures = load_fixtures(&cli.fixtures)?;
    if fixtures.is_empty() {
        anyhow::bail!("no fixtures found in {}", cli.fixtures.display());
    }
    eprintln!(
        "Scaling sweep: {} fixtures, ab_iters={}, c_iters={}, F={}",
        fixtures.len(),
        cli.ab_iters,
        cli.c_iters,
        F_DIM
    );

    // Collect raw timings first so we can compute ratios vs. the |E|=10 baseline.
    let mut raw: Vec<(FixtureEntry, (f64, f64, f64, f64, f64, f64))> =
        Vec::with_capacity(fixtures.len());
    for f in &fixtures {
        eprintln!(
            "  {}  |V|={} |E|={} d̄={:.1}",
            f.name, f.n_vertices, f.n_hyperedges, f.mean_arity
        );
        let timings = bench_fixture(&cli, f)?;
        eprintln!(
            "      compile med/p95 = {:.3}/{:.3} ms   expand = {:.3}/{:.3} ms   mp(F={}) = {:.3}/{:.3} ms",
            timings.0, timings.1, timings.2, timings.3, F_DIM, timings.4, timings.5
        );
        raw.push((f.clone(), timings));
    }

    // Anchor predictions at the smallest |E| data point (|E|=10 by construction).
    let (anchor_f, anchor_t) = raw
        .iter()
        .min_by_key(|(f, _)| f.n_hyperedges)
        .expect("at least one fixture");
    let anchor_e = anchor_f.n_hyperedges as f64;
    let (a_base, b_base, c_base) = (anchor_t.0, anchor_t.2, anchor_t.4);

    if let Some(parent) = cli.out.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut w = csv::Writer::from_path(&cli.out)?;

    for (f, t) in raw {
        let e = f.n_hyperedges as f64;
        let scale = e / anchor_e;

        for (op_name, med, p95, base) in [
            ("compile", t.0, t.1, a_base),
            ("star_expansion", t.2, t.3, b_base),
            ("mp_forward_F64", t.4, t.5, c_base),
        ] {
            let predicted = base * scale;
            let ratio = if predicted > 0.0 { med / predicted } else { 0.0 };
            w.serialize(Row {
                corpus: f.name.clone(),
                num_hyperedges: f.n_hyperedges,
                mean_arity: f.mean_arity,
                operation: op_name.to_string(),
                median_ms: med,
                p95_ms: p95,
                predicted_ms: predicted,
                ratio,
            })?;
        }
    }
    w.flush()?;
    eprintln!("Wrote {}", cli.out.display());
    Ok(())
}
