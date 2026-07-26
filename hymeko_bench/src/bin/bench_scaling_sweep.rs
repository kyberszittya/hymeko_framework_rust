//! SMC 2026 Task 2 — synthetic scaling sweep + scalability limit.
//!
//! Generates random signed hypergraphs on an increasing size ladder and, for
//! each point, measures generation, compile (parse+intern+resolve+lower), and
//! the star projection into COO (`|V|+|E|` Levi expansion), plus peak heap and
//! total COO non-zeros. Stops when a per-point wall budget, total budget, or a
//! soft memory ceiling is hit, and records what became impractical first — the
//! `limit_encountered` answer.
//!
//! Non-robot synthetic structures are compile+project only; the URDF/MJCF
//! emitters require a kinematic model and are exercised against real robot-like
//! fixtures by the separate `bench_scaling` binary (Prop. 4 is an IR-storage
//! claim, not an emitter-correctness claim — same stance as `bench_scaling`).
//!
//! Run (workspace root, release, single-threaded):
//!   cargo run --release -p hymeko_bench --bin bench_scaling_sweep -- \
//!       --out <run>/raw/bench_scaling_sweep.csv

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use hymeko::module_store::module_store::ModuleStore;
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
use hymeko::util::real_parser::RealParser;
use hymeko_hnn::traversal::hypergraphview::HyperGraphView;
use hymeko_hre::expansion::star_expansion_coo;

use hymeko_bench::corpus::random_hymeko_source;
use hymeko_bench::stats::{loglog_fit, median};
use hymeko_bench::{measure, peak_bytes};

#[global_allocator]
static GLOBAL: hymeko_bench::TrackingAlloc = hymeko_bench::TrackingAlloc;

const AGG: AggCfg = AggCfg {
    weight: WeightAgg::Sum,
    sign: SignAgg::PreferNonNeutral,
    clamp01: false,
};

/// (nodes, edges, density) ladder spanning ~10² to ~10⁶ target incidences.
const LADDER: &[(usize, usize, f64)] = &[
    (64, 32, 0.05),
    (128, 64, 0.05),
    (256, 128, 0.05),
    (384, 192, 0.05),
    (512, 256, 0.05),
    (768, 384, 0.05),
    (1024, 512, 0.05),
    (1536, 768, 0.04),
    (2048, 1024, 0.04),
    (3072, 1536, 0.03),
    (4096, 2048, 0.03),
    (6144, 3072, 0.025),
    (8192, 4096, 0.02),
    (12288, 6144, 0.015),
    (16384, 8192, 0.012),
    (24576, 12288, 0.01),
    (32768, 16384, 0.008),
    (49152, 24576, 0.006),
    (65536, 32768, 0.005),
    (98304, 49152, 0.004),
];

#[derive(Parser, Debug)]
#[command(about = "SMC2026 Task 2: synthetic scaling sweep + limit")]
struct Cli {
    #[arg(long, default_value = "bench_scaling_sweep.csv")]
    out: PathBuf,
    /// Repetitions at small sizes (reduced to 1 above `big_nnz`).
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Above this estimated nnz, drop to a single repetition.
    #[arg(long, default_value_t = 50_000)]
    big_nnz: usize,
    /// Per-point wall budget (seconds); exceeding it stops the sweep.
    #[arg(long, default_value_t = 120.0)]
    per_point_seconds: f64,
    /// Total wall budget (seconds).
    #[arg(long, default_value_t = 1500.0)]
    total_seconds: f64,
    /// Soft peak-heap ceiling (GiB); measuring above it stops the sweep.
    #[arg(long, default_value_t = 8.0)]
    mem_ceiling_gb: f64,
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,
}

#[derive(Debug, Clone)]
struct PointRow {
    nodes: usize,
    edges: usize,
    density: f64,
    nnz: usize,
    source_bytes: usize,
    gen_ms: f64,
    compile_ms: f64,
    view_ms: f64,
    coo_ms: f64,
    project_ms: f64,
    compile_peak_bytes: usize,
    project_peak_bytes: usize,
    us_per_nnz: f64,
}

/// One compile+project measurement of a freshly generated source.
fn run_point(
    nodes: usize,
    edges: usize,
    density: f64,
    seed: u64,
    inputs_dir: &std::path::Path,
) -> Result<PointRow> {
    // ── generate + persist source ──
    let t_gen = Instant::now();
    let src = random_hymeko_source(nodes, edges, density, seed);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
    let source_bytes = src.len();
    let path = inputs_dir.join(format!(
        "sweep_n{nodes}_e{edges}_d{:03}.hymeko",
        (density * 1000.0) as usize
    ));
    std::fs::write(&path, &src).with_context(|| format!("writing {}", path.display()))?;

    // ── compile (parse+intern+resolve+lower) ──
    let t_c = Instant::now();
    let (compiled_res, _r, compile_peak_bytes) = measure(|| {
        let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
        let compiled = store
            .compile(&path)
            .map_err(|e| anyhow::anyhow!("compile: {e:?}"));
        (store, compiled)
    });
    let compile_ms = t_c.elapsed().as_secs_f64() * 1e3;
    let (store, compiled) = compiled_res;
    let compiled = compiled?;
    let ir = &compiled.ir;

    // ── project: HyperGraphView (Levi) + star expansion into COO ──
    let ex = ScalarWeightExtractor;
    let t_v = Instant::now();
    let (hg, _r2, view_peak) =
        measure(|| HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(ir, &AGG, &ex));
    let view_ms = t_v.elapsed().as_secs_f64() * 1e3;

    let t_coo = Instant::now();
    let (coo, _r3, coo_peak) = measure(|| star_expansion_coo(&hg));
    let coo_ms = t_coo.elapsed().as_secs_f64() * 1e3;
    let nnz = coo.len();
    black_box(&coo);

    let project_ms = view_ms + coo_ms;
    let project_peak_bytes = view_peak.max(coo_peak);
    let us_per_nnz = if nnz == 0 {
        f64::NAN
    } else {
        (compile_ms + project_ms) * 1e3 / nnz as f64
    };

    drop(coo);
    drop(hg);
    drop(store);
    let _ = std::fs::remove_file(&path);

    Ok(PointRow {
        nodes,
        edges,
        density,
        nnz,
        source_bytes,
        gen_ms,
        compile_ms,
        view_ms,
        coo_ms,
        project_ms,
        compile_peak_bytes,
        project_peak_bytes,
        us_per_nnz,
    })
}

fn median_row(runs: &[PointRow]) -> PointRow {
    // Median over reps for the timing fields; structural fields are identical.
    let pick = |f: &dyn Fn(&PointRow) -> f64| median(&runs.iter().map(f).collect::<Vec<_>>());
    let first = runs[0].clone();
    PointRow {
        gen_ms: pick(&|r| r.gen_ms),
        compile_ms: pick(&|r| r.compile_ms),
        view_ms: pick(&|r| r.view_ms),
        coo_ms: pick(&|r| r.coo_ms),
        project_ms: pick(&|r| r.project_ms),
        compile_peak_bytes: runs.iter().map(|r| r.compile_peak_bytes).max().unwrap_or(0),
        project_peak_bytes: runs.iter().map(|r| r.project_peak_bytes).max().unwrap_or(0),
        us_per_nnz: pick(&|r| r.us_per_nnz),
        ..first
    }
}

fn csv_header() -> &'static str {
    "nodes,edges,density,nnz,source_bytes,gen_ms,compile_ms,view_ms,coo_ms,\
     project_ms,compile_peak_bytes,project_peak_bytes,us_per_nnz,reps"
}

fn csv_line(r: &PointRow, reps: usize) -> String {
    format!(
        "{},{},{:.4},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.4},{}\n",
        r.nodes,
        r.edges,
        r.density,
        r.nnz,
        r.source_bytes,
        r.gen_ms,
        r.compile_ms,
        r.view_ms,
        r.coo_ms,
        r.project_ms,
        r.compile_peak_bytes,
        r.project_peak_bytes,
        r.us_per_nnz,
        reps,
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let inputs_dir = PathBuf::from("target/benchmarks/scaling_sweep_inputs");
    std::fs::create_dir_all(&inputs_dir).ok();
    if let Some(p) = cli.out.parent() {
        std::fs::create_dir_all(p).ok();
    }

    let mem_ceiling = (cli.mem_ceiling_gb * (1u64 << 30) as f64) as usize;
    let mut csv = String::new();
    csv.push_str(csv_header());
    csv.push('\n');
    std::fs::write(&cli.out, &csv)?;

    let mut points: Vec<PointRow> = Vec::new();
    let mut reps_used: Vec<usize> = Vec::new();
    let t_all = Instant::now();
    let mut limit = String::new();
    let mut last_ok_nnz = 0usize;
    let mut first_failed = String::from("none");

    println!(
        "{:>7} {:>7} {:>7} {:>9} {:>10} {:>10} {:>11} {:>11} {:>9}",
        "nodes",
        "edges",
        "density",
        "nnz",
        "compile_ms",
        "project_ms",
        "cPeakMB",
        "pPeakMB",
        "us/nnz"
    );

    for &(nodes, edges, density) in LADDER {
        let est_nnz = (edges as f64 * nodes as f64 * density) as usize;
        let reps = if est_nnz > cli.big_nnz { 1 } else { cli.reps };

        let t_point = Instant::now();
        let mut runs = Vec::with_capacity(reps);
        for rep in 0..reps {
            let seed = cli.seed ^ ((nodes as u64) << 20) ^ ((edges as u64) << 4) ^ rep as u64;
            match run_point(nodes, edges, density, seed, &inputs_dir) {
                Ok(r) => runs.push(r),
                Err(e) => {
                    first_failed = format!("nodes={nodes} edges={edges}: {e}");
                    limit = format!("compile/project error at nodes={nodes} edges={edges}: {e}");
                    break;
                }
            }
        }
        if runs.is_empty() {
            break;
        }
        let row = median_row(&runs);
        let point_elapsed = t_point.elapsed().as_secs_f64();

        println!(
            "{:>7} {:>7} {:>7.3} {:>9} {:>10.3} {:>10.3} {:>11.1} {:>11.1} {:>9.4}",
            row.nodes,
            row.edges,
            row.density,
            row.nnz,
            row.compile_ms,
            row.project_ms,
            row.compile_peak_bytes as f64 / 1e6,
            row.project_peak_bytes as f64 / 1e6,
            row.us_per_nnz,
        );

        // Persist immediately so partial progress survives a later stop.
        let line = csv_line(&row, reps);
        csv.push_str(&line);
        std::fs::write(&cli.out, &csv)?;

        last_ok_nnz = row.nnz;
        points.push(row.clone());
        reps_used.push(reps);

        // ── stop conditions ──
        if point_elapsed > cli.per_point_seconds {
            limit = format!(
                "per-point wall budget exceeded ({point_elapsed:.1}s > {:.1}s) at nnz={}",
                cli.per_point_seconds, row.nnz
            );
            first_failed = format!("next size after nnz={} (wall)", row.nnz);
            break;
        }
        let cur_peak = peak_bytes();
        if row.compile_peak_bytes.max(row.project_peak_bytes) > mem_ceiling
            || cur_peak > mem_ceiling
        {
            limit = format!(
                "soft memory ceiling exceeded ({} B > {} B) at nnz={}",
                row.compile_peak_bytes.max(row.project_peak_bytes),
                mem_ceiling,
                row.nnz
            );
            first_failed = format!("next size after nnz={} (memory)", row.nnz);
            break;
        }
        if t_all.elapsed().as_secs_f64() > cli.total_seconds {
            limit = format!(
                "total wall budget exceeded ({:.1}s) after nnz={}",
                cli.total_seconds, row.nnz
            );
            first_failed = format!("next size after nnz={} (total wall)", row.nnz);
            break;
        }
    }

    if limit.is_empty() {
        limit = format!(
            "ladder exhausted without hitting a resource limit; largest nnz={last_ok_nnz} completed within budgets"
        );
    }

    // ── scaling exponents against nnz (report, don't assert linearity) ──
    let nnzs: Vec<f64> = points.iter().map(|p| p.nnz as f64).collect();
    let (b_compile, r2_compile) = loglog_fit(
        &nnzs,
        &points.iter().map(|p| p.compile_ms).collect::<Vec<_>>(),
    );
    let (b_project, r2_project) = loglog_fit(
        &nnzs,
        &points.iter().map(|p| p.project_ms).collect::<Vec<_>>(),
    );

    let summary = serde_json::json!({
        "points": points.len(),
        "largest_nnz_processed": last_ok_nnz,
        "limit_encountered": limit,
        "first_failed_or_impractical": first_failed,
        "fitted_exponent": { "compile": b_compile, "project": b_project },
        "fit_r2": { "compile": r2_compile, "project": r2_project },
        "us_per_nnz_at_largest": points.last().map(|p| p.us_per_nnz),
    });
    let summary_path = cli.out.with_extension("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!("\n=== Task 2 summary ===");
    println!("points measured        : {}", points.len());
    println!("largest nnz processed  : {last_ok_nnz}");
    println!("compile exponent (R²)  : {b_compile:.3} ({r2_compile:.4})");
    println!("project exponent (R²)  : {b_project:.3} ({r2_project:.4})");
    println!("limit_encountered      : {limit}");
    println!("wrote {} and {}", cli.out.display(), summary_path.display());
    Ok(())
}
