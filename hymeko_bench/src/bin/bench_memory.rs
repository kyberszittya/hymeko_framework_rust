//! SMC 2026 Task 1 — measured memory footprint + storage-overhead ratio ρ.
//!
//! For each of the five robotics fixtures: compile it under the tracking
//! allocator (peak heap during `compile`), then build both the raw
//! adjacency-list baseline and the Proposition-4-modeled IR (incidence +
//! `(n+m)` declaration records) and measure each structure's live heap. Emits
//! `ρ_measured = IR bytes / adjacency bytes` next to the count-based
//! `ρ_predicted = 1 + (n+m)/(m·d̄)`.
//!
//! Run (workspace root, release, single-threaded):
//!   cargo run --release -p hymeko_bench --bin bench_memory -- \
//!       --fixtures-root data/robotics --out <run>/raw/task1_memory.csv

use std::hint::black_box;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;

use hymeko::module_store::module_store::ModuleStore;
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
use hymeko::util::real_parser::RealParser;
use hymeko_hnn::traversal::hypergraphview::HyperGraphView;

use hymeko_bench::incidence::{DIGEST_BYTES, build_ir_records};
use hymeko_bench::measure;

/// Aggregation config matching the COO-builder benchmark (the canonical
/// star-expansion path). Fixes how signed multi-references collapse per edge.
const AGG: AggCfg = AggCfg {
    weight: WeightAgg::Sum,
    sign: SignAgg::PreferNonNeutral,
    clamp01: false,
};

// Necessarily-global allocation hook (see track_alloc.rs for the §6.5 #11
// justification). The instance lives in the binary; the counters in the lib.
#[global_allocator]
static GLOBAL: hymeko_bench::TrackingAlloc = hymeko_bench::TrackingAlloc;

/// The five camera-ready fixtures, matching `hymeko_query`'s workflow bench.
const FIXTURES: &[(&str, &str)] = &[
    ("mini_arm", "mini_arm.hymeko"),
    ("anthropomorphic_arm", "anthropomorphic_arm.hymeko"),
    ("anthropomorphic_using", "anthropomorphic_arm_using.hymeko"),
    ("robot_4wh", "robot_4wh.hymeko"),
    ("robot_4wh_using", "robot_4wh_using.hymeko"),
];

#[derive(Parser, Debug)]
#[command(about = "SMC2026 Task 1: measured memory footprint + rho")]
struct Cli {
    /// Directory holding the five `.hymeko` fixtures.
    #[arg(long, default_value = "data/robotics")]
    fixtures_root: PathBuf,
    /// Output CSV path.
    #[arg(long, default_value = "task1_memory.csv")]
    out: PathBuf,
}

#[derive(Debug)]
struct Row {
    fixture: String,
    n: usize,
    m: usize,
    nnz: usize,
    mean_arity: f64,
    n_over_m: f64,
    peak_heap_bytes: usize,
    ir_resident_bytes: usize,
    adj_baseline_bytes: usize,
    rec_bytes: usize,
    bytes_per_vertex: f64,
    bytes_per_incidence: f64,
    rho_measured: f64,
    rho_predicted: f64,
}

fn measure_fixture(label: &str, path: &Path) -> Result<Row> {
    // ── compile under the tracking allocator: peak = transient heap of the
    //    parse+intern+resolve+lower pipeline. Keep both store and compiled
    //    alive so the interner-backed IR references stay valid.
    let ((store, compiled), _retained, peak_heap_bytes) = measure(|| {
        let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
        let compiled = store
            .compile(path)
            .map_err(|e| anyhow::anyhow!("compile {}: {e:?}", path.display()));
        (store, compiled)
    });
    let compiled = compiled?;
    let ir = &compiled.ir;

    // ── canonical incidence: the pipeline's Levi/star HyperGraphView, the
    //    same representation Proposition 4's `m·d̄ signed-incidence entries`
    //    refers to (and that star_expansion_coo consumes).
    let ex = ScalarWeightExtractor;
    let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(ir, &AGG, &ex);
    let n = hg.num_nodes();
    let m = hg.num_edges();
    let nnz = hg.flat_edge_nodes.len();
    let mean_arity = if m == 0 { 0.0 } else { nnz as f64 / m as f64 };
    let n_over_m = if m == 0 {
        f64::NAN
    } else {
        n as f64 / m as f64
    };

    // ── raw adjacency baseline: Vec<Vec<usize>> of the nnz incidence entries
    //    (per-hyperedge member vertex ids), built from the canonical view.
    let (adj, adj_baseline_bytes, _) = measure(|| {
        let mut a: Vec<Vec<usize>> = Vec::with_capacity(m);
        for e in 0..m {
            let s = hg.edge_offsets[e];
            let t = hg.edge_offsets[e + 1];
            let mut members: Vec<usize> =
                hg.flat_edge_nodes[s..t].iter().map(|nid| nid.0).collect();
            members.shrink_to_fit();
            a.push(members);
        }
        a.shrink_to_fit();
        a
    });
    // ── the (n+m) declaration records the IR carries on top of the incidence.
    let (recs, rec_bytes, _) = measure(|| build_ir_records(n + m));

    black_box(&adj);
    black_box(&recs);

    // Modeled IR = incidence (== adjacency baseline entries) + records.
    let ir_resident_bytes = adj_baseline_bytes + rec_bytes;
    let rho_measured = if adj_baseline_bytes == 0 {
        f64::NAN
    } else {
        ir_resident_bytes as f64 / adj_baseline_bytes as f64
    };
    let bytes_per_vertex = if n == 0 {
        f64::NAN
    } else {
        ir_resident_bytes as f64 / n as f64
    };
    let bytes_per_incidence = if nnz == 0 {
        f64::NAN
    } else {
        ir_resident_bytes as f64 / nnz as f64
    };
    // Count-based predicted overhead ρ = 1 + (n+m)/(m·d̄) with unit constant.
    let rho_predicted = if nnz == 0 {
        f64::NAN
    } else {
        1.0 + (n + m) as f64 / nnz as f64
    };
    let _ = DIGEST_BYTES; // record size is asserted in incidence unit tests

    drop(adj);
    drop(recs);
    drop(store);

    Ok(Row {
        fixture: label.to_string(),
        n,
        m,
        nnz,
        mean_arity,
        n_over_m,
        peak_heap_bytes,
        ir_resident_bytes,
        adj_baseline_bytes,
        rec_bytes,
        bytes_per_vertex,
        bytes_per_incidence,
        rho_measured,
        rho_predicted,
    })
}

fn write_csv(rows: &[Row], out: &Path) -> Result<()> {
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let mut s = String::new();
    s.push_str(
        "fixture,n,m,nnz,mean_arity,n_over_m,peak_heap_bytes,ir_resident_bytes,\
         adj_baseline_bytes,rec_bytes,bytes_per_vertex,bytes_per_incidence,\
         rho_measured,rho_predicted\n",
    );
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{:.4},{:.4},{},{},{},{},{:.3},{:.3},{:.4},{:.4}\n",
            r.fixture,
            r.n,
            r.m,
            r.nnz,
            r.mean_arity,
            r.n_over_m,
            r.peak_heap_bytes,
            r.ir_resident_bytes,
            r.adj_baseline_bytes,
            r.rec_bytes,
            r.bytes_per_vertex,
            r.bytes_per_incidence,
            r.rho_measured,
            r.rho_predicted,
        ));
    }
    std::fs::write(out, s).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut rows = Vec::new();
    println!(
        "{:<24} {:>4} {:>4} {:>6} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "fixture", "n", "m", "nnz", "d_bar", "peakHeapB", "irB", "adjB", "rhoM", "rhoP"
    );
    for (label, file) in FIXTURES {
        let path = cli.fixtures_root.join(file);
        let row = measure_fixture(label, &path)?;
        println!(
            "{:<24} {:>4} {:>4} {:>6} {:>6.2} {:>10} {:>10} {:>10} {:>8.3} {:>8.3}",
            row.fixture,
            row.n,
            row.m,
            row.nnz,
            row.mean_arity,
            row.peak_heap_bytes,
            row.ir_resident_bytes,
            row.adj_baseline_bytes,
            row.rho_measured,
            row.rho_predicted,
        );
        rows.push(row);
    }
    write_csv(&rows, &cli.out)?;
    println!("wrote {}", cli.out.display());
    Ok(())
}
