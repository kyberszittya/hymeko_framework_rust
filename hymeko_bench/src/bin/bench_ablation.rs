//! Experiment #6 of the SMC paper — *in-process vs. subprocess generation*.
//!
//! Decomposes HyMeKo's codegen-cost advantage into two terms by timing three
//! ways to turn the same compiled IR into a robot description:
//!
//! * **C1 — signed-hypergraph, in-process.** The native path:
//!   [`hymeko_formats::generate_description`] renders directly from the IR.
//! * **C2 — binary mock, in-process.** The IR's contextual hyperedges are
//!   clique-expanded to a [`BinaryGraph`] and emitted by its mock generator —
//!   in-process, but with no hyperedge structure to exploit.
//! * **C3 — subprocess toolchain.** Shelling out to `xacro`/`gz`/`mujoco`.
//!   Those binaries are absent here, so C3 is an **analytic** constant
//!   (`~464 ms`, the documented process-spawn + parse estimate), tagged as such.
//!
//! `C1 − C2` is the representational gain; `C2 − C3` is the architectural gain.
//! Output: `hymeko_bench/results/ablation.csv` with median/IQR/worst per config.
//!
//! Run from the repo root (the codegen needs `transforms/`):
//!   cargo run --release -p hymeko_bench --bin bench_ablation -- \
//!       --input examples/paper/hymeko_robot.hymeko
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use hymeko::common::ids::DeclId;
use hymeko::ir::ir::{DeclKind, Ir};
use hymeko::module_store::module_store::ModuleStore;
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::resolution::interner::Interner;
use hymeko::util::real_parser::RealParser;
use hymeko_formats::{BinaryGraph, OutputFormat, generate_description};

/// Documented process-spawn + parse estimate for the subprocess toolchain
/// (xacro→URDF), used when the external binaries are not installed.
const C3_ANALYTIC_MS: f64 = 464.0;

#[derive(Parser, Debug)]
#[command(about = "SMC #6: in-process vs subprocess generation ablation")]
struct Cli {
    /// Canonical HyMeKo source.
    #[arg(long, default_value = "examples/paper/hymeko_robot.hymeko")]
    input: PathBuf,
    /// Output CSV.
    #[arg(long, default_value = "hymeko_bench/results/ablation.csv")]
    out: PathBuf,
    /// Hyperedge base types counted as "contextual" (the binary graph's edges).
    #[arg(long, default_value = "interpretation,aggregation")]
    contextual_bases: String,
    /// Timing iterations after warm-up (>= 5 required by CLAUDE.md §3).
    #[arg(long, default_value_t = 200)]
    iters: usize,
}

/// median, inter-quartile range, and worst (max) of a timing sample.
#[derive(Debug, Clone, Copy)]
struct Stats {
    median_ms: f64,
    iqr_ms: f64,
    worst_ms: f64,
    n: usize,
}

fn summarize(times: &mut [f64]) -> Stats {
    debug_assert!(!times.is_empty(), "empty timing sample");
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |frac: f64| times[((times.len() as f64 * frac) as usize).min(times.len() - 1)];
    Stats {
        median_ms: q(0.5),
        iqr_ms: q(0.75) - q(0.25),
        worst_ms: *times.last().unwrap(),
        n: times.len(),
    }
}

/// Build the contextual binary graph: per-hyperedge member vertex indices
/// (remapped to a contiguous `0..n_verts`) and a representative sign.
///
/// # Postconditions
/// - returned `members.len() == signs.len()`; every member index `< n_verts`.
fn gather_binary_graph(ir: &Ir, it: &Interner, bases: &[String]) -> BinaryGraph {
    let mut vert_index: std::collections::BTreeMap<DeclId, usize> =
        std::collections::BTreeMap::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut signs: Vec<i8> = Vec::new();

    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        let Some(eid) = ir.as_edge(DeclId::new(i)) else {
            continue;
        };
        let is_contextual = ir.edges[eid.0].bases.iter().any(|b| {
            let name = it.resolve(ir.decl_nodes[b.target().0].name);
            bases.iter().any(|cb| cb == name)
        });
        if !is_contextual {
            continue;
        }
        let mut this: Vec<usize> = Vec::new();
        let mut sign: i8 = 1;
        let mut first = true;
        for &aid in &ir.edges[eid.0].arcs {
            for r in &ir.arcs[aid.0].refs {
                let next = vert_index.len();
                let idx = *vert_index.entry(r.target()).or_insert(next);
                this.push(idx);
                if first {
                    sign = r.sign();
                    first = false;
                }
            }
        }
        members.push(this);
        signs.push(sign);
    }
    BinaryGraph::from_hyperedges(vert_index.len(), &members, &signs)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    assert!(cli.iters >= 5, "need >= 5 timing iterations (CLAUDE.md §3)");
    let bases: Vec<String> = cli
        .contextual_bases
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // ─── Compile once ───
    let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
    let compiled = store
        .compile(&cli.input)
        .map_err(|e| anyhow::anyhow!("compiling {}: {e:?}", cli.input.display()))?;
    let ir = &compiled.ir;

    let bg = gather_binary_graph(ir, &store.it, &bases);
    println!(
        "=== SMC #6 ablation: {} ===\n  binary graph: |V|={}, |E|={} (clique-expanded contextual)",
        cli.input.display(),
        bg.n_verts,
        bg.edges.len()
    );

    // ─── C1: signed-hypergraph, in-process (real codegen) ───
    // Validate once (surfaces a codegen error before the timing loop).
    let c1_out = generate_description(ir, &store.it, "hymeko_robot", OutputFormat::Urdf)
        .map_err(|e| anyhow::anyhow!("C1 generate_description: {e}"))?;
    let _ = generate_description(ir, &store.it, "hymeko_robot", OutputFormat::Urdf); // warm-up
    let mut c1 = Vec::with_capacity(cli.iters);
    for _ in 0..cli.iters {
        let t = Instant::now();
        let s = generate_description(ir, &store.it, "hymeko_robot", OutputFormat::Urdf)
            .map_err(|e| anyhow::anyhow!("C1: {e}"))?;
        c1.push(t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(s);
    }
    let c1s = summarize(&mut c1);

    // ─── C2: binary mock, in-process ───
    let _ = bg.mock_emit(); // warm-up
    let mut c2 = Vec::with_capacity(cli.iters);
    for _ in 0..cli.iters {
        let t = Instant::now();
        let s = bg.mock_emit();
        c2.push(t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(s);
    }
    let c2s = summarize(&mut c2);

    // ─── C3: subprocess toolchain (analytic — binaries absent) ───
    let c3_median = C3_ANALYTIC_MS;

    // Performance assertions (§3): real measurements, >= 5 iters, positive.
    assert!(c1s.median_ms > 0.0 && c1s.n >= 5, "C1 measurement invalid");
    assert!(c2s.median_ms > 0.0 && c2s.n >= 5, "C2 measurement invalid");

    println!(
        "  C1 hypergraph in-proc : median {:.4} ms  IQR {:.4}  worst {:.4}  (n={}, urdf {} B)",
        c1s.median_ms,
        c1s.iqr_ms,
        c1s.worst_ms,
        c1s.n,
        c1_out.len()
    );
    println!(
        "  C2 binary mock in-proc: median {:.4} ms  IQR {:.4}  worst {:.4}  (n={})",
        c2s.median_ms, c2s.iqr_ms, c2s.worst_ms, c2s.n
    );
    println!(
        "  C3 subprocess         : {:.1} ms  (ANALYTIC — xacro/gz absent)",
        c3_median
    );
    println!(
        "  representational gain (C1-C2): {:+.4} ms | architectural gain (C2-C3): {:+.4} ms",
        c1s.median_ms - c2s.median_ms,
        c2s.median_ms - c3_median
    );

    // ─── CSV ───
    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(&cli.out)?;
    wtr.write_record([
        "config",
        "median_ms",
        "iqr_ms",
        "worst_ms",
        "n_iters",
        "kind",
    ])?;
    let row = |w: &mut csv::Writer<std::fs::File>, cfg: &str, s: Stats, kind: &str| -> Result<()> {
        w.write_record([
            cfg,
            &format!("{:.6}", s.median_ms),
            &format!("{:.6}", s.iqr_ms),
            &format!("{:.6}", s.worst_ms),
            &s.n.to_string(),
            kind,
        ])?;
        Ok(())
    };
    row(&mut wtr, "C1_hypergraph_inproc", c1s, "measured")?;
    row(&mut wtr, "C2_binary_mock_inproc", c2s, "measured")?;
    wtr.write_record([
        "C3_subprocess",
        &format!("{:.6}", c3_median),
        "0.0",
        &format!("{:.6}", c3_median),
        "0",
        "analytic",
    ])?;
    wtr.flush()?;
    println!("\nwrote {}", cli.out.display());
    Ok(())
}
