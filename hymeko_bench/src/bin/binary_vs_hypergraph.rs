//! Experiment 1 of the MDPI Technologies paper —
//! *Binary vs. Hypergraph representational cost.*
//!
//! Reads a compiled HyMeKo IR, filters to the paper's "contextual" subset
//! (hyperedges carrying the aggregation/interpretation/constraint types),
//! and measures representational cost under four encodings:
//!
//! 1. **Hypergraph (native)** — |V|, |E|, nnz(B) directly from the IR.
//! 2. **Star expansion** — |V| + |E| vertices (hyperedges become auxiliary
//!    vertices), and one signed edge per incidence. Polarity preserved.
//! 3. **Clique expansion** — |V| unchanged; each hyperedge becomes a clique
//!    of `C(arity, 2)` undirected edges. Polarity lost.
//! 4. **Binary pairwise** — same edge count as clique, but labelled as a
//!    classical graph (no hyperedge identity retained).
//!
//! Produces `hymeko_bench/results/binary_vs_hypergraph.csv` with per-example
//! and per-arity rows, plus an arity-sweep corpus for Table 7.
//!
//! Run:
//!   cargo run --release -p hymeko_bench --bin binary_vs_hypergraph -- \
//!       --input examples/paper/hymeko_robot.hymeko \
//!       --out   hymeko_bench/results/binary_vs_hypergraph.csv

use std::collections::BTreeSet;
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

type ContextualHyperedge = (DeclId, usize, Vec<i8>);

#[derive(Parser, Debug)]
#[command(about = "Binary-vs-hypergraph representational-cost benchmark")]
struct Cli {
    /// Path to the canonical HyMeKo source.
    #[arg(long, default_value = "examples/paper/hymeko_robot.hymeko")]
    input: PathBuf,

    /// Output CSV.
    #[arg(long, default_value = "hymeko_bench/results/binary_vs_hypergraph.csv")]
    out: PathBuf,

    /// Hyperedge types considered "contextual" (only these count toward
    /// the paper's Table 6 headline E=10 figure). Comma-separated.
    #[arg(long, default_value = "interpretation,aggregation")]
    contextual_bases: String,

    /// Number of timing iterations for the build-time measurement.
    #[arg(long, default_value_t = 1000)]
    iters: usize,
}

/// Summary of the four encodings' representational cost for a given set
/// of hyperedges.
#[derive(Debug, Clone, Copy)]
struct EncodingStats {
    n_verts: usize,
    n_edges: usize,
    signed_entries: usize,
    polarity: bool,
}

fn compute_encodings(edge_arities: &[usize], n_verts: usize, nnz: usize) -> [EncodingStats; 4] {
    let n_edges = edge_arities.len();
    let hypergraph = EncodingStats {
        n_verts,
        n_edges,
        signed_entries: nnz,
        polarity: true,
    };
    let star = EncodingStats {
        // Each hyperedge becomes an auxiliary vertex.
        n_verts: n_verts + n_edges,
        // One signed bipartite edge per incidence entry.
        n_edges: nnz,
        signed_entries: nnz,
        polarity: true,
    };
    let clique_edges: usize = edge_arities
        .iter()
        .map(|&k| k * (k.saturating_sub(1)) / 2)
        .sum();
    let clique = EncodingStats {
        n_verts,
        n_edges: clique_edges,
        signed_entries: 0,
        polarity: false,
    };
    let binary = EncodingStats {
        n_verts,
        n_edges: clique_edges,
        signed_entries: 0,
        polarity: false,
    };
    [hypergraph, star, clique, binary]
}

/// Collect contextual hyperedges — those whose base type's name is in
/// `contextual_bases`. Returns a Vec of (edge_decl, arity) and the set
/// of participating vertex DeclIds.
fn gather_contextual_edges(
    ir: &Ir,
    it: &Interner,
    contextual_bases: &[String],
) -> (Vec<ContextualHyperedge>, BTreeSet<DeclId>) {
    let mut edges = Vec::new();
    let mut vertices = std::collections::BTreeSet::new();
    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        let e_did = DeclId::new(i);
        // Look up the edge's inherited bases; match against contextual set.
        let Some(eid) = ir.as_edge(e_did) else {
            continue;
        };
        let bases = &ir.edges[eid.0].bases;
        let is_contextual = bases.iter().any(|b| {
            let name = it.resolve(ir.decl_nodes[b.target().0].name);
            contextual_bases.iter().any(|cb| cb == name)
        });
        if !is_contextual {
            continue;
        }
        // Arity = total number of signed references in this edge's arcs.
        let mut signs = Vec::new();
        for &aid in &ir.edges[eid.0].arcs {
            for r in &ir.arcs[aid.0].refs {
                signs.push(r.sign());
                vertices.insert(r.target());
            }
        }
        let arity = signs.len();
        edges.push((e_did, arity, signs));
    }
    (edges, vertices)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let contextual_bases: Vec<String> = cli
        .contextual_bases
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // ─── Compile the canonical source ───
    let t0 = Instant::now();
    let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
    let compiled = store
        .compile(&cli.input)
        .map_err(|e| anyhow::anyhow!("compiling {}: {e:?}", cli.input.display()))?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ir = &compiled.ir;
    let it = &store.it;

    // ─── Canonical example stats ───
    let (ctx_edges, ctx_vertices) = gather_contextual_edges(ir, it, &contextual_bases);
    let arity_multiset: Vec<usize> = {
        let mut a: Vec<usize> = ctx_edges.iter().map(|(_, k, _)| *k).collect();
        a.sort();
        a
    };
    let nnz: usize = arity_multiset.iter().sum();
    let n_verts = ctx_vertices.len();

    let encodings = compute_encodings(&arity_multiset, n_verts, nnz);
    let names = ["hypergraph", "star", "clique", "binary"];

    println!("=== Canonical example: {} ===", cli.input.display());
    println!("  compile time:   {:.2} ms", compile_ms);
    println!("  |V| (ctx):      {}", n_verts);
    println!("  |E| (ctx):      {}", arity_multiset.len());
    println!("  nnz(B):         {}", nnz);
    println!("  Arity multiset: {:?}", arity_multiset);
    println!();
    println!(
        "  {:<12} {:>6} {:>6} {:>8} Polarity",
        "Encoding", "Nodes", "Edges", "SignedE"
    );
    for (name, s) in names.iter().zip(encodings.iter()) {
        println!(
            "  {:<12} {:>6} {:>6} {:>8} {}",
            name, s.n_verts, s.n_edges, s.signed_entries, s.polarity
        );
    }

    // ─── Build-time micro-benchmark ───
    // We time the COMPILE+LOWER step as the "build" latency, since that's
    // how long it takes to go from .hymeko source to an in-memory IR.
    let mut build_times = Vec::with_capacity(cli.iters);
    for _ in 0..cli.iters {
        let mut fresh_store = ModuleStore::new(StdFsProvider::new(), RealParser);
        let t = Instant::now();
        let _c = fresh_store
            .compile(&cli.input)
            .map_err(|e| anyhow::anyhow!("compile: {e:?}"))?;
        build_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    build_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let build_median = build_times[cli.iters / 2];
    let build_p95 = build_times[(cli.iters * 95) / 100];
    println!();
    println!("Build time (n={} iters):", cli.iters);
    println!(
        "  median: {:.2} ms   p95: {:.2} ms",
        build_median, build_p95
    );

    // ─── Arity sweep corpus ───
    // For Table 7, synthesise single-hyperedge stats at varied arities.
    // This is a purely analytic sweep — no IR construction needed.
    let arities: Vec<usize> = vec![2, 3, 4, 5, 6, 8, 10];
    let mut sweep = Vec::new();
    for &k in &arities {
        // k vertices, 1 hyperedge, k signed entries.
        let stats = compute_encodings(&[k], k, k);
        sweep.push((k, stats));
    }
    println!();
    println!("Arity sweep (Table 7):");
    println!("  k  | hyper(V,E,nnz) | star(V,E) | clique(E) | polarity_lost?");
    for (k, s) in &sweep {
        println!(
            "  {:>2} |  ({},{},{})     | ({},{})     | {}           | {}",
            k,
            s[0].n_verts,
            s[0].n_edges,
            s[0].signed_entries,
            s[1].n_verts,
            s[1].n_edges,
            s[2].n_edges,
            !s[2].polarity && *k >= 3,
        );
    }

    // ─── Write CSV ───
    std::fs::create_dir_all(cli.out.parent().unwrap())?;
    let mut wtr = csv::Writer::from_path(&cli.out)?;
    wtr.write_record([
        "corpus",
        "arity_k",
        "encoding",
        "n_verts",
        "n_edges",
        "signed_entries",
        "polarity",
        "build_median_ms",
        "build_p95_ms",
    ])?;

    // Canonical example rows.
    for (name, s) in names.iter().zip(encodings.iter()) {
        wtr.write_record([
            "canonical",
            "",
            name,
            &s.n_verts.to_string(),
            &s.n_edges.to_string(),
            &s.signed_entries.to_string(),
            &s.polarity.to_string(),
            &format!("{:.4}", build_median),
            &format!("{:.4}", build_p95),
        ])?;
    }

    // Arity sweep rows.
    for (k, stats) in &sweep {
        for (name, s) in names.iter().zip(stats.iter()) {
            wtr.write_record([
                "arity_sweep",
                &k.to_string(),
                name,
                &s.n_verts.to_string(),
                &s.n_edges.to_string(),
                &s.signed_entries.to_string(),
                &s.polarity.to_string(),
                "",
                "",
            ])?;
        }
    }

    wtr.flush()?;
    println!();
    println!("Wrote {}", cli.out.display());

    // ─── IR Blake3 hash for deterministic reporting ───
    // The compiled program's canonical hash is already in `compiled.ir_hash`
    // (or exposed via pretty_print_compiled). We surface it to stdout so
    // the downstream integration report can quote it verbatim.
    if let Some(hash) = &compiled.ir.doc_hash {
        println!("IR doc hash: {:?}", hash);
    }

    Ok(())
}
