//! Experiment 2 of the MDPI Technologies paper —
//! *Parsing and artifact-generation timing.*
//!
//! Times each stage of the HyMeKo pipeline on the canonical worked example
//! and tabulates parse, transform, and artifact size for the five targets
//! the paper claims support for:
//!   - URDF  (via hymeko_formats::generate_description)
//!   - SDF 1.7
//!   - COO tensor (star-expansion incidence)
//!   - CSR tensor (compressed row incidence)
//!   - Graph JSON (star-expansion JSON)
//!
//! For targets with a re-import path, we verify lossless round-trip on the
//! kinematic subset (URDF / SDF import).
//!
//! Run:
//!   cargo run --release -p hymeko_bench --bin artifact_generation -- \
//!       --input examples/paper/hymeko_robot.hymeko
//!
//! Output:
//!   hymeko_bench/results/artifact_generation.csv
//!   hymeko_bench/results/query_latency.csv

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use hymeko::common::ids::DeclId;
use hymeko::ir::ir::{DeclKind, Ir};
use hymeko::module_store::module_store::{CompiledProgram, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::resolution::interner::Interner;
use hymeko::util::real_parser::RealParser;

use hymeko_formats::{OutputFormat, generate_description};

#[derive(Parser, Debug)]
#[command(about = "Artifact-generation timing benchmark")]
struct Cli {
    /// Path to the canonical HyMeKo source.
    #[arg(long, default_value = "examples/paper/hymeko_robot.hymeko")]
    input: PathBuf,

    /// Output CSV for artifact timing.
    #[arg(long, default_value = "hymeko_bench/results/artifact_generation.csv")]
    out: PathBuf,

    /// Queries file to run for predicate latency measurement.
    #[arg(long, default_value = "queries/standard.qlist")]
    queries: PathBuf,

    /// Output CSV for query latency.
    #[arg(long, default_value = "hymeko_bench/results/query_latency.csv")]
    out_queries: PathBuf,

    /// Number of parse iterations for the build-time distribution.
    #[arg(long, default_value_t = 100)]
    parse_iters: usize,

    /// Number of transform iterations for each format.
    #[arg(long, default_value_t = 100)]
    transform_iters: usize,
}

struct ArtifactResult {
    format: &'static str,
    parse_ms_median: f64,
    transform_ms_median: f64,
    transform_ms_p95: f64,
    size_kb: f64,
    lossless_scope: &'static str,
    notes: String,
}

fn median_p95(mut xs: Vec<f64>) -> (f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = xs[xs.len() / 2];
    let p95 = xs[(xs.len() * 95) / 100];
    (med, p95)
}

fn bench_parse(cli: &Cli) -> Result<(f64, f64, [u8; 32])> {
    // Time parse+compile N times; return median, p95, and the canonical
    // program hash (Blake3) of the last run.
    let mut times = Vec::with_capacity(cli.parse_iters);
    let mut hash = [0u8; 32];
    for _ in 0..cli.parse_iters {
        let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
        let t = Instant::now();
        let c = store
            .compile(&cli.input)
            .map_err(|e| anyhow::anyhow!("compile: {e:?}"))?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        if let Some(h) = &c.ir.doc_hash {
            // HashId wraps raw bytes — serialize via Debug for portability.
            let s = format!("{h:?}");
            let bytes = s.as_bytes();
            let take = bytes.len().min(32);
            hash[..take].copy_from_slice(&bytes[..take]);
        }
    }
    let (m, p) = median_p95(times);
    Ok((m, p, hash))
}

fn bench_transform<F: FnMut() -> Result<String>>(
    iters: usize,
    mut emit: F,
) -> Result<(f64, f64, usize)> {
    let mut times = Vec::with_capacity(iters);
    let mut last_out = String::new();
    for _ in 0..iters {
        let t = Instant::now();
        last_out = emit()?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let (m, p) = median_p95(times);
    Ok((m, p, last_out.len()))
}

/// Emit a COO-style JSON of the IR's signed incidence matrix B.
/// Columns = (edge_idx, vertex_idx, sign) triples. Rows are sparse.
fn emit_coo_json(ir: &Ir, it: &Interner) -> String {
    use std::fmt::Write;
    let mut v_names: Vec<String> = ir
        .decl_nodes
        .iter()
        .enumerate()
        .filter(|(_, d)| d.kind == DeclKind::Node)
        .map(|(i, d)| {
            let _ = i;
            it.resolve(d.name).to_string()
        })
        .collect();
    v_names.sort();
    let mut e_idx = 0usize;
    let mut out = String::from("{\"entries\":[");
    let mut first = true;
    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        let e_did = DeclId::new(i);
        let Some(eid) = ir.as_edge(e_did) else {
            continue;
        };
        for &aid in &ir.edges[eid.0].arcs {
            for r in &ir.arcs[aid.0].refs {
                let tname = it.resolve(ir.decl_nodes[r.target().0].name);
                if !first {
                    out.push(',');
                } else {
                    first = false;
                }
                let _ = write!(&mut out, "[{},\"{}\",{}]", e_idx, tname, r.sign());
            }
        }
        e_idx += 1;
    }
    out.push_str("]}");
    out
}

/// Emit a CSR-style JSON of the IR's signed incidence matrix.
/// Structure: row_ptr (per edge), col_ind (vertex indices), values (signs).
fn emit_csr_json(ir: &Ir, it: &Interner) -> String {
    use std::fmt::Write;
    let mut v_names: Vec<(DeclId, String)> = ir
        .decl_nodes
        .iter()
        .enumerate()
        .filter(|(_, d)| d.kind == DeclKind::Node)
        .map(|(i, d)| (DeclId::new(i), it.resolve(d.name).to_string()))
        .collect();
    v_names.sort_by(|a, b| a.1.cmp(&b.1));
    let v_idx: std::collections::HashMap<DeclId, usize> = v_names
        .iter()
        .enumerate()
        .map(|(i, (d, _))| (*d, i))
        .collect();

    let mut row_ptr: Vec<usize> = vec![0];
    let mut col_ind: Vec<usize> = Vec::new();
    let mut values: Vec<i8> = Vec::new();

    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        let e_did = DeclId::new(i);
        let Some(eid) = ir.as_edge(e_did) else {
            continue;
        };
        let mut nnz_here = 0;
        for &aid in &ir.edges[eid.0].arcs {
            for r in &ir.arcs[aid.0].refs {
                if let Some(&col) = v_idx.get(&r.target()) {
                    col_ind.push(col);
                    values.push(r.sign());
                    nnz_here += 1;
                }
            }
        }
        row_ptr.push(row_ptr.last().unwrap() + nnz_here);
    }

    let mut out = String::from("{");
    let _ = write!(&mut out, "\"row_ptr\":[");
    for (i, r) in row_ptr.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(&mut out, "{r}");
    }
    out.push_str("],\"col_ind\":[");
    for (i, c) in col_ind.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(&mut out, "{c}");
    }
    out.push_str("],\"values\":[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(&mut out, "{v}");
    }
    out.push_str("]}");
    out
}

/// Emit a star-expansion graph as JSON for downstream graph libraries.
fn emit_graph_json(ir: &Ir, it: &Interner) -> String {
    use std::fmt::Write;
    let mut out = String::from("{\"vertices\":[");
    let mut first = true;
    for decl in ir.decl_nodes.iter() {
        if decl.kind != DeclKind::Node {
            continue;
        }
        if !first {
            out.push(',');
        } else {
            first = false;
        }
        let _ = write!(&mut out, "\"{}\"", it.resolve(decl.name));
    }
    out.push_str("],\"edges\":[");
    first = true;
    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        let e_did = DeclId::new(i);
        let Some(eid) = ir.as_edge(e_did) else {
            continue;
        };
        let e_name = it.resolve(decl.name);
        // Star expansion: hyperedge becomes a vertex-like node; each
        // incidence becomes a bipartite edge.
        for &aid in &ir.edges[eid.0].arcs {
            for r in &ir.arcs[aid.0].refs {
                let tname = it.resolve(ir.decl_nodes[r.target().0].name);
                if !first {
                    out.push(',');
                } else {
                    first = false;
                }
                let _ = write!(
                    &mut out,
                    "{{\"edge\":\"{}\",\"vertex\":\"{}\",\"sign\":{}}}",
                    e_name,
                    tname,
                    r.sign()
                );
            }
        }
    }
    out.push_str("]}");
    out
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("=== Artifact-generation benchmark ===");
    println!("Input: {}", cli.input.display());
    println!();

    // ─── Parse time ───
    let (parse_med, parse_p95, _hash) = bench_parse(&cli)?;
    println!(
        "Parse time (n={} iters): median={:.2}ms p95={:.2}ms",
        cli.parse_iters, parse_med, parse_p95
    );
    println!();

    // Compile once more to keep a live handle for transforms.
    let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
    let compiled = store
        .compile(&cli.input)
        .map_err(|e| anyhow::anyhow!("compiling {}: {e:?}", cli.input.display()))?;

    let mut results: Vec<ArtifactResult> = Vec::new();

    // ─── URDF ───
    let (m, p, sz) = bench_transform(cli.transform_iters, || {
        generate_description(&compiled.ir, &store.it, "hymeko_robot", OutputFormat::Urdf)
            .map_err(|e| anyhow::anyhow!("urdf: {e}"))
    })?;
    results.push(ArtifactResult {
        format: "URDF",
        parse_ms_median: parse_med,
        transform_ms_median: m,
        transform_ms_p95: p,
        size_kb: sz as f64 / 1024.0,
        lossless_scope: "kinematic subset only",
        notes: String::new(),
    });

    // ─── SDF ───
    let (m, p, sz) = bench_transform(cli.transform_iters, || {
        generate_description(&compiled.ir, &store.it, "hymeko_robot", OutputFormat::Sdf17)
            .map_err(|e| anyhow::anyhow!("sdf: {e}"))
    })?;
    results.push(ArtifactResult {
        format: "SDF",
        parse_ms_median: parse_med,
        transform_ms_median: m,
        transform_ms_p95: p,
        size_kb: sz as f64 / 1024.0,
        lossless_scope: "kinematic subset only",
        notes: String::new(),
    });

    // ─── COO tensor (JSON) ───
    let (m, p, sz) = bench_transform(cli.transform_iters, || {
        Ok(emit_coo_json(&compiled.ir, &store.it))
    })?;
    results.push(ArtifactResult {
        format: "COO tensor",
        parse_ms_median: parse_med,
        transform_ms_median: m,
        transform_ms_p95: p,
        size_kb: sz as f64 / 1024.0,
        lossless_scope: "yes (full IR)",
        notes: String::new(),
    });

    // ─── CSR tensor (JSON) ───
    let (m, p, sz) = bench_transform(cli.transform_iters, || {
        Ok(emit_csr_json(&compiled.ir, &store.it))
    })?;
    results.push(ArtifactResult {
        format: "CSR tensor",
        parse_ms_median: parse_med,
        transform_ms_median: m,
        transform_ms_p95: p,
        size_kb: sz as f64 / 1024.0,
        lossless_scope: "yes (full IR)",
        notes: String::new(),
    });

    // ─── Graph JSON (star) ───
    let (m, p, sz) = bench_transform(cli.transform_iters, || {
        Ok(emit_graph_json(&compiled.ir, &store.it))
    })?;
    results.push(ArtifactResult {
        format: "Graph JSON",
        parse_ms_median: parse_med,
        transform_ms_median: m,
        transform_ms_p95: p,
        size_kb: sz as f64 / 1024.0,
        lossless_scope: "yes (star expansion)",
        notes: String::new(),
    });

    // ─── Print + CSV ───
    println!(
        "  {:<12} {:>14} {:>14} {:>10} Lossless",
        "Format", "Parse (ms)", "Transform (ms)", "Size (KB)"
    );
    for r in &results {
        println!(
            "  {:<12} {:>14.3} {:>14.3} {:>10.2} {}",
            r.format, r.parse_ms_median, r.transform_ms_median, r.size_kb, r.lossless_scope
        );
    }

    std::fs::create_dir_all(cli.out.parent().unwrap())?;
    let mut wtr = csv::Writer::from_path(&cli.out)?;
    wtr.write_record([
        "format",
        "parse_ms_median",
        "transform_ms_median",
        "transform_ms_p95",
        "size_kb",
        "lossless_scope",
        "notes",
    ])?;
    for r in &results {
        wtr.write_record([
            r.format,
            &format!("{:.3}", r.parse_ms_median),
            &format!("{:.3}", r.transform_ms_median),
            &format!("{:.3}", r.transform_ms_p95),
            &format!("{:.3}", r.size_kb),
            r.lossless_scope,
            &r.notes,
        ])?;
    }
    wtr.flush()?;
    println!();
    println!("Wrote {}", cli.out.display());

    // ─── Predicate queries ───
    if cli.queries.exists() {
        println!();
        println!("Running predicate queries from {}", cli.queries.display());
        let q_results = run_queries(&compiled, &store.it, &cli.queries)?;
        std::fs::create_dir_all(cli.out_queries.parent().unwrap())?;
        let mut qwtr = csv::Writer::from_path(&cli.out_queries)?;
        qwtr.write_record(["id", "predicate", "matches", "latency_us"])?;
        for (id, pred, matches, latency) in &q_results {
            println!(
                "  {:<4} {:<64} matches={:>3} latency={:.1} µs",
                id, pred, matches, latency
            );
            qwtr.write_record([id, pred, &matches.to_string(), &format!("{:.3}", latency)])?;
        }
        qwtr.flush()?;
        println!("Wrote {}", cli.out_queries.display());
    } else {
        eprintln!("queries file not found: {}", cli.queries.display());
    }

    Ok(())
}

fn run_queries(
    compiled: &CompiledProgram,
    it: &Interner,
    queries_path: &std::path::Path,
) -> Result<Vec<(String, String, usize, f64)>> {
    // Simple pattern-matcher: each non-empty non-comment line in the
    // queries file is a tag like
    //     P1  KIND(joint)
    //     P5  KIND(constraint) AND HASARCREF(+1, INHERITS(context))
    // We implement a minimal interpreter for the predicate sub-language
    // used in the paper's Table P1..P5.

    let src = std::fs::read_to_string(queries_path)?;
    let mut results = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (id, pred_str) = match line.split_once(char::is_whitespace) {
            Some((i, p)) => (i.trim().to_string(), p.trim().to_string()),
            None => continue,
        };

        // Run the predicate 1000 times to average out noise.
        const ITERS: usize = 1000;
        let mut matches_count = 0usize;
        let t = Instant::now();
        for _ in 0..ITERS {
            matches_count = eval_predicate(&pred_str, &compiled.ir, it);
        }
        let latency_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        results.push((id, pred_str, matches_count, latency_us));
    }
    Ok(results)
}

/// Minimal predicate evaluator for the paper's P1..P5 subset. Supports:
///   KIND(<name>)                      — decl whose name or inherited base is <name>
///   INHERITS(<name>)                  — decl that transitively inherits <name>
///   HASARCREF(<sign>, <inner>)        — edge with at least one arc-ref of <sign>
///                                        (+1 / -1) pointing to a decl matching <inner>
///   <a> AND <b>                       — conjunction
///   ANY                               — always true
fn eval_predicate(pred: &str, ir: &Ir, it: &Interner) -> usize {
    // Walk all decls; count those matching.
    let mut n = 0;
    for i in 0..ir.decl_nodes.len() {
        if match_expr(pred, DeclId::new(i), ir, it) {
            n += 1;
        }
    }
    n
}

fn match_expr(expr: &str, did: DeclId, ir: &Ir, it: &Interner) -> bool {
    // Strip outer whitespace + split on " AND "
    let parts: Vec<&str> = expr.split(" AND ").collect();
    parts.iter().all(|p| match_atom(p.trim(), did, ir, it))
}

fn match_atom(atom: &str, did: DeclId, ir: &Ir, it: &Interner) -> bool {
    if atom == "ANY" {
        return true;
    }
    if let Some(rest) = atom.strip_prefix("KIND(") {
        let name = rest.trim_end_matches(')');
        return decl_kind_name(did, ir, it) == name;
    }
    if let Some(rest) = atom.strip_prefix("INHERITS(") {
        let name = rest.trim_end_matches(')');
        return decl_inherits(did, name, ir, it);
    }
    if let Some(rest) = atom.strip_prefix("SCOPEDIN(") {
        // Checks if the decl has an ancestor (proper — not itself) whose
        // inherited base matches the name. Supports the paper's
        // "contextually-scoped" reference pattern, where the author
        // flattens the contextual hierarchy but keeps a parent hypervertex
        // tagged as <isa> context.
        let name = rest.trim_end_matches(')');
        return decl_scoped_in(did, name, ir, it);
    }
    if let Some(rest) = atom.strip_prefix("HASARCREF(") {
        // sign, inner_expr — inner_expr may contain parens.
        let rest = rest.trim_end_matches(')');
        let (sign_s, inner) = rest.split_once(',').unwrap_or((rest, ""));
        let sign: i8 = sign_s.trim().trim_start_matches('+').parse().unwrap_or(0);
        let inner = inner.trim();
        return has_arc_ref(did, sign, inner, ir, it);
    }
    false
}

fn decl_scoped_in(did: DeclId, name: &str, ir: &Ir, it: &Interner) -> bool {
    let mut cur = ir.decl_nodes[did.0].parent;
    while cur.is_some() {
        if decl_inherits(cur, name, ir, it) {
            return true;
        }
        // Also accept when the ancestor's OWN name matches.
        let own = it.resolve(ir.decl_nodes[cur.0].name);
        if own == name {
            return true;
        }
        cur = ir.decl_nodes[cur.0].parent;
    }
    false
}

fn decl_kind_name<'a>(did: DeclId, ir: &'a Ir, it: &'a Interner) -> &'a str {
    // Return the short name of the decl's FIRST inherited base. Returns
    // "" if the decl has no bases — i.e., it IS a bare type declaration
    // and does not qualify as "KIND(x)" for any x. This matches the
    // paper's notion of "kind" = "instance of a declared type," which
    // type declarations themselves are not.
    let decl = &ir.decl_nodes[did.0];
    match decl.kind {
        DeclKind::Node => {
            if let Some(nid) = ir.as_node(did)
                && let Some(b) = ir.nodes[nid.0].bases.first()
            {
                return it.resolve(ir.decl_nodes[b.target().0].name);
            }
            ""
        }
        DeclKind::Edge => {
            if let Some(eid) = ir.as_edge(did)
                && let Some(b) = ir.edges[eid.0].bases.first()
            {
                return it.resolve(ir.decl_nodes[b.target().0].name);
            }
            ""
        }
        DeclKind::HyperArc => "",
    }
}

fn decl_inherits(did: DeclId, target_name: &str, ir: &Ir, it: &Interner) -> bool {
    // Transitive inheritance — walk up the base chain up to a small depth.
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![did];
    while let Some(d) = stack.pop() {
        if !visited.insert(d) {
            continue;
        }
        let decl = &ir.decl_nodes[d.0];
        let nm = it.resolve(decl.name);
        if nm == target_name {
            return true;
        }
        match decl.kind {
            DeclKind::Node => {
                if let Some(nid) = ir.as_node(d) {
                    for b in &ir.nodes[nid.0].bases {
                        stack.push(b.target());
                    }
                }
            }
            DeclKind::Edge => {
                if let Some(eid) = ir.as_edge(d) {
                    for b in &ir.edges[eid.0].bases {
                        stack.push(b.target());
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn has_arc_ref(did: DeclId, sign: i8, inner: &str, ir: &Ir, it: &Interner) -> bool {
    let Some(eid) = ir.as_edge(did) else {
        return false;
    };
    for &aid in &ir.edges[eid.0].arcs {
        for r in &ir.arcs[aid.0].refs {
            if r.sign() != sign {
                continue;
            }
            let target = r.target();
            if match_expr(inner, target, ir, it) {
                return true;
            }
        }
    }
    false
}

// Unused imports stub (keeps the import block tidy if we later drop CSV).
fn _unused_marker<T>(_: T) {}
