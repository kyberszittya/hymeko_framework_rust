//! Scaling benchmark harness for the HyMeKo pipeline.
//!
//! Adapted from the draft artefact in `input/benchmarks/files (5).zip` to
//! the real crate layout of this workspace:
//!
//!   compile: `ModuleStore::compile(path)` — full
//!            parse + intern + resolve + lower pipeline
//!   urdf  : `hymeko_formats::urdf::generate_urdf`
//!   sdf   : `hymeko_formats::sdf::generate_sdf`
//!   gazebo: `hymeko_formats::gazebo::generate_gazebo_world`
//!   mjcf  : `TransformRegistry::get("mjcf")!.emit(&ModelView, &cfg)`
//!   dot   : `TransformRegistry::get("dot")!.emit(&ModelView, &cfg)`
//!   mermaid: `TransformRegistry::get("mermaid")!.emit(&ModelView, &cfg)`
//!
//! Output: one CSV row per (fixture × rep × stage), consumed by
//! `scripts/scaling/analyze_scaling.py`.
//!
//! Usage:
//!     cargo run --release --bin bench_scaling -- \
//!         --fixtures ./fixtures --out scaling_results.csv --reps 30
//!
//! `highArity` fixtures run compile only — their output shape isn't
//! well-typed for the robotics emitters (Prop. 4 is an IR-storage
//! claim, not an emitter-correctness claim).

use std::{
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use hymeko::module_store::module_store::{HymekoParser, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use parser::ast::AstStr;

use hymeko_query::engine::QueryEngine;
use hymeko_query::kinematics::kinematic::extract_kinematic_model;
use hymeko_query::transforms::{ModelView, TransformConfig};

use hymeko_formats::gazebo::generate_gazebo_world;
use hymeko_formats::sdf::generate_sdf;
use hymeko_formats::urdf::generate_urdf;

// ───────────────────────────────────────────────────────────────────────────
// Parser glue (mirror of test_helpers::LalrpopParser so the bench stands
// alone without a dev-dependency on the hymeko_query integration tests)
// ───────────────────────────────────────────────────────────────────────────

struct LalrpopParser;

impl HymekoParser for LalrpopParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        parser::parse_description(src).map_err(|e| format!("{e:?}"))
    }
}

// ───────────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
struct FixtureEntry {
    family: String,
    name: String,
    n_vertices: usize,
    n_hyperedges: usize,
    mean_arity: f64,
    source_bytes: usize,
    path: String,
    /// Top-level decl name the emitters are asked to render. Mirrors the
    /// `robot_name` argument already threaded through
    /// `bench_workflow.rs`.
    robot_name: String,
}

#[derive(Serialize)]
struct Row<'a> {
    family: &'a str,
    name: &'a str,
    n_vertices: usize,
    n_hyperedges: usize,
    mean_arity: f64,
    source_bytes: usize,
    rep: usize,
    stage: &'a str,
    wall_ns: u128,
    output_bytes: usize,
}

fn time_once<F: FnOnce() -> R, R>(f: F) -> (R, u128) {
    let t0 = Instant::now();
    let r = f();
    let dt = t0.elapsed().as_nanos();
    (r, dt)
}

/// Fresh `ModuleStore` per call — the production compile path caches
/// parsed modules, so reusing a store would only measure cold-compile
/// once. This matches the `bench_workflow.rs` protocol.
fn compile_fresh(path: &Path) -> anyhow::Result<
    std::sync::Arc<hymeko::module_store::module_store::CompiledProgram>,
> {
    let mut store = ModuleStore::new(StdFsProvider::new(), LalrpopParser);
    let compiled = store
        .compile(path)
        .map_err(|e| anyhow::anyhow!("compile {}: {:?}", path.display(), e))?;
    // We need the store alive to keep the interner backing the IR
    // references, but the caller only wants the CompiledProgram. Wrap
    // both in a tuple is cleaner — but `ModuleStore` is not Clone. For
    // the bench we accept that each timing run also reconstructs the
    // store (that IS the point of measuring cold compile).
    //
    // The interner lives behind a `Rc<RefCell<…>>` reachable from the
    // CompiledProgram's ArcSwap-like graph; the `store` we drop here
    // holds a separate `StdFsProvider`, which is fine. The IR itself
    // references `SymId`s, resolution happens via the per-run
    // interner that CompiledProgram keeps.
    let _ = store;
    Ok(compiled)
}

fn bench_fixture<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    fixtures_root: &Path,
    entry: &FixtureEntry,
    reps: usize,
    warmup: usize,
) -> anyhow::Result<()> {
    let src_path = fixtures_root.join(&entry.path);
    let run_emitters = entry.family != "highArity";

    // ── warm-up ───────────────────────────────────────────────────────
    for _ in 0..warmup {
        let compiled = compile_fresh(&src_path)?;
        if run_emitters {
            // exercise the interner + rendering paths once
            let mut store = ModuleStore::new(StdFsProvider::new(), LalrpopParser);
            let _ = store.compile(&src_path);
            black_box(generate_urdf(&compiled.ir, &store.it, &entry.robot_name));
        }
    }

    for rep in 0..reps {
        // ── compile ───────────────────────────────────────────────────
        // Measure the full ModuleStore path: fresh store, compile, done.
        // We keep the store alive across the emit stages below to reuse
        // its interner (required for NameResolver).
        let t0 = Instant::now();
        let mut store = ModuleStore::new(StdFsProvider::new(), LalrpopParser);
        let compiled = store
            .compile(&src_path)
            .map_err(|e| anyhow::anyhow!("compile: {:?}", e))?;
        let dt_compile = t0.elapsed().as_nanos();

        writer.serialize(Row {
            family: &entry.family,
            name: &entry.name,
            n_vertices: entry.n_vertices,
            n_hyperedges: entry.n_hyperedges,
            mean_arity: entry.mean_arity,
            source_bytes: entry.source_bytes,
            rep,
            stage: "compile",
            wall_ns: dt_compile,
            output_bytes: 0,
        })?;

        if !run_emitters {
            continue;
        }

        // ── URDF / SDF / Gazebo via free functions ────────────────────
        let (urdf, dt_urdf) =
            time_once(|| generate_urdf(&compiled.ir, &store.it, &entry.robot_name));
        writer.serialize(Row {
            family: &entry.family, name: &entry.name,
            n_vertices: entry.n_vertices, n_hyperedges: entry.n_hyperedges,
            mean_arity: entry.mean_arity, source_bytes: entry.source_bytes,
            rep, stage: "urdf", wall_ns: dt_urdf, output_bytes: urdf.len(),
        })?;
        black_box(urdf);

        let (sdf, dt_sdf) =
            time_once(|| generate_sdf(&compiled.ir, &store.it, &entry.robot_name));
        writer.serialize(Row {
            family: &entry.family, name: &entry.name,
            n_vertices: entry.n_vertices, n_hyperedges: entry.n_hyperedges,
            mean_arity: entry.mean_arity, source_bytes: entry.source_bytes,
            rep, stage: "sdf", wall_ns: dt_sdf, output_bytes: sdf.len(),
        })?;
        black_box(sdf);

        let (gazebo, dt_gz) = time_once(|| {
            generate_gazebo_world(&compiled.ir, &store.it, &entry.robot_name, "empty")
        });
        writer.serialize(Row {
            family: &entry.family, name: &entry.name,
            n_vertices: entry.n_vertices, n_hyperedges: entry.n_hyperedges,
            mean_arity: entry.mean_arity, source_bytes: entry.source_bytes,
            rep, stage: "gazebo", wall_ns: dt_gz, output_bytes: gazebo.len(),
        })?;
        black_box(gazebo);

        // ── MJCF / DOT / Mermaid via TransformRegistry ────────────────
        // These formats don't expose a free function; the canonical
        // path is registry lookup + `emit(&ModelView, &cfg)`. We
        // extract the kinematic model once and reuse it across the
        // three formats to match bench_workflow.rs.
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let km = extract_kinematic_model(&engine, &entry.robot_name);
        let model_view = ModelView::Kinematic(km);
        let cfg = TransformConfig::default().with_name(&entry.robot_name);
        let reg = hymeko_formats::default_registry();

        for stage_name in &["mjcf", "dot", "mermaid"] {
            let t = reg.get(stage_name).expect("registry should contain stage");
            let (out, dt) = time_once(|| {
                t.emit(&model_view, &cfg).unwrap_or_default()
            });
            writer.serialize(Row {
                family: &entry.family, name: &entry.name,
                n_vertices: entry.n_vertices, n_hyperedges: entry.n_hyperedges,
                mean_arity: entry.mean_arity, source_bytes: entry.source_bytes,
                rep, stage: stage_name, wall_ns: dt, output_bytes: out.len(),
            })?;
            black_box(out);
        }
    }
    Ok(())
}

#[derive(clap::Parser)]
struct Args {
    /// Fixtures directory (containing `index.json` + family subdirs).
    #[arg(long)]
    fixtures: PathBuf,
    /// CSV output path.
    #[arg(long, default_value = "scaling_results.csv")]
    out: PathBuf,
    /// Release-profile repetitions per fixture.
    #[arg(long, default_value_t = 30)]
    reps: usize,
    /// Warm-up iterations (not recorded).
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Optional family filter ("chain" / "tree" / "highArity").
    #[arg(long)]
    family: Option<String>,
    /// Skip fixtures with size > this threshold (|V|+|E|).
    #[arg(long)]
    max_size: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;
    let args = Args::parse();

    let manifest_path = args.fixtures.join("index.json");
    let manifest: Vec<FixtureEntry> = serde_json::from_str(
        &fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?,
    )?;

    let mut writer = csv::Writer::from_path(&args.out)
        .with_context(|| format!("opening {}", args.out.display()))?;

    let filtered: Vec<_> = manifest
        .into_iter()
        .filter(|e| args.family.as_deref().map_or(true, |f| f == e.family))
        .filter(|e| {
            args.max_size
                .map_or(true, |m| e.n_vertices + e.n_hyperedges <= m)
        })
        .collect();

    let total = filtered.len();
    for (i, entry) in filtered.iter().enumerate() {
        eprintln!(
            "[{}/{}] bench: {} ({} V, {} E, d̄={:.2})",
            i + 1, total, entry.name,
            entry.n_vertices, entry.n_hyperedges, entry.mean_arity,
        );
        bench_fixture(&mut writer, &args.fixtures, entry, args.reps, args.warmup)?;
        writer.flush()?;
    }
    eprintln!("done → {}", args.out.display());
    Ok(())
}
