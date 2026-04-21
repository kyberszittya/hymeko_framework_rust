mod repl;

use std::path::PathBuf;
use std::sync::Arc;
use std::fs;

use clap::{Parser, Subcommand};

use hymeko::module_store::module_store::{CompiledProgram, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::resolution::interner::Interner;
use hymeko::util::pretty_print::pretty_print_compiled;
use hymeko::util::real_parser::RealParser;

use hymeko_formats::{generate_description, OutputFormat};
use hymeko_query::engine::QueryEngine;
use hymeko_query::entropy::{compute_entropy_hierarchical, StructuralEntropy};
use hymeko_query::interpret::interpret_transform_queries;
use hymeko_query::rewrite::{execute_transform, TransformSpec};

use parser::parse_description;

#[derive(Parser)]
#[command(name = "hymeko", version, about = "HyMeKo hypergraph description compiler")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a .hymeko file and generate robot description output
    Compile {
        /// Input .hymeko file
        input: PathBuf,

        /// Output format: urdf, sdf, mjcf, dot
        #[arg(short, long, default_value = "urdf")]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Robot/model name
        #[arg(short, long, default_value = "robot")]
        name: String,
    },

    /// Validate a .hymeko file (parse + resolve + check topology)
    Validate {
        /// Input .hymeko file
        input: PathBuf,
    },

    /// Pretty-print the compiled IR
    Inspect {
        /// Input .hymeko file
        input: PathBuf,
    },

    /// Start the interactive console (also the default with no args)
    Console,

    /// Run queries from a .hymeko query file against a compiled description
    Query {
        /// Input .hymeko description file
        input: PathBuf,

        /// Query file (.hymeko with pattern descriptions)
        #[arg(short = 'q', long)]
        query_file: PathBuf,
    },

    /// Run a template-driven transform (loads queries + template from transforms/ dir)
    Transform {
        /// Input .hymeko description file
        input: PathBuf,

        /// Transform name (e.g., urdf, sdf, dot — must match a directory under transforms/)
        #[arg(short = 't', long)]
        transform: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Robot/model name
        #[arg(short, long, default_value = "robot")]
        name: String,

        /// Directory containing transform definitions (default: transforms/)
        #[arg(long, default_value = "transforms")]
        transforms_dir: String,
    },

    /// Emit a format by **rendering a template** from the `transforms/`
    /// directory through the shared query + rendering pipeline. This is
    /// the canonical data-driven entry point — no hard-coded Rust string
    /// builders — and is the recommended alternative to `compile` for
    /// any format with a registered `transforms/<name>/` directory.
    ///
    /// Equivalent to the `transform` subcommand but uses
    /// `TransformRegistry::render_from_templates` directly, which picks
    /// up every format registered in the default registry (urdf, sdf,
    /// mjcf, dot, gazebo, mermaid).
    Emit {
        /// Input .hymeko description file
        input: PathBuf,

        /// Format name — one of the registered transforms (urdf, sdf,
        /// mjcf, dot, gazebo, mermaid).
        #[arg(short, long)]
        format: String,

        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Robot/model name
        #[arg(short, long, default_value = "robot")]
        name: String,

        /// Gazebo world name (used only by the `gazebo` format)
        #[arg(long, default_value = "empty")]
        world: String,

        /// Directory containing transform definitions (default: transforms/)
        #[arg(long, default_value = "transforms")]
        transforms_dir: String,
    },

    /// Compute per-scope structural entropy on the compiled IR.
    ///
    /// Walks every scope (module root + every hypervertex body) that
    /// contains at least one hyperedge, and emits the three-component
    /// Shannon metric `H_struct = (H_arity + H_sign + H_degree) / 3`
    /// (in nats) along with per-component values, vertex and edge
    /// counts. This is step 2 of the entropy hot-swap plan; the metric
    /// is defined in `docs/structural_entropy_ir.md`.
    Entropy {
        /// Input .hymeko description file
        input: PathBuf,

        /// Restrict output to scopes whose decl name matches exactly.
        /// Can be passed multiple times to keep a subset.
        #[arg(long = "scope")]
        scopes: Vec<String>,

        /// Emit machine-readable JSON instead of the human table.
        #[arg(long)]
        json: bool,
    },

    /// Propose a k=2 split of a hypervertex body using k-means on
    /// incidence-row signatures. Step 3 of the entropy hot-swap plan:
    /// the proposer reads the structural entropy (see `hymeko entropy`)
    /// and clusters vertices by which edges they participate in.
    ///
    /// Without `--scope`, auto-picks the scope with the highest
    /// `H_sign` (the three-role sign-mixing component). The
    /// `--weight-metric` flag controls how multi-component arc
    /// weights (axes, RPY, per-channel gains) collapse to a single
    /// participation strength in the feature space.
    Rewrite {
        /// Input .hymeko description file
        input: PathBuf,

        /// Restrict the proposal to the named scope instead of the
        /// auto-picked highest-`H_sign` one.
        #[arg(long = "scope")]
        scope: Option<String>,

        /// How to collapse arc weight vectors to a scalar. One of
        /// `l2` (default, Euclidean), `l1` (Manhattan), `linf`
        /// (max-norm), or `first` (first scalar only). All degenerate
        /// to `|scalar|` for single-scalar weights.
        #[arg(long = "weight-metric", default_value = "l2")]
        weight_metric: String,

        /// Emit machine-readable JSON instead of the human table.
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        None | Some(Commands::Console) => repl::interactive_console(),
        Some(cmd) => run_command(cmd),
    }
}

// ═══════════════════════════════════════════════════════════════
// One-shot command execution
// ═══════════════════════════════════════════════════════════════

fn run_command(cmd: Commands) {
    match cmd {
        Commands::Compile { input, format, output, name } => {
            let fmt = parse_format(&format);
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);

            let result = generate_description(&compiled.ir, &ms.it, &name, fmt)
                .unwrap_or_else(|e| {
                    eprintln!("Code generation failed: {e}");
                    std::process::exit(1);
                });

            match output {
                Some(path) => {
                    fs::write(&path, &result).unwrap_or_else(|e| {
                        eprintln!("Failed to write {}: {e}", path.display());
                        std::process::exit(1);
                    });
                    eprintln!("Wrote {} bytes to {}", result.len(), path.display());
                }
                None => print!("{result}"),
            }
        }

        Commands::Validate { input } => {
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            match ms.compile(&input) {
                Ok(compiled) => {
                    let warnings = hymeko_formats::urdf::validate_robot_schema(
                        &compiled.ir, &ms.it,
                    );
                    if warnings.is_empty() {
                        eprintln!("✅ {} is valid", input.display());
                    } else {
                        eprintln!("⚠️  {} compiled with {} warnings:", input.display(), warnings.len());
                        for w in &warnings {
                            eprintln!("  - {w}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ {} failed: {e:?}", input.display());
                    std::process::exit(1);
                }
            }
        }

        Commands::Inspect { input } => {
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);
            pretty_print_compiled(&ms.it, &compiled);
        }

        Commands::Console => unreachable!(),

        Commands::Query { input, query_file } => {
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);

            let query_src = fs::read_to_string(&query_file).unwrap_or_else(|e| {
                eprintln!("Failed to read {}: {e}", query_file.display());
                std::process::exit(1);
            });

            run_query_source(&compiled, &ms.it, &query_src, &query_file.to_string_lossy());
        }

        Commands::Transform { input, transform, output, name, transforms_dir } => {
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);

            let spec = load_transform_spec(&transforms_dir, &transform);

            let mut config = std::collections::HashMap::new();
            config.insert("robot_name".into(), name);

            let result = execute_transform(&compiled.ir, &ms.it, &spec, &config)
                .unwrap_or_else(|e| {
                    eprintln!("Transform failed: {e}");
                    std::process::exit(1);
                });

            match output {
                Some(path) => {
                    fs::write(&path, &result).unwrap_or_else(|e| {
                        eprintln!("Failed to write {}: {e}", path.display());
                        std::process::exit(1);
                    });
                    eprintln!("Wrote {} bytes to {}", result.len(), path.display());
                }
                None => print!("{result}"),
            }
        }

        Commands::Entropy { input, scopes, json } => {
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);

            let rows = resolve_entropy_rows(&compiled.ir, &ms.it, &scopes);
            if json {
                print!("{}", entropy_rows_to_json(&rows));
            } else {
                print_entropy_table(&rows);
            }
        }

        Commands::Rewrite { input, scope, weight_metric, json } => {
            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);

            let agg = parse_weight_metric(&weight_metric).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            let proposal = resolve_split_proposal(&compiled.ir, &ms.it, scope.as_deref(), agg)
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });

            if json {
                print!("{}", split_proposal_to_json(&compiled.ir, &ms.it, &proposal));
            } else {
                print_split_proposal(&compiled.ir, &ms.it, &proposal);
            }
        }

        Commands::Emit { input, format, output, name, world, transforms_dir } => {
            use hymeko_query::transforms::TransformConfig;

            let mut ms = ModuleStore::new(StdFsProvider::new(), RealParser);
            let compiled = compile_or_exit(&mut ms, &input);

            let reg = hymeko_formats::default_registry();
            let cfg = TransformConfig::default()
                .with_name(&name)
                .with_option("world_name", &world);

            let transforms_root = PathBuf::from(&transforms_dir);
            let result = reg
                .render_from_templates(&format, &compiled.ir, &ms.it, &cfg, &transforms_root)
                .unwrap_or_else(|| {
                    eprintln!(
                        "Unknown format: `{format}`. Registered template-driven formats: {:?}",
                        reg.available()
                    );
                    std::process::exit(1);
                })
                .unwrap_or_else(|e| {
                    eprintln!("Render failed: {e}");
                    std::process::exit(1);
                });

            match output {
                Some(path) => {
                    fs::write(&path, &result).unwrap_or_else(|e| {
                        eprintln!("Failed to write {}: {e}", path.display());
                        std::process::exit(1);
                    });
                    eprintln!("Wrote {} bytes to {}", result.len(), path.display());
                }
                None => print!("{result}"),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Entropy output helpers (shared by the subcommand and the REPL)
// ═══════════════════════════════════════════════════════════════

/// One row of the per-scope entropy report, with the scope resolved to
/// a human-readable name for display. `scope_name` is `"<root>"` for
/// the module root (DeclId::NONE) and `"<anonymous#<id>>"` for any
/// unnamed decl (rare but possible).
pub(crate) struct EntropyRow {
    scope_id: usize,
    scope_name: String,
    entropy: StructuralEntropy,
}

pub(crate) fn resolve_entropy_rows(
    ir: &hymeko::ir::ir::Ir,
    it: &Interner,
    keep: &[String],
) -> Vec<EntropyRow> {
    let keep_set: Option<std::collections::HashSet<&str>> = if keep.is_empty() {
        None
    } else {
        Some(keep.iter().map(String::as_str).collect())
    };

    compute_entropy_hierarchical(ir)
        .into_iter()
        .map(|(did, entropy)| {
            let (scope_id, scope_name) = if did.is_none() {
                (usize::MAX, "<root>".to_string())
            } else {
                let decl = &ir.decl_nodes[did.raw()];
                let name = it.resolve(decl.name).to_string();
                let name = if name.is_empty() {
                    format!("<anonymous#{}>", did.raw())
                } else {
                    name
                };
                (did.raw(), name)
            };
            EntropyRow { scope_id, scope_name, entropy }
        })
        .filter(|row| match &keep_set {
            Some(set) => set.contains(row.scope_name.as_str()),
            None => true,
        })
        .collect()
}

pub(crate) fn print_entropy_table(rows: &[EntropyRow]) {
    if rows.is_empty() {
        eprintln!("No scopes with hyperedges in this IR.");
        return;
    }
    // Pad scope name to the longest present, min 12.
    let name_w = rows.iter().map(|r| r.scope_name.len()).max().unwrap_or(12).max(12);
    println!(
        "{:<nw$}  {:>4} {:>4}  {:>9} {:>9} {:>9}  {:>9}",
        "scope", "V", "E", "H_arity", "H_sign", "H_degree", "H_total",
        nw = name_w,
    );
    println!("{}", "─".repeat(name_w + 2 + 4 + 1 + 4 + 2 + 9 + 1 + 9 + 1 + 9 + 2 + 9));
    for row in rows {
        let e = &row.entropy;
        println!(
            "{:<nw$}  {:>4} {:>4}  {:>9.4} {:>9.4} {:>9.4}  {:>9.4}",
            row.scope_name,
            e.n_vertices,
            e.n_edges,
            e.h_arity,
            e.h_sign,
            e.h_degree,
            e.h_total,
            nw = name_w,
        );
    }
}

/// Minimal hand-rolled JSON emitter — avoids pulling in `serde_json`
/// just for this one surface. f64 values use Rust's default Display
/// which round-trips through f64 parsers.
pub(crate) fn entropy_rows_to_json(rows: &[EntropyRow]) -> String {
    let mut out = String::from("[\n");
    for (i, row) in rows.iter().enumerate() {
        let e = &row.entropy;
        let scope_id_json = if row.scope_id == usize::MAX {
            "null".to_string()
        } else {
            row.scope_id.to_string()
        };
        out.push_str("  {");
        out.push_str(&format!("\"scope_id\": {scope_id_json}, "));
        out.push_str(&format!(
            "\"scope_name\": \"{}\", ",
            json_escape(&row.scope_name)
        ));
        out.push_str(&format!("\"n_vertices\": {}, ", e.n_vertices));
        out.push_str(&format!("\"n_edges\": {}, ", e.n_edges));
        out.push_str(&format!("\"h_arity\": {}, ", e.h_arity));
        out.push_str(&format!("\"h_sign\": {}, ", e.h_sign));
        out.push_str(&format!("\"h_degree\": {}, ", e.h_degree));
        out.push_str(&format!("\"h_total\": {}", e.h_total));
        out.push('}');
        if i + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("]\n");
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════
// Split-proposal output helpers
// ═══════════════════════════════════════════════════════════════

use hymeko_query::rewrite::{
    propose_split_for_highest_h_sign_with, propose_split_with,
    Cluster, SplitProposal, WeightAggregation,
};
use hymeko::common::ids::DeclId;

fn parse_weight_metric(s: &str) -> Result<WeightAggregation, String> {
    match s.to_lowercase().as_str() {
        "l1"    => Ok(WeightAggregation::L1Norm),
        "l2"    => Ok(WeightAggregation::L2Norm),
        "linf"  => Ok(WeightAggregation::LinfNorm),
        "first" => Ok(WeightAggregation::FirstScalar),
        other => Err(format!(
            "Unknown weight metric: `{other}`. Use one of: l1, l2, linf, first."
        )),
    }
}

fn resolve_split_proposal(
    ir: &hymeko::ir::ir::Ir,
    it: &Interner,
    scope_name: Option<&str>,
    weight_agg: WeightAggregation,
) -> Result<SplitProposal, String> {
    match scope_name {
        Some(name) => {
            let did = find_scope_by_name(ir, it, name)
                .ok_or_else(|| format!("No hypervertex scope named `{name}` found."))?;
            propose_split_with(ir, did, weight_agg).ok_or_else(|| format!(
                "Scope `{name}` is not splittable (needs ≥2 vertices and ≥1 edge)."
            ))
        }
        None => propose_split_for_highest_h_sign_with(ir, weight_agg).ok_or_else(|| {
            "No scope in this IR has enough structure to split (need ≥2 vertices + ≥1 edge).".to_string()
        }),
    }
}

fn find_scope_by_name(ir: &hymeko::ir::ir::Ir, it: &Interner, name: &str) -> Option<DeclId> {
    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind == hymeko::ir::ir::DeclKind::Node && it.resolve(decl.name) == name {
            return Some(DeclId::new(i));
        }
    }
    None
}

fn decl_name<'a>(ir: &'a hymeko::ir::ir::Ir, it: &'a Interner, did: DeclId) -> &'a str {
    if did.is_none() {
        return "<root>";
    }
    it.resolve(ir.decl_nodes[did.raw()].name)
}

fn print_split_proposal(
    ir: &hymeko::ir::ir::Ir,
    it: &Interner,
    p: &SplitProposal,
) {
    println!("Proposed split of `{}`:", decl_name(ir, it, p.target_scope));
    println!("  Cluster A ({} vertices): {}",
             p.cluster_a.len(),
             join_names(ir, it, &p.cluster_a));
    println!("  Cluster B ({} vertices): {}",
             p.cluster_b.len(),
             join_names(ir, it, &p.cluster_b));
    println!("  Edge assignments ({} total, {} cross):",
             p.edge_assignments.len(), p.n_cross_edges);
    for (e_did, cluster) in &p.edge_assignments {
        let tag = match cluster {
            Cluster::A     => "A",
            Cluster::B     => "B",
            Cluster::Cross => "cross",
        };
        println!("    {:<28} -> {}", decl_name(ir, it, *e_did), tag);
    }
    println!("  Inertia: {:.4}", p.inertia);
}

fn join_names(ir: &hymeko::ir::ir::Ir, it: &Interner, dids: &[DeclId]) -> String {
    let mut names: Vec<&str> = dids.iter().map(|d| decl_name(ir, it, *d)).collect();
    names.sort();
    names.join(", ")
}

fn split_proposal_to_json(
    ir: &hymeko::ir::ir::Ir,
    it: &Interner,
    p: &SplitProposal,
) -> String {
    let mut out = String::from("{\n");
    out.push_str(&format!(
        "  \"target_scope\": \"{}\",\n",
        json_escape(decl_name(ir, it, p.target_scope)),
    ));
    out.push_str("  \"cluster_a\": [");
    out.push_str(&p.cluster_a.iter()
        .map(|d| format!("\"{}\"", json_escape(decl_name(ir, it, *d))))
        .collect::<Vec<_>>()
        .join(", "));
    out.push_str("],\n");
    out.push_str("  \"cluster_b\": [");
    out.push_str(&p.cluster_b.iter()
        .map(|d| format!("\"{}\"", json_escape(decl_name(ir, it, *d))))
        .collect::<Vec<_>>()
        .join(", "));
    out.push_str("],\n");
    out.push_str("  \"edge_assignments\": [\n");
    for (i, (e_did, cluster)) in p.edge_assignments.iter().enumerate() {
        let tag = match cluster {
            Cluster::A     => "A",
            Cluster::B     => "B",
            Cluster::Cross => "cross",
        };
        out.push_str(&format!(
            "    {{\"edge\": \"{}\", \"cluster\": \"{}\"}}{}\n",
            json_escape(decl_name(ir, it, *e_did)),
            tag,
            if i + 1 < p.edge_assignments.len() { "," } else { "" },
        ));
    }
    out.push_str("  ],\n");
    out.push_str(&format!("  \"n_cross_edges\": {},\n", p.n_cross_edges));
    out.push_str(&format!("  \"inertia\": {}\n", p.inertia));
    out.push_str("}\n");
    out
}

fn parse_format(s: &str) -> OutputFormat {
    match s.to_lowercase().as_str() {
        "urdf" => OutputFormat::Urdf,
        "sdf"  => OutputFormat::Sdf17,
        "mjcf" => OutputFormat::Mjcf,
        "dot"  => OutputFormat::DotGraph,
        "torch" | "torch_dataflow" => OutputFormat::TorchDataflow,
        other => {
            eprintln!("Unknown format: {other}. Use: urdf, sdf, mjcf, dot, torch_dataflow");
            std::process::exit(1);
        }
    }
}

fn compile_or_exit(ms: &mut ModuleStore<StdFsProvider, RealParser>, path: &PathBuf) -> Arc<CompiledProgram> {
    ms.compile(path).unwrap_or_else(|e| {
        eprintln!("Compilation failed: {e:?}");
        std::process::exit(1);
    })
}


// ═══════════════════════════════════════════════════════════════
// Transform loading helpers
// ═══════════════════════════════════════════════════════════════

/// Load a transform spec from a directory: <dir>/<name>/queries.hymeko + template.*
pub(crate) fn load_transform_spec(transforms_dir: &str, name: &str) -> TransformSpec {
    let dir = PathBuf::from(transforms_dir).join(name);

    let query_path = dir.join("queries.hymeko");
    let query_source = fs::read_to_string(&query_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", query_path.display());
        std::process::exit(1);
    });

    // Find the template file (any file starting with "template.")
    let template_source = find_template_file(&dir).unwrap_or_else(|| {
        eprintln!("No template.* file found in {}", dir.display());
        std::process::exit(1);
    });

    TransformSpec {
        name: name.to_string(),
        query_source,
        template_source,
    }
}

/// Find and read the template file in a transform directory.
fn find_template_file(dir: &PathBuf) -> Option<String> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("template.") {
            return fs::read_to_string(entry.path()).ok();
        }
    }
    None
}

/// List available transforms in a directory.
pub(crate) fn list_available_transforms(transforms_dir: &str) {
    let dir = PathBuf::from(transforms_dir);
    if !dir.is_dir() {
        println!("  (directory '{}' not found)", transforms_dir);
        return;
    }
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                let has_queries = entry.path().join("queries.hymeko").exists();
                let has_template = find_template_file(&entry.path()).is_some();
                let status = match (has_queries, has_template) {
                    (true, true)   => "✅",
                    (true, false)  => "⚠️  (missing template)",
                    (false, true)  => "⚠️  (missing queries.hymeko)",
                    (false, false) => "❌ (empty)",
                };
                println!("    {status} {name}");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Query execution helpers
// ═══════════════════════════════════════════════════════════════

/// Run an inline query pattern against the loaded IR.
/// Wraps the pattern in a minimal description block for the parser.
pub(crate) fn run_inline_query(compiled: &CompiledProgram, interner: &Interner, pattern: &str) {
    // Wrap the pattern in a description block so the parser accepts it.
    // "q" is the query-set name.
    let src = format!("q {{\n  {pattern}\n}}");
    run_query_source(compiled, interner, &src, "<inline>");
}

/// Parse a source string as queries and run them against the compiled IR.
pub(crate) fn run_query_source(compiled: &CompiledProgram, interner: &Interner, src: &str, label: &str) {
    let ast = match parse_description(src) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("  Parse error in {label}: {e:?}");
            return;
        }
    };

    // Transform-aware: unwraps `context { … }` if present, otherwise
    // falls back to plain top-level interpretation.
    let queries = interpret_transform_queries(&ast);
    if queries.is_empty() {
        println!("  No query patterns found in {label}.");
        return;
    }

    let engine = QueryEngine::new(&compiled.ir, interner);
    let batch = engine.query_batch(&queries);

    let total_matches: usize = batch.iter().map(|(_, v)| v.len()).sum();
    println!("  Ran {} queries, {} total matches", batch.len(), total_matches);
    println!();

    for (label, matches) in &batch {
        println!("  ── {label} ({} matches) ──", matches.len());
        if matches.is_empty() {
            println!("    (no matches)");
        }
        for m in matches {
            let kind_str = match m.kind {
                hymeko::ir::ir::DeclKind::Node => "node",
                hymeko::ir::ir::DeclKind::Edge => "edge",
                hymeko::ir::ir::DeclKind::HyperArc => "arc ",
            };
            let depth_pad = "  ".repeat(m.depth.min(6));
            print!("    [{kind_str}] {depth_pad}{}", m.name);

            if !m.arc_bindings.is_empty() {
                print!("  {{");
                for (i, ab) in m.arc_bindings.iter().enumerate() {
                    let sign_ch = match ab.sign {
                        1 => "+",
                        -1 => "-",
                        _ => "~",
                    };
                    if i > 0 { print!(", "); }
                    print!("{sign_ch}{}", ab.target_name);
                }
                print!("}}");
            }
            println!();
        }
        println!();
    }
}
