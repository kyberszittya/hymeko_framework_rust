use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::fs;

use clap::{Parser, Subcommand};

use hymeko::module_store::module_store::{CompiledProgram, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::resolution::interner::Interner;
use hymeko::util::real_parser::RealParser;
use hymeko::util::pretty_print::pretty_print_compiled;

use hymeko_query::codegen::{generate_description, OutputFormat};
use hymeko_query::engine::QueryEngine;
use hymeko_query::interpret::interpret_as_queries;

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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        None | Some(Commands::Console) => interactive_console(),
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
                    let warnings = hymeko_query::formats::urdf::validate_robot_schema(
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
    }
}

fn parse_format(s: &str) -> OutputFormat {
    match s.to_lowercase().as_str() {
        "urdf" => OutputFormat::Urdf,
        "sdf"  => OutputFormat::Sdf17,
        "mjcf" => OutputFormat::Mjcf,
        "dot"  => OutputFormat::DotGraph,
        other => {
            eprintln!("Unknown format: {other}. Use: urdf, sdf, mjcf, dot");
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
// Interactive REPL
// ═══════════════════════════════════════════════════════════════

struct Session {
    ms: ModuleStore<StdFsProvider, RealParser>,
    compiled: Option<Arc<CompiledProgram>>,
    loaded_path: Option<PathBuf>,
    robot_name: String,
}

impl Session {
    fn new() -> Self {
        Self {
            ms: ModuleStore::new(StdFsProvider::new(), RealParser),
            compiled: None,
            loaded_path: None,
            robot_name: "robot".into(),
        }
    }

    fn interner(&self) -> &Interner {
        &self.ms.it
    }

    fn ensure_loaded(&self) -> Option<&Arc<CompiledProgram>> {
        if self.compiled.is_none() {
            eprintln!("  No file loaded. Use: load <path.hymeko>");
            return None;
        }
        self.compiled.as_ref()
    }
}

fn interactive_console() {
    let mut session = Session::new();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  HyMeKo Interactive Console                 ║");
    println!("║  Type 'help' for available commands          ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        let label = session.loaded_path.as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "no file".into());

        print!("hymeko [{label}]> ");
        stdout.flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();
        let cmd = parts[0].to_lowercase();

        match cmd.as_str() {
            "help" | "h" | "?" => print_help(),

            "load" | "open" => {
                if parts.len() < 2 {
                    eprintln!("  Usage: load <path.hymeko>");
                    continue;
                }
                let path = PathBuf::from(parts[1]);
                if !path.exists() {
                    eprintln!("  File not found: {}", path.display());
                    continue;
                }
                match session.ms.compile(&path) {
                    Ok(compiled) => {
                        let n_decls = compiled.ir.decl_nodes.len();
                        println!("  ✅ Loaded {} ({} declarations)", path.display(), n_decls);
                        session.compiled = Some(compiled);
                        session.loaded_path = Some(path);
                    }
                    Err(e) => eprintln!("  ❌ Compilation failed: {e:?}"),
                }
            }

            "reload" | "r" => {
                if let Some(path) = session.loaded_path.clone() {
                    match session.ms.compile(&path) {
                        Ok(compiled) => {
                            println!("  ✅ Reloaded {}", path.display());
                            session.compiled = Some(compiled);
                        }
                        Err(e) => eprintln!("  ❌ Reload failed: {e:?}"),
                    }
                } else {
                    eprintln!("  No file loaded. Use: load <path.hymeko>");
                }
            }

            "name" => {
                if parts.len() >= 2 {
                    session.robot_name = parts[1].to_string();
                    println!("  Robot name set to: {}", session.robot_name);
                } else {
                    println!("  Current robot name: {}", session.robot_name);
                }
            }

            "compile" | "gen" | "generate" => {
                let Some(compiled) = session.ensure_loaded() else { continue };
                let format_str = if parts.len() >= 2 { parts[1] } else { "urdf" };
                let fmt = match format_str.to_lowercase().as_str() {
                    "urdf" => OutputFormat::Urdf,
                    "sdf"  => OutputFormat::Sdf17,
                    "mjcf" => OutputFormat::Mjcf,
                    "dot"  => OutputFormat::DotGraph,
                    other => {
                        eprintln!("  Unknown format: {other}. Use: urdf, sdf, mjcf, dot");
                        continue;
                    }
                };

                match generate_description(&compiled.ir, session.interner(), &session.robot_name, fmt) {
                    Ok(result) => {
                        if parts.len() >= 3 {
                            let out_path = PathBuf::from(parts[2]);
                            match fs::write(&out_path, &result) {
                                Ok(_) => println!("  Wrote {} bytes to {}", result.len(), out_path.display()),
                                Err(e) => eprintln!("  Write failed: {e}"),
                            }
                        } else {
                            println!("{result}");
                        }
                    }
                    Err(e) => eprintln!("  Code generation failed: {e}"),
                }
            }

            "validate" | "check" => {
                let Some(compiled) = session.ensure_loaded() else { continue };
                let warnings = hymeko_query::formats::urdf::validate_robot_schema(
                    &compiled.ir, session.interner(),
                );
                if warnings.is_empty() {
                    println!("  ✅ Valid (no warnings)");
                } else {
                    println!("  ⚠️  {} warnings:", warnings.len());
                    for w in &warnings {
                        println!("    - {w}");
                    }
                }
            }

            "inspect" | "ir" | "dump" => {
                let Some(compiled) = session.ensure_loaded() else { continue };
                pretty_print_compiled(session.interner(), compiled);
            }

            "info" | "status" => {
                match &session.loaded_path {
                    Some(p) => {
                        let compiled = session.compiled.as_ref().unwrap();
                        println!("  File:          {}", p.display());
                        println!("  Robot name:    {}", session.robot_name);
                        println!("  Declarations:  {}", compiled.ir.decl_nodes.len());
                        println!("  Nodes:         {}", compiled.ir.nodes.len());
                        println!("  Edges:         {}", compiled.ir.edges.len());
                        println!("  Arcs:          {}", compiled.ir.arcs.len());
                    }
                    None => println!("  No file loaded."),
                }
            }

            "formats" => {
                println!("  Available output formats:");
                println!("    urdf  — URDF (ROS/Gazebo robot description)");
                println!("    sdf   — SDFormat 1.7 (Gazebo world description)");
                println!("    mjcf  — MuJoCo MJCF (MuJoCo simulator)");
                println!("    dot   — Graphviz DOT (visualization)");
            }

            "query" | "q!" => {
                let Some(compiled) = session.ensure_loaded() else { continue };
                if parts.len() < 2 {
                    eprintln!("  Usage: query <pattern>");
                    eprintln!("  Examples:");
                    eprintln!("    query _ : link           — find all nodes inheriting from 'link'");
                    eprintln!("    query _ : joint          — find all joints");
                    eprintln!("    query wheel_*            — find nodes with name prefix 'wheel_'");
                    continue;
                }
                // Rejoin everything after "query" as the pattern text
                let pattern_text = line[parts[0].len()..].trim();
                run_inline_query(&compiled, session.interner(), pattern_text);
            }

            "qfile" | "query-file" => {
                let Some(compiled) = session.ensure_loaded() else { continue };
                if parts.len() < 2 {
                    eprintln!("  Usage: qfile <query_file.hymeko>");
                    continue;
                }
                let qpath = PathBuf::from(parts[1]);
                if !qpath.exists() {
                    eprintln!("  File not found: {}", qpath.display());
                    continue;
                }
                match fs::read_to_string(&qpath) {
                    Ok(src) => run_query_source(&compiled, session.interner(), &src, &qpath.to_string_lossy()),
                    Err(e) => eprintln!("  Read failed: {e}"),
                }
            }

            "daemon" => {
                eprintln!("  Daemon integration not yet connected.");
                eprintln!("  Use hymeko_client for direct iceoryx2 IPC.");
                eprintln!("  (Planned: daemon connect/push/status/disconnect)");
            }

            "cd" => {
                if parts.len() < 2 {
                    // cd with no args → home directory
                    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
                        if let Err(e) = std::env::set_current_dir(&home) {
                            eprintln!("  cd failed: {e}");
                        }
                    } else {
                        eprintln!("  Usage: cd <directory>");
                    }
                } else {
                    let target = parts[1];
                    if let Err(e) = std::env::set_current_dir(target) {
                        eprintln!("  cd: {target}: {e}");
                    }
                }
            }

            "pwd" => {
                match std::env::current_dir() {
                    Ok(p) => println!("  {}", p.display()),
                    Err(e) => eprintln!("  pwd failed: {e}"),
                }
            }

            "ls" => {
                let dir = if parts.len() >= 2 { parts[1] } else { "." };
                match std::fs::read_dir(dir) {
                    Ok(entries) => {
                        let mut names: Vec<String> = Vec::new();
                        for entry in entries.flatten() {
                            let name = entry.file_name().to_string_lossy().to_string();
                            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                                names.push(format!("{name}/"));
                            } else {
                                names.push(name);
                            }
                        }
                        names.sort();
                        for n in &names {
                            if n.ends_with(".hymeko") {
                                println!("  \x1b[1;32m{n}\x1b[0m"); // green+bold for .hymeko
                            } else if n.ends_with('/') {
                                println!("  \x1b[1;34m{n}\x1b[0m"); // blue for dirs
                            } else {
                                println!("  {n}");
                            }
                        }
                    }
                    Err(e) => eprintln!("  ls: {dir}: {e}"),
                }
            }

            "exit" | "quit" | "q" => {
                println!("  Bye.");
                break;
            }

            other => {
                if other.ends_with(".hymeko") {
                    let path = PathBuf::from(other);
                    if path.exists() {
                        match session.ms.compile(&path) {
                            Ok(compiled) => {
                                println!("  ✅ Loaded {}", path.display());
                                session.compiled = Some(compiled);
                                session.loaded_path = Some(path);
                            }
                            Err(e) => eprintln!("  ❌ Compilation failed: {e:?}"),
                        }
                    } else {
                        eprintln!("  File not found: {other}");
                    }
                } else {
                    eprintln!("  Unknown command: '{other}'. Type 'help' for available commands.");
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Query execution helpers
// ═══════════════════════════════════════════════════════════════

/// Run an inline query pattern against the loaded IR.
/// Wraps the pattern in a minimal description block for the parser.
fn run_inline_query(compiled: &CompiledProgram, interner: &Interner, pattern: &str) {
    // Wrap the pattern in a description block so the parser accepts it.
    // "q" is the query-set name.
    let src = format!("q {{\n  {pattern}\n}}");
    run_query_source(compiled, interner, &src, "<inline>");
}

/// Parse a source string as queries and run them against the compiled IR.
fn run_query_source(compiled: &CompiledProgram, interner: &Interner, src: &str, label: &str) {
    let ast = match parse_description(src) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("  Parse error in {label}: {e:?}");
            return;
        }
    };

    let queries = interpret_as_queries(&ast);
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

fn print_help() {
    println!(r#"
  ┌─────────────────────────────────────────────────────────┐
  │ HyMeKo Interactive Console — Commands                  │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  load <file.hymeko>       Load and compile a file       │
  │  reload / r               Recompile the current file    │
  │  <file.hymeko>            Shortcut: load the file       │
  │                                                         │
  │  compile [fmt] [outfile]  Generate output               │
  │    gen urdf               Print URDF to stdout          │
  │    gen sdf robot.sdf      Write SDF to file             │
  │    gen mjcf               Print MJCF to stdout          │
  │    gen dot graph.dot      Write DOT to file             │
  │                                                         │
  │  validate / check         Run schema validation         │
  │  inspect / ir             Pretty-print compiled IR      │
  │  info                     Show current session info     │
  │  name [new_name]          Get/set robot model name      │
  │  formats                  List available output formats  │
  │                                                         │
  │  query <pattern>          Run inline query               │
  │    query _ : link         Find all link nodes            │
  │    query _ : joint        Find all joint edges           │
  │  qfile <file.hymeko>      Run queries from a file        │
  │                                                         │
  │  cd [dir]                 Change directory               │
  │  pwd                      Print working directory        │
  │  ls [dir]                 List files (.hymeko in green)  │
  │                                                         │
  │  help / h / ?             This help                     │
  │  exit / quit / q          Exit the console              │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
"#);
}