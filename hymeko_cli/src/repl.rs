//! Interactive REPL. Dispatches a whitespace-separated command line to one
//! handler per command; handlers are free `fn cmd_*`s living in this module,
//! each returning `CommandOutcome::{Continue, Break}`. Extracted from the
//! original monolithic `interactive_console` per the 2026-04-21 quality audit
//! (Phase 1.1 of `docs/quality/improvement_plan.md`).

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use rustyline::{Cmd, Config, CompletionType, Context, Editor, KeyCode, KeyEvent, Modifiers};
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::Validator;
use rustyline::Helper;

use hymeko::module_store::module_store::{CompiledProgram, ModuleStore};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::resolution::interner::Interner;
use hymeko::util::real_parser::RealParser;
use hymeko::util::pretty_print::pretty_print_compiled;

use hymeko_formats::{generate_description, OutputFormat};
use hymeko_query::rewrite::execute_transform;

use crate::{
    entropy_rows_to_json, list_available_transforms, load_transform_spec,
    print_entropy_table, resolve_entropy_rows, run_inline_query,
    run_query_source,
};

pub enum CommandOutcome { Continue, Break }

pub struct Session {
    pub ms: ModuleStore<StdFsProvider, RealParser>,
    pub compiled: Option<Arc<CompiledProgram>>,
    pub loaded_path: Option<PathBuf>,
    pub robot_name: String,
    pub transforms_dir: String,
}

impl Session {
    pub fn new() -> Self {
        Self {
            ms: ModuleStore::new(StdFsProvider::new(), RealParser),
            compiled: None,
            loaded_path: None,
            robot_name: "robot".into(),
            transforms_dir: "transforms".into(),
        }
    }

    pub fn interner(&self) -> &Interner { &self.ms.it }

    pub fn ensure_loaded(&self) -> Option<&Arc<CompiledProgram>> {
        if self.compiled.is_none() {
            eprintln!("  No file loaded. Use: load <path.hymeko>");
            return None;
        }
        self.compiled.as_ref()
    }
}

// ─── Entry point ────────────────────────────────────────────────────

pub fn interactive_console() {
    let mut session = Session::new();
    print_banner();
    let Some(mut rl) = setup_editor(&session) else { return };
    let history_path = history_file_path();
    if let Some(p) = &history_path {
        let _ = rl.load_history(p);
    }

    run_repl_loop(&mut session, &mut rl);

    if let Some(p) = &history_path {
        let _ = rl.save_history(p);
    }
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  HyMeKo Interactive Console                 ║");
    println!("║  Type 'help' for available commands          ║");
    println!("║  Tab to complete, ↑/↓ for history            ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
}

fn setup_editor(session: &Session) -> Option<Editor<ReplHelper, FileHistory>> {
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .history_ignore_space(true)
        .auto_add_history(false)
        .build();
    let helper = ReplHelper {
        fs: FilenameCompleter::new(),
        transforms_dir: session.transforms_dir.clone(),
    };
    let mut rl: Editor<ReplHelper, FileHistory> = match Editor::with_config(config) {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to initialise readline: {e}. Falling back to basic mode.");
            return None;
        }
    };
    rl.set_helper(Some(helper));
    rl.bind_sequence(
        KeyEvent(KeyCode::Up, Modifiers::NONE),
        Cmd::HistorySearchBackward,
    );
    rl.bind_sequence(
        KeyEvent(KeyCode::Down, Modifiers::NONE),
        Cmd::HistorySearchForward,
    );
    Some(rl)
}

fn run_repl_loop(session: &mut Session, rl: &mut Editor<ReplHelper, FileHistory>) {
    loop {
        let prompt = build_prompt(session);
        let line = match rl.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => {
                println!("  Bye.");
                break;
            }
            Err(e) => {
                eprintln!("  readline error: {e}");
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        rl.add_history_entry(trimmed).ok();
        if matches!(dispatch_line(session, rl, trimmed), CommandOutcome::Break) {
            break;
        }
    }
}

fn build_prompt(session: &Session) -> String {
    let label = session.loaded_path.as_ref()
        .and_then(|p| p.file_stem())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "no file".into());
    format!("hymeko [{label}]> ")
}

fn dispatch_line(
    session: &mut Session,
    rl: &mut Editor<ReplHelper, FileHistory>,
    line: &str,
) -> CommandOutcome {
    let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();
    let cmd = parts[0].to_lowercase();
    match cmd.as_str() {
        "help" | "h" | "?"                   => { print_help(); CommandOutcome::Continue }
        "load" | "open"                      => cmd_load(session, &parts),
        "reload" | "r"                       => cmd_reload(session),
        "name"                               => cmd_name(session, &parts),
        "compile" | "gen" | "generate"       => cmd_compile(session, &parts),
        "validate" | "check"                 => cmd_validate(session),
        "inspect" | "ir" | "dump"            => cmd_inspect(session),
        "entropy"                            => cmd_entropy(session, &parts),
        "info" | "status"                    => cmd_info(session),
        "formats"                            => { cmd_formats(); CommandOutcome::Continue }
        "query" | "q!"                       => cmd_query(session, &parts, line),
        "qfile" | "query-file"               => cmd_qfile(session, &parts),
        "daemon"                             => { cmd_daemon(); CommandOutcome::Continue }
        "transform" | "tf"                   => cmd_transform(session, &parts),
        "tdir" | "transforms-dir"            => cmd_tdir(session, rl, &parts),
        "cd"                                 => { cmd_cd(&parts); CommandOutcome::Continue }
        "pwd"                                => { cmd_pwd(); CommandOutcome::Continue }
        "ls"                                 => { cmd_ls(&parts); CommandOutcome::Continue }
        "exit" | "quit" | "q"                => { println!("  Bye."); CommandOutcome::Break }
        other                                => cmd_fallback(session, other),
    }
}

// ─── Handlers ───────────────────────────────────────────────────────

fn cmd_load(session: &mut Session, parts: &[&str]) -> CommandOutcome {
    if parts.len() < 2 {
        eprintln!("  Usage: load <path.hymeko>");
        return CommandOutcome::Continue;
    }
    let path = PathBuf::from(parts[1]);
    if !path.exists() {
        eprintln!("  File not found: {}", path.display());
        return CommandOutcome::Continue;
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
    CommandOutcome::Continue
}

fn cmd_reload(session: &mut Session) -> CommandOutcome {
    let Some(path) = session.loaded_path.clone() else {
        eprintln!("  No file loaded. Use: load <path.hymeko>");
        return CommandOutcome::Continue;
    };
    match session.ms.compile(&path) {
        Ok(compiled) => {
            println!("  ✅ Reloaded {}", path.display());
            session.compiled = Some(compiled);
        }
        Err(e) => eprintln!("  ❌ Reload failed: {e:?}"),
    }
    CommandOutcome::Continue
}

fn cmd_name(session: &mut Session, parts: &[&str]) -> CommandOutcome {
    if parts.len() >= 2 {
        session.robot_name = parts[1].to_string();
        println!("  Robot name set to: {}", session.robot_name);
    } else {
        println!("  Current robot name: {}", session.robot_name);
    }
    CommandOutcome::Continue
}

fn cmd_compile(session: &mut Session, parts: &[&str]) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    let format_str = if parts.len() >= 2 { parts[1] } else { "urdf" };
    let fmt = match format_str.to_lowercase().as_str() {
        "urdf" => OutputFormat::Urdf,
        "sdf"  => OutputFormat::Sdf17,
        "mjcf" => OutputFormat::Mjcf,
        "dot"  => OutputFormat::DotGraph,
        other => {
            eprintln!("  Unknown format: {other}. Use: urdf, sdf, mjcf, dot");
            return CommandOutcome::Continue;
        }
    };
    let result = match generate_description(&compiled.ir, session.interner(), &session.robot_name, fmt) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  Code generation failed: {e}");
            return CommandOutcome::Continue;
        }
    };
    write_or_print(&result, parts.get(2).map(|s| PathBuf::from(*s)).as_ref());
    CommandOutcome::Continue
}

fn cmd_validate(session: &mut Session) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    let warnings = hymeko_formats::urdf::validate_robot_schema(&compiled.ir, session.interner());
    if warnings.is_empty() {
        println!("  ✅ Valid (no warnings)");
        return CommandOutcome::Continue;
    }
    println!("  ⚠️  {} warnings:", warnings.len());
    for w in &warnings {
        println!("    - {w}");
    }
    CommandOutcome::Continue
}

fn cmd_inspect(session: &mut Session) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    pretty_print_compiled(session.interner(), compiled);
    CommandOutcome::Continue
}

fn cmd_entropy(session: &mut Session, parts: &[&str]) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    let rest: Vec<&str> = parts.iter().skip(1).copied().collect();
    let (as_json, scopes) = match rest.as_slice() {
        []        => (false, Vec::<String>::new()),
        ["json"]  => (true, Vec::new()),
        names     => (false, names.iter().map(|s| s.to_string()).collect()),
    };
    let rows = resolve_entropy_rows(&compiled.ir, session.interner(), &scopes);
    if as_json {
        print!("{}", entropy_rows_to_json(&rows));
    } else {
        print_entropy_table(&rows);
    }
    CommandOutcome::Continue
}

fn cmd_info(session: &Session) -> CommandOutcome {
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
    CommandOutcome::Continue
}

fn cmd_formats() {
    println!("  Available output formats:");
    println!("    urdf  — URDF (ROS/Gazebo robot description)");
    println!("    sdf   — SDFormat 1.7 (Gazebo world description)");
    println!("    mjcf  — MuJoCo MJCF (MuJoCo simulator)");
    println!("    dot   — Graphviz DOT (visualization)");
}

fn cmd_query(session: &Session, parts: &[&str], line: &str) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    if parts.len() < 2 {
        eprintln!("  Usage: query <pattern>");
        eprintln!("  Examples:");
        eprintln!("    query _ : link           — find all nodes inheriting from 'link'");
        eprintln!("    query _ : joint          — find all joints");
        eprintln!("    query wheel_*            — find nodes with name prefix 'wheel_'");
        return CommandOutcome::Continue;
    }
    let pattern_text = line[parts[0].len()..].trim();
    run_inline_query(compiled, session.interner(), pattern_text);
    CommandOutcome::Continue
}

fn cmd_qfile(session: &Session, parts: &[&str]) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    if parts.len() < 2 {
        eprintln!("  Usage: qfile <query_file.hymeko>");
        return CommandOutcome::Continue;
    }
    let qpath = PathBuf::from(parts[1]);
    if !qpath.exists() {
        eprintln!("  File not found: {}", qpath.display());
        return CommandOutcome::Continue;
    }
    match fs::read_to_string(&qpath) {
        Ok(src) => run_query_source(compiled, session.interner(), &src, &qpath.to_string_lossy()),
        Err(e) => eprintln!("  Read failed: {e}"),
    }
    CommandOutcome::Continue
}

fn cmd_daemon() {
    eprintln!("  Daemon integration not yet connected.");
    eprintln!("  Use hymeko_client for direct iceoryx2 IPC.");
    eprintln!("  (Planned: daemon connect/push/status/disconnect)");
}

fn cmd_transform(session: &Session, parts: &[&str]) -> CommandOutcome {
    let Some(compiled) = session.ensure_loaded() else { return CommandOutcome::Continue };
    if parts.len() < 2 {
        eprintln!("  Usage: transform <name> [output_file]");
        eprintln!("  Available transforms:");
        list_available_transforms(&session.transforms_dir);
        return CommandOutcome::Continue;
    }
    let tf_name = parts[1];
    let tf_dir = PathBuf::from(&session.transforms_dir).join(tf_name);
    if !tf_dir.is_dir() {
        eprintln!("  Transform '{tf_name}' not found in {}", session.transforms_dir);
        eprintln!("  Available:");
        list_available_transforms(&session.transforms_dir);
        return CommandOutcome::Continue;
    }
    let spec = load_transform_spec(&session.transforms_dir, tf_name);
    let mut config = std::collections::HashMap::new();
    config.insert("robot_name".into(), session.robot_name.clone());
    match execute_transform(&compiled.ir, session.interner(), &spec, &config) {
        Ok(result) => {
            write_or_print(&result, parts.get(2).map(|s| PathBuf::from(*s)).as_ref());
        }
        Err(e) => eprintln!("  Transform failed: {e}"),
    }
    CommandOutcome::Continue
}

fn cmd_tdir(
    session: &mut Session,
    rl: &mut Editor<ReplHelper, FileHistory>,
    parts: &[&str],
) -> CommandOutcome {
    if parts.len() >= 2 {
        session.transforms_dir = parts[1].to_string();
        if let Some(h) = rl.helper_mut() {
            h.transforms_dir = session.transforms_dir.clone();
        }
        println!("  Transforms directory set to: {}", session.transforms_dir);
    } else {
        println!("  Current transforms directory: {}", session.transforms_dir);
        list_available_transforms(&session.transforms_dir);
    }
    CommandOutcome::Continue
}

fn cmd_cd(parts: &[&str]) {
    if parts.len() < 2 {
        let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
            eprintln!("  Usage: cd <directory>");
            return;
        };
        if let Err(e) = std::env::set_current_dir(&home) {
            eprintln!("  cd failed: {e}");
        }
        return;
    }
    let target = parts[1];
    if let Err(e) = std::env::set_current_dir(target) {
        eprintln!("  cd: {target}: {e}");
    }
}

fn cmd_pwd() {
    match std::env::current_dir() {
        Ok(p) => println!("  {}", p.display()),
        Err(e) => eprintln!("  pwd failed: {e}"),
    }
}

fn cmd_ls(parts: &[&str]) {
    let dir = if parts.len() >= 2 { parts[1] } else { "." };
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  ls: {dir}: {e}");
            return;
        }
    };
    let mut names: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        names.push(if is_dir { format!("{name}/") } else { name });
    }
    names.sort();
    for n in &names {
        print_ls_entry(n);
    }
}

fn print_ls_entry(n: &str) {
    if n.ends_with(".hymeko") {
        println!("  \x1b[1;32m{n}\x1b[0m");
    } else if n.ends_with('/') {
        println!("  \x1b[1;34m{n}\x1b[0m");
    } else {
        println!("  {n}");
    }
}

fn cmd_fallback(session: &mut Session, other: &str) -> CommandOutcome {
    if !other.ends_with(".hymeko") {
        eprintln!("  Unknown command: '{other}'. Type 'help' for available commands.");
        return CommandOutcome::Continue;
    }
    let path = PathBuf::from(other);
    if !path.exists() {
        eprintln!("  File not found: {other}");
        return CommandOutcome::Continue;
    }
    match session.ms.compile(&path) {
        Ok(compiled) => {
            println!("  ✅ Loaded {}", path.display());
            session.compiled = Some(compiled);
            session.loaded_path = Some(path);
        }
        Err(e) => eprintln!("  ❌ Compilation failed: {e:?}"),
    }
    CommandOutcome::Continue
}

// ─── Output helpers ──────────────────────────────────────────────────

fn write_or_print(result: &str, out_path: Option<&PathBuf>) {
    match out_path {
        Some(path) => match fs::write(path, result) {
            Ok(_) => println!("  Wrote {} bytes to {}", result.len(), path.display()),
            Err(e) => eprintln!("  Write failed: {e}"),
        },
        None => println!("{result}"),
    }
}

// ─── REPL completion helper ─────────────────────────────────────────

const REPL_COMMANDS: &[&str] = &[
    "help", "load", "open", "reload", "name",
    "compile", "gen", "generate",
    "validate", "check",
    "inspect", "ir", "dump",
    "entropy",
    "info", "status", "formats",
    "query", "qfile", "query-file",
    "daemon", "transform", "tf",
    "tdir", "transforms-dir",
    "cd", "pwd", "ls",
    "exit", "quit",
];

const FORMAT_NAMES: &[&str] = &["urdf", "sdf", "mjcf", "dot"];

pub struct ReplHelper {
    pub fs: FilenameCompleter,
    pub transforms_dir: String,
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let before = &line[..pos];
        let word_start = before
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        let current_word = &before[word_start..];
        let word_index = current_word_index(before);

        if word_index == 0 {
            return Ok((word_start, match_prefix_pairs(REPL_COMMANDS, current_word)));
        }

        let cmd = before.split_whitespace().next().unwrap_or("").to_lowercase();
        complete_for_command(&self.fs, &self.transforms_dir, cmd.as_str(), word_index,
                              current_word, word_start, line, pos, ctx)
    }
}

fn current_word_index(before: &str) -> usize {
    let starts_new_word = before.ends_with(char::is_whitespace);
    let prior = before.split_whitespace().count();
    if starts_new_word || prior == 0 { prior } else { prior - 1 }
}

fn match_prefix_pairs(options: &[&str], current: &str) -> Vec<Pair> {
    options.iter()
        .filter(|c| c.starts_with(current))
        .map(|c| Pair { display: c.to_string(), replacement: format!("{c} ") })
        .collect()
}

fn complete_for_command(
    fs: &FilenameCompleter,
    transforms_dir: &str,
    cmd: &str,
    word_index: usize,
    current_word: &str,
    word_start: usize,
    line: &str,
    pos: usize,
    ctx: &Context<'_>,
) -> rustyline::Result<(usize, Vec<Pair>)> {
    match (cmd, word_index) {
        ("load" | "open" | "qfile" | "query-file" | "cd" | "ls" | "tdir" | "transforms-dir", _) => {
            fs.complete(line, pos, ctx)
        }
        ("compile" | "gen" | "generate", 1) => {
            Ok((word_start, match_prefix_pairs(FORMAT_NAMES, current_word)))
        }
        ("compile" | "gen" | "generate", _) => fs.complete(line, pos, ctx),
        ("transform" | "tf", 1) => {
            let names = list_transform_names(transforms_dir);
            let pairs = names.iter()
                .filter(|t| t.starts_with(current_word))
                .map(|t| Pair { display: t.clone(), replacement: format!("{t} ") })
                .collect();
            Ok((word_start, pairs))
        }
        ("transform" | "tf", _) => fs.complete(line, pos, ctx),
        _ => Ok((word_start, vec![])),
    }
}

impl Hinter for ReplHelper { type Hint = String; }
impl Highlighter for ReplHelper {}
impl Validator for ReplHelper {}
impl Helper for ReplHelper {}

fn list_transform_names(transforms_dir: &str) -> Vec<String> {
    let dir = PathBuf::from(transforms_dir);
    let Ok(entries) = fs::read_dir(&dir) else { return Vec::new() };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| e.path().join("queries.hymeko").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    names
}

fn history_file_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".hymeko_history"))
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
  │  entropy [scope|json]     Per-scope structural entropy  │
  │  info                     Show current session info     │
  │  name [new_name]          Get/set robot model name      │
  │  formats                  List available output formats  │
  │                                                         │
  │  query <pattern>          Run inline query               │
  │    query _ : link         Find all link nodes            │
  │    query _ : joint        Find all joint edges           │
  │  qfile <file.hymeko>      Run queries from a file        │
  │                                                         │
  │  transform <name> [out]   Run a template-driven transform│
  │    tf urdf                Print URDF to stdout           │
  │    tf sdf robot.sdf       Write SDF to file              │
  │    tf dot graph.dot       Write DOT to file              │
  │  tdir [dir]               Get/set transforms directory   │
  │  daemon                   (planned) IPC bridge           │
  │                                                         │
  │  cd [dir]                 Change directory               │
  │  pwd                      Print working directory        │
  │  ls [dir]                 List files (.hymeko in green)  │
  │                                                         │
  │  help / h / ?             This help                     │
  │  exit / quit / q          Exit the console              │
  │                                                         │
  │  Line editing:                                          │
  │    Tab                    Complete cmd / file / format  │
  │    ↑ / ↓                  Prefix-search command history │
  │    Ctrl-R                 Reverse search history        │
  │    Ctrl-C                 Discard current line          │
  │    Ctrl-D                 Exit (on empty line)          │
  │  History persists in ~/.hymeko_history                  │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
"#);
}
