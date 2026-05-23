//! CLI: dump MSG / SSG / ABB analysis for a P-graph file.
//!
//! Accepts `.hymeko` source or `.pgip` (P-graph Studio SQLite) input,
//! auto-detected by file extension.
//!
//! ```text
//! hymeko_pgraph_dump path/to/graph.{hymeko|pgip}
//!     [--algorithm msg|ssg|abb]
//!     [--weights "w1,w2,...,wD"]
//!     [--relaxed-msg]
//!     [--write-pgip path/to/out.pgip]
//! ```
//!
//! Writes one JSON document to stdout (pretty-printed). Exit code 0 on
//! success, 2 when parse/lower fails, 1 on usage / IO errors.
//!
//! Stage P-mo (2026-05-19): `--weights` for multi-objective ABB.
//! Stage P-io (2026-05-19): `.pgip` direct read; `--write-pgip` emits
//! a P-graph Studio-loadable file with the ABB result baked in.

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use hymeko_pgraph::abb::AbbOptions;
use hymeko_pgraph::msg::MaximalStructureOptions;
use hymeko_pgraph::{
    DumpAlgorithm, analyze_lowered_with_full_options, analyze_source_with_full_options,
    read_pgip, write_pgip,
};

fn parse_weights(s: &str) -> Result<Vec<f64>, String> {
    let v: Result<Vec<f64>, _> = s
        .split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .map_err(|e| format!("--weights entry {t:?}: {e}"))
        })
        .collect();
    let v = v?;
    if v.is_empty() {
        return Err("--weights requires at least one comma-separated value".into());
    }
    if v.iter().any(|w| *w < 0.0) {
        return Err(
            "--weights entries must be non-negative (admissibility of the inclusion bound)".into(),
        );
    }
    Ok(v)
}

fn main() -> std::process::ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).map(|s| s.to_string()).collect();
    if argv.is_empty() {
        eprintln!(
            "usage: hymeko_pgraph_dump <file.hymeko> [--algorithm msg|ssg|abb] [--weights \"w1,w2,...\"]"
        );
        return std::process::ExitCode::from(1u8);
    }
    let path = argv[0].clone();
    let mut algorithm = DumpAlgorithm::Ssg;
    let mut cost_weights: Option<Vec<f64>> = None;
    let mut relaxed_msg = false;
    let mut write_pgip_path: Option<PathBuf> = None;
    let mut i = 1usize;
    while i < argv.len() {
        if argv[i] == "--algorithm" {
            if i + 1 >= argv.len() {
                eprintln!("--algorithm requires a value");
                return std::process::ExitCode::from(1u8);
            }
            match DumpAlgorithm::from_str(argv[i + 1].trim()) {
                Ok(x) => algorithm = x,
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1u8);
                }
            }
            i += 2;
            continue;
        }
        if let Some(rest) = argv[i].strip_prefix("--algorithm=") {
            match DumpAlgorithm::from_str(rest.trim()) {
                Ok(x) => algorithm = x,
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1u8);
                }
            }
            i += 1;
            continue;
        }
        if argv[i] == "--weights" {
            if i + 1 >= argv.len() {
                eprintln!("--weights requires a comma-separated list");
                return std::process::ExitCode::from(1u8);
            }
            match parse_weights(&argv[i + 1]) {
                Ok(v) => cost_weights = Some(v),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1u8);
                }
            }
            i += 2;
            continue;
        }
        if let Some(rest) = argv[i].strip_prefix("--weights=") {
            match parse_weights(rest) {
                Ok(v) => cost_weights = Some(v),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1u8);
                }
            }
            i += 1;
            continue;
        }
        if argv[i] == "--relaxed-msg" {
            relaxed_msg = true;
            i += 1;
            continue;
        }
        if argv[i] == "--write-pgip" {
            if i + 1 >= argv.len() {
                eprintln!("--write-pgip requires a path argument");
                return std::process::ExitCode::from(1u8);
            }
            write_pgip_path = Some(PathBuf::from(&argv[i + 1]));
            i += 2;
            continue;
        }
        if let Some(rest) = argv[i].strip_prefix("--write-pgip=") {
            write_pgip_path = Some(PathBuf::from(rest));
            i += 1;
            continue;
        }
        eprintln!("unexpected argument: {}", argv[i]);
        return std::process::ExitCode::from(1u8);
    }
    // --relaxed-msg flips BOTH the MSG backward-pass criterion AND the
    // SSG/ABB leaf-feasibility check; the two must agree to match
    // P-graph Studio's relaxed-no-excess regime.
    let opts = AbbOptions {
        cost_weights,
        strict_no_excess: !relaxed_msg,
        ..AbbOptions::default()
    };
    let msg_opts = MaximalStructureOptions {
        strict_no_excess: !relaxed_msg,
    };

    // Auto-detect input format by extension. `.pgip` reads the
    // SQLite directly; anything else is treated as `.hymeko` source.
    let path_buf = PathBuf::from(&path);
    let is_pgip = path_buf
        .extension()
        .map(|e| e.eq_ignore_ascii_case("pgip"))
        .unwrap_or(false);

    let (out, abb_solution, graph_for_write) = if is_pgip {
        // Stage P-io: direct .pgip read.
        let graph = match read_pgip(Path::new(&path)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("read_pgip {path}: {e}");
                return std::process::ExitCode::from(1u8);
            }
        };
        let description = path_buf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pgip_input")
            .to_string();
        let (json, abb) = analyze_lowered_with_full_options(
            &graph, description, algorithm, msg_opts, opts,
        );
        (json, abb, Some(graph))
    } else {
        let src = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("read {path}: {e}");
                return std::process::ExitCode::from(1u8);
            }
        };
        // For --write-pgip we also need the lowered graph; re-parse on
        // demand to keep the back-compat shim hot-path unchanged.
        if write_pgip_path.is_some() {
            let desc = match parser::parse_description(&src) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("parse for --write-pgip: {e:?}");
                    return std::process::ExitCode::from(1u8);
                }
            };
            let description = desc.name.to_string();
            let p = match hymeko_pgraph::lower(&desc) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("lower for --write-pgip: {e}");
                    return std::process::ExitCode::from(1u8);
                }
            };
            let (json, abb) = analyze_lowered_with_full_options(
                &p, description, algorithm, msg_opts, opts,
            );
            (json, abb, Some(p))
        } else {
            let json = analyze_source_with_full_options(&src, algorithm, msg_opts, opts);
            (json, None, None)
        }
    };
    match serde_json::to_string_pretty(&out) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("json: {e}");
            return std::process::ExitCode::from(1u8);
        }
    }

    // Stage P-io: --write-pgip emits a P-graph-Studio-loadable file.
    if let Some(out_path) = write_pgip_path {
        let graph = match graph_for_write {
            Some(ref g) => g,
            None => {
                // Should not happen: we only set write_pgip_path on
                // input paths that produce graph_for_write.
                eprintln!("--write-pgip: internal — no lowered graph available");
                return std::process::ExitCode::from(1u8);
            }
        };
        if let Err(e) = write_pgip(graph, &out_path, abb_solution.as_ref()) {
            eprintln!("write_pgip {}: {e}", out_path.display());
            return std::process::ExitCode::from(1u8);
        }
        eprintln!("wrote {}", out_path.display());
    }

    if out.ok {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::from(2u8)
    }
}
