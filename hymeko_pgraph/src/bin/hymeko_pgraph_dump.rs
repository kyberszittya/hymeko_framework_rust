//! CLI: dump MSG / SSG / ABB analysis for a P-graph `.hymeko` file.
//!
//! ```text
//! hymeko_pgraph_dump path/to/graph.hymeko [--algorithm msg|ssg|abb]
//! ```
//!
//! Writes one JSON document to stdout (pretty-printed). Exit code 0 on
//! success, 2 when parse/lower fails, 1 on usage / IO errors.

use std::fs;
use std::str::FromStr;

use hymeko_pgraph::{DumpAlgorithm, analyze_source};

fn main() -> std::process::ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).map(|s| s.to_string()).collect();
    if argv.is_empty() {
        eprintln!("usage: hymeko_pgraph_dump <file.hymeko> [--algorithm msg|ssg|abb]");
        return std::process::ExitCode::from(1u8);
    }
    let path = argv[0].clone();
    let mut algorithm = DumpAlgorithm::Ssg;
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
        eprintln!("unexpected argument: {}", argv[i]);
        return std::process::ExitCode::from(1u8);
    }
    let src = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return std::process::ExitCode::from(1u8);
        }
    };
    let out = analyze_source(&src, algorithm);
    match serde_json::to_string_pretty(&out) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("json: {e}");
            return std::process::ExitCode::from(1u8);
        }
    }
    if out.ok {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::from(2u8)
    }
}
