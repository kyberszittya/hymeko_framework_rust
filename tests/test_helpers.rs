
use std::path::{Path};
use std::sync::Arc;
use hymeko_framework::module_store::module_store::{CompiledProgram, HymekoParser, ModuleLoadError, ModuleStore};
use hymeko_framework::module_store::source_provider::{StdFsProvider};
use parser::ast::AstStr;

/// Load, parse, resolve, and lower a HyMeKo module using the production ModuleStore pipeline.
/// Returns the lowered IR, the resolved index, and a reference to the shared interner.
///
/// This is meant for tests only.
///

pub struct LalrpopParser;

impl HymekoParser for LalrpopParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        parser::parse_description(src).map_err(|e| format!("{e:?}"))
    }
}

/// Test-only helper: build ModuleStore, compile root, and return both.
/// Returning the store lets tests access `store.it` without cloning.
pub fn load_and_lower(
    root_path: impl AsRef<Path>,
) -> Result<(ModuleStore<StdFsProvider, LalrpopParser>, Arc<CompiledProgram>), ModuleLoadError> {
    let fs = StdFsProvider::new();
    let parser = LalrpopParser;
    let mut store = ModuleStore::new(fs, parser);

    let compiled = store.compile(root_path.as_ref())?;
    Ok((store, compiled))
}

pub fn print_dense_matrix(m: &[Vec<f32>], title: &str) {
    println!("{title} ({}x{}):", m.len(), if m.is_empty() { 0 } else { m[0].len() });
    for row in m {
        for &x in row {
            if (x - x.round()).abs() < 1e-6 {
                print!("{:>3} ", x.round() as i32);
            } else {
                print!("{:>6.2} ", x);
            }
        }
        println!();
    }
}