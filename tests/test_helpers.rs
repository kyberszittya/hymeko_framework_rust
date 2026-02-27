
use std::path::{Path};
use std::sync::Arc;
use hymeko_framework::common::ids::{DeclId, SymId};
use hymeko_framework::ir::ir::{DeclKind, Ir, NodeRec, SignedRefR, ValueR};
use hymeko_framework::module_store::module_store::{CompiledProgram, HymekoParser, ModuleLoadError, ModuleStore};
use hymeko_framework::module_store::source_provider::{StdFsProvider};
use hymeko_framework::resolution::interner::Interner;
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

pub fn find_decl(ir: &Ir, it: &Interner, name: &str, kind: DeclKind) -> DeclId {
    for (i, &sym) in ir.decl_name.iter().enumerate() {
        if ir.decl_kind[i] == kind && it.resolve(sym) == name {
            return DeclId(i);
        }
    }
    panic!("decl not found: {name}");
}

pub fn get_node<'a>(
    ir: &'a Ir,
    did: DeclId,
) -> &'a NodeRec {
    let nid = ir.as_node(did).expect("decl should be a node");
    &ir.nodes[nid.0 as usize]
}

pub fn has_tag(
    it: &Interner,
    tags: &[SymId],
    name: &str,
) -> bool {
    let tid = it.get_id(name).expect("tag should be interned");
    tags.contains(&tid)
}

pub fn weight0(r: &SignedRefR) -> f64 {
    let atom = match r {
        SignedRefR::Plus(a) | SignedRefR::Minus(a) | SignedRefR::Neutral(a) => a,
    };

    // 1) Ha weights: Option<Vec<ValueR>>
    #[allow(unreachable_patterns)]
    match atom.weights.as_ref() {
        Some(ws) => {
            // ValueR-s változat
            #[allow(unreachable_patterns)]
            match &ws[0] {
                ValueR::Num(n) => *n,
                other => panic!("expected numeric weight, got {:?}", other),
            }
        }
        None => panic!("expected weight(s), got None"),
    }
}
