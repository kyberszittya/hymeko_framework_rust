//! Parse / compile / query / emit surface for the browser.
//!
//! This module is native-testable (no wasm-bindgen here); the
//! `wasm::CompiledIR` façade in `crate::wasm` forwards to the methods
//! here. Everything is built against the `MemProvider` source backend
//! so no filesystem access is needed — the caller passes a raw
//! `.hymeko` source string and gets back a compiled IR wrapper.

use std::sync::Arc;

use hymeko::common::ids::DeclId;
use hymeko::module_store::module_store::{CompiledProgram, HymekoParser, ModuleStore};
use hymeko::module_store::source_provider::MemProvider;
use hymeko::resolution::string_table::StringTable;
use hymeko_formats::sdf::generate_sdf;
use hymeko_formats::urdf::generate_urdf;
use parser::ast::AstStr;

/// Parser glue — mirrors `hymeko_py::interface_python::api`'s
/// `RealParser` but staying inside the wasm crate so we can keep
/// `hymeko_core` free of its optional `util` module if we ever prune it.
pub struct LalrpopParser;

impl HymekoParser for LalrpopParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        parser::parse_description(src).map_err(|e| format!("{e:?}"))
    }
}

/// Compiled-IR wrapper exposed to both native tests and (via a thin
/// `wasm_bindgen` façade) the browser.
pub struct CompiledDoc {
    pub compiled: Arc<CompiledProgram>,
    pub strings:  StringTable,
}

/// Parse a single-file `.hymeko` source string and return a compiled IR.
///
/// The virtual filename used internally is `inline.hymeko`; it's
/// exposed here so the error messages are predictable.
pub fn compile_source(source: &str) -> Result<CompiledDoc, String> {
    let mut store = ModuleStore::new(MemProvider::default(), LalrpopParser);
    store.provider_mut().insert_file("inline.hymeko", source);
    let compiled = store
        .compile(std::path::Path::new("inline.hymeko"))
        .map_err(|e| format!("compile error: {e:?}"))?;
    let strings = StringTable::from_interner(&store.it);
    Ok(CompiledDoc { compiled, strings })
}

// --------------------------------------------------------------------- //
// Snapshot JSON — shape tuned for a force-directed graph viewer.
// --------------------------------------------------------------------- //

// Snapshot DTOs live in hymeko_formats::snapshot — re-exported here so
// existing wasm consumers (`pub use compile::SnapshotDto` in lib.rs,
// `docs/demo/demo.js` deserialising the JSON) keep their import paths.
pub use hymeko_formats::snapshot::{ArcDto, NodeDto, SnapshotDto};

impl CompiledDoc {
    pub fn node_count(&self) -> usize { self.compiled.ir.nodes.len() }
    pub fn edge_count(&self) -> usize { self.compiled.ir.edges.len() }
    pub fn arc_count(&self) -> usize  { self.compiled.ir.arcs.len() }

    pub fn snapshot(&self) -> SnapshotDto {
        hymeko_formats::snapshot::snapshot(&self.compiled.ir, &self.strings)
    }

    pub fn snapshot_json(&self) -> Result<String, String> {
        hymeko_formats::snapshot::snapshot_json(&self.compiled.ir, &self.strings)
    }

    pub fn to_urdf(&self, robot_name: &str) -> String {
        generate_urdf(&self.compiled.ir, &self.strings, robot_name)
    }

    pub fn to_sdf(&self, model_name: &str) -> String {
        generate_sdf(&self.compiled.ir, &self.strings, model_name)
    }

    pub fn to_dot(&self, graph_name: &str) -> String {
        hymeko_formats::snapshot::emit_dot_graph(
            &self.compiled.ir, &self.strings, graph_name,
        )
    }

    // ------------------------ predicate queries ------------------------

    /// Evaluate the `queries/standard.qlist` predicate language and
    /// return the names of all matching decls.
    ///
    /// Supported atoms:
    ///   KIND(<name>)                 — first inherited base equals <name>
    ///   INHERITS(<name>)             — transitively inherits <name>
    ///   SCOPEDIN(<name>)             — ancestor inherits <name>
    ///   HASARCREF(<sign>, <inner>)   — edge with sign-matching arc-ref
    ///                                  pointing at a decl matching <inner>
    ///   <a> AND <b>                  — conjunction
    ///   ANY                          — always true
    pub fn query(&self, predicate: &str) -> Vec<String> {
        let ir = &self.compiled.ir;
        let st = &self.strings;
        let mut out = Vec::new();
        for i in 0..ir.decl_nodes.len() {
            let did = DeclId::new(i);
            if pred_match_expr(predicate, did, ir, st) {
                out.push(st.resolve(ir.decl_nodes[i].name).to_string());
            }
        }
        out
    }

    pub fn query_count(&self, predicate: &str) -> usize {
        let ir = &self.compiled.ir;
        (0..ir.decl_nodes.len())
            .filter(|i| pred_match_expr(predicate, DeclId::new(*i), ir, &self.strings))
            .count()
    }
}

// ------------------------------------------------------------------- //
// Predicate-string evaluator — single source of truth in
// hymeko_query::predicate_expr; re-imported under legacy `pred_*`
// names for unchanged call-site compilation.
// ------------------------------------------------------------------- //

use hymeko_query::predicate_expr::match_expr as pred_match_expr;
