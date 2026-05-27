//! Native P-graph application logic for the browser binding.
//!
//! Pure functions over *source strings* (no filesystem), so they are ergonomic
//! to unit-test natively; the `cfg(wasm32)` [`crate::wasm`] module wraps them
//! as `#[wasm_bindgen]` entry points. All P-graph work is delegated to
//! `hymeko_pgraph` — this module only adapts in-memory sources and serialises
//! results.

use hymeko_pgraph::{
    AbbOptions, DumpAlgorithm, LoweredPGraph, MaximalStructureOptions, MetaResolveError,
    analyze_lowered_with_full_options, compile_sources, lower, render_pgraph, to_dot,
};

/// In-memory name of the entry source.
const ROOT: &str = "input.hymeko";
/// Include name the meta-model source is keyed under (matches the canonical
/// `@"meta_pgraph.hymeko"` include).
const META_NAME: &str = "meta_pgraph.hymeko";

/// Resolve `(instance, meta)` source strings to a [`LoweredPGraph`].
///
/// Tries the meta-model path ([`compile_sources`]); a literal-tag instance
/// (no pgraph archetypes) falls back to [`lower`], so `meta` may be empty for
/// such files.
fn load(instance: &str, meta: &str) -> Result<LoweredPGraph, String> {
    match compile_sources(ROOT, &[(ROOT, instance), (META_NAME, meta)]) {
        Ok(g) => Ok(g),
        Err(MetaResolveError::MissingArchetype(_)) => {
            let desc = parser::parse_description(instance).map_err(|e| format!("parse: {e:?}"))?;
            lower(&desc).map_err(|e| format!("lower: {e}"))
        }
        Err(e) => Err(format!("{e}")),
    }
}

/// Solve the P-graph and return the MSG/SSG/ABB analysis as a JSON string.
pub fn solve_json(instance: &str, meta: &str) -> Result<String, String> {
    let g = load(instance, meta)?;
    let (out, _) = analyze_lowered_with_full_options(
        &g,
        "pgraph".to_string(),
        DumpAlgorithm::Abb,
        MaximalStructureOptions::default(),
        AbbOptions::default(),
    );
    serde_json::to_string(&out).map_err(|e| format!("json: {e}"))
}

/// Render the bipartite P-graph (M/O partition + signed incidence) as text.
pub fn transform_text(instance: &str, meta: &str) -> Result<String, String> {
    let g = load(instance, meta)?;
    Ok(render_pgraph(&g, "pgraph"))
}

/// Render the P-graph as Graphviz DOT.
pub fn dot(instance: &str, meta: &str) -> Result<String, String> {
    let g = load(instance, meta)?;
    Ok(to_dot(&g, "pgraph"))
}
