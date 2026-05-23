//! Shared integration-test helpers for `hymeko_hre`.
//!
//! Mirrors the minimal part of `hymeko_core/tests/test_helpers.rs` that we
//! need to drive fixture-based end-to-end tests: a `HymekoParser`
//! implementation backed by `parser::parse_description` plus a
//! `load_and_lower` convenience that runs the full `ModuleStore` pipeline.
//!
//! Kept deliberately small — richer helpers (dense-matrix printers, weight
//! extractors, etc.) live in `hymeko_core/tests/test_helpers.rs` and can be
//! inlined here on demand when a test wants them.

#![allow(dead_code)]

use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use hymeko::module_store::module_store::{
    CompiledProgram, HymekoParser, ModuleLoadError, ModuleStore,
};
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
use hymeko_hnn::traversal::hypergraphview::HyperGraphView;
use parser::ast::AstStr;

/// LALRPOP parser thin wrapper that implements `HymekoParser`, letting a
/// `ModuleStore` drive parsing via the production SIMD pipeline.
pub struct LalrpopParser;

impl HymekoParser for LalrpopParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        parser::parse_description(src).map_err(|e| format!("{e:?}"))
    }
}

/// Parse, resolve, and lower a `.hymeko` module using the production
/// `ModuleStore` pipeline. Returns the store (so the caller can hold onto
/// the interner via `store.it`) and an `Arc<CompiledProgram>` containing
/// the lowered `Ir`.
pub fn load_and_lower(
    root: impl AsRef<Path>,
) -> Result<
    (
        ModuleStore<StdFsProvider, LalrpopParser>,
        Arc<CompiledProgram>,
    ),
    ModuleLoadError,
> {
    let fs = StdFsProvider::new();
    let parser = LalrpopParser;
    let mut store = ModuleStore::new(fs, parser);
    let compiled = store.compile(root.as_ref())?;
    Ok((store, compiled))
}

/// Default aggregation config matching what `HypergraphEngine` uses when
/// building its internal view.
pub fn default_agg_cfg() -> AggCfg {
    AggCfg {
        sign: SignAgg::PreferNonNeutral,
        weight: WeightAgg::Sum,
        clamp01: false,
    }
}

/// Build a `HyperGraphView<f32, EdgeWScalar<f32>, f32>` from a lowered
/// program. The concrete type matches the one `HypergraphEngine::
/// compile_star_expansion_core::<f32>` operates on.
pub fn view_f32(compiled: &CompiledProgram) -> HyperGraphView<f32, EdgeWScalar<f32>, f32> {
    let cfg = default_agg_cfg();
    let ex = ScalarWeightExtractor;
    HyperGraphView::from_ir(&compiled.ir, &cfg, &ex)
}

/// Build a `HyperGraphView<f64, EdgeWScalar<f64>, f64>` for tests that want
/// double-precision.
pub fn view_f64(compiled: &CompiledProgram) -> HyperGraphView<f64, EdgeWScalar<f64>, f64> {
    let cfg = default_agg_cfg();
    let ex = ScalarWeightExtractor;
    HyperGraphView::from_ir(&compiled.ir, &cfg, &ex)
}

// `PhantomData` is pulled in to keep this module's types aligned with
// `hymeko_core`'s definitions across potential future generics churn.
const _: PhantomData<()> = PhantomData;
