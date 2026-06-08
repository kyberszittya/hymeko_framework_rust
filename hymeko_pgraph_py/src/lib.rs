//! # `hymeko_pgraph` (Python extension module)
//!
//! Thin PyO3 wrapper around the Rust `hymeko_pgraph` crate. Exposes:
//!
//! * [`PyLoweredPGraph`] -- opaque handle around `LoweredPGraph`.
//! * [`PyMaximalStructure`] / [`PyAbbSolution`] / [`PySolutionStructure`] --
//!   wrapped result types; expose names rather than raw `DeclId`s so the
//!   Python side sees stable identifiers.
//! * Free functions: `from_hymeko_text`, `maximal_structure_rs`,
//!   `enumerate_ssg_rs`, `solve_abb_rs`, `solve_top_k_abb_rs`.
//!
//! The wrapper adds **zero** algorithm logic; every call delegates to
//! the Rust crate. The `name_lookup` field on result types caches the
//! `DeclId -> String` map once at construction so per-name lookups on
//! the Python side never re-traverse the schema.
//!
//! See `docs/plans/2026-06-04-hymeko-pgraph-py/plan.{tex,pdf,tikz,mmd}`.

#![forbid(unsafe_code)]
// PyO3 boilerplate sometimes triggers these.
#![allow(clippy::needless_pass_by_value)]

use std::collections::BTreeMap;

use hymeko_pgraph::{
    abb::{
        AbbOptions as RsAbbOptions, AbbSolution as RsAbbSolution,
        solve_top_k_with_regime, solve_with_regime,
    },
    lowering::{LoweredPGraph, lower},
    msg::{MaximalStructure, maximal_structure_with_regime},
    regime::{Canonical, NoExcess, Regime},
    ssg::SolutionStructure,
    ssg_dm::{SsgDmOptions, enumerate_with_options},
};
use parser::parse_description;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ----------------------------------------------------------------------
// Wrappers
// ----------------------------------------------------------------------

/// Opaque handle around a `hymeko_pgraph::LoweredPGraph`.
#[pyclass(name = "LoweredPGraph", module = "hymeko_pgraph", frozen)]
pub struct PyLoweredPGraph {
    inner: LoweredPGraph,
}

#[pymethods]
impl PyLoweredPGraph {
    /// Number of M-nodes (materials).
    fn n_materials(&self) -> usize {
        self.inner.materials.len()
    }

    /// Number of O-nodes (operating units).
    fn n_units(&self) -> usize {
        self.inner.units.len()
    }

    /// Sorted list of material names.
    fn material_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .materials
            .iter()
            .filter_map(|d| self.inner.decl_to_name.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    /// Sorted list of unit names.
    fn unit_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .units
            .iter()
            .filter_map(|d| self.inner.decl_to_name.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    /// Sorted raw-material names.
    fn raw_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .raws
            .iter()
            .filter_map(|d| self.inner.decl_to_name.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    /// Sorted required-product names.
    fn product_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .products
            .iter()
            .filter_map(|d| self.inner.decl_to_name.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    /// Cost of a single unit by name (default 1.0 if absent).
    fn cost_of(&self, unit: &str) -> PyResult<f64> {
        let id = self
            .inner
            .name_to_decl
            .get(unit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown unit `{unit}`")))?;
        Ok(self.inner.costs.get(id).copied().unwrap_or(1.0))
    }

    fn __repr__(&self) -> String {
        format!(
            "<LoweredPGraph n_materials={} n_units={} n_raws={} n_products={}>",
            self.inner.materials.len(),
            self.inner.units.len(),
            self.inner.raws.len(),
            self.inner.products.len(),
        )
    }
}

/// Result of [`maximal_structure_rs`]: MSG unit set + name cache.
#[pyclass(name = "MaximalStructure", module = "hymeko_pgraph", frozen)]
pub struct PyMaximalStructure {
    inner: MaximalStructure,
    name_lookup: BTreeMap<hymeko::common::ids::DeclId, String>,
}

#[pymethods]
impl PyMaximalStructure {
    fn unit_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .units
            .iter()
            .filter_map(|d| self.name_lookup.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    fn material_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .materials
            .iter()
            .filter_map(|d| self.name_lookup.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    fn n_units(&self) -> usize {
        self.inner.units.len()
    }

    fn n_materials(&self) -> usize {
        self.inner.materials.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<MaximalStructure n_units={} n_materials={}>",
            self.inner.units.len(),
            self.inner.materials.len(),
        )
    }
}

/// Result of [`enumerate_ssg_rs`]: a single Friedler solution-structure.
#[pyclass(name = "SolutionStructure", module = "hymeko_pgraph", frozen)]
pub struct PySolutionStructure {
    inner: SolutionStructure,
    name_lookup: BTreeMap<hymeko::common::ids::DeclId, String>,
}

#[pymethods]
impl PySolutionStructure {
    fn unit_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .units
            .iter()
            .filter_map(|d| self.name_lookup.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    fn n_units(&self) -> usize {
        self.inner.units.len()
    }

    fn __len__(&self) -> usize {
        self.inner.units.len()
    }

    fn __repr__(&self) -> String {
        format!("<SolutionStructure n_units={}>", self.inner.units.len())
    }
}

/// Result of [`solve_abb_rs`] / [`solve_top_k_abb_rs`].
#[pyclass(name = "AbbSolution", module = "hymeko_pgraph", frozen)]
pub struct PyAbbSolution {
    inner: RsAbbSolution,
    name_lookup: BTreeMap<hymeko::common::ids::DeclId, String>,
}

#[pymethods]
impl PyAbbSolution {
    fn unit_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .inner
            .units
            .iter()
            .filter_map(|d| self.name_lookup.get(d).cloned())
            .collect();
        out.sort();
        out
    }

    #[getter]
    fn cost(&self) -> f64 {
        self.inner.cost
    }

    #[getter]
    fn explored(&self) -> u64 {
        self.inner.explored
    }

    #[getter]
    fn pruned_by_inclusion(&self) -> u64 {
        self.inner.pruned_by_inclusion
    }

    #[getter]
    fn pruned_by_reachability(&self) -> u64 {
        self.inner.pruned_by_reachability
    }

    #[getter]
    fn pruned(&self) -> u64 {
        self.inner.pruned_by_inclusion + self.inner.pruned_by_reachability
    }

    fn n_units(&self) -> usize {
        self.inner.units.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<AbbSolution cost={:.4} n_units={} explored={} pruned={}>",
            self.inner.cost,
            self.inner.units.len(),
            self.inner.explored,
            self.inner.pruned_by_inclusion + self.inner.pruned_by_reachability,
        )
    }
}

// ----------------------------------------------------------------------
// Free functions
// ----------------------------------------------------------------------

/// Parse + lower a HyMeKo P-graph source string into a `LoweredPGraph`.
#[pyfunction]
fn from_hymeko_text(text: &str) -> PyResult<PyLoweredPGraph> {
    let d = parse_description(text)
        .map_err(|e| PyValueError::new_err(format!("parse error: {e:?}")))?;
    let inner = lower(&d).map_err(|e| PyValueError::new_err(format!("lower error: {e:?}")))?;
    Ok(PyLoweredPGraph { inner })
}

/// Resolve the regime trait object once at the wrapper boundary
/// (so the Python surface stays a single bool flag; no string typing).
fn pick_regime(strict_no_excess: bool) -> &'static dyn Regime {
    if strict_no_excess {
        &NoExcess
    } else {
        &Canonical
    }
}

/// Run MSG over a lowered P-graph.
#[pyfunction]
#[pyo3(signature = (g, strict_no_excess = false))]
fn maximal_structure_rs(
    g: &PyLoweredPGraph,
    strict_no_excess: bool,
) -> PyResult<PyMaximalStructure> {
    let regime = pick_regime(strict_no_excess);
    let msg = maximal_structure_with_regime(&g.inner, regime);
    Ok(PyMaximalStructure {
        inner: msg,
        name_lookup: g.inner.decl_to_name.clone(),
    })
}

/// Enumerate every Friedler solution-structure inside the MSG via the
/// canonical decision-mapping recursion.
///
/// `max_structures = 0` is "unlimited" (matches the Rust enum default).
#[pyfunction]
#[pyo3(signature = (g, msg, max_structures = 0))]
fn enumerate_ssg_rs(
    g: &PyLoweredPGraph,
    msg: &PyMaximalStructure,
    max_structures: usize,
) -> PyResult<Vec<PySolutionStructure>> {
    let opts = SsgDmOptions {
        max_structures,
        ..Default::default()
    };
    let r = enumerate_with_options(&g.inner, &msg.inner, opts);
    if r.capped {
        // Capped is a soft cutoff -- not an error, but worth reporting.
        // (Python side can inspect `len(result)` against `max_structures`.)
    }
    Ok(r.structures
        .into_iter()
        .map(|s| PySolutionStructure {
            inner: s,
            name_lookup: g.inner.decl_to_name.clone(),
        })
        .collect())
}

/// Cost-minimal feasible structure via ABB.
///
/// `opts` is a dict; recognised keys: `max_explored` (u64, 0=unlimited),
/// `strict_no_excess` (bool, default false). Anything else is ignored.
#[pyfunction]
#[pyo3(signature = (g, msg, opts = None))]
fn solve_abb_rs(
    g: &PyLoweredPGraph,
    msg: &PyMaximalStructure,
    opts: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<PyAbbSolution>> {
    let (rs_opts, regime) = parse_opts(opts)?;
    let sol = solve_with_regime(&g.inner, &msg.inner, rs_opts, regime);
    Ok(sol.map(|s| PyAbbSolution {
        inner: s,
        name_lookup: g.inner.decl_to_name.clone(),
    }))
}

/// Top-K cost-ranked feasible structures via the decision-mapping SSG
/// + per-structure cost scan (`hymeko_pgraph::abb::solve_top_k_with_regime`).
#[pyfunction]
#[pyo3(signature = (g, msg, k, opts = None))]
fn solve_top_k_abb_rs(
    g: &PyLoweredPGraph,
    msg: &PyMaximalStructure,
    k: usize,
    opts: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<PyAbbSolution>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let (rs_opts, regime) = parse_opts(opts)?;
    let sols = solve_top_k_with_regime(&g.inner, &msg.inner, k, rs_opts, regime);
    Ok(sols
        .into_iter()
        .map(|s| PyAbbSolution {
            inner: s,
            name_lookup: g.inner.decl_to_name.clone(),
        })
        .collect())
}

/// Shared opts parser for `solve_abb_rs` / `solve_top_k_abb_rs`.
fn parse_opts(
    opts: Option<&Bound<'_, PyDict>>,
) -> PyResult<(RsAbbOptions, &'static dyn Regime)> {
    let mut rs_opts = RsAbbOptions::default();
    let mut strict_no_excess = false;
    if let Some(d) = opts {
        if let Some(v) = d.get_item("max_explored")? {
            rs_opts.max_explored = v
                .extract::<u64>()
                .map_err(|_| PyValueError::new_err("max_explored must be u64"))?;
        }
        if let Some(v) = d.get_item("strict_no_excess")? {
            strict_no_excess = v
                .extract::<bool>()
                .map_err(|_| PyValueError::new_err("strict_no_excess must be bool"))?;
        }
    }
    Ok((rs_opts, pick_regime(strict_no_excess)))
}

// ----------------------------------------------------------------------
// Module entry
// ----------------------------------------------------------------------

/// `#[pymodule]` entry. The function name controls the Python module
/// name (must match the cdylib output -- pyo3 0.28 reads the symbol).
/// We name the symbol `hymeko_pgraph_py` to avoid clashing with the
/// `hymeko_pgraph` Rust crate; Python users `import hymeko_pgraph_py`
/// (mirrors the cdylib `name` in Cargo.toml).
#[pymodule]
fn hymeko_pgraph_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoweredPGraph>()?;
    m.add_class::<PyMaximalStructure>()?;
    m.add_class::<PySolutionStructure>()?;
    m.add_class::<PyAbbSolution>()?;

    m.add_function(wrap_pyfunction!(from_hymeko_text, m)?)?;
    m.add_function(wrap_pyfunction!(maximal_structure_rs, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_ssg_rs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_abb_rs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_top_k_abb_rs, m)?)?;

    // Suppress unused-import warning in release builds: PyRuntimeError
    // is reserved for future stage extensions (cycle detection in the
    // MSG, capped SSG report, etc.).
    let _ = PyRuntimeError::new_err::<&str>("(reserved)");
    Ok(())
}
