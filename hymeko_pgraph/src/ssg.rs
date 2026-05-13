//! Solution Structure Generation (SSG).
//!
//! Given the maximal structure produced by [`crate::msg`], SSG
//! enumerates every **combinatorially feasible** solution structure —
//! a subset $O' \subseteq O_{\max}$ of operating units such that:
//!
//! 1. every required product is producible from the raw set using
//!    only units in $O'$;
//! 2. every input of every $u \in O'$ is raw or produced by some
//!    other unit in $O'$ (consistency);
//! 3. (P-graph "no excess" property) every produced material that is
//!    not a product is consumed by some unit in $O'$.
//!
//! Condition (3) is the strict P-graph rule; relaxing it gives SSG'.
//! We expose both via [`enumerate_with_options`].
//!
//! For pedagogical clarity this implementation uses a
//! *generate-and-test* enumeration over $2^{|O_{\max}|}$ subsets. On
//! the small process-synthesis graphs that originally motivated
//! P-graphs ($|O_{\max}| \leq 32$ in the textbook examples) this is
//! ample. ABB ([`crate::abb`]) is the right tool when $|O_{\max}|$ is
//! large.

use std::collections::BTreeSet;

use hymeko::common::ids::DeclId;

use crate::lowering::LoweredPGraph;
use crate::msg::{MaximalStructure, close_producible};

/// One feasible solution structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SolutionStructure {
    /// Operating units selected.
    pub units: BTreeSet<DeclId>,
}

/// Knobs controlling SSG.
#[derive(Debug, Clone, Copy)]
pub struct SsgOptions {
    /// Strict P-graph: every produced non-product material must be
    /// consumed by some included unit.  When `false`, "excess"
    /// by-products are allowed (SSG').
    pub strict_no_excess: bool,
    /// Exclude the empty structure even when products are themselves
    /// raw (default `true`; an empty structure is rarely meaningful).
    pub require_at_least_one_unit: bool,
}

impl Default for SsgOptions {
    fn default() -> Self {
        Self {
            strict_no_excess: true,
            require_at_least_one_unit: true,
        }
    }
}

/// Enumerate every feasible solution structure inside the maximal
/// structure.
pub fn enumerate(p: &LoweredPGraph, msg: &MaximalStructure) -> Vec<SolutionStructure> {
    enumerate_with_options(p, msg, SsgOptions::default())
}

/// Enumerate with explicit options.
pub fn enumerate_with_options(
    p: &LoweredPGraph,
    msg: &MaximalStructure,
    opts: SsgOptions,
) -> Vec<SolutionStructure> {
    let units: Vec<DeclId> = msg.units.iter().copied().collect();
    let n = units.len();
    if n > 30 {
        // Refuse to silently chew through 2^31 subsets.
        // Caller should use ABB instead.
        return Vec::new();
    }

    let mut out: Vec<SolutionStructure> = Vec::new();
    let total: u32 = 1u32 << n;
    for mask in 0u32..total {
        let mut sel: BTreeSet<DeclId> = BTreeSet::new();
        for (i, u) in units.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                sel.insert(*u);
            }
        }
        if opts.require_at_least_one_unit && sel.is_empty() {
            continue;
        }
        if is_feasible(p, &sel, opts) {
            out.push(SolutionStructure { units: sel });
        }
    }
    out
}

/// True iff `sel` is a combinatorially feasible solution structure
/// for `p` under `opts`.
pub fn is_feasible(p: &LoweredPGraph, sel: &BTreeSet<DeclId>, opts: SsgOptions) -> bool {
    // (a) Every input of every selected unit must be raw or produced
    //     by another selected unit.
    let producible = close_producible(p, sel, &p.raws);
    for u in sel {
        let inputs = p.unit_inputs.get(u).cloned().unwrap_or_default();
        for m in &inputs {
            if !producible.contains(m) {
                return false;
            }
        }
    }
    // (b) Every required product is producible.
    for prod in &p.products {
        if !producible.contains(prod) {
            return false;
        }
    }
    // (c) Strict P-graph rule (optional): every produced non-product
    //     non-raw material is consumed by some selected unit.
    if opts.strict_no_excess {
        let mut produced: BTreeSet<DeclId> = BTreeSet::new();
        for u in sel {
            if let Some(out) = p.unit_outputs.get(u) {
                produced.extend(out.iter().copied());
            }
        }
        let mut consumed: BTreeSet<DeclId> = BTreeSet::new();
        for u in sel {
            if let Some(ins) = p.unit_inputs.get(u) {
                consumed.extend(ins.iter().copied());
            }
        }
        for m in &produced {
            if p.products.contains(m) || p.raws.contains(m) {
                continue;
            }
            if !consumed.contains(m) {
                return false;
            }
        }
    }
    true
}
