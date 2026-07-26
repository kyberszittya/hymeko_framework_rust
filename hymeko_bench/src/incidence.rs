//! Structural accounting for the measured storage-overhead ratio ρ
//! (SMC 2026 Task 1, witness for Proposition 4).
//!
//! Proposition 4 models the canonical IR as `(n+m)` fixed-size declaration
//! records (a Blake3 digest + a name-index entry each) plus `m·d̄`
//! signed-incidence entries, against a raw adjacency list holding just the
//! `m·d̄` incidence entries, giving `ρ = 1 + c(n+m)/(m·d̄)`. This module builds
//! the two representations from a compiled `Ir` so their live-heap sizes can be
//! measured with the tracking allocator.

use hymeko::common::ids::DeclId;
use hymeko::ir::ir::{DeclKind, Ir};

/// Size of a Blake3 digest as stored in a declaration record (`HashId([u8;32])`).
pub const DIGEST_BYTES: usize = 32;

/// Structural counts of a compiled hypergraph IR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureCounts {
    /// `n = |V|`: number of vertex (node) declarations.
    pub n: usize,
    /// `m = |E|`: number of hyperedge declarations.
    pub m: usize,
    /// `m·d̄`: total signed-incidence entries.
    pub nnz: usize,
    /// Mean hyperedge arity `d̄ = nnz/m` (0 if `m == 0`).
    pub mean_arity: f64,
}

impl StructureCounts {
    /// `n/m` ratio (the evidence for Proposition 4's `n = O(m log n)`
    /// assumption). `NaN` if `m == 0`.
    #[must_use]
    pub fn n_over_m(&self) -> f64 {
        if self.m == 0 {
            f64::NAN
        } else {
            self.n as f64 / self.m as f64
        }
    }

    /// Count-based predicted overhead `ρ = 1 + (n+m)/(m·d̄)` (the closed form
    /// with unit constant `c = 1`). `NaN` if `nnz == 0`.
    #[must_use]
    pub fn rho_predicted_unit(&self) -> f64 {
        if self.nnz == 0 {
            f64::NAN
        } else {
            1.0 + (self.n + self.m) as f64 / self.nnz as f64
        }
    }
}

/// Count vertices, hyperedges, and total incidences of a compiled IR.
#[must_use]
pub fn structure_counts(ir: &Ir) -> StructureCounts {
    let n = ir
        .decl_nodes
        .iter()
        .filter(|d| d.kind == DeclKind::Node)
        .count();
    let mut m = 0usize;
    let mut nnz = 0usize;
    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        m += 1;
        if let Some(eid) = ir.as_edge(DeclId::new(i)) {
            for &aid in &ir.edges[eid.0].arcs {
                nnz += ir.arcs[aid.0].refs.len();
            }
        }
    }
    let mean_arity = if m > 0 { nnz as f64 / m as f64 } else { 0.0 };
    StructureCounts {
        n,
        m,
        nnz,
        mean_arity,
    }
}

/// Build the naive raw-adjacency baseline: one `Vec<usize>` of member vertex
/// ids per hyperedge (vertex id = its declaration-arena index). Total stored
/// entries = `nnz`. Each inner and the outer `Vec` is `shrink_to_fit` so the
/// measured heap reflects the minimal representation, not growth slack.
///
/// # Postconditions
/// `result.iter().map(Vec::len).sum() == structure_counts(ir).nnz`.
#[must_use]
pub fn build_adjacency(ir: &Ir) -> Vec<Vec<usize>> {
    let mut inc: Vec<Vec<usize>> = Vec::new();
    for (i, decl) in ir.decl_nodes.iter().enumerate() {
        if decl.kind != DeclKind::Edge {
            continue;
        }
        let did = DeclId::new(i);
        let Some(eid) = ir.as_edge(did) else {
            inc.push(Vec::new());
            continue;
        };
        let mut members: Vec<usize> = Vec::new();
        for &aid in &ir.edges[eid.0].arcs {
            for r in &ir.arcs[aid.0].refs {
                members.push(r.target().0);
            }
        }
        members.shrink_to_fit();
        inc.push(members);
    }
    inc.shrink_to_fit();
    inc
}

/// Build the `(n+m)` declaration records the canonical IR carries on top of the
/// incidence: a Blake3 digest and a name-index entry per declaration. Returns
/// `(digests, name_index)`, both `shrink_to_fit`.
///
/// # Preconditions
/// `n_plus_m` is `structure_counts(ir).n + structure_counts(ir).m`.
#[must_use]
pub fn build_ir_records(n_plus_m: usize) -> (Vec<[u8; DIGEST_BYTES]>, Vec<u32>) {
    let mut digests: Vec<[u8; DIGEST_BYTES]> = Vec::with_capacity(n_plus_m);
    let mut name_index: Vec<u32> = Vec::with_capacity(n_plus_m);
    for i in 0..n_plus_m {
        digests.push([0u8; DIGEST_BYTES]);
        name_index.push(i as u32);
    }
    digests.shrink_to_fit();
    name_index.shrink_to_fit();
    (digests, name_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_have_expected_length() {
        let (d, ni) = build_ir_records(7);
        assert_eq!(d.len(), 7);
        assert_eq!(ni.len(), 7);
        assert_eq!(d.capacity(), 7, "digests not shrunk");
    }

    #[test]
    fn rho_predicted_monotone_in_arity() {
        // Higher mean arity (more nnz for fixed n,m) → ρ closer to 1.
        let low = StructureCounts {
            n: 100,
            m: 50,
            nnz: 100,
            mean_arity: 2.0,
        };
        let high = StructureCounts {
            n: 100,
            m: 50,
            nnz: 5000,
            mean_arity: 100.0,
        };
        assert!(low.rho_predicted_unit() > high.rho_predicted_unit());
        assert!(high.rho_predicted_unit() > 1.0);
    }
}
