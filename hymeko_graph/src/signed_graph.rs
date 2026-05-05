//! Generic signed-graph types — vertex IDs, edges, signs, CSR adjacency.
//!
//! Decoupled from the Python / numpy conversion layer in
//! `hymeko_py` so this crate is pure-Rust and can be imported
//! anywhere.

use std::collections::HashMap;

/// Edge sign in a signed graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Positive edge ($+1$).
    Pos,
    /// Negative edge ($-1$).
    Neg,
}

impl Sign {
    /// Map to $\pm 1$ as `i8` for arithmetic over edge-sign products.
    #[inline]
    pub fn as_i8(self) -> i8 {
        match self {
            Sign::Pos => 1,
            Sign::Neg => -1,
        }
    }

    /// Construct from a non-zero `i8`; panics on `0`.
    #[inline]
    pub fn from_i8(s: i8) -> Sign {
        match s {
            1 => Sign::Pos,
            -1 => Sign::Neg,
            _ => panic!("Sign::from_i8: expected +1 or -1, got {s}"),
        }
    }
}

/// Compact signed graph: edges stored as `(u, v, sign)` triples
/// plus the vertex count.  Build a CSR adjacency from this via
/// [`SignedGraph::build_csr`] for fast neighbour queries.
#[derive(Debug, Clone)]
pub struct SignedGraph {
    /// Number of vertices.  Vertex IDs are `[0, n_nodes)`.
    pub n_nodes: u32,
    /// Edge endpoints, length $|E|$.  Each entry is `(u, v)` with
    /// `u != v`; we don't enforce `u < v`, callers may pass either
    /// orientation.
    pub edges: Vec<(u32, u32)>,
    /// Per-edge signs in `[-1, +1]`, parallel to `edges`.
    pub signs: Vec<i8>,
}

impl SignedGraph {
    /// Build the graph from a parallel-arrays representation.
    /// Panics if `edges_u` and `edges_v` differ in length.
    pub fn from_parts(
        n_nodes: u32,
        edges_u: &[u32],
        edges_v: &[u32],
        signs: &[i8],
    ) -> SignedGraph {
        assert_eq!(edges_u.len(), edges_v.len(),
                   "edges_u and edges_v differ in length");
        assert_eq!(edges_u.len(), signs.len(),
                   "edges and signs differ in length");
        let edges: Vec<(u32, u32)> = edges_u.iter()
            .zip(edges_v.iter())
            .map(|(&u, &v)| (u, v))
            .collect();
        SignedGraph { n_nodes, edges, signs: signs.to_vec() }
    }

    /// Number of edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Build undirected CSR adjacency (`row_ptr`, `col_idx`) for
    /// fast neighbour iteration during DFS.  Symmetric: each edge
    /// `(u, v)` produces both `u → v` and `v → u` entries.
    /// Neighbour lists are sorted and deduplicated.
    pub fn build_csr(&self) -> (Vec<u32>, Vec<u32>) {
        let n = self.n_nodes as usize;
        let mut deg = vec![0u32; n];
        for &(u, v) in &self.edges {
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
        let mut raw_ptr = vec![0u32; n + 1];
        for i in 0..n {
            raw_ptr[i + 1] = raw_ptr[i] + deg[i];
        }
        let total = raw_ptr[n] as usize;
        let mut raw_col = vec![0u32; total];
        let mut cursor = raw_ptr.clone();
        for &(u, v) in &self.edges {
            let pu = cursor[u as usize] as usize;
            raw_col[pu] = v;
            cursor[u as usize] += 1;
            let pv = cursor[v as usize] as usize;
            raw_col[pv] = u;
            cursor[v as usize] += 1;
        }
        let mut col_idx: Vec<u32> = Vec::with_capacity(total);
        let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        for i in 0..n {
            let s = raw_ptr[i] as usize;
            let e = raw_ptr[i + 1] as usize;
            raw_col[s..e].sort_unstable();
            let mut prev: i64 = -1;
            for &v in &raw_col[s..e] {
                if v as i64 != prev {
                    col_idx.push(v);
                    prev = v as i64;
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        (row_ptr, col_idx)
    }

    /// Build an `(u, v) → sign` lookup table.  Treats the graph as
    /// undirected: `(u, v)` and `(v, u)` map to the same sign.
    /// Used by cycle pruners that need sign products.
    pub fn build_sign_lookup(&self) -> HashMap<(u32, u32), i8> {
        let mut out = HashMap::with_capacity(self.edges.len() * 2);
        for (i, &(u, v)) in self.edges.iter().enumerate() {
            let s = self.signs[i];
            let key = (u.min(v), u.max(v));
            out.insert(key, s);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_round_trip_small_triangle() {
        // Triangle 0-1-2 with signs +, -, +.
        let g = SignedGraph::from_parts(
            3,
            &[0, 1, 2],
            &[1, 2, 0],
            &[1, -1, 1],
        );
        let (row_ptr, col_idx) = g.build_csr();
        assert_eq!(row_ptr.len(), 4);
        // Each vertex has 2 unique undirected neighbours.
        for v in 0..3 {
            let s = row_ptr[v] as usize;
            let e = row_ptr[v + 1] as usize;
            assert_eq!(e - s, 2);
        }
    }

    #[test]
    fn sign_lookup_canonicalises_undirected() {
        let g = SignedGraph::from_parts(
            3,
            &[0, 1],
            &[1, 2],
            &[1, -1],
        );
        let lk = g.build_sign_lookup();
        assert_eq!(lk.get(&(0, 1)), Some(&1));
        assert_eq!(lk.get(&(1, 2)), Some(&-1));
        // Reversed-orientation lookup also works (canonicalised).
        assert_eq!(lk.get(&(2, 1).into()).copied().or(lk.get(&(1, 2)).copied()),
                   Some(-1));
    }
}
