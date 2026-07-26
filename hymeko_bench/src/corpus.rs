//! Deterministic synthetic-hypergraph source generation for the scaling
//! sweep (SMC 2026 Task 2).
//!
//! Emits `.hymeko` source with `nodes` vertex declarations and `edges`
//! signed hyperedges whose membership is drawn i.i.d. Bernoulli(`density`).
//! This mirrors the private generator in
//! `hymeko_core/tests/benchmarks/bench_coo_builder_random.rs` (which is not
//! importable across crates) so the two agree on the corpus shape; kept here
//! as the single shared generator for the `hymeko_bench` binaries rather than
//! copied per-bin (CLAUDE.md §6.1).

/// Xorshift64* PRNG — small, deterministic, seedable. Not cryptographic; used
/// only to lay out synthetic incidence structure reproducibly.
#[derive(Clone, Copy, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state: s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn gen_index(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive == 0 {
            return 0;
        }
        (self.next_u64() as usize) % upper_exclusive
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// A `(sign, weight)` pair with sign uniform in {+,−} and weight in
    /// [0.1, 5.0).
    #[inline]
    pub fn signed_weight(&mut self) -> (&'static str, f64) {
        let sign = if self.bernoulli(0.5) { "+" } else { "-" };
        let weight = 0.1 + self.next_f64() * 4.9;
        (sign, weight)
    }
}

/// Build a deterministic `.hymeko` source with `nodes` vertices and `edges`
/// signed hyperedges at membership probability `density`.
///
/// # Preconditions
/// `nodes >= 1`. Every emitted hyperedge has arity `>= 1` (an empty draw is
/// repaired by adding one random member), so the compiled structure never has
/// a dangling edge.
///
/// # Postconditions
/// Returns UTF-8 source parseable by `parser::parse_description`.
#[must_use]
pub fn random_hymeko_source(nodes: usize, edges: usize, density: f64, seed: u64) -> String {
    let mut rng = Xorshift64::new(seed);
    // Pre-size roughly: header + per-node line + expected per-edge members.
    let approx = 32 + nodes * 12 + edges * (16 + (nodes as f64 * density * 18.0) as usize);
    let mut out = String::with_capacity(approx);
    out.push_str("BenchmarkCase\n{}\ncontext\n{\n");

    for n in 0..nodes {
        out.push_str("    node");
        out.push_str(&n.to_string());
        out.push_str("{}\n");
    }

    for e in 0..edges {
        let mut refs: Vec<String> = Vec::new();
        for n in 0..nodes {
            if rng.bernoulli(density) {
                let (sign, w) = rng.signed_weight();
                refs.push(format!("{sign} node{n}[{w:.4}]"));
            }
        }
        if refs.is_empty() {
            let n = rng.gen_index(nodes.max(1));
            let (sign, w) = rng.signed_weight();
            refs.push(format!("{sign} node{n}[{w:.4}]"));
        }
        out.push_str(&format!("    @e{} {{({});}}\n", e, refs.join(", ")));
    }

    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let a = random_hymeko_source(32, 16, 0.1, 42);
        let b = random_hymeko_source(32, 16, 0.1, 42);
        assert_eq!(a, b, "generation must be seed-deterministic");
    }

    #[test]
    fn different_seed_differs() {
        let a = random_hymeko_source(64, 32, 0.1, 1);
        let b = random_hymeko_source(64, 32, 0.1, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn emits_all_nodes_and_edges() {
        let src = random_hymeko_source(10, 5, 0.2, 7);
        assert!(src.matches("node").count() >= 10);
        // 5 hyperedges named @e0..@e4
        for e in 0..5 {
            assert!(src.contains(&format!("@e{e} ")), "missing @e{e}");
        }
    }

    #[test]
    fn no_empty_edges() {
        // Zero density still repairs each edge to arity >= 1.
        let src = random_hymeko_source(8, 4, 0.0, 3);
        for e in 0..4 {
            let line = src
                .lines()
                .find(|l| l.contains(&format!("@e{e} ")))
                .unwrap();
            assert!(line.contains("node"), "edge @e{e} has no member");
        }
    }
}
