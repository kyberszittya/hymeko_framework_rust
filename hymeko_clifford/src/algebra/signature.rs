//! Metric signature $(p, q)$ for the Clifford algebra $Cl(p, q)$.

/// $(p, q)$ metric signature.
///
/// Basis vectors $e_1, \dots, e_p$ square to $+1$; basis vectors
/// $e_{p+1}, \dots, e_{p+q}$ square to $-1$. Null directions are not
/// supported in this revision (deferred per Phase 5 of the plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signature {
    /// Number of positive-square basis vectors.
    pub p: usize,
    /// Number of negative-square basis vectors.
    pub q: usize,
}

impl Signature {
    /// Construct a Euclidean signature $Cl(n, 0)$ — every basis vector
    /// squares to $+1$.
    pub const fn euclidean(n: usize) -> Self {
        Self { p: n, q: 0 }
    }

    /// Construct a Lorentzian signature $Cl(p, q)$.
    pub const fn lorentzian(p: usize, q: usize) -> Self {
        Self { p, q }
    }

    /// Total dimension $n = p + q$.
    pub const fn n(&self) -> usize {
        self.p + self.q
    }

    /// Square of the basis vector $e_i$ (1-indexed). Returns $+1$ for
    /// the first $p$ vectors and $-1$ for the next $q$.
    ///
    /// # Panics
    /// If `i` is zero or larger than `p + q`.
    pub fn basis_square(&self, i: usize) -> f64 {
        debug_assert!(i >= 1, "basis vectors are 1-indexed; got 0");
        debug_assert!(
            i <= self.n(),
            "basis index {} out of range for signature ({}, {})",
            i,
            self.p,
            self.q
        );
        if i <= self.p { 1.0 } else { -1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_squares_to_plus_one() {
        let s = Signature::euclidean(4);
        for i in 1..=4 {
            assert_eq!(s.basis_square(i), 1.0);
        }
    }

    #[test]
    fn lorentzian_3_1() {
        let s = Signature::lorentzian(3, 1);
        assert_eq!(s.basis_square(1), 1.0);
        assert_eq!(s.basis_square(2), 1.0);
        assert_eq!(s.basis_square(3), 1.0);
        assert_eq!(s.basis_square(4), -1.0);
    }
}
