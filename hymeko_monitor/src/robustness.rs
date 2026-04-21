//! Robustness combinators — pure functions over real-valued robustness.
//!
//! These live in a separate module so they can be unit-tested exhaustively
//! independent of the hypergraph machinery. All combinators handle `NaN`
//! defensively: if any input is `NaN`, the combinator returns `NaN`.

/// Conjunction: `min`.
pub fn and(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() { f64::NAN } else { f64::min(x, y) }
}

/// Disjunction: `max`.
pub fn or(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() { f64::NAN } else { f64::max(x, y) }
}

/// Negation: `-x`.
pub fn not(x: f64) -> f64 {
    -x
}

/// Supremum over an iterator of values. `-∞` if the iterator is empty.
pub fn sup<I: IntoIterator<Item = f64>>(it: I) -> f64 {
    it.into_iter().fold(f64::NEG_INFINITY, |acc, v| {
        if acc.is_nan() || v.is_nan() { f64::NAN } else { f64::max(acc, v) }
    })
}

/// Infimum over an iterator of values. `+∞` if the iterator is empty.
pub fn inf<I: IntoIterator<Item = f64>>(it: I) -> f64 {
    it.into_iter().fold(f64::INFINITY, |acc, v| {
        if acc.is_nan() || v.is_nan() { f64::NAN } else { f64::min(acc, v) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn de_morgan() {
        for &a in &[-1.0f64, 0.0, 1.0, 2.5] {
            for &b in &[-1.0f64, 0.0, 1.0, 2.5] {
                assert_eq!(not(and(a, b)), or(not(a), not(b)));
            }
        }
    }

    #[test]
    fn nan_propagates() {
        assert!(and(f64::NAN, 1.0).is_nan());
        assert!(or(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn sup_empty_is_neg_inf() {
        let v: Vec<f64> = vec![];
        assert_eq!(sup(v), f64::NEG_INFINITY);
    }

    #[test]
    fn inf_empty_is_pos_inf() {
        let v: Vec<f64> = vec![];
        assert_eq!(inf(v), f64::INFINITY);
    }
}
