//! Signal Temporal Logic AST and semantics.
//!
//! The type `Stl<P>` is parameterised by the predicate type `P`. Callers
//! typically use it with `Arc<dyn HypergraphPredicate<H>>` or a concrete
//! predicate enum. The robustness semantics follow Donze-Maler 2010,
//! lifted from scalar signals to hypergraph traces (see `paper_outline.tex`
//! §3.3).

use std::sync::Arc;

use crate::predicate::{HypergraphPredicate, HypergraphState};
use crate::trace::Timestamp;

/// STL AST. `P` is the predicate type; for hypergraph predicates it is
/// typically `Arc<dyn HypergraphPredicate<H>>`.
#[derive(Debug, Clone)]
pub enum Stl<P> {
    /// Boolean constant true.
    True,
    /// Boolean constant false.
    False,
    /// Atomic proposition — a hypergraph predicate.
    Pred(P),
    /// Negation `¬φ`.
    Not(Box<Stl<P>>),
    /// Conjunction `φ ∧ ψ`.
    And(Box<Stl<P>>, Box<Stl<P>>),
    /// Disjunction `φ ∨ ψ`.
    Or(Box<Stl<P>>, Box<Stl<P>>),
    /// Bounded eventually `F_{[a,b]} φ`.
    Eventually { a: Timestamp, b: Timestamp, phi: Box<Stl<P>> },
    /// Bounded always `G_{[a,b]} φ`.
    Always { a: Timestamp, b: Timestamp, phi: Box<Stl<P>> },
    /// Bounded until `φ U_{[a,b]} ψ`.
    Until { a: Timestamp, b: Timestamp, phi: Box<Stl<P>>, psi: Box<Stl<P>> },
}

impl<P> Stl<P> {
    /// Upper bound on the look-ahead required to evaluate the formula at a
    /// given time. For a bounded-horizon formula this is finite.
    pub fn horizon(&self) -> Timestamp {
        match self {
            Stl::True | Stl::False | Stl::Pred(_) => 0.0,
            Stl::Not(inner) => inner.horizon(),
            Stl::And(l, r) | Stl::Or(l, r) => f64::max(l.horizon(), r.horizon()),
            Stl::Eventually { b, phi, .. } | Stl::Always { b, phi, .. } => {
                b + phi.horizon()
            }
            Stl::Until { b, phi, psi, .. } => {
                b + f64::max(phi.horizon(), psi.horizon())
            }
        }
    }

    /// Returns true if every temporal operator in the formula has a
    /// finite upper bound. Bounded-memory online monitoring requires
    /// this.
    pub fn is_bounded_horizon(&self) -> bool {
        match self {
            Stl::True | Stl::False | Stl::Pred(_) => true,
            Stl::Not(inner) => inner.is_bounded_horizon(),
            Stl::And(l, r) | Stl::Or(l, r) => {
                l.is_bounded_horizon() && r.is_bounded_horizon()
            }
            Stl::Eventually { b, phi, .. } | Stl::Always { b, phi, .. } => {
                b.is_finite() && phi.is_bounded_horizon()
            }
            Stl::Until { b, phi, psi, .. } => {
                b.is_finite()
                    && phi.is_bounded_horizon()
                    && psi.is_bounded_horizon()
            }
        }
    }
}

// -------------------------------------------------------------------- //
// Combinators — the user-facing DSL                                     //
// -------------------------------------------------------------------- //
//
// Usage:
//     let phi = always(
//         implies(has_tag("collab"),
//                 always_bounded(0.0, 0.1,
//                     forall(kind("joint"), within_limits()))));
//
// These produce STL AST nodes. Predicate combinators live in
// `crate::predicate` / the predicate builders module (TODO).

/// `φ ∧ ψ`.
pub fn and<P>(phi: Stl<P>, psi: Stl<P>) -> Stl<P> {
    Stl::And(Box::new(phi), Box::new(psi))
}

/// `φ ∨ ψ`.
pub fn or<P>(phi: Stl<P>, psi: Stl<P>) -> Stl<P> {
    Stl::Or(Box::new(phi), Box::new(psi))
}

/// `¬φ`.
pub fn not<P>(phi: Stl<P>) -> Stl<P> {
    Stl::Not(Box::new(phi))
}

/// `φ → ψ`, desugared to `¬φ ∨ ψ`.
pub fn implies<P: Clone>(phi: Stl<P>, psi: Stl<P>) -> Stl<P> {
    or(not(phi), psi)
}

/// `G_{[a,b]} φ`.
pub fn always_bounded<P>(a: Timestamp, b: Timestamp, phi: Stl<P>) -> Stl<P> {
    Stl::Always { a, b, phi: Box::new(phi) }
}

/// `F_{[a,b]} φ`.
pub fn eventually_bounded<P>(a: Timestamp, b: Timestamp, phi: Stl<P>) -> Stl<P> {
    Stl::Eventually { a, b, phi: Box::new(phi) }
}

/// `φ U_{[a,b]} ψ`.
pub fn until_bounded<P>(
    a: Timestamp,
    b: Timestamp,
    phi: Stl<P>,
    psi: Stl<P>,
) -> Stl<P> {
    Stl::Until { a, b, phi: Box::new(phi), psi: Box::new(psi) }
}

// -------------------------------------------------------------------- //
// Robustness evaluation on a single state                              //
//                                                                      //
// Note: temporal operators' robustness requires the *trace*, not a     //
// single state. Evaluation over traces is implemented by the sliding-  //
// window monitor in `crate::monitor::stl`. The function below is the   //
// base case (no temporal lookahead) used by the monitor at the         //
// subformula level.                                                    //
// -------------------------------------------------------------------- //

/// Robustness of a non-temporal sub-formula on a single state.
///
/// Panics (debug-only) if called on an STL node with temporal content;
/// the sliding-window monitor is responsible for decomposing temporal
/// operators and invoking this function per sample.
pub fn robustness_pointwise<H, P>(phi: &Stl<Arc<P>>, h: &H) -> f64
where
    H: HypergraphState,
    P: HypergraphPredicate<H> + ?Sized,
{
    match phi {
        Stl::True => f64::INFINITY,
        Stl::False => f64::NEG_INFINITY,
        Stl::Pred(p) => p.robustness(h),
        Stl::Not(inner) => -robustness_pointwise(inner, h),
        Stl::And(l, r) => {
            f64::min(robustness_pointwise(l, h), robustness_pointwise(r, h))
        }
        Stl::Or(l, r) => {
            f64::max(robustness_pointwise(l, h), robustness_pointwise(r, h))
        }
        Stl::Eventually { .. } | Stl::Always { .. } | Stl::Until { .. } => {
            debug_assert!(
                false,
                "robustness_pointwise called on temporal operator; \
                 use the sliding-window monitor"
            );
            f64::NAN
        }
    }
}

// -------------------------------------------------------------------- //
// Tests — only the non-temporal/structural laws for now                 //
// -------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_constant_is_zero() {
        let phi: Stl<()> = Stl::True;
        assert_eq!(phi.horizon(), 0.0);
    }

    #[test]
    fn horizon_nested_always_is_sum() {
        let phi: Stl<()> = always_bounded(0.0, 0.1,
                              always_bounded(0.0, 0.2, Stl::True));
        assert!((phi.horizon() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn bounded_fragment_detection() {
        let phi: Stl<()> = Stl::Eventually {
            a: 0.0, b: f64::INFINITY, phi: Box::new(Stl::True),
        };
        assert!(!phi.is_bounded_horizon());
    }
}
