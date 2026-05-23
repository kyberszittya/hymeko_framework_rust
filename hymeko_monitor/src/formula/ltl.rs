//! LTL AST — reduced to unbounded-horizon STL.
//!
//! This module exists for grammar compatibility with the paper's
//! semantics section. In v0.1 the online monitor rejects unbounded
//! temporal operators; LTL formulas must be translated to a bounded
//! horizon before monitoring, either by the caller or by a future
//! bounded-approximation transform.

use crate::formula::stl::Stl;

/// LTL formulas are represented as STL with the distinguished upper
/// bound `+∞`. The online monitor will refuse these at construction
/// time via [`Stl::is_bounded_horizon`].
pub fn always<P>(phi: Stl<P>) -> Stl<P> {
    Stl::Always { a: 0.0, b: f64::INFINITY, phi: Box::new(phi) }
}

/// LTL `F φ` = `F_{[0, +∞)} φ`.
pub fn eventually<P>(phi: Stl<P>) -> Stl<P> {
    Stl::Eventually { a: 0.0, b: f64::INFINITY, phi: Box::new(phi) }
}

/// LTL `φ U ψ` = `φ U_{[0, +∞)} ψ`.
pub fn until<P>(phi: Stl<P>, psi: Stl<P>) -> Stl<P> {
    Stl::Until { a: 0.0, b: f64::INFINITY, phi: Box::new(phi), psi: Box::new(psi) }
}
