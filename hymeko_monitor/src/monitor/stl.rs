//! Online STL monitor — sliding-window DP over the bounded-horizon
//! fragment.
//!
//! ## Algorithm
//!
//! The formula tree is flattened in post-order at construction time so
//! every node has a unique `NodeId` with the invariant that children's
//! IDs precede their parent's. Each node owns one [`SlidingWindow`]
//! which holds its robustness signal `(semantic_time, ρ_node)`.
//!
//! On each `observe(state, τ_obs)`:
//!   1. Iterate node IDs in ascending order (children first).
//!   2. For each node `N` with delay `d_N = N.horizon()`, if
//!      `τ_obs ≥ d_N` push `(t* = τ_obs − d_N, ρ_N(t*))` into `N`'s
//!      window. If `τ_obs < d_N`, skip — not enough data yet.
//!   3. Compute `ρ_N(t*)` by reading children's windows:
//!      - atomic / `True` / `False` — direct;
//!      - `¬φ`, `∧`, `∨` — pointwise on children's value at `t*`;
//!      - `G_{[a,b]}(φ)`, `F_{[a,b]}(φ)` — `inf` / `sup` over child's
//!        values in `[t* + a, t* + b]`;
//!      - `φ U_{[a,b]} ψ` — full Donze–Maler combinator.
//!   4. The root's latest pushed value is the verdict.
//!
//! ## Window sizing
//!
//! Every node's window is sized to the root's horizon (a safe over-
//! allocation). With per-observe pushes for atomic nodes and per-
//! settled-time pushes for everything else, all required look-back is
//! always retained. A tighter per-node sizing is a future
//! performance-tuning TODO.

use std::sync::Arc;

use crate::formula::stl::Stl;
use crate::monitor::{Monitor, Verdict};
use crate::predicate::{HypergraphPredicate, HypergraphState};
use crate::trace::{Sample, Timestamp};
use crate::window::SlidingWindow;
use crate::MonitorError;

/// Tolerance for time-equality comparisons when looking up a child's
/// value at a specific time. Sized to be loose enough to absorb
/// floating-point accumulation across nested temporal operators while
/// tight enough that a 1 ms sampling cadence is still distinguishable.
const TIME_EPS: f64 = 1.0e-9;

// ─── Flattened AST representation ────────────────────────────────────

/// A single AST node in the flattened post-order layout. Children are
/// referenced by `NodeId` (an index into the monitor's `nodes` vector).
#[derive(Debug, Clone)]
enum NodeRecord<P: ?Sized> {
    True,
    False,
    Pred(Arc<P>),
    Not(usize),
    And(usize, usize),
    Or(usize, usize),
    Eventually { a: f64, b: f64, child: usize },
    Always { a: f64, b: f64, child: usize },
    Until { a: f64, b: f64, phi: usize, psi: usize },
}

/// Result of flattening the formula tree: parallel vectors of node
/// records, per-node delays (horizons), and pre-allocated sliding
/// windows.
struct Flattened<P: ?Sized> {
    nodes: Vec<NodeRecord<P>>,
    delays: Vec<f64>,
    windows: Vec<SlidingWindow>,
}

/// Walk the formula tree in post-order, assigning each node a
/// monotonically increasing `NodeId`. Children's IDs are guaranteed
/// to precede their parent's.
fn flatten<P>(
    phi: &Stl<Arc<P>>,
    nodes: &mut Vec<NodeRecord<P>>,
    delays: &mut Vec<f64>,
) -> usize
where
    P: HypergraphPredicate<Dummy> + ?Sized,
{
    let _ = std::marker::PhantomData::<P>;  // suppress unused warning on bound
    let (record, delay) = match phi {
        Stl::True => (NodeRecord::True, 0.0),
        Stl::False => (NodeRecord::False, 0.0),
        Stl::Pred(p) => (NodeRecord::Pred(p.clone()), 0.0),
        Stl::Not(inner) => {
            let cid = flatten(inner, nodes, delays);
            (NodeRecord::Not(cid), delays[cid])
        }
        Stl::And(l, r) => {
            let lid = flatten(l, nodes, delays);
            let rid = flatten(r, nodes, delays);
            (NodeRecord::And(lid, rid), delays[lid].max(delays[rid]))
        }
        Stl::Or(l, r) => {
            let lid = flatten(l, nodes, delays);
            let rid = flatten(r, nodes, delays);
            (NodeRecord::Or(lid, rid), delays[lid].max(delays[rid]))
        }
        Stl::Eventually { a, b, phi } => {
            let cid = flatten(phi, nodes, delays);
            (NodeRecord::Eventually { a: *a, b: *b, child: cid }, b + delays[cid])
        }
        Stl::Always { a, b, phi } => {
            let cid = flatten(phi, nodes, delays);
            (NodeRecord::Always { a: *a, b: *b, child: cid }, b + delays[cid])
        }
        Stl::Until { a, b, phi, psi } => {
            let pid = flatten(phi, nodes, delays);
            let qid = flatten(psi, nodes, delays);
            (
                NodeRecord::Until { a: *a, b: *b, phi: pid, psi: qid },
                b + delays[pid].max(delays[qid]),
            )
        }
    };
    let id = nodes.len();
    nodes.push(record);
    delays.push(delay);
    id
}

/// Dummy `HypergraphState` used only to satisfy `flatten`'s type
/// parameter without requiring it at call sites. The actual state
/// type is constrained at the [`StlMonitor`] level.
enum Dummy {}
impl HypergraphState for Dummy {
    type VertexId = ();
    type EdgeId = ();
    type TypeId = ();
    type Attr = ();
    fn vertices(&self) -> Box<dyn Iterator<Item = ()> + '_> { Box::new(std::iter::empty()) }
    fn edges(&self) -> Box<dyn Iterator<Item = ()> + '_> { Box::new(std::iter::empty()) }
    fn incidences(&self, _e: ()) -> Box<dyn Iterator<Item = ((), crate::predicate::Sign)> + '_> {
        Box::new(std::iter::empty())
    }
    fn vertex_type(&self, _v: ()) {}
    fn edge_type(&self, _e: ()) {}
    fn inherits(&self, _t: (), _b: ()) -> bool { false }
    fn vertex_attr(&self, _v: (), _k: &str) -> Option<&()> { None }
    fn edge_attr(&self, _e: (), _k: &str) -> Option<&()> { None }
}

// ─── Monitor ─────────────────────────────────────────────────────────

/// Online STL monitor for a bounded-horizon formula.
#[allow(missing_debug_implementations)]
pub struct StlMonitor<H: HypergraphState, P: ?Sized + HypergraphPredicate<H>> {
    /// Flattened post-order AST nodes.
    nodes: Vec<NodeRecord<P>>,
    /// Per-node delay (horizon required to evaluate it).
    delays: Vec<f64>,
    /// Per-node robustness-signal sliding window. Indexed by `NodeId`.
    windows: Vec<SlidingWindow>,
    /// Most recent observed timestamp; used to enforce monotonic time.
    last_t: Option<Timestamp>,
    /// Latest verdict ready to emit, if the root has settled.
    latest_verdict: Option<Verdict>,
    _phantom: std::marker::PhantomData<fn(&H)>,
}

impl<H, P> StlMonitor<H, P>
where
    H: HypergraphState,
    P: HypergraphPredicate<H> + ?Sized,
{
    /// Construct a monitor for the given formula.
    ///
    /// Returns [`MonitorError::UnboundedHorizon`] if the formula
    /// contains any unbounded temporal operator (monitoring requires
    /// bounded look-ahead to preserve the bounded-memory guarantee).
    pub fn new(formula: Stl<Arc<P>>) -> Result<Self, MonitorError> {
        if !formula.is_bounded_horizon() {
            return Err(MonitorError::UnboundedHorizon);
        }
        let (nodes, delays, windows) = build_flattened(&formula);
        Ok(Self {
            nodes,
            delays,
            windows,
            last_t: None,
            latest_verdict: None,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The horizon (look-ahead) of the root formula.
    pub fn horizon(&self) -> Timestamp {
        *self.delays.last().unwrap_or(&0.0)
    }

    /// Number of AST nodes (for tests and introspection).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Build the flattened representation + windows. Wrapper over the
/// generic `flatten` to centralise the windowing decision.
fn build_flattened<H, P>(
    phi: &Stl<Arc<P>>,
) -> (Vec<NodeRecord<P>>, Vec<f64>, Vec<SlidingWindow>)
where
    H: HypergraphState,
    P: HypergraphPredicate<H> + ?Sized,
{
    let mut nodes: Vec<NodeRecord<P>> = Vec::new();
    let mut delays: Vec<f64> = Vec::new();
    flatten_concrete(phi, &mut nodes, &mut delays);
    let root_horizon = delays.last().copied().unwrap_or(0.0);
    let windows = (0..nodes.len())
        .map(|_| SlidingWindow::new(root_horizon))
        .collect();
    (nodes, delays, windows)
}

/// `flatten` specialised to drop the `Dummy` parameter — same logic,
/// concrete on the predicate type only. The caller doesn't have a
/// concrete `H`, just the predicate, which is all flatten needs.
fn flatten_concrete<P: ?Sized>(
    phi: &Stl<Arc<P>>,
    nodes: &mut Vec<NodeRecord<P>>,
    delays: &mut Vec<f64>,
) -> usize {
    let (record, delay) = match phi {
        Stl::True => (NodeRecord::True, 0.0),
        Stl::False => (NodeRecord::False, 0.0),
        Stl::Pred(p) => (NodeRecord::Pred(p.clone()), 0.0),
        Stl::Not(inner) => {
            let cid = flatten_concrete(inner, nodes, delays);
            (NodeRecord::Not(cid), delays[cid])
        }
        Stl::And(l, r) => {
            let lid = flatten_concrete(l, nodes, delays);
            let rid = flatten_concrete(r, nodes, delays);
            (NodeRecord::And(lid, rid), delays[lid].max(delays[rid]))
        }
        Stl::Or(l, r) => {
            let lid = flatten_concrete(l, nodes, delays);
            let rid = flatten_concrete(r, nodes, delays);
            (NodeRecord::Or(lid, rid), delays[lid].max(delays[rid]))
        }
        Stl::Eventually { a, b, phi } => {
            let cid = flatten_concrete(phi, nodes, delays);
            (NodeRecord::Eventually { a: *a, b: *b, child: cid }, b + delays[cid])
        }
        Stl::Always { a, b, phi } => {
            let cid = flatten_concrete(phi, nodes, delays);
            (NodeRecord::Always { a: *a, b: *b, child: cid }, b + delays[cid])
        }
        Stl::Until { a, b, phi, psi } => {
            let pid = flatten_concrete(phi, nodes, delays);
            let qid = flatten_concrete(psi, nodes, delays);
            (
                NodeRecord::Until { a: *a, b: *b, phi: pid, psi: qid },
                b + delays[pid].max(delays[qid]),
            )
        }
    };
    let id = nodes.len();
    nodes.push(record);
    delays.push(delay);
    id
}

impl<H, P> Monitor<H> for StlMonitor<H, P>
where
    H: HypergraphState,
    P: HypergraphPredicate<H> + ?Sized,
{
    fn observe(&mut self, sample: Sample<H>) {
        if let Some(prev) = self.last_t {
            debug_assert!(
                sample.t > prev,
                "non-monotonic timestamp: prev={prev}, new={}", sample.t
            );
        }
        self.last_t = Some(sample.t);
        let tau_obs = sample.t;

        // Process nodes in post-order (= ascending NodeId, since
        // `flatten_concrete` assigns IDs post-order).
        for id in 0..self.nodes.len() {
            let delay = self.delays[id];
            if tau_obs < delay - TIME_EPS {
                continue; // not enough data yet for this node
            }
            let t_settled = tau_obs - delay;

            let value = match &self.nodes[id] {
                NodeRecord::True => f64::INFINITY,
                NodeRecord::False => f64::NEG_INFINITY,
                NodeRecord::Pred(p) => p.robustness(&sample.state),
                NodeRecord::Not(c) => -value_at(&self.windows[*c], t_settled),
                NodeRecord::And(l, r) => f64::min(
                    value_at(&self.windows[*l], t_settled),
                    value_at(&self.windows[*r], t_settled),
                ),
                NodeRecord::Or(l, r) => f64::max(
                    value_at(&self.windows[*l], t_settled),
                    value_at(&self.windows[*r], t_settled),
                ),
                NodeRecord::Always { a, b, child } => {
                    sup_or_inf_in_range(
                        &self.windows[*child],
                        t_settled + a, t_settled + b,
                        /*sup=*/ false,
                    )
                }
                NodeRecord::Eventually { a, b, child } => {
                    sup_or_inf_in_range(
                        &self.windows[*child],
                        t_settled + a, t_settled + b,
                        /*sup=*/ true,
                    )
                }
                NodeRecord::Until { a, b, phi, psi } => {
                    until_robustness(
                        &self.windows[*phi],
                        &self.windows[*psi],
                        t_settled, *a, *b,
                    )
                }
            };
            self.windows[id].push(t_settled, value);

            // Root's freshly pushed value is the verdict.
            if id + 1 == self.nodes.len() {
                self.latest_verdict = Some(Verdict {
                    t: t_settled,
                    robustness: value,
                });
            }
        }
    }

    fn verdict(&self) -> Option<Verdict> {
        self.latest_verdict
    }
}

// ─── Window-lookup helpers ───────────────────────────────────────────

/// Find the value in `w` whose timestamp is closest to `t`. Returns
/// `NaN` if the window is empty. With per-observe pushes the closest
/// timestamp is normally an exact match (within `TIME_EPS`).
fn value_at(w: &SlidingWindow, t: Timestamp) -> f64 {
    let mut best: Option<(f64, f64)> = None;
    for (t_w, v) in w.iter() {
        let dist = (t_w - t).abs();
        match best {
            None => best = Some((dist, v)),
            Some((bd, _)) if dist < bd => best = Some((dist, v)),
            _ => {}
        }
    }
    best.map(|(_, v)| v).unwrap_or(f64::NAN)
}

/// Compute the supremum or infimum of `w`'s values whose timestamps
/// fall in `[lo, hi]` (inclusive, modulo `TIME_EPS`). Returns the
/// neutral element (`NEG_INFINITY` for sup, `INFINITY` for inf) if the
/// range contains no samples.
fn sup_or_inf_in_range(
    w: &SlidingWindow, lo: Timestamp, hi: Timestamp, sup: bool,
) -> f64 {
    let neutral = if sup { f64::NEG_INFINITY } else { f64::INFINITY };
    let mut acc = neutral;
    for (t, v) in w.iter() {
        if t >= lo - TIME_EPS && t <= hi + TIME_EPS {
            acc = if sup { acc.max(v) } else { acc.min(v) };
        }
    }
    acc
}

/// Until robustness:
///
/// `ρ(φ U_{[a,b]} ψ, t) = sup_{t' ∈ [t+a, t+b]} min(ρ(ψ, t'),
///                          inf_{t'' ∈ [t, t']} ρ(φ, t''))`
fn until_robustness(
    phi_win: &SlidingWindow, psi_win: &SlidingWindow,
    t: Timestamp, a: f64, b: f64,
) -> f64 {
    let mut acc = f64::NEG_INFINITY;
    for (t_prime, psi_v) in psi_win.iter() {
        if t_prime < t + a - TIME_EPS || t_prime > t + b + TIME_EPS { continue; }
        // inf over phi in [t, t']
        let phi_inf = sup_or_inf_in_range(phi_win, t, t_prime, /*sup=*/ false);
        acc = acc.max(psi_v.min(phi_inf));
    }
    acc
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::stl::*;
    use crate::predicate::{Dependencies, HypergraphPredicate, HypergraphState, Sign};
    use crate::trace::Sample;

    /// Minimal test fixture: a single-vertex hypergraph carrying one
    /// scalar attribute, mutable across samples.
    struct ScalarState { x: f64 }
    impl HypergraphState for ScalarState {
        type VertexId = u32; type EdgeId = u32; type TypeId = u32; type Attr = f64;
        fn vertices(&self) -> Box<dyn Iterator<Item = u32> + '_> { Box::new(std::iter::once(0)) }
        fn edges(&self) -> Box<dyn Iterator<Item = u32> + '_> { Box::new(std::iter::empty()) }
        fn incidences(&self, _e: u32) -> Box<dyn Iterator<Item = (u32, Sign)> + '_> {
            Box::new(std::iter::empty())
        }
        fn vertex_type(&self, _v: u32) -> u32 { 0 }
        fn edge_type(&self, _e: u32) -> u32 { 0 }
        fn inherits(&self, _t: u32, _b: u32) -> bool { false }
        fn vertex_attr(&self, _v: u32, _k: &str) -> Option<&f64> { Some(&self.x) }
        fn edge_attr(&self, _e: u32, _k: &str) -> Option<&f64> { None }
    }

    /// Predicate `x > 0`: robustness = x.
    struct Positive;
    impl HypergraphPredicate<ScalarState> for Positive {
        fn eval(&self, h: &ScalarState) -> bool { h.x > 0.0 }
        fn robustness(&self, h: &ScalarState) -> f64 { h.x }
        fn dependencies(&self, _h: &ScalarState) -> Dependencies<ScalarState> {
            Dependencies::global()
        }
    }

    fn p() -> Stl<Arc<dyn HypergraphPredicate<ScalarState>>> {
        Stl::Pred(Arc::new(Positive))
    }

    #[test]
    fn atomic_predicate_matches_robustness() {
        let mut m: StlMonitor<ScalarState, dyn HypergraphPredicate<ScalarState>> =
            StlMonitor::new(p()).unwrap();
        m.observe(Sample { state: ScalarState { x: 0.5 }, t: 0.0 });
        let v = m.verdict().expect("verdict ready");
        assert_eq!(v.t, 0.0);
        assert!((v.robustness - 0.5).abs() < 1e-12);
    }

    #[test]
    fn always_bounded_settles_after_horizon() {
        // G_{[0,1]}(x > 0) on a trace where x flips negative at t=0.5.
        let phi = always_bounded(0.0, 1.0, p());
        let mut m: StlMonitor<ScalarState, dyn HypergraphPredicate<ScalarState>> =
            StlMonitor::new(phi).unwrap();
        // Before horizon (t<1.0), no verdict.
        m.observe(Sample { state: ScalarState { x: 1.0 }, t: 0.0 });
        m.observe(Sample { state: ScalarState { x: 1.0 }, t: 0.4 });
        assert!(m.verdict().is_none());
        // After horizon, verdict at settled time.
        m.observe(Sample { state: ScalarState { x: -0.5 }, t: 0.5 });
        m.observe(Sample { state: ScalarState { x: 1.0 }, t: 0.9 });
        m.observe(Sample { state: ScalarState { x: 1.0 }, t: 1.2 });
        // At τ_obs=1.2, settled time = 0.2. Window [0.2, 1.2].
        // Values seen: 1.0@0.0, 1.0@0.4, -0.5@0.5, 1.0@0.9, 1.0@1.2.
        // Filter to [0.2, 1.2]: 1.0@0.4, -0.5@0.5, 1.0@0.9, 1.0@1.2. inf = -0.5.
        let v = m.verdict().expect("verdict ready");
        assert!((v.robustness - (-0.5)).abs() < 1e-12,
                "expected robustness=-0.5, got {}", v.robustness);
    }

    #[test]
    fn eventually_bounded_finds_the_peak() {
        // F_{[0,1]}(x > 0) — robustness is the max of x over [t, t+1].
        let phi = eventually_bounded(0.0, 1.0, p());
        let mut m: StlMonitor<ScalarState, dyn HypergraphPredicate<ScalarState>> =
            StlMonitor::new(phi).unwrap();
        m.observe(Sample { state: ScalarState { x: -1.0 }, t: 0.0 });
        m.observe(Sample { state: ScalarState { x: -2.0 }, t: 0.5 });
        m.observe(Sample { state: ScalarState { x:  3.0 }, t: 1.0 });
        // At τ_obs=1.0, settled time=0.0. Window [0, 1]. Values: -1, -2, 3. sup=3.
        let v = m.verdict().expect("verdict ready");
        assert!((v.robustness - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_unbounded_horizon() {
        let phi: Stl<Arc<dyn HypergraphPredicate<ScalarState>>> = Stl::Eventually {
            a: 0.0, b: f64::INFINITY, phi: Box::new(p()),
        };
        // Avoid `.unwrap_err()` so `StlMonitor` doesn't need `Debug`.
        assert!(matches!(
            StlMonitor::<ScalarState, _>::new(phi),
            Err(MonitorError::UnboundedHorizon)
        ));
    }

    #[test]
    fn nested_combinator_settles_correctly() {
        // not(F_{[0,1]} (x > 0))  — robustness = -sup_{t' in [t, t+1]} x(t').
        let phi = not(eventually_bounded(0.0, 1.0, p()));
        let mut m: StlMonitor<ScalarState, dyn HypergraphPredicate<ScalarState>> =
            StlMonitor::new(phi).unwrap();
        m.observe(Sample { state: ScalarState { x:  1.0 }, t: 0.0 });
        m.observe(Sample { state: ScalarState { x: -2.0 }, t: 0.5 });
        m.observe(Sample { state: ScalarState { x: -0.5 }, t: 1.0 });
        // sup over [0,1] = 1.0; not = -1.0.
        let v = m.verdict().expect("verdict ready");
        assert!((v.robustness - (-1.0)).abs() < 1e-12);
    }
}
