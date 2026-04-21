//! Online STL monitor — sliding-window DP over the bounded-horizon
//! fragment.
//!
//! ## Algorithm (paper §4)
//!
//! 1. Parse the formula into a tree. Each node has a unique `NodeId`.
//! 2. Compute per-node horizon `H_node` (required look-ahead for nodes
//!    above it). Pre-allocate one [`SlidingWindow`] per temporal node,
//!    sized to its horizon.
//! 3. On each sample `(H_i, t_i)`:
//!    a. Evaluate every atomic predicate; push into its per-predicate
//!       window.
//!    b. Bottom-up, compute pointwise robustness of each non-temporal
//!       subformula (and, or, not) at `t_i`.
//!    c. For each temporal subformula `G_{[a,b]} φ`,
//!       `F_{[a,b]} φ`, `φ U_{[a,b]} ψ`, compute robustness at the
//!       settled time `t_i - b` using the subformula's sliding window.
//! 4. Emit verdict at the root formula's settled time.
//!
//! ## Status
//!
//! **SKELETON**. The structure here is the finished target; the bodies
//! marked `todo!()` are the work items. See `SPEC.md` §"Deliverables"
//! and §"Pitfalls to Watch".

use std::sync::Arc;

use crate::formula::stl::Stl;
use crate::monitor::{Monitor, Verdict};
use crate::predicate::{HypergraphPredicate, HypergraphState};
use crate::trace::{Sample, Timestamp};
use crate::window::SlidingWindow;
use crate::MonitorError;

/// Online STL monitor for a bounded-horizon formula.
#[derive(Debug)]
pub struct StlMonitor<H: HypergraphState, P: ?Sized + HypergraphPredicate<H>> {
    /// The formula, with predicates boxed so the tree is heterogeneous.
    formula: Stl<Arc<P>>,

    /// Per-temporal-subformula window. Indexed by a flattened tree id.
    /// Computed at construction from [`Stl::horizon`].
    windows: Vec<SlidingWindow>,

    /// Most recent observed timestamp; used to enforce monotonic time.
    last_t: Option<Timestamp>,

    /// Latest verdict ready to emit, if any.
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
    /// Returns [`MonitorError::UnboundedHorizon`] if the formula contains
    /// any unbounded temporal operator (monitoring requires bounded
    /// look-ahead to preserve the bounded-memory guarantee).
    pub fn new(formula: Stl<Arc<P>>) -> Result<Self, MonitorError> {
        if !formula.is_bounded_horizon() {
            return Err(MonitorError::UnboundedHorizon);
        }
        let windows = allocate_windows(&formula);
        Ok(Self {
            formula,
            windows,
            last_t: None,
            latest_verdict: None,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The horizon (look-ahead) of the root formula.
    pub fn horizon(&self) -> Timestamp {
        self.formula.horizon()
    }
}

impl<H, P> Monitor<H> for StlMonitor<H, P>
where
    H: HypergraphState,
    P: HypergraphPredicate<H> + ?Sized,
{
    fn observe(&mut self, sample: Sample<H>) {
        // Monotonic-time check — panic in debug, silently drop in release.
        // (Or return error — but `observe` currently returns unit;
        // decide whether to upgrade the signature to Result<>.)
        if let Some(prev) = self.last_t {
            debug_assert!(
                sample.t > prev,
                "non-monotonic timestamp: prev={prev}, new={}", sample.t
            );
        }
        self.last_t = Some(sample.t);

        // (1) Evaluate atomic predicates, push into per-predicate windows.
        // (2) Bottom-up pointwise combinators for non-temporal nodes.
        // (3) Sliding sup/inf for temporal nodes at settled time.
        // (4) Update self.latest_verdict if root has settled.
        //
        // This is the main work. See SPEC.md §"Pitfalls to Watch" —
        // especially the floating-point epsilon for boundary cases and
        // the incremental-evaluation correctness note.
        todo!("implement sliding-window DP over self.formula and self.windows")
    }

    fn verdict(&self) -> Option<Verdict> {
        self.latest_verdict
    }
}

/// Walk the formula and pre-allocate one [`SlidingWindow`] per temporal
/// subformula, sized to the formula's required look-ahead.
fn allocate_windows<P: ?Sized>(_phi: &Stl<Arc<P>>) -> Vec<SlidingWindow> {
    // Walk the tree, collect (NodeId, horizon) for every temporal node,
    // allocate a window per node. Index by the same NodeId used by the
    // observe() evaluation pass.
    todo!("walk phi, emit one SlidingWindow per temporal node")
}
