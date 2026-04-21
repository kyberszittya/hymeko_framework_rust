//! Trace primitives: samples and timestamps.
//!
//! A trace is a sequence of `(state, timestamp)` pairs with strictly
//! increasing timestamps. The monitor consumes samples one at a time via
//! [`Monitor::observe`](crate::Monitor::observe). It does not retain
//! states across observe calls; it retains only predicate verdicts and
//! robustness values in a bounded sliding window.

/// Timestamp in seconds from an epoch defined by the caller.
pub type Timestamp = f64;

/// One observation in a hypergraph trace.
#[derive(Debug)]
pub struct Sample<H> {
    /// The hypergraph state at `t`.
    pub state: H,
    /// Timestamp of this sample.
    pub t: Timestamp,
}

impl<H> Sample<H> {
    /// Construct a sample.
    pub fn new(state: H, t: Timestamp) -> Self {
        Self { state, t }
    }
}
