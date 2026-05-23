//! Sliding-window buffer.
//!
//! Used by the STL monitor to retain the robustness values of each
//! subformula across the time horizon required by its enclosing
//! temporal operator. The buffer supports amortised O(1) insert and
//! O(k) sup/inf over the active window (where k is the window length).
//!
//! A more elaborate monotonic-queue implementation (Lemire 2006) would
//! give O(1) sup/inf; left as a performance-tuning TODO.

use std::collections::VecDeque;

use crate::trace::Timestamp;

/// A time-indexed ring buffer. Values older than `horizon` behind the
/// most recent timestamp are evicted on insert.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    horizon: Timestamp,
    samples: VecDeque<(Timestamp, f64)>,
}

impl SlidingWindow {
    /// Construct a window with the given horizon (in the same unit as
    /// the timestamps passed to [`SlidingWindow::push`]).
    pub fn new(horizon: Timestamp) -> Self {
        debug_assert!(horizon >= 0.0 && horizon.is_finite());
        Self { horizon, samples: VecDeque::new() }
    }

    /// Insert a new sample and evict anything older than
    /// `t - self.horizon`.
    pub fn push(&mut self, t: Timestamp, value: f64) {
        let cutoff = t - self.horizon;
        while let Some(&(old_t, _)) = self.samples.front() {
            if old_t < cutoff { self.samples.pop_front(); } else { break; }
        }
        self.samples.push_back((t, value));
    }

    /// Supremum of values in the window. `-∞` if empty.
    pub fn sup(&self) -> f64 {
        self.samples.iter().map(|&(_, v)| v).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Infimum of values in the window. `+∞` if empty.
    pub fn inf(&self) -> f64 {
        self.samples.iter().map(|&(_, v)| v).fold(f64::INFINITY, f64::min)
    }

    /// Number of samples currently retained.
    pub fn len(&self) -> usize { self.samples.len() }

    /// Iterate over retained `(timestamp, value)` pairs, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = (Timestamp, f64)> + '_ {
        self.samples.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_old_samples() {
        let mut w = SlidingWindow::new(1.0);
        w.push(0.0, 1.0);
        w.push(0.5, 2.0);
        w.push(1.2, 3.0);  // evicts 0.0
        assert_eq!(w.len(), 2);
        assert_eq!(w.sup(), 3.0);
        assert_eq!(w.inf(), 2.0);
    }

    #[test]
    fn sup_inf_empty() {
        let w = SlidingWindow::new(1.0);
        assert_eq!(w.sup(), f64::NEG_INFINITY);
        assert_eq!(w.inf(), f64::INFINITY);
    }
}
