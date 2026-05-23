//! Online monitor trait and the STL implementation.

pub mod stl;

use crate::trace::{Sample, Timestamp};

/// Verdict emitted by a monitor at the trailing edge of its window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    /// The timestamp at which this verdict is reported (i.e., the time
    /// whose robustness the verdict describes — typically trailing the
    /// most recent sample by the formula horizon).
    pub t: Timestamp,
    /// Quantitative robustness. Positive = property holds with margin,
    /// negative = property violated with margin, zero = boundary.
    pub robustness: f64,
}

impl Verdict {
    /// Boolean reading of the verdict (true iff robustness > 0).
    pub fn satisfied(&self) -> bool { self.robustness > 0.0 }
}

/// An online monitor.
pub trait Monitor<H> {
    /// Feed one observation to the monitor.
    fn observe(&mut self, sample: Sample<H>);

    /// Latest settled verdict, if the window has filled enough to emit
    /// one. `None` while the monitor is still warming up.
    fn verdict(&self) -> Option<Verdict>;
}
