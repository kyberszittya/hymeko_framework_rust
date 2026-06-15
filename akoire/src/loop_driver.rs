//! The closed cognitive loop — wires the three SysML flows into one driver.
//!
//! synthesis flow → gatekeeper → {success loop | error loop}, bounded by a hard
//! `max_rounds` cap so an agent that never produces parseable HyQL cannot spin
//! forever.

use crate::context::{CognitiveContext, Goal, Intent, Kyosei, Objectives};
use crate::engine::{ErrorFeedback, EvalOutcome, HymekoEngine};
use crate::synthesize::CognitiveSynthesizer;

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// Objectives satisfied.
    Converged,
    /// Hit the `max_rounds` cap without converging.
    CapReached,
    /// The synthesizer reported it had nothing left to propose.
    Exhausted,
}

/// Structured outcome of a loop run.
#[derive(Debug, Clone)]
pub struct LoopReport {
    /// Terminal condition.
    pub termination: Termination,
    /// Rounds actually executed (a round = one synthesize attempt resolved).
    pub rounds: usize,
    /// Refinements accepted by the gatekeeper.
    pub accepted: usize,
    /// Refinements rejected by the gatekeeper.
    pub rejected: usize,
}

/// Drives [`CognitiveSynthesizer`] against [`HymekoEngine`] until the objectives
/// are met, the synthesizer is exhausted, or the cap is reached.
#[derive(Debug)]
pub struct CognitiveLoop<S: CognitiveSynthesizer> {
    engine: HymekoEngine,
    synth: S,
    max_rounds: usize,
}

impl<S: CognitiveSynthesizer> CognitiveLoop<S> {
    /// Construct a loop.
    ///
    /// # Preconditions
    /// `max_rounds >= 1` (a zero-round loop can never make progress).
    #[must_use]
    pub fn new(engine: HymekoEngine, synth: S, max_rounds: usize) -> Self {
        debug_assert!(max_rounds >= 1, "max_rounds must be >= 1");
        Self {
            engine,
            synth,
            max_rounds,
        }
    }

    /// Borrow the engine (e.g. to inspect the final ambience after a run).
    #[must_use]
    pub fn engine(&self) -> &HymekoEngine {
        &self.engine
    }

    /// Run the closed loop.
    ///
    /// `objectives` is both a synthesizer input and the termination predicate
    /// (it implements [`Goal`]), so there is a single source of truth for the
    /// target.
    ///
    /// # Postconditions
    /// - `report.rounds <= max_rounds` (the cap is never exceeded).
    /// - `report.termination == Converged` ⇒ `objectives.satisfied(ambience)`.
    pub fn run(&mut self, intent: &Intent, objectives: &Objectives, kyosei: &Kyosei) -> LoopReport {
        let mut accepted = 0usize;
        let mut rejected = 0usize;
        let mut last_error: Option<ErrorFeedback> = None;

        for round in 1..=self.max_rounds {
            // Immutable view for synthesis. The borrow of `self.engine` ends at
            // the `synthesize` call (NLL), freeing the later `&mut self.engine`.
            let refinement = {
                let ctx = CognitiveContext {
                    ambience: self.engine.ambience(),
                    intent,
                    objectives,
                    kyosei,
                    last_error: last_error.as_ref(),
                };
                match self.synth.synthesize(&ctx) {
                    Some(r) => r,
                    None => {
                        return LoopReport {
                            termination: Termination::Exhausted,
                            rounds: round - 1,
                            accepted,
                            rejected,
                        };
                    }
                }
            };

            match self.engine.evaluate(&refinement) {
                EvalOutcome::Accepted { .. } => {
                    accepted += 1;
                    last_error = None; // success loop: clear stale feedback
                    if objectives.satisfied(self.engine.ambience()) {
                        return LoopReport {
                            termination: Termination::Converged,
                            rounds: round,
                            accepted,
                            rejected,
                        };
                    }
                }
                EvalOutcome::Rejected(feedback) => {
                    rejected += 1;
                    last_error = Some(feedback); // error loop: feed next round
                }
            }
        }

        LoopReport {
            termination: Termination::CapReached,
            rounds: self.max_rounds,
            accepted,
            rejected,
        }
    }
}
