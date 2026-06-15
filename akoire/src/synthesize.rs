//! The cognitive seam: where AKOIRE (the agent) produces HyQL refinements.
//!
//! [`CognitiveSynthesizer`] is the Strategy interface for the `synthesizeRefinement`
//! action. The loop is agnostic to *how* a refinement is produced:
//!
//! - [`ScriptedSynthesizer`] replays a fixed queue — deterministic, used by tests
//!   and benchmarks, and the stand-in for "the agent already decided its moves".
//! - An LLM-backed synthesizer (sketched below) is the production binding: it
//!   reads `ctx.ambience.source()`, `ctx.intent`, `ctx.objectives`, `ctx.kyosei`
//!   and `ctx.last_error`, and returns the next HyQL string. Because the loop
//!   only depends on the trait, dropping such an impl in requires no loop change.
//!
//! ```ignore
//! struct LlmSynthesizer<C> { client: C }
//! impl<C: ChatModel> CognitiveSynthesizer for LlmSynthesizer<C> {
//!     fn synthesize(&mut self, ctx: &CognitiveContext<'_>) -> Option<Refinement> {
//!         let prompt = build_prompt(ctx);          // intent + ambience + last_error
//!         self.client.complete(&prompt).ok().map(Refinement)
//!     }
//! }
//! ```

use std::collections::VecDeque;

use crate::context::CognitiveContext;

/// A proposed structural mutation: one HyQL description string.
///
/// This is the value crossing the Layer-2 → Layer-1 boundary
/// (`akoire.proposedRefinement → hymeko.incomingRefinement`).
#[derive(Debug, Clone)]
pub struct Refinement(pub String);

/// Produces the next refinement given the current cognitive context.
///
/// # Contract
/// - Returning `Some(r)` offers `r` to the gatekeeper.
/// - Returning `None` signals the synthesizer has nothing more to propose; the
///   loop then terminates with [`crate::Termination::Exhausted`]. A synthesizer
///   must not panic to signal exhaustion.
pub trait CognitiveSynthesizer {
    /// Synthesize the next refinement, or `None` if exhausted.
    fn synthesize(&mut self, ctx: &CognitiveContext<'_>) -> Option<Refinement>;
}

/// Replays a fixed sequence of refinements, in order, then reports exhaustion.
#[derive(Debug, Clone, Default)]
pub struct ScriptedSynthesizer {
    queue: VecDeque<Refinement>,
}

impl ScriptedSynthesizer {
    /// Build from any iterable of HyQL source strings.
    pub fn new<I, S>(refinements: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            queue: refinements
                .into_iter()
                .map(|s| Refinement(s.into()))
                .collect(),
        }
    }

    /// Number of refinements still queued.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }
}

impl CognitiveSynthesizer for ScriptedSynthesizer {
    /// Pops the next queued refinement. Ignores the context — the script is
    /// fixed in advance.
    fn synthesize(&mut self, _ctx: &CognitiveContext<'_>) -> Option<Refinement> {
        self.queue.pop_front()
    }
}
