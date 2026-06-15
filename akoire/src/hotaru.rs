//! HOTARU planning over HIVE-delta space.
//!
//! HOTARU does not commit structure directly. It proposes named deltas over the
//! current HIVE state, and this adapter lowers each delta to a HyQL refinement
//! that must still pass through AKOIRE's gatekeeper.

use std::collections::VecDeque;

use crate::context::CognitiveContext;
use crate::synthesize::{CognitiveSynthesizer, Refinement};

/// A named candidate mutation in HIVE-delta space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiveDelta {
    name: String,
    op: HiveDeltaOp,
}

impl HiveDelta {
    /// Replace the whole rendered HIVE state with `source`.
    #[must_use]
    pub fn replace_source(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            op: HiveDeltaOp::ReplaceSource(source.into()),
        }
    }

    /// Append a HyQL source fragment to the current rendered HIVE state.
    #[must_use]
    pub fn append_source(name: impl Into<String>, fragment: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            op: HiveDeltaOp::AppendSource(fragment.into()),
        }
    }

    /// Stable planner-facing delta label.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    fn lower(&self, current_source: &str) -> Refinement {
        match &self.op {
            HiveDeltaOp::ReplaceSource(source) => Refinement(source.clone()),
            HiveDeltaOp::AppendSource(fragment) => {
                if current_source.trim().is_empty() {
                    Refinement(fragment.clone())
                } else {
                    Refinement(format!("{}\n{}", current_source.trim_end(), fragment))
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HiveDeltaOp {
    ReplaceSource(String),
    AppendSource(String),
}

/// Chooses the next HOTARU delta from the current cognitive context.
pub trait HotaruPlanner {
    /// Return the next candidate delta, or `None` when planning is exhausted.
    fn next_delta(&mut self, ctx: &CognitiveContext<'_>) -> Option<HiveDelta>;
}

/// Deterministic HOTARU stand-in for tests, demos, and benches.
#[derive(Debug, Clone, Default)]
pub struct ScriptedHotaru {
    queue: VecDeque<HiveDelta>,
}

impl ScriptedHotaru {
    /// Build a planner from a fixed delta sequence.
    pub fn new<I>(deltas: I) -> Self
    where
        I: IntoIterator<Item = HiveDelta>,
    {
        Self {
            queue: deltas.into_iter().collect(),
        }
    }

    /// Number of candidate deltas still queued.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }
}

impl HotaruPlanner for ScriptedHotaru {
    fn next_delta(&mut self, _ctx: &CognitiveContext<'_>) -> Option<HiveDelta> {
        self.queue.pop_front()
    }
}

/// AKOIRE synthesizer adapter for a HOTARU planner.
#[derive(Debug, Clone)]
pub struct HotaruSynthesizer<P: HotaruPlanner> {
    planner: P,
    last_delta_name: Option<String>,
}

impl<P: HotaruPlanner> HotaruSynthesizer<P> {
    /// Wrap a planner so it can drive [`crate::CognitiveLoop`].
    #[must_use]
    pub fn new(planner: P) -> Self {
        Self {
            planner,
            last_delta_name: None,
        }
    }

    /// Name of the most recently lowered delta, if any.
    #[must_use]
    pub fn last_delta_name(&self) -> Option<&str> {
        self.last_delta_name.as_deref()
    }
}

impl<P: HotaruPlanner> CognitiveSynthesizer for HotaruSynthesizer<P> {
    fn synthesize(&mut self, ctx: &CognitiveContext<'_>) -> Option<Refinement> {
        let delta = self.planner.next_delta(ctx)?;
        self.last_delta_name = Some(delta.name().to_string());
        Some(delta.lower(ctx.ambience.source()))
    }
}
