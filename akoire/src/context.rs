//! Cognitive inputs and current-state types for the AKOIRE layer.
//!
//! These mirror the input ports of `CognitiveOperatingSystem` in
//! `architecture/akoire/overview.sysml`: `externalIntent`, `systemObjectives`,
//! `physicsKyosei`, and `currentAmbience`. [`Ambience`] additionally plays the
//! role of HyMeKo's frozen `TensorCsr` state — here represented by the owned,
//! always-parseable source plus a monotonic generation counter (the real system
//! delegates the COO→CSR commit to `hymeko_core`).

use crate::engine::ErrorFeedback;

/// High-level task description handed to the cognitive synthesizer.
///
/// Maps to the `externalIntent` port. Opaque to the loop; only the
/// [`crate::synthesize::CognitiveSynthesizer`] interprets it.
#[derive(Debug, Clone)]
pub struct Intent(pub String);

impl Intent {
    /// Borrow the intent text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.0
    }
}

/// Target topology the refinement must reach.
///
/// Maps to the `systemObjectives` port. Also serves as the loop's termination
/// predicate via its [`Goal`] implementation, so there is a single source of
/// truth for "what success means" (no duplicate goal object).
#[derive(Debug, Clone, Default)]
pub struct Objectives {
    /// Edge names that must be present in the accepted description for the
    /// objectives to count as met. Empty ⇒ "any non-empty accepted state".
    pub required_edges: Vec<String>,
}

impl Objectives {
    /// True when every required edge name appears in `edge_names`.
    ///
    /// The edge-set core of [`Goal::satisfied`], factored out (single source of truth, CLAUDE.md
    /// §6.1) so the search planner can score a *candidate's* edge names without constructing an
    /// [`Ambience`]. With no required edges this is vacuously true — [`Goal::satisfied`] adds the
    /// "state actually committed" condition on top.
    #[must_use]
    pub fn satisfied_edges(&self, edge_names: &[String]) -> bool {
        self.required_edges
            .iter()
            .all(|req| edge_names.iter().any(|e| e == req))
    }

    /// How many required edges are still missing from `edge_names`.
    ///
    /// The admissible A\* heuristic for HOTARU's delta search: each menu delta adds at most one
    /// required edge and every delta costs one, so this never overestimates the remaining steps.
    #[must_use]
    pub fn missing_edges(&self, edge_names: &[String]) -> usize {
        self.required_edges
            .iter()
            .filter(|req| !edge_names.iter().any(|e| e == *req))
            .count()
    }
}

/// Physical limits the synthesizer must respect (`physicsKyosei` port).
///
/// Carried into the [`CognitiveContext`] so an agent-backed synthesizer can
/// honour them. The scripted reference synthesizer ignores it.
#[derive(Debug, Clone)]
pub struct Kyosei {
    /// Maximum hyperedge arity the physical target admits.
    pub max_arity: usize,
}

impl Default for Kyosei {
    fn default() -> Self {
        // No constraint by default: usize::MAX is the "unbounded arity" sentinel.
        Self {
            max_arity: usize::MAX,
        }
    }
}

/// The current mathematical reality — HyMeKo's frozen state, as seen by AKOIRE.
///
/// # Invariants
/// - `source` always parses under `parser::parse_description` once `generation > 0`
///   (only accepted refinements are ever committed).
/// - `generation` is monotonically non-decreasing and increments by exactly one
///   per accepted refinement.
/// - `edge_names` is the set of edge names extracted from `source` at commit time.
#[derive(Debug, Clone, Default)]
pub struct Ambience {
    source: String,
    generation: u64,
    edge_names: Vec<String>,
}

impl Ambience {
    /// The empty ambience: no committed state, generation 0.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// The committed HyQL source (the last accepted refinement).
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The monotonic generation counter (number of accepted refinements).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Edge names present in the committed state.
    #[must_use]
    pub fn edge_names(&self) -> &[String] {
        &self.edge_names
    }

    /// Commit an accepted refinement as the new state.
    ///
    /// # Preconditions
    /// `source` must already have been validated by the engine
    /// (`parser::parse_description` returned `Ok`), and `edge_names` must be the
    /// edge names extracted from that same parse. Crate-internal so the
    /// precondition cannot be violated from outside.
    ///
    /// # Postconditions
    /// `generation` increases by one; `source` and `edge_names` reflect the new
    /// state.
    pub(crate) fn commit(&mut self, source: String, edge_names: Vec<String>) {
        self.source = source;
        self.edge_names = edge_names;
        self.generation += 1;
    }
}

/// The bundled input the synthesizer reads each round.
///
/// Mirrors the `synthesizeRefinement` action's inputs, plus `last_error` which
/// carries the error-loop feedback (`hymeko.errorFeedback → akoire.currentAmbience`).
pub struct CognitiveContext<'a> {
    /// Current frozen state (`currentAmbience`).
    pub ambience: &'a Ambience,
    /// Task (`externalIntent`).
    pub intent: &'a Intent,
    /// Target topology (`systemObjectives`).
    pub objectives: &'a Objectives,
    /// Physical limits (`physicsKyosei`).
    pub kyosei: &'a Kyosei,
    /// Feedback from the previous round's syntax failure, if any.
    pub last_error: Option<&'a ErrorFeedback>,
}

/// Termination predicate over the accepted state (Strategy).
///
/// One canonical implementation ([`Objectives`]) ships today; the trait exists
/// so future criteria (node-count, balance, entropy thresholds — cf.
/// `hymeko_query::entropy`) can be swapped in without touching the loop.
pub trait Goal {
    /// Does the current ambience satisfy this goal?
    fn satisfied(&self, ambience: &Ambience) -> bool;
}

impl Goal for Objectives {
    /// Satisfied when every required edge name is present in the accepted state.
    /// With no required edges, satisfied as soon as any state is committed.
    fn satisfied(&self, ambience: &Ambience) -> bool {
        if self.required_edges.is_empty() {
            return ambience.generation() > 0;
        }
        self.satisfied_edges(ambience.edge_names())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wants(edges: &[&str]) -> Objectives {
        Objectives {
            required_edges: edges.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn satisfied_edges_normal_and_negative() {
        let obj = wants(&["joint", "gripper"]);
        let present = [
            "joint".to_string(),
            "gripper".to_string(),
            "extra".to_string(),
        ];
        assert!(obj.satisfied_edges(&present));
        assert!(!obj.satisfied_edges(&["joint".to_string()])); // gripper missing
        assert!(!obj.satisfied_edges(&[])); // both missing
    }

    #[test]
    fn satisfied_edges_empty_required_is_vacuous() {
        // The edge-set core is vacuously true with no requirements; `Goal::satisfied` adds the
        // "state committed" condition on top (tested via the loop). Boundary case.
        assert!(wants(&[]).satisfied_edges(&[]));
    }

    #[test]
    fn missing_edges_counts_absent_requirements() {
        let obj = wants(&["joint", "gripper"]);
        assert_eq!(obj.missing_edges(&[]), 2);
        assert_eq!(obj.missing_edges(&["joint".to_string()]), 1);
        assert_eq!(
            obj.missing_edges(&["joint".to_string(), "gripper".to_string()]),
            0
        );
        assert_eq!(wants(&[]).missing_edges(&[]), 0); // no requirements ⇒ nothing missing
    }
}
