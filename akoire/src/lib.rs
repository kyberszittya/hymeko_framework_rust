//! # AKOIRE — Layer-2 Cognitive OS harness
//!
//! AKOIRE plays the *Cognitive Operating System* of
//! `architecture/akoire/overview.sysml`: it turns probabilistic intent into a
//! deterministic HyQL refinement string, which the **HyMeKo** Layer-1 engine
//! (the CORE `parser` crate) gate-keeps. Three flows close the system:
//!
//! 1. **synthesis** — [`CognitiveSynthesizer`] → [`HymekoEngine`];
//! 2. **success loop** — accepted source → [`Ambience`] → next context;
//! 3. **error loop** — [`ErrorFeedback`] → next context (self-correction).
//!
//! [`CognitiveLoop`] drives the three flows under a hard `max_rounds` cap. The
//! agent (an LLM, or the test-only [`ScriptedSynthesizer`]) supplies the
//! refinements; the engine decides what becomes reality.
//!
//! This crate is **non-core**: it only depends on `parser`'s public API and
//! never edits a `CORE.YAML`-locked file.

pub mod context;
pub mod engine;
pub mod hotaru;
pub mod loop_driver;
pub mod synthesize;

pub use context::{Ambience, CognitiveContext, Goal, Intent, Kyosei, Objectives};
pub use engine::{ErrorFeedback, EvalOutcome, HymekoEngine};
pub use hotaru::{HiveDelta, HotaruPlanner, HotaruSynthesizer, ScriptedHotaru};
pub use loop_driver::{CognitiveLoop, LoopReport, Termination};
pub use synthesize::{CognitiveSynthesizer, Refinement, ScriptedSynthesizer};

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid HyQL description with one edge named `joint`.
    const VALID_WITH_JOINT: &str = "RobotArm {\n  base;\n  link1;\n}\n@joint : base, link1 { }";

    /// A valid HyQL description with no edges.
    const VALID_NO_EDGE: &str = "RobotArm {\n  base;\n}";

    /// Malformed: header item `base` is missing its terminating `;`.
    const MALFORMED: &str = "RobotArm { base }";

    fn objectives_joint() -> Objectives {
        Objectives {
            required_edges: vec!["joint".to_string()],
        }
    }

    // ---- engine (gatekeeper) -------------------------------------------

    #[test]
    fn engine_accepts_valid() {
        let mut engine = HymekoEngine::new();
        let outcome = engine.evaluate(&Refinement(VALID_WITH_JOINT.to_string()));
        match outcome {
            EvalOutcome::Accepted { generation } => assert_eq!(generation, 1),
            EvalOutcome::Rejected(fb) => panic!("expected Accepted, got {fb:?}"),
        }
        assert_eq!(engine.ambience().generation(), 1);
        assert!(engine.ambience().edge_names().iter().any(|e| e == "joint"));
    }

    #[test]
    fn engine_rejects_malformed() {
        let mut engine = HymekoEngine::new();
        let outcome = engine.evaluate(&Refinement(MALFORMED.to_string()));
        match outcome {
            EvalOutcome::Rejected(fb) => assert!(!fb.message.is_empty()),
            EvalOutcome::Accepted { .. } => panic!("expected Rejected"),
        }
        // Rejection must not mutate state.
        assert_eq!(engine.ambience().generation(), 0);
    }

    // ---- synthesizer ---------------------------------------------------

    #[test]
    fn scripted_exhausts() {
        let ambience = Ambience::empty();
        let intent = Intent("build an arm".to_string());
        let objectives = Objectives::default();
        let kyosei = Kyosei::default();
        let ctx = CognitiveContext {
            ambience: &ambience,
            intent: &intent,
            objectives: &objectives,
            kyosei: &kyosei,
            last_error: None,
        };
        let mut synth = ScriptedSynthesizer::new(["a", "b"]);
        assert_eq!(synth.remaining(), 2);
        assert!(synth.synthesize(&ctx).is_some());
        assert!(synth.synthesize(&ctx).is_some());
        assert!(synth.synthesize(&ctx).is_none());
    }

    // ---- loop ----------------------------------------------------------

    #[test]
    fn loop_converges_first_try() {
        let synth = ScriptedSynthesizer::new([VALID_WITH_JOINT]);
        let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, 5);
        let report = cog.run(
            &Intent("build an arm".to_string()),
            &objectives_joint(),
            &Kyosei::default(),
        );
        assert_eq!(report.termination, Termination::Converged);
        assert_eq!(report.rounds, 1);
        assert_eq!(report.accepted, 1);
        assert_eq!(report.rejected, 0);
    }

    #[test]
    fn loop_self_corrects() {
        // Error loop: first proposal fails, second (corrected) succeeds.
        let synth = ScriptedSynthesizer::new([MALFORMED, VALID_WITH_JOINT]);
        let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, 5);
        let report = cog.run(
            &Intent("build an arm".to_string()),
            &objectives_joint(),
            &Kyosei::default(),
        );
        assert_eq!(report.termination, Termination::Converged);
        assert_eq!(report.rounds, 2);
        assert_eq!(report.accepted, 1);
        assert_eq!(report.rejected, 1);
    }

    #[test]
    fn loop_respects_cap() {
        // Regression test for the max_rounds contract: all proposals malformed,
        // objectives never met — the loop must stop at the cap, not spin.
        let cap = 3;
        let synth = ScriptedSynthesizer::new([MALFORMED, MALFORMED, MALFORMED, MALFORMED]);
        let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, cap);
        let report = cog.run(
            &Intent("build an arm".to_string()),
            &objectives_joint(),
            &Kyosei::default(),
        );
        assert_eq!(report.termination, Termination::CapReached);
        assert_eq!(report.rounds, cap);
        assert_eq!(report.accepted, 0);
        assert_eq!(report.rejected, cap);
    }

    #[test]
    fn loop_exhausts_when_synthesizer_empties() {
        // Accept a valid-but-insufficient state, then run dry before the goal.
        let synth = ScriptedSynthesizer::new([VALID_NO_EDGE]);
        let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, 5);
        let report = cog.run(
            &Intent("build an arm".to_string()),
            &objectives_joint(),
            &Kyosei::default(),
        );
        assert_eq!(report.termination, Termination::Exhausted);
        assert_eq!(report.rounds, 1); // one accepted round, then exhausted
        assert_eq!(report.accepted, 1);
    }
}
