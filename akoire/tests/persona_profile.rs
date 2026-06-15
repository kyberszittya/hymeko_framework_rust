//! Parser-level witness for the LLM persona/profile HyMeKo graph.

use akoire::{EvalOutcome, HymekoEngine, Refinement};

const META_PERSONA: &str = include_str!("../../data/persona/meta_llm_persona.hymeko");
const CODEX_PROFILE: &str = include_str!("../../data/persona/codex_llm_profile.hymeko");

#[test]
fn llm_persona_vocabulary_parses() {
    parser::parse_description(META_PERSONA).expect("meta persona vocabulary must parse");
}

#[test]
fn codex_llm_profile_parses_and_exposes_rules_as_edges() {
    let mut engine = HymekoEngine::new();
    let outcome = engine.evaluate(&Refinement(CODEX_PROFILE.to_string()));
    assert!(matches!(outcome, EvalOutcome::Accepted { .. }));

    let edge_names = engine.ambience().edge_names();
    assert!(edge_names
        .iter()
        .any(|name| name == "rule_inspect_before_edit"));
    assert!(edge_names
        .iter()
        .any(|name| name == "rule_execute_when_concrete"));
    assert!(edge_names
        .iter()
        .any(|name| name == "rule_question_when_risky"));
    assert!(edge_names
        .iter()
        .any(|name| name == "rule_finish_with_evidence"));
}
