//! End-to-end: AKOIRE self-corrects a syntax error, then incrementally reaches
//! a two-edge objective. Exercises the full public surface (synthesizer →
//! engine → loop) through the three SysML flows.

use akoire::{
    CognitiveLoop, HiveDelta, HotaruSynthesizer, HymekoEngine, Intent, Kyosei, Objectives,
    ScriptedHotaru, ScriptedSynthesizer, Termination,
};

/// Malformed: edge is missing its base list after `:`.
const BROKEN: &str = "RobotArm {\n  base;\n  link1;\n}\n@joint : { }";

/// Valid, but only the `joint` edge — objective not yet met.
const ONE_EDGE: &str = "RobotArm {\n  base;\n  link1;\n}\n@joint : base, link1 { }";

/// Valid with both `joint` and `gripper` — objective met.
const TWO_EDGES: &str = "RobotArm {\n  base;\n  link1;\n  tool;\n}\n\
                         @joint : base, link1 { }\n@gripper : link1, tool { }";

#[test]
fn self_correction_then_incremental_convergence() {
    // The agent's scripted moves: a broken proposal (error loop), a partial
    // valid one (success loop, goal unmet), then the complete one.
    let synth = ScriptedSynthesizer::new([BROKEN, ONE_EDGE, TWO_EDGES]);
    let objectives = Objectives {
        required_edges: vec!["joint".to_string(), "gripper".to_string()],
    };

    let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, 10);
    let report = cog.run(
        &Intent("assemble a 2-joint arm".to_string()),
        &objectives,
        &Kyosei { max_arity: 4 },
    );

    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(
        report.rejected, 1,
        "the broken proposal must be rejected once"
    );
    assert_eq!(report.accepted, 2, "two valid proposals committed");
    assert_eq!(report.rounds, 3);

    // Final frozen state must carry both objective edges.
    let names = cog.engine().ambience().edge_names();
    assert!(names.iter().any(|e| e == "joint"));
    assert!(names.iter().any(|e| e == "gripper"));
    assert_eq!(cog.engine().ambience().generation(), 2);
}

#[test]
fn hotaru_delta_sequence_mutates_hive_only_through_akoire_gate() {
    let planner = ScriptedHotaru::new([
        HiveDelta::replace_source("bootstrap-arm", "RobotArm {\n  base;\n  link1;\n  tool;\n}"),
        HiveDelta::append_source("bad-joint", "@joint : { }"),
        HiveDelta::append_source("joint", "@joint : base, link1 { }"),
        HiveDelta::append_source("gripper", "@gripper : link1, tool { }"),
    ]);
    let synth = HotaruSynthesizer::new(planner);
    let objectives = Objectives {
        required_edges: vec!["joint".to_string(), "gripper".to_string()],
    };

    let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, 10);
    let report = cog.run(
        &Intent("plan a two-edge arm in HIVE-delta space".to_string()),
        &objectives,
        &Kyosei { max_arity: 4 },
    );

    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(report.accepted, 3);
    assert_eq!(report.rejected, 1);
    assert_eq!(report.rounds, 4);

    let ambience = cog.engine().ambience();
    assert_eq!(ambience.generation(), 3);
    assert!(ambience.edge_names().iter().any(|e| e == "joint"));
    assert!(ambience.edge_names().iter().any(|e| e == "gripper"));
}
