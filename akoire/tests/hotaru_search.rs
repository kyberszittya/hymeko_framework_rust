//! End-to-end: an A\*-*planned* HOTARU drives the AKOIRE cognitive loop to convergence.
//!
//! This is the "kickoff HOTARU with the planner framework" path exercised through the full public
//! surface: [`SearchHotaru`] plans a delta sequence by implicit-graph A\*, [`HotaruSynthesizer`]
//! streams it, and the unchanged [`CognitiveLoop`]/[`HymekoEngine`] gate-keeps each delta — no
//! hand-written script anywhere, unlike `tests/integration.rs` (which drives the same loop from a
//! fixed `ScriptedHotaru`/`ScriptedSynthesizer`).

use akoire::{
    CognitiveLoop, HiveDelta, HotaruSynthesizer, HymekoEngine, Intent, Kyosei, Objectives,
    SearchHotaru, Termination,
};

/// The HIVE-delta menu HOTARU searches over: a base host block, two wanted edges, one distractor.
fn menu() -> Vec<HiveDelta> {
    vec![
        HiveDelta::replace_source("base", "Rig {\n  a;\n  b;\n  c;\n}"),
        HiveDelta::append_source("e_ab", "@e_ab : a, b { }"),
        HiveDelta::append_source("e_bc", "@e_bc : b, c { }"),
        HiveDelta::append_source("dist", "@dist : a, c { }"),
    ]
}

fn wants(edges: &[&str]) -> Objectives {
    Objectives {
        required_edges: edges.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn planned_hotaru_drives_loop_to_convergence() {
    let objectives = wants(&["e_ab", "e_bc"]);
    let planner = SearchHotaru::plan("", &menu(), &objectives, &Kyosei::default(), 256)
        .expect("goal reachable");
    let plan_len = planner.remaining();
    assert_eq!(plan_len, 3, "optimal plan = base, e_ab, e_bc");

    let mut cog = CognitiveLoop::new(HymekoEngine::new(), HotaruSynthesizer::new(planner), 10);
    let report = cog.run(
        &Intent("assemble the rig".into()),
        &objectives,
        &Kyosei::default(),
    );

    // Every planned delta is feasible by construction ⇒ zero rejections, and the loop converges in
    // exactly the plan's length (one accepted round per delta).
    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(report.rounds, plan_len);
    assert_eq!(report.accepted, plan_len);
    assert_eq!(report.rejected, 0);

    // The final committed ambience holds both required edges.
    let edges = cog.engine().ambience().edge_names();
    assert!(edges.iter().any(|e| e == "e_ab"));
    assert!(edges.iter().any(|e| e == "e_bc"));
    assert!(
        !edges.iter().any(|e| e == "dist"),
        "distractor never committed"
    );
}

#[test]
fn planner_planned_for_less_exhausts_against_stricter_goal() {
    // Planner plans only for `e_ab`; the loop's objective additionally needs `e_bc`. Every planned
    // delta is still accepted, but the plan runs dry before the stricter goal ⇒ Exhausted (not a
    // spin, not a false Converged) — the loop's Exhausted branch, reached through a real planner.
    let planner = SearchHotaru::plan("", &menu(), &wants(&["e_ab"]), &Kyosei::default(), 256)
        .expect("reachable");
    let plan_len = planner.remaining();
    assert_eq!(plan_len, 2, "base, e_ab");

    let stricter = wants(&["e_ab", "e_bc"]);
    let mut cog = CognitiveLoop::new(HymekoEngine::new(), HotaruSynthesizer::new(planner), 10);
    let report = cog.run(
        &Intent("assemble the rig".into()),
        &stricter,
        &Kyosei::default(),
    );

    assert_eq!(report.termination, Termination::Exhausted);
    assert_eq!(report.accepted, plan_len);
    assert_eq!(report.rejected, 0);
}
