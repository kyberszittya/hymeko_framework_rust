//! Per-round cost of the cognitive loop, dominated by `parse_description`.
//!
//! Run: `cargo bench -p akoire`. Reports median / IQR over the criterion
//! sample (CLAUDE.md §3 benchmark-stability rule). Budget (see plan §
//! "Performance budget"): median per converged 1-round run < 200 µs on the dev
//! host. This file asserts no budget itself — criterion's regression tracking
//! does that across runs; promote to a hard `assert!` in CI if desired.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use akoire::{
    CognitiveLoop, HiveDelta, HymekoEngine, Intent, Kyosei, Objectives, ScriptedSynthesizer,
    SearchHotaru,
};

const VALID_WITH_JOINT: &str = "RobotArm {\n  base;\n  link1;\n}\n@joint : base, link1 { }";

fn bench_loop_round(c: &mut Criterion) {
    let intent = Intent("build an arm".to_string());
    let objectives = Objectives {
        required_edges: vec!["joint".to_string()],
    };
    let kyosei = Kyosei::default();

    c.bench_function("cognitive_loop_converge_1_round", |b| {
        b.iter(|| {
            // Fresh engine + synth each iteration so the parse is real work.
            let synth = ScriptedSynthesizer::new([VALID_WITH_JOINT]);
            let mut cog = CognitiveLoop::new(HymekoEngine::new(), synth, 4);
            let report = cog.run(
                black_box(&intent),
                black_box(&objectives),
                black_box(&kyosei),
            );
            black_box(report);
        });
    });
}

/// A\* planning cost for HOTARU: the kickoff path (`SearchHotaru::plan` over the delta menu).
/// Reports median / IQR; the asserted numeric budget lives in the unit test
/// (`search_meets_expansion_budget`, a deterministic expansion count) — this is wall-time provenance.
fn bench_hotaru_plan(c: &mut Criterion) {
    let menu = vec![
        HiveDelta::replace_source("base", "Rig {\n  a;\n  b;\n  c;\n}"),
        HiveDelta::append_source("e_ab", "@e_ab : a, b { }"),
        HiveDelta::append_source("e_bc", "@e_bc : b, c { }"),
        HiveDelta::append_source("dist", "@dist : a, c { }"),
    ];
    let objectives = Objectives {
        required_edges: vec!["e_ab".to_string(), "e_bc".to_string()],
    };

    c.bench_function("hotaru_search_plan_two_edge", |b| {
        b.iter(|| {
            let planner =
                SearchHotaru::plan(black_box(""), black_box(&menu), black_box(&objectives), 256);
            black_box(planner);
        });
    });
}

criterion_group!(benches, bench_loop_round, bench_hotaru_plan);
criterion_main!(benches);
