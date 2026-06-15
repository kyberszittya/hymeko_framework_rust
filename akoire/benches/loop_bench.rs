//! Per-round cost of the cognitive loop, dominated by `parse_description`.
//!
//! Run: `cargo bench -p akoire`. Reports median / IQR over the criterion
//! sample (CLAUDE.md §3 benchmark-stability rule). Budget (see plan §
//! "Performance budget"): median per converged 1-round run < 200 µs on the dev
//! host. This file asserts no budget itself — criterion's regression tracking
//! does that across runs; promote to a hard `assert!` in CI if desired.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use akoire::{CognitiveLoop, HymekoEngine, Intent, Kyosei, Objectives, ScriptedSynthesizer};

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

criterion_group!(benches, bench_loop_round);
criterion_main!(benches);
