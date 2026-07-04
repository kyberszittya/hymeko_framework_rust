#[allow(dead_code)]
#[path = "../examples/holonomy_entropy_toys.rs"]
mod holonomy_entropy_toys;

use holonomy_entropy_toys::{
    clifford_probability_error, make_dataset, quaternion_periodic_features, run_suite, Config, Task,
};

#[test]
fn quaternion_periodic_features_are_finite_and_wide() {
    let data = make_dataset(Task::Moons, 3, 5, 123);
    let features = quaternion_periodic_features(&data.x, data.samples, data.points);
    assert_eq!(features.len(), data.samples * data.points * 7);
    assert!(features.iter().all(|v| v.is_finite()));
}

#[test]
fn clifford_error_is_zero_for_perfect_logits() {
    let logits = [10.0, -10.0, -10.0, 10.0];
    let labels = [0, 1];
    assert!(clifford_probability_error(&logits, &labels) < 1e-6);
}

#[test]
fn holonomy_suite_smoke_runs_with_fused_update() {
    let cfg = Config {
        tasks: vec![Task::Moons],
        n_train: 24,
        n_test: 12,
        n_points: 8,
        hidden: 8,
        epochs: 2,
        batch_size: 8,
        lr: 1.0e-3,
        seed: 5,
        out: None,
    };
    let rows = run_suite(&cfg);
    assert_eq!(rows.len(), 1);
    assert!((0.0..=1.0).contains(&rows[0].test.acc));
    assert!(rows[0].test.clifford_error.is_finite());
    assert!(rows[0].fused_update_bytes < rows[0].materialized_update_bytes);
    assert!(rows[0].timing.median_us_per_sample > 0.0);
}
