#[path = "../examples/global_pool_entropy_toys.rs"]
#[allow(dead_code)]
mod global_pool_entropy_toys;

use global_pool_entropy_toys::{global_pool, make_dataset, run_suite, Config, Task};

#[test]
fn generated_toy_dataset_has_expected_shape() {
    let split = make_dataset(Task::Moons, 12, 8, 16, 7);
    assert_eq!(split.train.x.len(), 12 * 16 * 2);
    assert_eq!(split.train.y.len(), 12);
    assert_eq!(split.test.x.len(), 8 * 16 * 2);
    assert_eq!(split.test.y.len(), 8);
}

#[test]
fn global_pool_emits_mean_std_max_lanes() {
    let h = vec![1.0, 2.0, 3.0, 6.0, 5.0, 10.0];
    let pooled = global_pool(&h, 1, 3, 2);
    assert_eq!(pooled.len(), 6);
    assert!((pooled[0] - 3.0).abs() < 1e-6);
    assert!((pooled[1] - 6.0).abs() < 1e-6);
    assert!((pooled[4] - 5.0).abs() < 1e-6);
    assert!((pooled[5] - 10.0).abs() < 1e-6);
}

#[test]
fn nagare_entropy_feedback_suite_smoke() {
    let cfg = Config {
        tasks: vec![Task::Moons],
        n_train: 32,
        n_test: 16,
        n_points: 16,
        hidden: 8,
        epochs: 2,
        batch_size: 16,
        ..Config::default()
    };
    let rows = run_suite(&cfg);
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert!((0.0..=1.0).contains(&row.baseline.test.acc));
    assert!((0.0..=1.0).contains(&row.entropy_feedback.test.acc));
    assert!(row.baseline.timing.median_us_per_sample > 0.0);
    assert!(row.entropy_feedback.timing.median_us_per_sample > 0.0);
    assert!(row.baseline.n_params > 0);
    assert!(row.entropy_feedback.n_params > row.baseline.n_params);
}
