#[allow(dead_code)]
#[path = "../examples/entropy_pool_learning_compare.rs"]
mod entropy_pool_learning_compare;

use entropy_pool_learning_compare::{
    corrupt_dataset, evaluate_local, learn_holonomy_projection_basis, make_dataset,
    project_onto_holonomy_axis, run_stress_ablation, run_suite, structural_pool_features, Config,
    EntropyPoolLocalLearner, GateMode, Task,
};

#[test]
fn structural_pool_features_are_finite() {
    let data = make_dataset(Task::Moons, 4, 6, 11);
    let pooled = structural_pool_features(&data);
    assert_eq!(pooled.len(), 4 * 28);
    assert!(pooled.iter().all(|v| v.is_finite()));
}

#[test]
fn local_update_reduces_loss_on_tiny_batch() {
    let data = make_dataset(Task::Xor, 24, 8, 12);
    let mut model = EntropyPoolLocalLearner::new(7);
    let before = evaluate_local(&model, &data).loss;
    model.train(&data, 8, 8, 0.05, 7);
    let after = evaluate_local(&model, &data).loss;
    assert!(after < before, "before={before} after={after}");
}

#[test]
fn constant_gate_update_reduces_loss_on_tiny_batch() {
    let data = make_dataset(Task::Xor, 24, 8, 12);
    let mut model = EntropyPoolLocalLearner::new_with_gate(7, GateMode::Constant);
    let before = evaluate_local(&model, &data).loss;
    model.train(&data, 8, 8, 0.05, 7);
    let after = evaluate_local(&model, &data).loss;
    assert!(after < before, "before={before} after={after}");
}

#[test]
fn projection_gate_update_reduces_loss_on_tiny_batch() {
    let data = make_dataset(Task::Xor, 24, 8, 12);
    let mut model = EntropyPoolLocalLearner::new_with_gate(7, GateMode::Projection);
    let before = evaluate_local(&model, &data).loss;
    model.train(&data, 8, 8, 0.05, 7);
    let after = evaluate_local(&model, &data).loss;
    assert!(after < before, "before={before} after={after}");
}

#[test]
fn projection_gate_preserves_shape_and_finiteness() {
    let mut phi = vec![0.1; 29];
    phi[3] = 0.4;
    phi[4] = -0.2;
    phi[5] = 0.8;
    phi[6] = 0.3;
    project_onto_holonomy_axis(&mut phi);
    assert_eq!(phi.len(), 29);
    assert!(phi.iter().all(|v| v.is_finite()));
}

#[test]
fn learned_projection_basis_is_finite_and_orthogonal() {
    let data = make_dataset(Task::Moons, 24, 8, 13);
    let structural = structural_pool_features(&data);
    let basis = learn_holonomy_projection_basis(&structural, &data.y);
    assert!(basis.iter().flatten().all(|v| v.is_finite()));
    for (i, lhs) in basis.iter().enumerate() {
        let lhs_norm = lhs.iter().map(|v| v * v).sum::<f32>();
        assert!(lhs_norm <= 1.0001);
        for rhs in basis.iter().skip(i + 1) {
            let dot = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                .abs();
            assert!(dot < 1.0e-4, "basis dot={dot}");
        }
    }
}

#[test]
fn corruption_preserves_shape_and_finiteness() {
    let data = make_dataset(Task::Spiral, 5, 7, 21);
    let corrupted = corrupt_dataset(&data, 0.2, 0.4, 22);
    assert_eq!(corrupted.x.len(), data.x.len());
    assert_eq!(corrupted.y, data.y);
    assert!(corrupted.x.iter().all(|v| v.is_finite()));
}

#[test]
fn comparison_suite_smoke_runs() {
    let cfg = Config {
        tasks: vec![Task::Moons],
        n_train: 24,
        n_test: 12,
        n_points: 8,
        hidden: 8,
        epochs: 2,
        batch_size: 8,
        lr: 0.05,
        seed: 3,
        out: None,
    };
    let rows = run_suite(&cfg);
    assert_eq!(rows.len(), 1);
    assert!((0.0..=1.0).contains(&rows[0].local.acc));
    assert!((0.0..=1.0).contains(&rows[0].backprop_acc));
    assert!(rows[0].local_timing.median_us_per_sample > 0.0);
    assert!(rows[0].local_params < rows[0].backprop_params);
}

#[test]
fn stress_ablation_suite_smoke_runs() {
    let cfg = Config {
        tasks: vec![Task::Moons],
        n_train: 24,
        n_test: 12,
        n_points: 8,
        hidden: 8,
        epochs: 2,
        batch_size: 8,
        lr: 0.05,
        seed: 3,
        out: None,
    };
    let rows = run_stress_ablation(&cfg);
    assert_eq!(rows.len(), 4);
    for row in rows {
        assert!((0.0..=1.0).contains(&row.entropy_metrics.acc));
        assert!((0.0..=1.0).contains(&row.constant_metrics.acc));
        assert!((0.0..=1.0).contains(&row.projection_metrics.acc));
        assert!(row.entropy_timing.median_us_per_sample > 0.0);
        assert!(row.constant_timing.median_us_per_sample > 0.0);
        assert!(row.projection_timing.median_us_per_sample > 0.0);
    }
}
