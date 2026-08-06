//! Entropy-pool local learner.
//!
//! A linear readout over globally pooled structural features, trained with
//! a *local* update rule only — no reverse-mode propagation into the
//! feature generator or pooling:
//!
//! ```text
//!   W <- W + lr * gate * phi * (y - p)
//! ```
//!
//! Gate variants: scalar predictive-entropy gate, constant gate, and the
//! fitted projection gate (class-mean basis, alpha-mixed projection of the
//! pooled features; see `holonomy::projection`).

use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

use super::datasets::Dataset;
use super::metrics::{Metrics, clifford_probability_error, cross_entropy_eval, entropy2, softmax2};
use super::pooling::{STRUCTURAL_FEATURES, structural_pool_features};
use super::projection::{
    PROJECTION_ALPHA, PROJECTION_RANK, ProjectionBasis, default_holonomy_basis,
    fit_class_mean_basis,
};

/// Learner input width: pooled structural features plus one gate slot.
pub const LOCAL_FEATURES: usize = STRUCTURAL_FEATURES + 1;

/// Update-gate variant of the local learner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateMode {
    /// Scale updates by `0.25 + H(p)` and feed entropy as a feature.
    Entropy,
    /// Constant gate `1.0` and constant feature (ablation control).
    Constant,
    /// Constant gate, but alpha-mix-project the pooled features onto the
    /// fitted holonomy subspace.
    Projection,
}

impl GateMode {
    /// Stable lowercase name used in JSON output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entropy => "entropy",
            Self::Constant => "constant",
            Self::Projection => "projection",
        }
    }
}

/// Linear local learner over pooled holonomy features.
#[derive(Clone, Debug)]
pub struct EntropyPoolLocalLearner {
    w: Vec<f32>,
    b: [f32; 2],
    gate_mode: GateMode,
    projection_basis: ProjectionBasis,
}

impl EntropyPoolLocalLearner {
    /// Entropy-gated learner with seeded near-zero weights.
    pub fn new(seed: u64) -> Self {
        Self::new_with_gate(seed, GateMode::Entropy)
    }

    /// Learner with an explicit gate mode.
    pub fn new_with_gate(seed: u64, gate_mode: GateMode) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w = vec![0.0; LOCAL_FEATURES * 2];
        for value in &mut w {
            *value = (rng.random::<f32>() * 2.0 - 1.0) * 0.01;
        }
        Self {
            w,
            b: [0.0, 0.0],
            gate_mode,
            projection_basis: default_holonomy_basis(),
        }
    }

    /// Two-class logits for one feature row.
    ///
    /// # Preconditions
    /// * `phi.len() == LOCAL_FEATURES`.
    pub fn logits_one(&self, phi: &[f32]) -> [f32; 2] {
        assert_eq!(phi.len(), LOCAL_FEATURES);
        let mut out = self.b;
        for (i, &v) in phi.iter().enumerate() {
            out[0] += v * self.w[2 * i];
            out[1] += v * self.w[2 * i + 1];
        }
        out
    }

    /// Logits for every sample of `data`, `(samples, 2)` row-major.
    pub fn predict_dataset(&self, data: &Dataset) -> Vec<f32> {
        let structural = structural_pool_features(data);
        let mut logits = vec![0.0; data.samples * 2];
        for sample in 0..data.samples {
            let phi = self.gated_phi(
                &structural[sample * STRUCTURAL_FEATURES..(sample + 1) * STRUCTURAL_FEATURES],
            );
            let row = self.logits_one(&phi);
            logits[2 * sample] = row[0];
            logits[2 * sample + 1] = row[1];
        }
        logits
    }

    /// Train with the local update rule; deterministic in `seed`.
    ///
    /// In [`GateMode::Projection`] the basis is fitted from the training
    /// pool before the update epochs.
    pub fn train(&mut self, data: &Dataset, epochs: usize, batch_size: usize, lr: f32, seed: u64) {
        let structural = structural_pool_features(data);
        if self.gate_mode == GateMode::Projection {
            self.projection_basis = fit_class_mean_basis(&structural, &data.y);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..data.samples).collect();
        for _ in 0..epochs {
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(batch_size) {
                for &sample in chunk {
                    let base = sample * STRUCTURAL_FEATURES;
                    let phi = self.gated_phi(&structural[base..base + STRUCTURAL_FEATURES]);
                    let logits = self.logits_one(&phi);
                    let (p0, p1) = softmax2(logits[0], logits[1]);
                    let y0 = f32::from(data.y[sample] == 0);
                    let y1 = f32::from(data.y[sample] == 1);
                    let entropy = entropy2(p0, p1);
                    let gate = match self.gate_mode {
                        GateMode::Entropy => 0.25 + entropy,
                        GateMode::Constant | GateMode::Projection => 1.0,
                    };
                    let delta = [y0 - p0, y1 - p1];
                    for (i, &value) in phi.iter().enumerate() {
                        self.w[2 * i] += lr * gate * value * delta[0];
                        self.w[2 * i + 1] += lr * gate * value * delta[1];
                    }
                    self.b[0] += lr * gate * delta[0];
                    self.b[1] += lr * gate * delta[1];
                }
            }
        }
    }

    /// Parameter count, including the fitted basis when it is in play.
    pub fn n_params(&self) -> usize {
        let projection_params = if self.gate_mode == GateMode::Projection {
            STRUCTURAL_FEATURES * PROJECTION_RANK
        } else {
            0
        };
        self.w.len() + self.b.len() + projection_params
    }

    /// Build the gated feature row: pooled features plus the gate feature
    /// (predictive entropy or constant), projection-mixed when configured.
    fn gated_phi(&self, structural: &[f32]) -> Vec<f32> {
        let mut warm = vec![0.0; LOCAL_FEATURES];
        warm[..STRUCTURAL_FEATURES].copy_from_slice(structural);
        warm[STRUCTURAL_FEATURES] = 1.0;
        let logits = self.logits_one(&warm);
        let (p0, p1) = softmax2(logits[0], logits[1]);
        warm[STRUCTURAL_FEATURES] = match self.gate_mode {
            GateMode::Entropy => entropy2(p0, p1),
            GateMode::Constant | GateMode::Projection => 1.0,
        };
        if self.gate_mode == GateMode::Projection {
            self.projection_basis
                .apply_alpha_mix(&mut warm[..STRUCTURAL_FEATURES], PROJECTION_ALPHA);
        }
        warm
    }
}

/// Evaluate a local learner on a dataset (CE, accuracy, entropy, Clifford
/// probability error).
pub fn evaluate_local(model: &EntropyPoolLocalLearner, data: &Dataset) -> Metrics {
    let logits = model.predict_dataset(data);
    let ce = cross_entropy_eval(&logits, &data.y);
    Metrics {
        acc: ce.acc,
        loss: ce.loss,
        entropy: ce.entropy,
        clifford_error: clifford_probability_error(&logits, &data.y),
    }
}
