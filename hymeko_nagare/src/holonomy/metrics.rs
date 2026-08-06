//! Shared evaluation metrics for the holonomy experiments.
//!
//! Cross entropy / accuracy / predictive entropy over 2-logit outputs, the
//! Clifford `Cl(2,0)` probability-vector error, and the shared forward
//! timing protocol (20 warm-ups, median/mean/max over repeats).

use std::time::Instant;

use hymeko_clifford::{Multivector, Signature};

/// Scalar evaluation metrics for a 2-class model.
#[derive(Clone, Debug)]
pub struct Metrics {
    /// Classification accuracy in `[0, 1]`.
    pub acc: f32,
    /// Mean cross-entropy loss.
    pub loss: f32,
    /// Mean predictive entropy (bits).
    pub entropy: f32,
    /// Mean Clifford `Cl(2,0)` probability error.
    pub clifford_error: f32,
}

/// Forward-latency summary per sample (microseconds).
#[derive(Clone, Debug)]
pub struct Timing {
    /// Median over repeats.
    pub median_us_per_sample: f64,
    /// Mean over repeats.
    pub mean_us_per_sample: f64,
    /// Worst case over repeats.
    pub max_us_per_sample: f64,
}

/// Cross-entropy summary (no gradients) for 2-logit outputs.
#[derive(Clone, Debug)]
pub struct CrossEntropyEval {
    /// Mean cross-entropy loss.
    pub loss: f32,
    /// Classification accuracy.
    pub acc: f32,
    /// Mean predictive entropy (bits).
    pub entropy: f32,
}

/// Evaluate mean cross entropy, accuracy, and predictive entropy.
///
/// # Preconditions
/// * `logits.len() == 2 * labels.len()`, labels in `{0, 1}`.
pub fn cross_entropy_eval(logits: &[f32], labels: &[usize]) -> CrossEntropyEval {
    assert_eq!(logits.len(), labels.len() * 2);
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut entropy = 0.0;
    for i in 0..labels.len() {
        let (p0, p1) = softmax2(logits[2 * i], logits[2 * i + 1]);
        loss += if labels[i] == 0 { -p0.ln() } else { -p1.ln() };
        correct += usize::from((p1 > p0) == (labels[i] == 1));
        entropy += entropy2(p0, p1);
    }
    CrossEntropyEval {
        loss: loss / labels.len() as f32,
        acc: correct as f32 / labels.len() as f32,
        entropy: entropy / labels.len() as f32,
    }
}

/// Clifford `Cl(2,0)` probability-vector squared error.
///
/// Embeds the probability residual as a grade-1 vector and reads the scalar
/// part of its geometric square.
///
/// # Preconditions
/// * `logits.len() == 2 * labels.len()`.
pub fn clifford_probability_error(logits: &[f32], labels: &[usize]) -> f32 {
    assert_eq!(logits.len(), labels.len() * 2);
    let sig = Signature::euclidean(2);
    let mut sum = 0.0;
    for b in 0..labels.len() {
        let (p0, p1) = softmax2(logits[2 * b], logits[2 * b + 1]);
        let mut err = Multivector::zero(2);
        err.components[1] = (p0 - f32::from(labels[b] == 0)) as f64;
        err.components[2] = (p1 - f32::from(labels[b] == 1)) as f64;
        sum += err.geo(&err, &sig).components[0] as f32;
    }
    sum / labels.len() as f32
}

/// Numerically stable two-way softmax.
pub fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    (ea / (ea + eb), eb / (ea + eb))
}

/// Binary entropy (bits) of a two-way distribution given both probabilities.
pub fn entropy2(p0: f32, p1: f32) -> f32 {
    -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2
}

/// Binary entropy (bits) of a single probability `p`.
pub fn binary_entropy(p: f32) -> f32 {
    let q = 1.0 - p;
    -(p * p.max(1.0e-12).ln() + q * q.max(1.0e-12).ln()) / std::f32::consts::LN_2
}

/// Shared forward-latency protocol: 20 warm-up calls, then `repeats` timed
/// calls of `forward`, reported per sample.
///
/// # Preconditions
/// * `samples > 0`, `repeats > 0`.
pub fn forward_timing<F: FnMut()>(samples: usize, repeats: usize, mut forward: F) -> Timing {
    assert!(samples > 0 && repeats > 0);
    for _ in 0..20 {
        forward();
    }
    let mut values = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        forward();
        values.push(start.elapsed().as_secs_f64() * 1.0e6 / samples as f64);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Timing {
        median_us_per_sample: values[values.len() / 2],
        mean_us_per_sample: values.iter().sum::<f64>() / values.len() as f64,
        max_us_per_sample: values[values.len() - 1],
    }
}
