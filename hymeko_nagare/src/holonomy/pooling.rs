//! Global structural pooling over point sets.
//!
//! Pools the per-point quaternion feature lift into four per-channel
//! statistics — mean, std, max, and sign entropy — giving a fixed-size
//! structural descriptor per sample. Sign entropy is the binary entropy of
//! the fraction of non-negative values, the pooled analogue of the signed
//! structure Nagare cares about.

use super::datasets::Dataset;
use super::features::{VERTEX_FEATURES, quaternion_periodic_features};
use super::metrics::binary_entropy;

/// Number of pooled statistics per channel (mean, std, max, sign entropy).
pub const POOL_STATS: usize = 4;

/// Pooled descriptor width per sample.
pub const STRUCTURAL_FEATURES: usize = POOL_STATS * VERTEX_FEATURES;

/// Pool the quaternion feature lift of `data` into per-sample descriptors.
///
/// # Postconditions
/// * Returns `data.samples * STRUCTURAL_FEATURES` values laid out as
///   `[mean(7) | std(7) | max(7) | sign_entropy(7)]` per sample.
pub fn structural_pool_features(data: &Dataset) -> Vec<f32> {
    let lifted = quaternion_periodic_features(&data.x, data.samples, data.points);
    let mut out = vec![0.0; data.samples * STRUCTURAL_FEATURES];
    let inv_points = 1.0 / data.points as f32;
    for sample in 0..data.samples {
        for channel in 0..VERTEX_FEATURES {
            let mut sum = 0.0;
            let mut max_value = f32::NEG_INFINITY;
            let mut positive = 0usize;
            for point in 0..data.points {
                let value = lifted[(sample * data.points + point) * VERTEX_FEATURES + channel];
                sum += value;
                max_value = max_value.max(value);
                positive += usize::from(value >= 0.0);
            }
            let mean = sum * inv_points;
            let mut var = 0.0;
            for point in 0..data.points {
                let value = lifted[(sample * data.points + point) * VERTEX_FEATURES + channel];
                let d = value - mean;
                var += d * d;
            }
            let pos = positive as f32 * inv_points;
            let sign_entropy = binary_entropy(pos);
            let base = sample * STRUCTURAL_FEATURES;
            out[base + channel] = mean;
            out[base + VERTEX_FEATURES + channel] = (var * inv_points + 1.0e-6).sqrt();
            out[base + 2 * VERTEX_FEATURES + channel] = max_value;
            out[base + 3 * VERTEX_FEATURES + channel] = sign_entropy;
        }
    }
    out
}
