//! Generated point-set toy datasets (moons / spiral / xor) and corruption
//! modes for the holonomy learning experiments.
//!
//! All generators are deterministic in the provided seed (`StdRng`); the
//! seed-53 configuration is frozen as a content-hashed fixture in
//! `tests/fixtures/moons_spiral_xor_seed53.txt`.

use std::str::FromStr;

use rand::{Rng, SeedableRng, rngs::StdRng};

/// Toy point-set classification task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// Two interleaved half-moons.
    Moons,
    /// Two phase-shifted spirals.
    Spiral,
    /// Two diagonal cluster pairs (XOR layout).
    Xor,
}

impl Task {
    /// Stable lowercase name used in CLI flags and JSON output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Moons => "moons",
            Self::Spiral => "spiral",
            Self::Xor => "xor",
        }
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "moons" => Ok(Self::Moons),
            "spiral" => Ok(Self::Spiral),
            "xor" => Ok(Self::Xor),
            other => Err(format!("unknown task '{other}'")),
        }
    }
}

/// A batch of labelled 2-D point sets in SoA layout.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Flattened points, `(samples * points, 2)` row-major.
    pub x: Vec<f32>,
    /// Per-sample binary labels.
    pub y: Vec<usize>,
    /// Number of point sets.
    pub samples: usize,
    /// Points per set.
    pub points: usize,
}

/// Generate a labelled dataset for `task`, deterministic in `seed`.
///
/// # Preconditions
/// * `samples > 0`, `points > 0`.
///
/// # Postconditions
/// * `x.len() == samples * points * 2`, `y.len() == samples`, labels
///   alternate `0, 1, 0, 1, ...`.
pub fn make_dataset(task: Task, samples: usize, points: usize, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(samples * points * 2);
    let mut y = Vec::with_capacity(samples);
    for sample in 0..samples {
        let label = sample % 2;
        y.push(label);
        match task {
            Task::Moons => sample_moons(label, points, &mut rng, &mut x),
            Task::Spiral => sample_spiral(label, points, &mut rng, &mut x),
            Task::Xor => sample_xor(label, points, &mut rng, &mut x),
        }
    }
    Dataset {
        x,
        y,
        samples,
        points,
    }
}

/// Corrupt a dataset with per-point dropout (zeroing) and additive noise.
///
/// # Postconditions
/// * Shape and labels are preserved; only `x` values change. Deterministic
///   in `seed`.
pub fn corrupt_dataset(data: &Dataset, noise_std: f32, missing_rate: f32, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = data.clone();
    for sample in 0..out.samples {
        for point in 0..out.points {
            let base = (sample * out.points + point) * 2;
            if rng.random::<f32>() < missing_rate {
                out.x[base] = 0.0;
                out.x[base + 1] = 0.0;
            } else if noise_std > 0.0 {
                out.x[base] += normalish(&mut rng) * noise_std;
                out.x[base + 1] += normalish(&mut rng) * noise_std;
            }
        }
    }
    out
}

/// Gather a minibatch of point sets by sample indices.
///
/// # Preconditions
/// * Every index in `indices` is `< data.samples`.
pub fn gather_batch(data: &Dataset, indices: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let row = data.points * 2;
    let mut x = Vec::with_capacity(indices.len() * row);
    let mut y = Vec::with_capacity(indices.len());
    for &idx in indices {
        x.extend_from_slice(&data.x[idx * row..(idx + 1) * row]);
        y.push(data.y[idx]);
    }
    (x, y)
}

fn sample_moons(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    for _ in 0..points {
        let theta = rng.random::<f32>() * std::f32::consts::PI;
        let (x, y) = if label == 0 {
            (theta.cos(), theta.sin())
        } else {
            (1.0 - theta.cos(), 0.45 - theta.sin())
        };
        out.push(x + normalish(rng) * 0.055);
        out.push(y + normalish(rng) * 0.055);
    }
}

fn sample_spiral(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    for point in 0..points {
        let t =
            point as f32 / points as f32 * 3.5 * std::f32::consts::PI + rng.random::<f32>() * 0.15;
        let phase = if label == 0 {
            0.0
        } else {
            std::f32::consts::PI
        };
        let radius = 0.12 + 0.08 * t;
        out.push(radius * (t + phase).cos() + normalish(rng) * 0.04);
        out.push(radius * (t + phase).sin() + normalish(rng) * 0.04);
    }
}

fn sample_xor(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    let centers = if label == 0 {
        [(-0.75, -0.75), (0.75, 0.75)]
    } else {
        [(-0.75, 0.75), (0.75, -0.75)]
    };
    for _ in 0..points {
        let (cx, cy) = centers[rng.random_range(0..2)];
        out.push(cx + normalish(rng) * 0.12);
        out.push(cy + normalish(rng) * 0.12);
    }
}

/// Irwin–Hall(6) approximation of a unit normal (matches the historical
/// generator bit-for-bit; do not replace with a different sampler).
fn normalish(rng: &mut StdRng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..6 {
        sum += rng.random::<f32>() - 0.5;
    }
    sum * 0.816_496_6
}
