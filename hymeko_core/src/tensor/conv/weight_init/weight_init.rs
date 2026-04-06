use crate::tensor::common::Real;

pub trait WeightInit<F: Real> {
    fn init(&self, fan_in: usize, fan_out: usize, out: &mut [F]);
    fn create(&self, fan_in: usize, fan_out: usize) -> Vec<F>;
}

///  |--------------------
///  | Weight initializers
///  |--------------------

/// Xavier/Glorot uniform: scale = sqrt(6 / (fan_in + fan_out))
///
/// Alternating positive/negative values in a deterministic pattern.
/// For true random Xavier, use `XavierRandom` with an RNG.

pub struct Xavier;
/// Kaiming/He uniform: scale = sqrt(6 / fan_in)
///
/// Designed for ReLU activations (accounts for half the neurons being dead).
pub struct Kaiming;

/// LeCun uniform: scale = sqrt(3 / fan_in)
///
/// Designed for SELU/sigmoid activations.
pub struct LeCun;

/// Uniform constant: all weights = scale.
/// The original "Xavier-ish" placeholder.
pub struct Constant {
    pub value: f64,
}

/// All zeros (for bias vectors or testing).
pub struct Zeros;

/// All ones (for testing or identity-like initialization).
pub struct Ones;

// ─── RNG-based initializers ───────────────────────────────────────────────

/// Xavier uniform with a seedable PRNG (xoshiro256**).
///
/// Produces proper i.i.d. uniform samples in [-scale, +scale].
pub struct XavierRandom {
    pub seed: u64,
}

/// Kaiming uniform with a seedable PRNG.
pub struct KaimingRandom {
    pub seed: u64,
}


