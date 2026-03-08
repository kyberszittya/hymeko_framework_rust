use iceoryx2::prelude::*;

pub const MAX_NNZ: usize = 1_048_576; // 1M non-zero entries baseline

#[repr(C)]
#[derive(Debug, ZeroCopySend)]
pub struct HypergraphWeights {
    pub version: u64,
    pub nnz: usize,
    pub values: [f32; MAX_NNZ],
}

