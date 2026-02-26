use std::marker::PhantomData;
use crate::common::ids::{EdgeId, NodeId};
use crate::tensor::common::{Real};
use crate::tensor::tensor_val::IncVal;

/// Sparse 3D tensor in COO form (k, i, j, value).
/// Intended as an intermediate export format for JAX/PyTorch.
///
/// - `k` selects the slice (e.g., edge index / hyperedge id).
/// - `i`, `j` are row/col indices inside the slice.
/// - `v` is the weight/value (usually f32).
#[derive(Clone, Debug, Default)]
pub struct TensorCoo<F: Real> {
    /// Number of slices (e.g., |E|)
    pub num_slices: usize,
    /// First dimension size (e.g., |V*| or |V|)
    pub dim_i: usize,
    /// Second dimension size (e.g., |V*| or |V|)
    pub dim_j: usize,
    /// Slice index (0..num_slices)
    pub k: Vec<usize>,
    /// Row index (0..dim_i)
    pub i: Vec<usize>,
    /// Column index (0..dim_j)
    pub j: Vec<usize>,
    /// Value / weight (e.g., f(.) from the dissertation)
    pub v: Vec<F>,
}

#[derive(Clone, Copy, Debug)]
pub struct TensorBuildCfg {
    pub default_arc_weight: f32,
    pub default_edge_weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TensorInc<F: Real, V: IncVal<F>> {
    pub e: EdgeId,
    pub n: NodeId,
    pub s: i8,
    pub w: V,
    pub(crate) _pd: PhantomData<F>,
}

impl Default for TensorBuildCfg {
    fn default() -> Self {
        Self {
            default_arc_weight: 1.0,
            default_edge_weight: 1.0,
        }
    }
}
impl<F: Real> TensorCoo<F> {
    pub fn new(num_slices: usize, dim_i: usize, dim_j: usize) -> Self {
        Self { num_slices, dim_i, dim_j, k: vec![], i: vec![], j: vec![], v: vec![] }
    }

    #[inline(always)]
    pub fn with_meta(num_slices: usize, dim_i: usize, dim_j: usize) -> Self {
        Self {
            num_slices,
            dim_i,
            dim_j,
            k: Vec::new(),
            i: Vec::new(),
            j: Vec::new(),
            v: Vec::new(),

        }
    }

    /// Push one non-zero entry (k, i, j, value).
    #[inline(always)]
    pub fn push(&mut self, k: usize, i: usize, j: usize, value: F) {
        self.k.push(k);
        self.i.push(i);
        self.j.push(j);
        self.v.push(value);
    }

    #[inline(always)]
    pub fn reserve(&mut self, n: usize) {
        self.k.reserve(n);
        self.i.reserve(n);
        self.j.reserve(n);
        self.v.reserve(n);
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.v.len() }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.v.is_empty() }
}
