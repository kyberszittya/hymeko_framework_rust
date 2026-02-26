use nalgebra::{DMatrix, Scalar};
use num_traits::Zero;
use nalgebra_sparse::{coo::CooMatrix, CsrMatrix};
use crate::tensor::common::{Real};
use crate::tensor::tensor_coo::TensorCoo;

pub struct JaxBcoo<F: Real> {
    pub shape: [usize; 3],          // [K, I, J]
    pub indices: Vec<[i32; 3]>,     // (nnz, 3)
    pub data: Vec<F>,             // (nnz,)
}

pub fn to_nalgebra_dense_slice<F: Real>(coo: &TensorCoo<F>, k_sel: usize) -> DMatrix<F>
where
    F: Real + Scalar + Zero,
{
    assert!(k_sel < coo.num_slices);
    let mut m = DMatrix::<F>::zeros(coo.dim_i, coo.dim_j);

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        m[(coo.i[t], coo.j[t])] += coo.v[t];
    }
    m
}

pub fn to_nalgebra_csr_slice<F: Real>(coo: &TensorCoo<F>, k_sel: usize) -> CsrMatrix<F>
where
    F: Real + Scalar + Zero,
{
    assert!(k_sel < coo.num_slices);

    let mut m = CooMatrix::new(coo.dim_i, coo.dim_j);

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        m.push(coo.i[t], coo.j[t], coo.v[t]);
    }
    CsrMatrix::from(&m) // will coalesce duplicates
}


pub fn to_jax_bcoo<F: Real>(coo: &TensorCoo<F>) -> JaxBcoo<F>
where
    F: Real + Scalar + Zero,
{
    let mut indices = Vec::with_capacity(coo.len());
    let mut data = Vec::with_capacity(coo.len());

    for t in 0..coo.len() {
        indices.push([
            coo.k[t] as i32,
            coo.i[t] as i32,
            coo.j[t] as i32,
        ]);
        data.push(coo.v[t]);
    }

    JaxBcoo {
        shape: [coo.num_slices, coo.dim_i, coo.dim_j],
        indices,
        data,
    }
}