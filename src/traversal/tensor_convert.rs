use nalgebra::DMatrix;
use nalgebra_sparse::{coo::CooMatrix, CsrMatrix};
use crate::traversal::tensor::TensorCoo;


pub struct JaxBcoo {
    pub shape: [usize; 3],          // [K, I, J]
    pub indices: Vec<[i32; 3]>,     // (nnz, 3)
    pub data: Vec<f32>,             // (nnz,)
}

pub fn to_nalgebra_dense_slice(coo: &TensorCoo, k_sel: usize) -> DMatrix<f32> {
    assert!(k_sel < coo.num_slices);
    let mut m = DMatrix::<f32>::zeros(coo.dim_i, coo.dim_j);

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        m[(coo.i[t], coo.j[t])] += coo.v[t];
    }
    m
}

pub fn to_nalgebra_csr_slice(coo: &TensorCoo, k_sel: usize) -> CsrMatrix<f32> {
    assert!(k_sel < coo.num_slices);

    let mut m = CooMatrix::new(coo.dim_i, coo.dim_j);

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        m.push(coo.i[t], coo.j[t], coo.v[t]);
    }
    CsrMatrix::from(&m) // will coalesce duplicates
}


pub fn to_jax_bcoo(coo: &TensorCoo) -> JaxBcoo {
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