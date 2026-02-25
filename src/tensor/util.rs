use crate::tensor::common::TensorCoo;

pub fn print_dense_block(
    coo: &TensorCoo,
    k_sel: usize,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
) {
    assert!(k_sel < coo.num_slices, "k out of range");
    assert!(row0 < coo.dim_i && col0 < coo.dim_j, "start out of range");

    let r = rows.min(coo.dim_i - row0);
    let c = cols.min(coo.dim_j - col0);

    let mut block = vec![vec![0.0f32; c]; r];

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        let i = coo.i[t];
        let j = coo.j[t];

        if i >= row0 && i < row0 + r && j >= col0 && j < col0 + c {
            block[i - row0][j - col0] += coo.v[t];
        }
    }

    println!(
        "slice k={k_sel}, block rows [{row0}..{}), cols [{col0}..{})",
        row0 + r,
        col0 + c
    );

    for i in 0..r {
        for j in 0..c {
            let x = block[i][j];
            if (x - x.round()).abs() < 1e-6 {
                print!("{:>3} ", x.round() as i32);
            } else {
                print!("{:>6.2} ", x);
            }
        }
        println!();
    }
}