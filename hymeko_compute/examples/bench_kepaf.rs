//! KEPAF §VII benchmark — wall-clock the Vulkan kernels at the same
//! three fixture scales the CPU baseline reports (canonical 21-vertex,
//! MNIST-MLP-adjacency $|V_L|\approx 7\,882$, synthetic $|V_L|=35\,000$).
//!
//! Run with:
//! ```bash
//! cargo run --release -p hymeko_compute --example bench_kepaf
//! ```
//!
//! Reports per-iteration and total wall-time for `force_directed` and
//! `signed_spmv` at each scale, comparable line-by-line to the
//! `kepaf_benchmark.py` CPU baseline.

use std::time::Instant;

use hymeko_compute::{
    VulkanContext,
    kernels::{
        force_directed::{Arc2D, LayoutParams, Position, run as fr_run},
        signed_spmv::{SignedIncidenceCsr, run as spmv_run},
    },
};

fn synthesise_layout_fixture(n: u32, m_arcs: u32, seed: u64) -> (Vec<Position>, Vec<Arc2D>) {
    let mut rng_state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((rng_state >> 33) as f32) * (1.0 / u32::MAX as f32)
    };

    let pos: Vec<Position> = (0..n)
        .map(|_| Position {
            x: 2.0 * next() - 1.0,
            y: 2.0 * next() - 1.0,
        })
        .collect();
    let arcs: Vec<Arc2D> = (0..m_arcs)
        .map(|_| {
            let s = (next() * n as f32) as u32 % n;
            let mut t = (next() * n as f32) as u32 % n;
            if t == s {
                t = (t + 1) % n;
            }
            Arc2D {
                src: s,
                dst: t,
                rest_len: 0.05,
                _pad: 0.0,
            }
        })
        .collect();
    (pos, arcs)
}

fn synthesise_csr_fixture(n_rows: u32, n_cols: u32, nnz_per_row: u32, seed: u64)
    -> (SignedIncidenceCsr, Vec<f32>)
{
    let mut rng_state = seed.wrapping_add(0xD1B5_4A32_D192_ED03);
    let mut next = || {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((rng_state >> 33) as f32) * (1.0 / u32::MAX as f32)
    };

    let mut row_ptr = Vec::with_capacity((n_rows + 1) as usize);
    let mut col_ind = Vec::with_capacity((n_rows * nnz_per_row) as usize);
    let mut val = Vec::with_capacity((n_rows * nnz_per_row) as usize);
    row_ptr.push(0u32);
    for _ in 0..n_rows {
        for _ in 0..nnz_per_row {
            let c = (next() * n_cols as f32) as u32 % n_cols;
            let s = if next() < 0.5 { -1.0 } else { 1.0 };
            col_ind.push(c);
            val.push(s);
        }
        row_ptr.push(col_ind.len() as u32);
    }
    let x: Vec<f32> = (0..n_cols).map(|_| 2.0 * next() - 1.0).collect();

    (
        SignedIncidenceCsr {
            row_ptr,
            col_ind,
            val,
        },
        x,
    )
}

fn bench_force_directed(ctx: &VulkanContext, label: &str, n: u32, m_arcs: u32, n_iter: u32) {
    let (mut pos, arcs) = synthesise_layout_fixture(n, m_arcs, 0xC0FFEE);
    let params = LayoutParams {
        n_vertices: n,
        n_arcs: m_arcs,
        k_repulsion: 0.05,
        k_attraction: 1.0,
        damping: 0.85,
        dt: 0.02,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    // Warm-up dispatch (1 iter) — pipeline + alloc + first-launch JIT.
    let _ = fr_run(ctx, &mut pos.clone(), &arcs, params, 1);

    let t0 = Instant::now();
    fr_run(ctx, &mut pos, &arcs, params, n_iter).expect("fr dispatch");
    let dt = t0.elapsed();
    let total_ms = dt.as_secs_f64() * 1e3;
    let per_iter_ms = total_ms / n_iter as f64;

    let nan_count = pos
        .iter()
        .filter(|p| !p.x.is_finite() || !p.y.is_finite())
        .count();
    println!(
        "  force_directed  {label:<22} |V|={n:>6}  |E|={m_arcs:>7}  iter={n_iter:>3}  total={total_ms:>10.3}ms  per_iter={per_iter_ms:>9.3}ms  nan/inf={nan_count}"
    );
}

fn bench_signed_spmv(
    ctx: &VulkanContext,
    label: &str,
    n_rows: u32,
    n_cols: u32,
    nnz_per_row: u32,
    n_repeat: u32,
) {
    let (b, x) = synthesise_csr_fixture(n_rows, n_cols, nnz_per_row, 0xBEEF);
    // Warm-up.
    let _ = spmv_run(ctx, &b, &x);

    let t0 = Instant::now();
    let mut last_y_len = 0;
    for _ in 0..n_repeat {
        let y = spmv_run(ctx, &b, &x).expect("spmv dispatch");
        last_y_len = y.len();
    }
    let dt = t0.elapsed();
    let total_ms = dt.as_secs_f64() * 1e3;
    let per_iter_ms = total_ms / n_repeat as f64;
    let nnz = b.val.len();
    println!(
        "  signed_spmv     {label:<22} |V|={n_rows:>6}  |E|={n_cols:>7}  nnz={nnz:>8}  rep={n_repeat:>3}  total={total_ms:>10.3}ms  per_call={per_iter_ms:>9.3}ms  |y|={last_y_len}"
    );
}

fn main() {
    let ctx = VulkanContext::new().expect("vulkan init");
    println!("== KEPAF §VII Vulkan benchmark ==");
    println!("device: {}", ctx.device_name());

    println!();
    println!("force_directed (Fruchterman-Reingold, naïve O(N^2) summation)");
    bench_force_directed(&ctx, "canonical-21",     31,        33, 100);
    bench_force_directed(&ctx, "mnist-adj",      7_882,    13_280,  50);
    bench_force_directed(&ctx, "synthetic-35k", 35_000,   100_000,  20);

    println!();
    println!("signed_spmv (signed-incidence y = B * x)");
    bench_signed_spmv(&ctx, "canonical-21",      31,         33,  4, 200);
    bench_signed_spmv(&ctx, "mnist-adj",       1_242,     2_000,  6,  50);
    bench_signed_spmv(&ctx, "synthetic-35k",  35_000,    25_000,  8,  20);
}
