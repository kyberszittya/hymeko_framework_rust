//! Read a (signed-incidence) CSR triplet on stdin, run the GPU
//! `signed_spmv` kernel, write per-call wall-clock on stdout.
//!
//! Stdin schema:
//! ```json
//! {
//!   "n_rows": 35000,
//!   "n_cols": 25000,
//!   "row_ptr": [0, 4, ...],
//!   "col_ind": [...],
//!   "val":     [...],
//!   "x":       [...],
//!   "n_repeat": 50
//! }
//! ```
//!
//! Stdout schema:
//! ```json
//! { "wall_ms": 12.3, "n_repeat": 50, "per_call_ms": 0.246,
//!   "device": "..." }
//! ```

use std::io::{self, Read, Write};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use hymeko_compute::{
    VulkanContext,
    kernels::signed_spmv::{SignedIncidenceCsr, run as spmv_run},
};

#[derive(Deserialize)]
struct CsrJson {
    n_rows: u32,
    #[serde(default)]
    n_cols: u32,
    row_ptr: Vec<u32>,
    col_ind: Vec<u32>,
    val:     Vec<f32>,
    x:       Vec<f32>,
    #[serde(default = "default_repeat")]
    n_repeat: u32,
}

fn default_repeat() -> u32 { 50 }

#[derive(Serialize)]
struct OutJson {
    wall_ms:     f64,
    n_repeat:    u32,
    per_call_ms: f64,
    device:      String,
}

fn main() {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf).expect("read stdin");
    let csr_in: CsrJson = serde_json::from_str(&buf).expect("parse csr");

    let csr = SignedIncidenceCsr {
        row_ptr: csr_in.row_ptr,
        col_ind: csr_in.col_ind,
        val:     csr_in.val,
    };
    let ctx = VulkanContext::new().expect("vulkan init");

    // Warm-up.
    let _ = spmv_run(&ctx, &csr, &csr_in.x);

    let t0 = Instant::now();
    for _ in 0..csr_in.n_repeat {
        let _ = spmv_run(&ctx, &csr, &csr_in.x).expect("dispatch");
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

    let out = OutJson {
        wall_ms,
        n_repeat:    csr_in.n_repeat,
        per_call_ms: wall_ms / csr_in.n_repeat as f64,
        device:      ctx.device_name(),
    };
    let s = serde_json::to_string(&out).unwrap();
    io::stdout().write_all(s.as_bytes()).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}
