//! Read a graph as JSON on stdin, run the GPU `force_directed` kernel,
//! write final positions as JSON on stdout. Used by
//! `scripts/kepaf_benchmark.py --gpu-layout` to replace
//! `nx.spring_layout` with the Vulkan kernel for figure generation,
//! so the KEPAF figures match what the paper measures.
//!
//! ## Wire format
//!
//! Stdin:
//! ```json
//! {
//!   "n_nodes":  35000,
//!   "n_iter":   50,
//!   "seed":     0,
//!   "edges":    [[0, 17], [0, 42], ...]
//! }
//! ```
//!
//! Stdout:
//! ```json
//! {
//!   "wall_ms":   1052.4,
//!   "n_iter":    50,
//!   "device":    "NVIDIA GeForce RTX 2070 SUPER",
//!   "positions": [[x0, y0], [x1, y1], ...]
//! }
//! ```

use std::io::{self, Read, Write};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use hymeko_compute::{
    VulkanContext,
    kernels::force_directed::{Arc2D, LayoutParams, Position, run as fr_run},
};

#[derive(Deserialize)]
struct GraphJson {
    n_nodes: u32,
    #[serde(default = "default_iter")]
    n_iter: u32,
    #[serde(default)]
    seed: u64,
    edges: Vec<[u32; 2]>,
    #[serde(default)]
    rest_len: Option<f32>,
    /// If `Some(K)`, dispatch the kernel one iteration at a time and
    /// record positions every K iterations (plus iteration 0).
    /// Result is returned via the optional `snapshots` field.
    #[serde(default)]
    dump_every: Option<u32>,
}

fn default_iter() -> u32 {
    50
}

#[derive(Serialize)]
struct PositionsJson {
    wall_ms: f64,
    n_iter: u32,
    device: String,
    positions: Vec<[f32; 2]>,
    /// Set when `dump_every` was provided. `snapshots[k]` is the
    /// positions at iteration `k * dump_every` (and the final one).
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshots: Option<Vec<SnapshotEntry>>,
}

#[derive(Serialize)]
struct SnapshotEntry {
    iter: u32,
    positions: Vec<[f32; 2]>,
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as f32) * (1.0 / u32::MAX as f32)
}

fn main() {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .expect("read stdin");
    let graph: GraphJson = serde_json::from_str(&buf).expect("parse graph json");

    let n = graph.n_nodes;
    let n_iter = graph.n_iter;

    // Deterministic initial positions in [-1, 1]^2 from the seed.
    let mut rng = graph.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut pos: Vec<Position> = (0..n)
        .map(|_| Position {
            x: 2.0 * lcg(&mut rng) - 1.0,
            y: 2.0 * lcg(&mut rng) - 1.0,
        })
        .collect();

    let arcs: Vec<Arc2D> = graph
        .edges
        .iter()
        .map(|e| Arc2D {
            src: e[0],
            dst: e[1],
            rest_len: graph.rest_len.unwrap_or(0.05),
            _pad: 0.0,
        })
        .collect();

    // Pick FR gains by graph size — same recipe used in the benchmark.
    // These are the defaults that produce visually reasonable layouts;
    // matching networkx.spring_layout's exact behaviour is not the goal.
    let params = LayoutParams {
        n_vertices: n,
        n_arcs: arcs.len() as u32,
        k_repulsion: 0.05,
        k_attraction: 1.0,
        damping: 0.85,
        dt: 0.02,
        _pad0: 0.0,
        _pad1: 0.0,
    };

    let ctx = VulkanContext::new().expect("vulkan init");
    // Warm-up dispatch (1 iter), then time the real run.
    let _ = fr_run(&ctx, &mut pos.clone(), &arcs, params, 1);

    let snapshots = if let Some(k) = graph.dump_every {
        // Per-iteration step so we can record snapshots at iter 0, k, 2k, ...
        let mut snaps = Vec::new();
        snaps.push(SnapshotEntry {
            iter: 0,
            positions: pos.iter().map(|p| [p.x, p.y]).collect(),
        });
        let t0 = Instant::now();
        for i in 1..=n_iter {
            fr_run(&ctx, &mut pos, &arcs, params, 1).expect("fr dispatch");
            if i % k == 0 || i == n_iter {
                snaps.push(SnapshotEntry {
                    iter: i,
                    positions: pos.iter().map(|p| [p.x, p.y]).collect(),
                });
            }
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        (Some(snaps), wall_ms)
    } else {
        let t0 = Instant::now();
        fr_run(&ctx, &mut pos, &arcs, params, n_iter).expect("fr dispatch");
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        (None, wall_ms)
    };
    let (snapshots, wall_ms) = snapshots;

    let out = PositionsJson {
        wall_ms,
        n_iter,
        device: ctx.device_name(),
        positions: pos.iter().map(|p| [p.x, p.y]).collect(),
        snapshots,
    };
    let s = serde_json::to_string(&out).expect("serialise positions");
    io::stdout().write_all(s.as_bytes()).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}
