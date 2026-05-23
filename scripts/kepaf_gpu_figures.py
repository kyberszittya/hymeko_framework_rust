"""Regenerate the KEPAF §VII figures using the Vulkan force_directed
kernel for layout instead of networkx.spring_layout — same fixtures,
same renderer, GPU positions.

Pipeline per fixture:
  1. Build the NetworkX graph via kepaf_benchmark.fixture_*.
  2. Map node labels → contiguous u32 ids; serialise edges as JSON.
  3. Pipe JSON to `target/release/examples/layout_from_json` (Rust
     binary that calls hymeko_compute::kernels::force_directed::run).
  4. Read positions JSON, rebuild dict {label: (x, y)}.
  5. Render via the existing kepaf_benchmark.render_layout — figures
     land at paper/kepaf_v1/figures/<name>_layout.{pdf,png},
     overwriting the spring_layout versions.

Usage:
    cargo build --release -p hymeko_compute --example layout_from_json
    python scripts/kepaf_gpu_figures.py
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Anchor cwd to the repo root so kepaf_benchmark's relative
# OUT_FIG_DIR ("paper/kepaf_v1/figures") resolves correctly no matter
# where the user invoked us from.
import os
os.chdir(REPO)
import sys
sys.path.insert(0, str(REPO / "scripts"))

from kepaf_benchmark import (
    fixture_canonical,
    fixture_mnist_adjacency,
    fixture_synthetic,
    render_layout,
)

LAYOUT_BIN = REPO / "target" / "release" / "examples" / "layout_from_json"


def gpu_layout(G, n_iter: int, seed: int = 0) -> tuple[dict, dict]:
    """Run the Vulkan FR kernel on G; return (positions_dict, info)."""
    if not LAYOUT_BIN.exists():
        raise SystemExit(
            f"Missing {LAYOUT_BIN}. Run:\n"
            f"  cargo build --release -p hymeko_compute "
            f"--example layout_from_json"
        )

    labels = list(G.nodes())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    edges = [[label_to_id[u], label_to_id[v]] for u, v in G.edges()]

    payload = json.dumps(
        {
            "n_nodes": len(labels),
            "n_iter": n_iter,
            "seed": seed,
            "edges": edges,
        }
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(LAYOUT_BIN)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    wall_py_ms = (time.perf_counter() - t0) * 1e3
    out = json.loads(proc.stdout)

    pos = {labels[i]: (xy[0], xy[1]) for i, xy in enumerate(out["positions"])}
    info = {
        "wall_kernel_ms": out["wall_ms"],
        "wall_total_ms": wall_py_ms,
        "n_iter": out["n_iter"],
        "device": out["device"],
    }
    return pos, info


def render_one(G, name: str, n_iter: int, max_edges: int = 3000):
    print(f"[{name}] |V|={len(G)} |E|={G.number_of_edges()}  GPU layout...")
    pos, info = gpu_layout(G, n_iter=n_iter, seed=0)
    pdf, png = render_layout(G, pos, name, max_edges=max_edges)
    print(
        f"  device={info['device']}  "
        f"kernel={info['wall_kernel_ms']:.1f}ms  "
        f"total(py+ipc)={info['wall_total_ms']:.1f}ms  "
        f"→ {pdf}"
    )
    return info


def main():
    print("== KEPAF §VII figure regeneration on GPU positions ==")

    # Canonical: 21-vertex Levi graph, 100 iter (matches paper).
    G, name = fixture_canonical()
    render_one(G, name, n_iter=100, max_edges=2000)

    # MNIST adjacency: ~7882 Levi nodes, 50 iter.
    G, name = fixture_mnist_adjacency()
    render_one(G, name, n_iter=50, max_edges=3000)

    # Synthetic |V|=10000: ~35000 Levi nodes, 20 iter.
    G, name = fixture_synthetic(N=10_000, M=25_000)
    render_one(G, name, n_iter=20, max_edges=5000)


if __name__ == "__main__":
    main()
