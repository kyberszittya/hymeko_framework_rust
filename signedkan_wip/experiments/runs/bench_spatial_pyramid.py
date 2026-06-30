"""Benchmark: batched matmul vs per-item loop for SpatialPyramidPool.

The classifier currently calls the readout per image (a Python loop → B tiny
kernel launches = dispatch-bound). Because the pyramid is a linear operator
(``cells = P @ features``), the whole batch is one fused matmul. This measures
the gap on CPU and CUDA (median / IQR / worst over warmed-up runs).

    python -m signedkan_wip.experiments.runs.bench_spatial_pyramid
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from signedkan_wip.src.vision.spatial_pyramid import SpatialPyramidPool, grid_positions


def _bench(fn: Callable[[], object], device: str, iters: int = 60,
           warmup: int = 15) -> dict[str, float]:
    """Median / IQR / worst wall time in microseconds (CUDA-synchronised)."""
    cuda = device == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts: list[float] = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return {
        "median_us": statistics.median(ts),
        "iqr_us": ts[3 * len(ts) // 4] - ts[len(ts) // 4],
        "worst_us": ts[-1],
    }


def run(h: int = 12, w: int = 12, d: int = 16, batch: int = 128) -> dict:
    n = h * w
    results: dict[str, dict] = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        dev = torch.device(device)
        pool = SpatialPyramidPool(d).to(dev)
        pool.set_fixed_positions(grid_positions(h, w).to(dev))
        feats = torch.randn(batch, n, d, device=dev)

        def loop() -> object:
            return torch.stack([pool(feats[b]) for b in range(batch)], dim=0)

        def batched() -> object:
            return pool(feats)

        # Correctness gate: the two must agree before timing.
        assert torch.allclose(loop(), batched(), atol=1e-5)
        loop_r = _bench(loop, device)
        batch_r = _bench(batched, device)
        speedup = loop_r["median_us"] / batch_r["median_us"]
        results[device] = {"loop": loop_r, "batched": batch_r, "speedup": speedup}
    return {"config": {"h": h, "w": w, "d": d, "batch": batch, "n_items": n},
            "results": results}


def main(out: str | None = None) -> int:
    report = run()
    cfg = report["config"]
    print(f"SpatialPyramidPool readout  (N={cfg['n_items']}, d={cfg['d']}, "
          f"batch={cfg['batch']})")
    for device, r in report["results"].items():
        print(f"  [{device}] per-item loop : {r['loop']['median_us']:8.1f} µs "
              f"(IQR {r['loop']['iqr_us']:.1f}, worst {r['loop']['worst_us']:.1f})")
        print(f"  [{device}] batched matmul: {r['batched']['median_us']:8.1f} µs "
              f"(IQR {r['batched']['iqr_us']:.1f}, worst {r['batched']['worst_us']:.1f})")
        print(f"  [{device}] speedup       : {r['speedup']:.1f}x")
    dest = Path(out) if out else Path("reports/spatial_pyramid_bench_20260630.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
