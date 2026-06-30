"""Benchmark: batched vs per-image-loop forward for WalkConvImageClassifier.

The classifier used to loop over the batch (one image at a time → many tiny
kernel launches = dispatch-bound). With the grid topology shared across the
batch, the whole forward now runs on ``(B, n_patches, d)`` in one pass. This
measures the end-to-end gap (median / IQR / worst, CUDA-synchronised).

    python -m signedkan_wip.experiments.runs.bench_walk_conv_batched
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
    Readout,
    WalkConvImageClassifier,
)


def _bench(fn: Callable[[], object], cuda: bool, iters: int = 30,
           warmup: int = 8) -> dict[str, float]:
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
        ts.append((time.perf_counter() - t0) * 1e3)            # ms
    ts.sort()
    return {"median_ms": statistics.median(ts),
            "iqr_ms": ts[3 * len(ts) // 4] - ts[len(ts) // 4],
            "worst_ms": ts[-1]}


def run(canvas: int = 48, batch: int = 64, readout: Readout = Readout.SPATIAL_TREE) -> dict:
    results: dict[str, dict] = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        dev = torch.device(device)
        model = WalkConvImageClassifier(
            image_h=canvas, image_w=canvas, patch_size=4, in_channels=1,
            d_hidden=16, n_classes=10, readout=readout).to(dev).eval()
        imgs = torch.randn(batch, 1, canvas, canvas, device=dev)

        def loop() -> object:
            with torch.no_grad():
                return torch.stack([model._forward_single(imgs[b])
                                    for b in range(batch)], dim=0)

        def batched() -> object:
            with torch.no_grad():
                return model(imgs)

        assert torch.allclose(loop(), batched(), atol=1e-4)
        loop_r = _bench(loop, device == "cuda")
        batch_r = _bench(batched, device == "cuda")
        results[device] = {"loop": loop_r, "batched": batch_r,
                           "speedup": loop_r["median_ms"] / batch_r["median_ms"]}
    return {"config": {"canvas": canvas, "batch": batch,
                       "n_patches": (canvas // 4) ** 2, "readout": readout.value},
            "results": results}


def main(out: str | None = None) -> int:
    report = run()
    c = report["config"]
    print(f"WalkConvImageClassifier forward  (canvas={c['canvas']}, "
          f"n_patches={c['n_patches']}, batch={c['batch']}, readout={c['readout']})")
    for device, r in report["results"].items():
        print(f"  [{device}] per-image loop : {r['loop']['median_ms']:8.2f} ms")
        print(f"  [{device}] batched        : {r['batched']['median_ms']:8.2f} ms")
        print(f"  [{device}] speedup        : {r['speedup']:.1f}x")
    dest = Path(out) if out else Path("reports/walk_conv_batched_bench_20260630.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
