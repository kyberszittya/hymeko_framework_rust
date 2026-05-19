"""Profile a single Config-E forward pass of RicciStimDetector.

Companion to ``docs/plans/2026-05-15-ricci-stim-opt-pass-5/plan.tex``,
section P5 (``Profile the unaccounted 10 ms``).

This script is paper-only until the in-flight ablation
(``ricci-stim-ablation-2026-05-15.service``) completes. It does NOT
modify any source under ``signedkan_wip/src/`` — it only invokes the
detector through its public forward.

Usage
-----
After the ablation lands::

    python -m signedkan_wip.experiments.profile_ricci_stim_config_e \\
        --out reports/2026-05-15-ricci-stim-opt-pass-5/profile/

What it captures
----------------
For Config E (``alpha=beta=0.05``, ``sdrf=True``) on a single 28x28
ClutteredMNIST-style image:

* Per-op CPU + (if available) CUDA timing via ``torch.profiler``.
* Top-N hot operators by self CPU time.
* Memory deltas per op (peak allocator-tracked).
* A flamegraph-compatible export (``.json`` chrome-trace) + the standard
  ``key_averages().table()`` text dump.

The intent is to identify the unaccounted ~10 ms hot ops on the per-image
forward path so P5 can replace them with cheaper equivalents.

Determinism
-----------
Random seed pinned (0). Single warm-up pass before measurement to amortise
allocator + first-touch costs. Five measured passes; median wall and a
single profiler run on the median-equivalent pass.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.profiler

from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_detector import (
    RicciStimDetector,
)


def _make_config_e_detector(device: torch.device) -> RicciStimDetector:
    """Build the Config-E variant (alpha=beta=0.05, sdrf=True).

    Matches the ablation runner's Config-E construction in
    ``run_ricci_stim_cluttered_mnist_ablation.sh`` /
    ``ricci_stim_train.py``.
    """
    det = RicciStimDetector(
        image_h=28,
        image_w=28,
        patch_size_initial=4,
        patch_size_min=1,
        in_channels=1,
        d_hidden=16,
        n_classes=10,
        max_depth=2,
        max_anchors=256,
        score_threshold=0.05,
        bochner_alpha=0.05,
        bochner_beta=0.05,
        use_sdrf=True,
        sdrf_max_iters=5,
        sdrf_kappa_target=-2.0,
    ).to(device)
    det.eval()
    return det


def _make_dummy_image(device: torch.device, seed: int = 0) -> torch.Tensor:
    """Single (1, 28, 28) image, deterministic across runs."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand((1, 28, 28), generator=g)
    return img.to(device)


def _measure_wall_ms(
    det: RicciStimDetector, img: torch.Tensor, n_passes: int,
) -> list[float]:
    """Return per-pass wall-clock times in ms (after one warm-up)."""
    with torch.no_grad():
        # Warm-up.
        _ = det(img)
        if img.is_cuda:
            torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(n_passes):
            t0 = time.perf_counter()
            _ = det(img)
            if img.is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _profile_one_pass(
    det: RicciStimDetector,
    img: torch.Tensor,
    out_dir: Path,
) -> None:
    """Run torch.profiler over one forward pass and dump artefacts."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if img.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = out_dir / "trace.json"
    table_path = out_dir / "profile_table.txt"
    summary_path = out_dir / "profile_summary.json"

    with torch.no_grad(), torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        _ = det(img)
        if img.is_cuda:
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))

    key_avgs = prof.key_averages()
    sort_key = (
        "self_cuda_time_total" if img.is_cuda else "self_cpu_time_total"
    )
    table_path.write_text(
        key_avgs.table(sort_by=sort_key, row_limit=40, header=str(sort_key)),
        encoding="utf-8",
    )

    summary: list[dict] = []
    for ev in key_avgs:
        summary.append(
            {
                "key": ev.key,
                "count": ev.count,
                "self_cpu_us": getattr(ev, "self_cpu_time_total", 0),
                "cpu_us": getattr(ev, "cpu_time_total", 0),
                "self_cuda_us": getattr(ev, "self_cuda_time_total", 0),
                "cuda_us": getattr(ev, "cuda_time_total", 0),
                "cpu_memory_usage": getattr(ev, "cpu_memory_usage", 0),
                "cuda_memory_usage": getattr(ev, "cuda_memory_usage", 0),
            }
        )
    summary.sort(key=lambda d: d[sort_key.replace("_total", "").replace(
        "_time", "_us"
    )], reverse=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/2026-05-15-ricci-stim-opt-pass-5/profile"),
        help="Output directory for profile artefacts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-wall-passes", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help=(
            "Profiling device. CPU is the canonical case for the unaccounted-"
            "ms P5 hypothesis (Python dispatch + sparse coalesce)."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    det = _make_config_e_detector(device)
    img = _make_dummy_image(device, seed=args.seed)

    n_params = det.n_parameters()
    wall_ms = _measure_wall_ms(det, img, args.n_wall_passes)
    wall_median = statistics.median(wall_ms)
    wall_iqr = (
        statistics.quantiles(wall_ms, n=4)[2]
        - statistics.quantiles(wall_ms, n=4)[0]
        if len(wall_ms) >= 4
        else 0.0
    )

    _profile_one_pass(det, img, out_dir)

    meta = {
        "device": str(device),
        "config": "E (alpha=beta=0.05, sdrf=True)",
        "n_params": n_params,
        "image_shape": list(img.shape),
        "seed": args.seed,
        "n_wall_passes": args.n_wall_passes,
        "wall_ms_per_pass": wall_ms,
        "wall_ms_median": wall_median,
        "wall_ms_iqr": wall_iqr,
        "wall_ms_worst": max(wall_ms),
        "torch_version": torch.__version__,
        "host_pid": os.getpid(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"[profile] device={device}")
    print(f"[profile] n_params={n_params}")
    print(
        f"[profile] wall ms (median/IQR/worst over {args.n_wall_passes}): "
        f"{wall_median:.2f} / {wall_iqr:.2f} / {max(wall_ms):.2f}"
    )
    print(f"[profile] artefacts → {out_dir}")


if __name__ == "__main__":
    main()
