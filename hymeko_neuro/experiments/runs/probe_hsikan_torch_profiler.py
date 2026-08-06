"""torch.profiler probe of HSiKAN forward+backward — gets CUDA-kernel-level
attribution (py-spy can't see GPU work). Confirms whether the dense
incidence einsum is the wall hotspot before any optimization.

Usage:
    PYTHONPATH=$PWD python -m hymeko_neuro.experiments.runs.probe_hsikan_torch_profiler
"""
from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from hymeko_neuro.experiments.vision.vision_bench_cell import build_model

B, H, W = 128, 28, 28
N_WARMUP = 3
N_ACTIVE = 10


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = build_model("hsikan", h=H, w=W, n_classes=10, hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(B, 1, H, W, device=device)
    y = torch.randint(0, 10, (B,), device=device)

    sched = schedule(wait=1, warmup=N_WARMUP, active=N_ACTIVE)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched, record_shapes=False) as prof:
        for _ in range(1 + N_WARMUP + N_ACTIVE):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            prof.step()

    print("\n=== Top CUDA kernels by self time (averaged over",
          N_ACTIVE, "iterations) ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20,
        max_name_column_width=70,
    ))


if __name__ == "__main__":
    main()
