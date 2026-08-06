"""PyTorch matched net for the Nagare forward-throughput scaling benchmark.

Same architecture as the Nagare `scaling_bench` example (embed D->H, global
pool, entropy feedback, fused update 4H+1->H, head), parameterized over
batch/points/input-dim/hidden, timed on CPU or CUDA over random data. Throughput
+ memory only, not accuracy.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


def global_pool(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([h.mean(1), h.std(1, unbiased=False), h.amax(1)], dim=-1)


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    ln2 = torch.log(torch.tensor(2.0, device=logits.device))
    return (-(prob * logp).sum(dim=-1, keepdim=True) / ln2).detach()


class Net(nn.Module):
    def __init__(self, d_in: int, hidden: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Linear(d_in, hidden)
        self.first = nn.Linear(3 * hidden, 2)
        self.update = nn.Linear(4 * hidden + 1, hidden)
        self.head = nn.Linear(3 * hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.embed(x))
        pooled = global_pool(h)
        ent = normalized_entropy(self.first(pooled))
        ctx = torch.cat([pooled, ent], dim=-1).unsqueeze(1).expand(-1, x.shape[1], -1)
        h2 = torch.relu(self.update(torch.cat([h, ctx], dim=-1)))
        return self.head(global_pool(h2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--points", type=int, default=64)
    ap.add_argument("--input-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    x = torch.randn(a.batch, a.points, a.input_dim, device=dev)
    net = Net(a.input_dim, a.hidden, 7).to(dev).eval()
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    with torch.inference_mode():
        for _ in range(5):
            net(x)
        sync()
        times = []
        for _ in range(a.reps):
            sync()
            s = time.perf_counter_ns()
            net(x)
            sync()
            times.append((time.perf_counter_ns() - s) / 1e9)
    times.sort()
    med = times[len(times) // 2]
    rows = a.batch * a.points
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if dev.type == "cuda" else float("nan")
    )
    print(
        f"engine=pytorch device={a.device} threads={a.threads} "
        f"batch={a.batch} points={a.points} input_dim={a.input_dim} hidden={a.hidden} | "
        f"median_ms={med * 1e3:.3f} us_per_sample={med * 1e6 / a.batch:.3f} "
        f"Mrows_per_s={rows / med / 1e6:.1f} gpu_peak_mb={peak_mb:.0f}"
    )


if __name__ == "__main__":
    main()
