"""Export PyTorch fixtures for Nagare global-pool entropy forward parity."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.global_pool_entropy_toys import make_dataset  # noqa: E402


def global_pool(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([h.mean(dim=1), h.std(dim=1, unbiased=False), h.amax(dim=1)], dim=-1)


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    return (-(prob * log_prob).sum(dim=-1, keepdim=True) / torch.log(torch.tensor(2.0))).detach()


class ParityEntropyNet(nn.Module):
    """Single-layer ReLU version mirrored by the Nagare/Rust parity harness."""

    def __init__(self, hidden: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Linear(2, hidden)
        self.first = nn.Linear(3 * hidden, 2)
        self.update = nn.Linear(4 * hidden + 1, hidden)
        self.head = nn.Linear(3 * hidden, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.relu(self.embed(x))
        pooled = global_pool(h)
        logits_first = self.first(pooled)
        ent = normalized_entropy(logits_first)
        context = torch.cat([pooled, ent], dim=-1).unsqueeze(1).expand(-1, x.shape[1], -1)
        h2 = torch.relu(self.update(torch.cat([h, context], dim=-1)))
        logits = self.head(global_pool(h2))
        return {"logits": logits, "logits_first": logits_first, "entropy": ent}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    values: torch.Tensor


def linear_weight_in_out(layer: nn.Linear) -> torch.Tensor:
    return layer.weight.detach().t().contiguous()


def tensor_specs(model: ParityEntropyNet, x: torch.Tensor, out: dict[str, torch.Tensor]) -> list[TensorSpec]:
    return [
        TensorSpec("x", x),
        TensorSpec("embed_w", linear_weight_in_out(model.embed)),
        TensorSpec("embed_b", model.embed.bias.detach()),
        TensorSpec("first_w", linear_weight_in_out(model.first)),
        TensorSpec("first_b", model.first.bias.detach()),
        TensorSpec("update_w", linear_weight_in_out(model.update)),
        TensorSpec("update_b", model.update.bias.detach()),
        TensorSpec("head_w", linear_weight_in_out(model.head)),
        TensorSpec("head_b", model.head.bias.detach()),
        TensorSpec("logits", out["logits"]),
        TensorSpec("logits_first", out["logits_first"]),
        TensorSpec("entropy", out["entropy"]),
    ]


def forward_timing_us(model: nn.Module, x: torch.Tensor, repeats: int) -> dict[str, float]:
    model.eval()
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(20):
            model(x)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(x)
            times.append((time.perf_counter_ns() - start) / 1000.0 / x.shape[0])
    return {
        "mean_us_per_sample": statistics.mean(times),
        "median_us_per_sample": statistics.median(times),
        "min_us_per_sample": min(times),
        "max_us_per_sample": max(times),
    }


def profiler_cpu_memory_bytes(model: nn.Module, x: torch.Tensor) -> int:
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        model(x)
    return int(sum(max(0, evt.self_cpu_memory_usage) for evt in prof.key_averages()))


def flatten_values(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.detach().cpu().contiguous().reshape(-1)]


def write_tensor(f, spec: TensorSpec) -> None:
    vals = flatten_values(spec.values)
    shape = ",".join(str(v) for v in spec.values.shape)
    f.write(f"tensor {spec.name} {len(vals)} {shape}\n")
    for start in range(0, len(vals), 8):
        f.write(" ".join(f"{v:.9g}" for v in vals[start : start + 8]) + "\n")
    f.write("endtensor\n")


def build_case(task: str, args: argparse.Namespace, index: int) -> tuple[str, list[TensorSpec], dict[str, Any]]:
    _, _, x_test, _ = make_dataset(
        task,
        n_train=args.n_train,
        n_test=args.n_test,
        n_points=args.n_points,
        seed=args.seed + 100 * index,
    )
    x = x_test.to(dtype=torch.float32)
    model = ParityEntropyNet(args.hidden, args.seed + 10 * index).eval()
    with torch.inference_mode():
        out = model(x)
    timing = forward_timing_us(model, x, args.repeats)
    memory_bytes = profiler_cpu_memory_bytes(model, x)
    specs = tensor_specs(model, x, out)
    summary = {
        "task": task,
        "samples": args.n_test,
        "points": args.n_points,
        "hidden": args.hidden,
        "forward_us": timing,
        "profiler_cpu_memory_bytes": memory_bytes,
        "n_params": sum(p.numel() for p in model.parameters()),
        "param_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "logits_checksum": float(out["logits"].sum().item()),
    }
    return task, specs, summary


def write_fixture(path: Path, cases: list[tuple[str, list[TensorSpec], dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("HYMEKO_GLOBAL_POOL_ENTROPY_PARITY_V1\n")
        f.write(f"cases {len(cases)}\n")
        for task, specs, summary in cases:
            f.write(f"case {task}\n")
            f.write(f"samples {summary['samples']}\n")
            f.write(f"points {summary['points']}\n")
            f.write(f"hidden {summary['hidden']}\n")
            for spec in specs:
                write_tensor(f, spec)
            f.write("endcase\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["moons", "rings", "xor"])
    parser.add_argument("--n-train", type=int, default=192)
    parser.add_argument("--n-test", type=int, default=96)
    parser.add_argument("--n-points", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(8)
    cases = [build_case(task, args, index) for index, task in enumerate(args.tasks)]
    write_fixture(args.fixture, cases)
    summary = {
        "engine": "pytorch",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "fixture": str(args.fixture),
        "cases": [case_summary for _, _, case_summary in cases],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
