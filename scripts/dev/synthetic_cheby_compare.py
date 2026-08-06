"""PyTorch fixture/exporter for synthetic Chebyshev classifier comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


CHEBYSHEV_DOMAIN_SCALE = 0.5


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def make_moons(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(n) % 2
    theta = torch.rand(n, generator=gen) * torch.pi
    x = torch.empty(n, 2)
    top = y == 0
    x[top, 0] = torch.cos(theta[top])
    x[top, 1] = torch.sin(theta[top])
    x[~top, 0] = 1.0 - torch.cos(theta[~top])
    x[~top, 1] = 0.45 - torch.sin(theta[~top])
    x += 0.055 * torch.randn(n, 2, generator=gen)
    return x, y.long()


def make_spiral(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(n) % 2
    t = torch.rand(n, generator=gen) * 3.5 * torch.pi
    radius = 0.12 + 0.08 * t
    phase = y.float() * torch.pi
    x = torch.stack([radius * torch.cos(t + phase), radius * torch.sin(t + phase)], dim=-1)
    x += 0.045 * torch.randn(n, 2, generator=gen)
    x = x / x.abs().amax().clamp_min(1e-6)
    return x, y.long()


def make_xor(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.tensor([[-0.75, -0.75], [0.75, 0.75], [-0.75, 0.75], [0.75, -0.75]])
    idx = torch.randint(0, 4, (n,), generator=gen)
    y = (idx >= 2).long()
    x = centers[idx] + 0.12 * torch.randn(n, 2, generator=gen)
    return x, y


def make_dataset(task: str, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = make_generator(seed)
    if task == "moons":
        return make_moons(n, gen)
    if task == "spiral":
        return make_spiral(n, gen)
    if task == "xor":
        return make_xor(n, gen)
    raise ValueError(f"unknown task {task!r}")


def chebyshev_terms(x: torch.Tensor, k: int) -> torch.Tensor:
    terms = [torch.ones_like(x)]
    if k > 1:
        terms.append(x)
    for _ in range(2, k):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return torch.stack(terms[:k], dim=-1)


class ChebyClassifier(nn.Module):
    def __init__(self, hidden: int, k: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.input = nn.Linear(2, hidden)
        self.cheb_coef = nn.Parameter(torch.randn(hidden, k) * 0.1)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input(x) * CHEBYSHEV_DOMAIN_SCALE
        h = (chebyshev_terms(z, self.cheb_coef.shape[1]) * self.cheb_coef).sum(dim=-1)
        return self.head(h)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    values: torch.Tensor


def linear_weight_in_out(layer: nn.Linear) -> torch.Tensor:
    return layer.weight.detach().t().contiguous()


def timing(model: nn.Module, x: torch.Tensor, repeats: int) -> dict[str, float]:
    values: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            model(x)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(x)
            values.append((time.perf_counter_ns() - start) / 1000.0 / x.shape[0])
    return {
        "mean_us_per_sample": statistics.mean(values),
        "median_us_per_sample": statistics.median(values),
        "max_us_per_sample": max(values),
        "min_us_per_sample": min(values),
    }


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def flatten_values(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.detach().cpu().contiguous().reshape(-1)]


def write_tensor(f, spec: TensorSpec) -> None:
    values = flatten_values(spec.values)
    shape = ",".join(str(v) for v in spec.values.shape)
    f.write(f"tensor {spec.name} {len(values)} {shape}\n")
    for start in range(0, len(values), 8):
        f.write(" ".join(f"{v:.9g}" for v in values[start : start + 8]) + "\n")
    f.write("endtensor\n")


def build_case(task: str, args: argparse.Namespace, index: int) -> tuple[str, list[TensorSpec], dict[str, Any]]:
    x, y = make_dataset(task, args.n_samples, args.seed + 100 * index)
    model = ChebyClassifier(args.hidden, args.k, args.seed + 10 * index).eval()
    with torch.inference_mode():
        logits = model(x)
    specs = [
        TensorSpec("x", x),
        TensorSpec("y", y.to(torch.float32)),
        TensorSpec("input_w", linear_weight_in_out(model.input)),
        TensorSpec("input_b", model.input.bias.detach()),
        TensorSpec("cheb_coef", model.cheb_coef.detach()),
        TensorSpec("head_w", linear_weight_in_out(model.head)),
        TensorSpec("head_b", model.head.bias.detach()),
        TensorSpec("logits", logits),
    ]
    return task, specs, {
        "task": task,
        "n_samples": args.n_samples,
        "hidden": args.hidden,
        "k": args.k,
        "activation_policy": "chebyshev_domain_rescale_only",
        "chebyshev_domain_scale": CHEBYSHEV_DOMAIN_SCALE,
        "forward_us": timing(model, x, args.repeats),
        "acc_random_weights": accuracy(logits, y),
        "n_params": sum(p.numel() for p in model.parameters()),
        "param_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "logits_checksum": float(logits.sum().item()),
    }


def write_fixture(path: Path, cases: list[tuple[str, list[TensorSpec], dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("HYMEKO_SYNTHETIC_CHEBY_COMPARE_V1\n")
        f.write(f"cases {len(cases)}\n")
        for task, specs, summary in cases:
            f.write(f"case {task}\n")
            f.write(f"samples {summary['n_samples']}\n")
            f.write(f"hidden {summary['hidden']}\n")
            f.write(f"k {summary['k']}\n")
            for spec in specs:
                write_tensor(f, spec)
            f.write("endcase\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["moons", "spiral", "xor"])
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(8)
    cases = [build_case(task, args, idx) for idx, task in enumerate(args.tasks)]
    write_fixture(args.fixture, cases)
    summary = {
        "engine": "pytorch",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "fixture": str(args.fixture),
        "cases": [case[2] for case in cases],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
