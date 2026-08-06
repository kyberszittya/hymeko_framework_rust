"""Toy point-cloud learning with global pooling and entropy feedback."""

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
import torch.nn.functional as F


def _make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _sample_moons(n_samples: int, n_points: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(n_samples) % 2
    theta = torch.rand(n_samples, n_points, generator=gen) * torch.pi
    noise = 0.055 * torch.randn(n_samples, n_points, 2, generator=gen)
    x = torch.empty(n_samples, n_points, 2)
    top = labels == 0
    x[top, :, 0] = torch.cos(theta[top])
    x[top, :, 1] = torch.sin(theta[top])
    x[~top, :, 0] = 1.0 - torch.cos(theta[~top])
    x[~top, :, 1] = 0.45 - torch.sin(theta[~top])
    return x + noise, labels.long()


def _sample_rings(n_samples: int, n_points: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(n_samples) % 2
    theta = torch.rand(n_samples, n_points, generator=gen) * (2.0 * torch.pi)
    radius = torch.where(
        labels[:, None] == 0,
        0.55 + 0.055 * torch.randn(n_samples, n_points, generator=gen),
        1.05 + 0.055 * torch.randn(n_samples, n_points, generator=gen),
    )
    x = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
    return x, labels.long()


def _sample_xor_clouds(n_samples: int, n_points: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(n_samples) % 2
    x = torch.empty(n_samples, n_points, 2)
    centers = {
        0: torch.tensor([[-0.75, -0.75], [0.75, 0.75]]),
        1: torch.tensor([[-0.75, 0.75], [0.75, -0.75]]),
    }
    for i in range(n_samples):
        choice = torch.randint(0, 2, (n_points,), generator=gen)
        x[i] = centers[int(labels[i])][choice] + 0.12 * torch.randn(n_points, 2, generator=gen)
    return x, labels.long()


def make_dataset(
    task: str,
    *,
    n_train: int,
    n_test: int,
    n_points: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = _make_generator(seed)
    if task == "moons":
        x, y = _sample_moons(n_train + n_test, n_points, gen)
    elif task == "rings":
        x, y = _sample_rings(n_train + n_test, n_points, gen)
    elif task == "xor":
        x, y = _sample_xor_clouds(n_train + n_test, n_points, gen)
    else:
        raise ValueError(f"unknown task {task!r}")
    perm = torch.randperm(x.shape[0], generator=gen)
    x = x[perm]
    y = y[perm]
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def global_pool(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        h.mean(dim=1),
        h.std(dim=1, unbiased=False),
        h.amax(dim=1),
    ], dim=-1)


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    denom = torch.log(torch.tensor(float(logits.shape[-1]), dtype=logits.dtype, device=logits.device))
    return (-(prob * log_prob).sum(dim=-1, keepdim=True) / denom.clamp_min(1e-6)).detach()


class DeepSetBaseline(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.point = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.point(x)
        logits = self.head(global_pool(h))
        return {"logits": logits, "logits_first": logits, "entropy": normalized_entropy(logits)}


class EntropyFeedbackSetNet(nn.Module):
    """Simultaneous point update conditioned on global pool and predictive entropy."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.first = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.update = nn.Sequential(
            nn.Linear(hidden + 3 * hidden + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.embed(x)
        pooled = global_pool(h)
        logits_first = self.first(pooled)
        ent = normalized_entropy(logits_first)
        global_context = torch.cat([pooled, ent], dim=-1).unsqueeze(1).expand(-1, h.shape[1], -1)
        h2 = self.update(torch.cat([h, global_context], dim=-1))
        logits = self.head(global_pool(h2))
        return {"logits": logits, "logits_first": logits_first, "entropy": ent}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    lr: float = 2e-3
    hidden: int = 32
    batch_size: int = 32


def _iter_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, gen: torch.Generator):
    perm = torch.randperm(x.shape[0], generator=gen)
    for start in range(0, x.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        yield x[idx], y[idx]


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        out = model(x)
        logits = out["logits"]
        pred = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, y)
        acc = (pred == y).float().mean()
        ent = normalized_entropy(logits).mean()
    return {
        "acc": float(acc.item()),
        "loss": float(loss.item()),
        "entropy": float(ent.item()),
        "pred": pred.tolist(),
    }


def _forward_timing_us(model: nn.Module, x: torch.Tensor, repeats: int = 300) -> dict[str, float]:
    model.eval()
    times = []
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


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: TrainConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    gen = _make_generator(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in _iter_batches(x_train, y_train, cfg.batch_size, gen):
            out = model(xb)
            loss = F.cross_entropy(out["logits"], yb)
            if out["logits_first"] is not out["logits"]:
                loss = loss + 0.25 * F.cross_entropy(out["logits_first"], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train = _evaluate(model, x_train, y_train)
    test = _evaluate(model, x_test, y_test)
    return {
        "train": train,
        "test": test,
        "forward_us": _forward_timing_us(model, x_test),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def run_suite(
    *,
    tasks: list[str],
    n_train: int,
    n_test: int,
    n_points: int,
    cfg: TrainConfig,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rows = {}
    for i, task in enumerate(tasks):
        x_train, y_train, x_test, y_test = make_dataset(
            task,
            n_train=n_train,
            n_test=n_test,
            n_points=n_points,
            seed=seed + 100 * i,
        )
        torch.manual_seed(seed + 10 * i)
        baseline = DeepSetBaseline(cfg.hidden)
        torch.manual_seed(seed + 10 * i)
        feedback = EntropyFeedbackSetNet(cfg.hidden)
        rows[task] = {
            "baseline": train_model(baseline, x_train, y_train, x_test, y_test, cfg, seed=seed + 1 + i),
            "entropy_feedback": train_model(feedback, x_train, y_train, x_test, y_test, cfg, seed=seed + 11 + i),
        }
    return {
        "tasks": tasks,
        "n_train": n_train,
        "n_test": n_test,
        "n_points": n_points,
        "global_pool": "concat(mean, std, max) over simultaneous point embeddings",
        "simultaneous_update": "all points receive the same pooled context and entropy scalar in one vectorized update",
        "entropy_feedback": "first-pass predictive entropy is fed into the second simultaneous point update during learning",
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["moons", "rings", "xor"])
    parser.add_argument("--n-train", type=int, default=192)
    parser.add_argument("--n-test", type=int, default=96)
    parser.add_argument("--n-points", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, hidden=args.hidden, batch_size=args.batch_size)
    result = run_suite(
        tasks=args.tasks,
        n_train=args.n_train,
        n_test=args.n_test,
        n_points=args.n_points,
        cfg=cfg,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
