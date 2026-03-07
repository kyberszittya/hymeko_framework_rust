"""Batch-evaluate COO tensor exports across node/edge grids.

Run directly via:
    python py/coo_tensor/coo_tensor_grid_eval.py
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Literal, Sequence

import numpy as np
import torch
from hymeko import PyHypergraphEngine

try:  # Support both `python -m` and direct execution.
    from .coo_tensor_pytorch import build_sparse_tensor  # type: ignore
except (ImportError, ValueError):
    from coo_tensor_pytorch import build_sparse_tensor  # type: ignore


ExpansionKind = Literal["star", "clique"]


@dataclass(frozen=True)
class TensorBenchmarkConfig:
    nodes: int
    edges: int
    density: float
    expansion: ExpansionKind

    @property
    def tag(self) -> str:
        density_pct = int(self.density * 100)
        return f"{self.expansion}_n{self.nodes}_e{self.edges}_d{density_pct}"


class TensorCooProxy:
    """Lightweight adapter so build_sparse_tensor can accept Python tuples."""

    def __init__(self, k_idx: Sequence[int], i_idx: Sequence[int], j_idx: Sequence[int], values: Sequence[float], shape: tuple[int, int, int]):
        stacked = np.stack([k_idx, i_idx, j_idx], axis=0).astype(np.int64, copy=False)
        self._indices = stacked
        self._values = np.asarray(values, dtype=np.float32)
        self._shape = shape

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def export_to_pytorch(self):  # Consumed by build_sparse_tensor
        return self._indices, self._values


def generate_hymeko_file(filepath: str, graph_name: str, num_nodes: int, num_edges: int, density: float, seed: int) -> None:
    random.seed(seed)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(f"{graph_name.capitalize()}{{}}\n")
        handle.write(f"{graph_name.lower()}\n{{\n")
        for i in range(num_nodes):
            handle.write(f"    n{i} {{}}\n")
        for j in range(num_edges):
            nodes_in_edge = [f"~n{i}" for i in range(num_nodes) if random.random() < density]
            if len(nodes_in_edge) < 2:
                sample_count = min(2, num_nodes)
                nodes_in_edge = [f"~n{i}" for i in random.sample(range(num_nodes), sample_count)]
            handle.write(f"    @e{j}{{ ( {', '.join(nodes_in_edge)} ); }}\n")
        handle.write("}\n")


def summarize(samples: Sequence[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean * 1000.0, stdev * 1000.0


def summarize_counts(samples: Sequence[int]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def compile_expansion(engine: PyHypergraphEngine, py_ir, expansion: ExpansionKind):
    if expansion == "star":
        return engine.compile_star_expansion(py_ir)
    return engine.compile_clique_tensor_expansion(py_ir)


def benchmark_configuration(engine: PyHypergraphEngine, config: TensorBenchmarkConfig, iterations: int) -> dict:
    metrics = {"parse": [], "extract": [], "tensor": [], "nnz": []}
    raw_records = []

    for trial in range(iterations):
        path = f"data/benchmarks/coo_grid_{config.tag}_trial_{trial}.hymeko"
        generate_hymeko_file(path, config.tag, config.nodes, config.edges, config.density, seed=trial)
        try:
            parse_start = time.perf_counter()
            py_ir = engine.load_file(path)
            parse_duration = time.perf_counter() - parse_start

            extract_start = time.perf_counter()
            k_idx, i_idx, j_idx, values, dims = compile_expansion(engine, py_ir, config.expansion)
            extract_duration = time.perf_counter() - extract_start

            tensor_proxy = TensorCooProxy(k_idx, i_idx, j_idx, values, dims)
            tensor_start = time.perf_counter()
            sparse_tensor = build_sparse_tensor(tensor_proxy)
            tensor_duration = time.perf_counter() - tensor_start

            nnz = sparse_tensor._nnz()

            metrics["parse"].append(parse_duration)
            metrics["extract"].append(extract_duration)
            metrics["tensor"].append(tensor_duration)
            metrics["nnz"].append(nnz)

            raw_records.append({
                "iteration": trial,
                "parse_ms": parse_duration * 1000.0,
                "extract_ms": extract_duration * 1000.0,
                "tensor_ms": tensor_duration * 1000.0,
                "nnz": nnz,
            })
        finally:
            if os.path.exists(path):
                os.remove(path)

    return {
        "config": config,
        "iterations": iterations,
        "parse": summarize(metrics["parse"]),
        "extract": summarize(metrics["extract"]),
        "tensor": summarize(metrics["tensor"]),
        "nnz": summarize_counts(metrics["nnz"]),
        "raw": raw_records,
    }


def benchmark_grid(node_grid: Iterable[int], edge_grid: Iterable[int], densities: Iterable[float], expansions: Iterable[ExpansionKind], iterations: int = 10) -> List[dict]:
    engine = PyHypergraphEngine()
    configs = [
        TensorBenchmarkConfig(nodes, edges, density, expansion)
        for nodes, edges, density, expansion in itertools.product(node_grid, edge_grid, densities, expansions)
    ]
    results = []
    for cfg in configs:
        print(f"[{cfg.tag}] iterations={iterations}")
        results.append(benchmark_configuration(engine, cfg, iterations))
    return results


def print_summary_table(results: Sequence[dict]) -> None:
    header = (
        "Config",
        "Iter",
        "Parse (ms)",
        "Extract (ms)",
        "Tensor Build (ms)",
        "NNZ",
    )
    print("\n" + " | ".join(header))
    print("-" * 80)
    for row in results:
        cfg = row["config"]
        fmt = (
            cfg.tag,
            str(row["iterations"]),
            f"{row['parse'][0]:7.2f} ± {row['parse'][1]:6.2f}",
            f"{row['extract'][0]:7.2f} ± {row['extract'][1]:6.2f}",
            f"{row['tensor'][0]:7.2f} ± {row['tensor'][1]:6.2f}",
            f"{row['nnz'][0]:8.0f} ± {row['nnz'][1]:6.0f}",
        )
        print(" | ".join(fmt))


def write_summary_csv(results: Sequence[dict], output_dir: str = "data/benchmarks", timestamp: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"coo_tensor_grid_{stamp}.csv")
    headers = [
        "expansion",
        "nodes",
        "edges",
        "density",
        "iterations",
        "parse_mean_ms",
        "parse_std_ms",
        "extract_mean_ms",
        "extract_std_ms",
        "tensor_mean_ms",
        "tensor_std_ms",
        "nnz_mean",
        "nnz_std",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in results:
            cfg = row["config"]
            writer.writerow([
                cfg.expansion,
                cfg.nodes,
                cfg.edges,
                cfg.density,
                row["iterations"],
                row["parse"][0],
                row["parse"][1],
                row["extract"][0],
                row["extract"][1],
                row["tensor"][0],
                row["tensor"][1],
                row["nnz"][0],
                row["nnz"][1],
            ])
    return path


def write_raw_csv(results: Sequence[dict], output_dir: str = "data/benchmarks", timestamp: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"coo_tensor_grid_raw_{stamp}.csv")
    headers = [
        "expansion",
        "nodes",
        "edges",
        "density",
        "iteration",
        "parse_ms",
        "extract_ms",
        "tensor_ms",
        "nnz",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in results:
            cfg = row["config"]
            for sample in row["raw"]:
                writer.writerow([
                    cfg.expansion,
                    cfg.nodes,
                    cfg.edges,
                    cfg.density,
                    sample["iteration"],
                    sample["parse_ms"],
                    sample["extract_ms"],
                    sample["tensor_ms"],
                    sample["nnz"],
                ])
    return path


def main() -> None:
    nodes = [32, 64, 128]
    edges = [32, 64, 128]
    densities = [0.05, 0.15, 0.30]
    expansions: tuple[ExpansionKind, ...] = ("star", "clique")
    iterations = 5

    results = benchmark_grid(nodes, edges, densities, expansions, iterations=iterations)
    print_summary_table(results)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary_path = write_summary_csv(results, timestamp=stamp)
    raw_path = write_raw_csv(results, timestamp=stamp)
    print(f"Summary CSV written to {summary_path}")
    print(f"Raw measurements CSV written to {raw_path}")


if __name__ == "__main__":
    main()

