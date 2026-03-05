"""Benchmark Hymeko expansions across a grid of node/edge sizes and densities.

This module is intentionally standalone so it can be run directly:
    python py/parsing/benchmarks/grid_expansion_bench.py
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
from typing import Iterable, List, Sequence

import torch
from hymeko import PyHypergraphEngine


@dataclass
class BenchmarkConfig:
    nodes: int
    edges: int
    density: float

    @property
    def tag(self) -> str:
        return f"n{self.nodes}_e{self.edges}_d{int(self.density * 100)}"


def generate_hymeko_file(filepath: str, graph_name: str, num_nodes: int, num_edges: int, density: float, seed: int) -> None:
    """Generate a pseudo-random Hymeko hypergraph on disk."""
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
                sampled = random.sample(range(num_nodes), sample_count)
                nodes_in_edge = [f"~n{i}" for i in sampled]
            handle.write(f"    @e{j}{{ ( {', '.join(nodes_in_edge)} ); }}\n")
        handle.write("}\n")


def summarize(samples: Sequence[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean * 1000.0, stdev * 1000.0  # seconds -> milliseconds


def summarize_counts(samples: Sequence[int]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def benchmark_configuration(engine: PyHypergraphEngine, config: BenchmarkConfig, iterations: int) -> dict:
    metrics = {
        "parse": [],
        "star_extract": [],
        "star_tensor": [],
        "star_nnz": [],
        "clique_extract": [],
        "clique_tensor": [],
        "clique_nnz": [],
    }
    raw_records = []

    for trial in range(iterations):
        path = f"data/benchmarks/grid_{config.tag}_trial_{trial}.hymeko"
        generate_hymeko_file(path, config.tag, config.nodes, config.edges, config.density, seed=trial)
        try:
            t0 = time.perf_counter()
            py_ir = engine.load_file(path)
            parse_duration = time.perf_counter() - t0
            metrics["parse"].append(parse_duration)

            t1 = time.perf_counter()
            star_idx, star_i, star_j, star_vals, star_dims = engine.compile_star_expansion(py_ir)
            star_extract_duration = time.perf_counter() - t1
            metrics["star_extract"].append(star_extract_duration)

            t2 = time.perf_counter()
            torch.sparse_coo_tensor(
                torch.tensor([star_idx, star_i, star_j], dtype=torch.int64),
                torch.tensor(star_vals, dtype=torch.float32),
                size=star_dims,
            )
            star_tensor_duration = time.perf_counter() - t2
            metrics["star_tensor"].append(star_tensor_duration)
            star_nnz = len(star_vals)
            metrics["star_nnz"].append(star_nnz)

            t3 = time.perf_counter()
            clique_idx, clique_i, clique_j, clique_vals, clique_dims = engine.compile_clique_tensor_expansion(py_ir)
            clique_extract_duration = time.perf_counter() - t3
            metrics["clique_extract"].append(clique_extract_duration)

            t4 = time.perf_counter()
            torch.sparse_coo_tensor(
                torch.tensor([clique_idx, clique_i, clique_j], dtype=torch.int64),
                torch.tensor(clique_vals, dtype=torch.float32),
                size=clique_dims,
            )
            clique_tensor_duration = time.perf_counter() - t4
            metrics["clique_tensor"].append(clique_tensor_duration)
            clique_nnz = len(clique_vals)
            metrics["clique_nnz"].append(clique_nnz)

            raw_records.append({
                "iteration": trial,
                "parse_ms": parse_duration * 1000.0,
                "star_extract_ms": star_extract_duration * 1000.0,
                "star_tensor_ms": star_tensor_duration * 1000.0,
                "clique_extract_ms": clique_extract_duration * 1000.0,
                "clique_tensor_ms": clique_tensor_duration * 1000.0,
                "star_nnz": star_nnz,
                "clique_nnz": clique_nnz,
            })
        finally:
            if os.path.exists(path):
                os.remove(path)

    return {
        "config": config,
        "iterations": iterations,
        "parse": summarize(metrics["parse"]),
        "star_extract": summarize(metrics["star_extract"]),
        "star_tensor": summarize(metrics["star_tensor"]),
        "star_nnz": summarize_counts(metrics["star_nnz"]),
        "clique_extract": summarize(metrics["clique_extract"]),
        "clique_tensor": summarize(metrics["clique_tensor"]),
        "clique_nnz": summarize_counts(metrics["clique_nnz"]),
        "raw": raw_records,
    }


def benchmark_grid(node_grid: Iterable[int], edge_grid: Iterable[int], densities: Iterable[float], iterations: int = 10) -> List[dict]:
    engine = PyHypergraphEngine()
    configs = [BenchmarkConfig(n, e, d) for n, e, d in itertools.product(node_grid, edge_grid, densities)]
    results = []
    for config in configs:
        print(f"Evaluating {config.nodes} nodes, {config.edges} edges, density {config.density:.2f} ({iterations} trials)...")
        result = benchmark_configuration(engine, config, iterations)
        results.append(result)
    return results


def print_summary_table(results: Sequence[dict]) -> None:
    header = (
        "Config",
        "Iterations",
        "Parse (ms)",
        "Star Extract (ms)",
        "Star Tensor (ms)",
        "Star NNZ",
        "Clique Extract (ms)",
        "Clique Tensor (ms)",
        "Clique NNZ",
    )
    print("\n" + " | ".join(header))
    print("-" * 110)
    for row in results:
        cfg = row["config"]
        fmt = (
            f"{cfg.tag}",
            f"{row['iterations']}",
            f"{row['parse'][0]:7.2f} ± {row['parse'][1]:6.2f}",
            f"{row['star_extract'][0]:7.2f} ± {row['star_extract'][1]:6.2f}",
            f"{row['star_tensor'][0]:7.2f} ± {row['star_tensor'][1]:6.2f}",
            f"{row['star_nnz'][0]:8.0f} ± {row['star_nnz'][1]:6.0f}",
            f"{row['clique_extract'][0]:7.2f} ± {row['clique_extract'][1]:6.2f}",
            f"{row['clique_tensor'][0]:7.2f} ± {row['clique_tensor'][1]:6.2f}",
            f"{row['clique_nnz'][0]:8.0f} ± {row['clique_nnz'][1]:6.0f}",
        )
        print(" | ".join(fmt))


def write_csv_results(results: Sequence[dict], output_dir: str = "data/benchmarks", timestamp: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"grid_expansion_{stamp}.csv"
    path = os.path.join(output_dir, filename)

    headers = [
        "nodes",
        "edges",
        "density",
        "iterations",
        "parse_mean_ms",
        "parse_std_ms",
        "star_extract_mean_ms",
        "star_extract_std_ms",
        "star_tensor_mean_ms",
        "star_tensor_std_ms",
        "star_nnz_mean",
        "star_nnz_std",
        "clique_extract_mean_ms",
        "clique_extract_std_ms",
        "clique_tensor_mean_ms",
        "clique_tensor_std_ms",
        "clique_nnz_mean",
        "clique_nnz_std",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in results:
            cfg = row["config"]
            writer.writerow([
                cfg.nodes,
                cfg.edges,
                cfg.density,
                row["iterations"],
                row["parse"][0],
                row["parse"][1],
                row["star_extract"][0],
                row["star_extract"][1],
                row["star_tensor"][0],
                row["star_tensor"][1],
                row["star_nnz"][0],
                row["star_nnz"][1],
                row["clique_extract"][0],
                row["clique_extract"][1],
                row["clique_tensor"][0],
                row["clique_tensor"][1],
                row["clique_nnz"][0],
                row["clique_nnz"][1],
            ])
    return path


def write_raw_measurements(results: Sequence[dict], output_dir: str = "data/benchmarks", timestamp: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"grid_expansion_raw_{stamp}.csv"
    path = os.path.join(output_dir, filename)

    headers = [
        "nodes",
        "edges",
        "density",
        "iteration",
        "parse_ms",
        "star_extract_ms",
        "star_tensor_ms",
        "clique_extract_ms",
        "clique_tensor_ms",
        "star_nnz",
        "clique_nnz",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in results:
            cfg = row["config"]
            for sample in row["raw"]:
                writer.writerow([
                    cfg.nodes,
                    cfg.edges,
                    cfg.density,
                    sample["iteration"],
                    sample["parse_ms"],
                    sample["star_extract_ms"],
                    sample["star_tensor_ms"],
                    sample["clique_extract_ms"],
                    sample["clique_tensor_ms"],
                    sample["star_nnz"],
                    sample["clique_nnz"],
                ])
    return path


def main() -> None:
    nodes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    edges = [5, 10, 20, 50, 100, 200, 250, 500]
    densities = [0.05, 0.15, 0.30]
    results = benchmark_grid(nodes, edges, densities, iterations=20)
    print_summary_table(results)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = write_csv_results(results, timestamp=stamp)
    raw_csv_path = write_raw_measurements(results, timestamp=stamp)
    print(f"Results written to {csv_path}")
    print(f"Raw measurements written to {raw_csv_path}")


if __name__ == "__main__":
    main()

