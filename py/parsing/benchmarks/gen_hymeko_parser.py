import os
import random
import time
import statistics
import torch
import numpy as np
import matplotlib.pyplot as plt
from hymeko import PyHypergraphEngine

def generate_hymeko_file(filepath, graph_name, num_nodes, num_edges, density):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"{graph_name.capitalize()}{{}}\n")
        f.write(f"{graph_name.lower()}\n{{\n")

        for i in range(num_nodes):
            f.write(f"    n{i} {{}}\n")
        f.write("\n")

        for j in range(num_edges):
            nodes_in_edge = [f"~n{i}" for i in range(num_nodes) if random.random() < density]
            if len(nodes_in_edge) < 2:
                sampled = random.sample(range(num_nodes), min(2, num_nodes))
                nodes_in_edge = [f"~n{i}" for i in sampled]

            f.write(f"    @e{j}{{ ( {', '.join(nodes_in_edge)} ); }}\n")
        f.write("}\n")

def visualize_scaling(results):
    """
    Plots the benchmark results using a grouped bar chart with a logarithmic Y-axis.
    """
    names = [r['name'] for r in results]
    parse_times = [r['parse'] for r in results]
    extract_times = [r['extract'] for r in results]
    tensor_times = [r['tensor'] for r in results]
    nnzs = [r['nnz'] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    # Grouped bars
    ax.bar(x - width, parse_times, width, label='Parse Time (Rust)', color='#4c72b0')
    ax.bar(x, extract_times, width, label='Extract Time (Rust)', color='#dd8452')
    ax.bar(x + width, tensor_times, width, label='Tensor Time (PyTorch)', color='#55a868')

    # Logarithmic scale to handle extreme variances between Parse and Tensor times
    ax.set_yscale('log')
    ax.set_ylabel('Execution Time (ms) [Log Scale]', fontsize=12, fontweight='bold')
    ax.set_title('3D Clique Tensor Compilation: Engine Scalability', fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(x)
    # Append the mathematically crucial NNZ count to the X-axis labels
    ax.set_xticklabels([f"{name}\n(NNZ: {nnz:,})" for name, nnz in zip(names, nnzs)], rotation=0, fontsize=10)

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def benchmark_clique_tensor(filepath, iterations=10):
    engine = PyHypergraphEngine()

    # --- WARM-UP PHASE ---
    # Discard the first run to bypass FFI and PyTorch cold-start penalties
    try:
        py_ir_warmup = engine.load_file(filepath)
        k, i, j, v, dims = engine.compile_clique_tensor_expansion(py_ir_warmup)
        _ = torch.sparse_coo_tensor(
            torch.tensor([k, i, j], dtype=torch.int64),
            torch.tensor(v, dtype=torch.float32),
            size=dims
        ).coalesce()
    except Exception as e:
        print(f"  Failed during warmup: {e}")
        return

    parse_times = []
    extract_times = []
    tensor_times = []
    nnz_count = 0

    # --- STATISTICAL RUNS ---
    for _ in range(iterations):
        # 1. Parsing
        t0 = time.perf_counter()
        py_ir = engine.load_file(filepath)
        parse_times.append(time.perf_counter() - t0)

        # 2. Rust Tensor Extraction
        t1 = time.perf_counter()
        k, i, j, v, dims = engine.compile_clique_tensor_expansion(py_ir)
        extract_times.append(time.perf_counter() - t1)

        # 3. PyTorch Memory Coalescing
        t2 = time.perf_counter()
        indices = torch.tensor([k, i, j], dtype=torch.int64)
        values = torch.tensor(v, dtype=torch.float32)
        clique_tensor_3d = torch.sparse_coo_tensor(indices, values, size=dims).coalesce()
        tensor_times.append(time.perf_counter() - t2)

        nnz_count = clique_tensor_3d._nnz()

    def print_stats(name, data):
        mean = statistics.mean(data) * 1000 # Convert to milliseconds
        stdev = statistics.stdev(data) * 1000 if len(data) > 1 else 0
        print(f"  {name:<12}: {mean:>8.2f} ms ± {stdev:>5.2f} ms")

    mean_parse = statistics.mean(parse_times) * 1000
    mean_extract = statistics.mean(extract_times) * 1000
    mean_tensor = statistics.mean(tensor_times) * 1000

    print(f"  Total NNZ   : {nnz_count:,}")
    print_stats("Parse Time", parse_times)
    print_stats("Extract Time", extract_times)
    print_stats("PyTorch Time", tensor_times)
    print("-" * 50)
    return mean_parse, mean_extract, mean_tensor, nnz_count

def run_scaling_tests():
    configs = [
        (100, 50, 0.1, "small_sparse"),
        (500, 200, 0.05, "medium_sparse"),
        (1000, 500, 0.01, "large_sparse"),
        (100, 50, 0.5, "small_dense"),
        (500, 200, 0.2, "medium_dense")
    ]

    runs_per_config = 10
    results = []

    for v, e, d, name in configs:
        filepath = f"data/benchmarks/{name}.hymeko"
        print(f"Generating {name} (V={v}, E={e}, Density={d})...")
        generate_hymeko_file(filepath, name, v, e, d)

        print(f"Benchmarking {name} ({runs_per_config} iterations)...")
        res = benchmark_clique_tensor(filepath, iterations=runs_per_config)

        if res:
            p_time, e_time, t_time, nnz = res
            results.append({
                'name': name,
                'parse': p_time,
                'extract': e_time,
                'tensor': t_time,
                'nnz': nnz
            })

    if results:
        visualize_scaling(results)


if __name__ == "__main__":
    run_scaling_tests()