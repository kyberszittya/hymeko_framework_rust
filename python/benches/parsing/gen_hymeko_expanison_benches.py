import os
import random
import time
import statistics
import torch
import numpy as np
import matplotlib.pyplot as plt
from hymeko import PyHypergraphEngine

def generate_hymeko_file(filepath, graph_name, num_nodes, num_edges, density, seed):
    """Generates a unique random hypergraph based on a seed."""
    random.seed(seed)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"{graph_name.capitalize()}{{}}\n")
        f.write(f"{graph_name.lower()}\n{{\n")
        for i in range(num_nodes):
            f.write(f"    n{i} {{}}\n")
        for j in range(num_edges):
            nodes_in_edge = [f"~n{i}" for i in range(num_nodes) if random.random() < density]
            if len(nodes_in_edge) < 2:
                sampled = random.sample(range(num_nodes), min(2, num_nodes))
                nodes_in_edge = [f"~n{i}" for i in sampled]
            f.write(f"    @e{j}{{ ( {', '.join(nodes_in_edge)} ); }}\n")
        f.write("}\n")

def benchmark_expansions(v, e, d, name, iterations=50):
    """
    Performs N iterations, generating a fresh graph for each to ensure
    statistical independence.
    """
    engine = PyHypergraphEngine()
    metrics = {
        'parse': [], 'star_ext': [], 'star_pt': [],
        'clique_ext': [], 'clique_pt': [], 'star_nnz': [], 'clique_nnz': []
    }

    print(f"Running {iterations} statistical trials for {name}...")

    for trial in range(iterations):
        path = f"data/benchmarks/{name}_trial_{trial}.hymeko"
        generate_hymeko_file(path, name, v, e, d, seed=trial)

        # 1. Parse Phase
        t0 = time.perf_counter()
        py_ir = engine.load_file(path)
        metrics['parse'].append(time.perf_counter() - t0)

        # 2. Star Expansion
        t1 = time.perf_counter()
        k_s, i_s, j_s, v_s, dims_s = engine.compile_star_expansion(py_ir)
        metrics['star_ext'].append(time.perf_counter() - t1)

        t2 = time.perf_counter()
        star_t = torch.sparse_coo_tensor(torch.tensor([k_s, i_s, j_s], dtype=torch.int64),
                                        torch.tensor(v_s, dtype=torch.float32), size=dims_s)
        metrics['star_pt'].append(time.perf_counter() - t2)
        metrics['star_nnz'].append(star_t._nnz())

        # 3. Clique Expansion
        t3 = time.perf_counter()
        k_c, i_c, j_c, v_c, dims_c = engine.compile_clique_tensor_expansion(py_ir)
        metrics['clique_ext'].append(time.perf_counter() - t3)

        t4 = time.perf_counter()
        clique_t = torch.sparse_coo_tensor(torch.tensor([k_c, i_c, j_c], dtype=torch.int64),
                                          torch.tensor(v_c, dtype=torch.float32), size=dims_c)
        metrics['clique_pt'].append(time.perf_counter() - t4)
        metrics['clique_nnz'].append(clique_t._nnz())

        os.remove(path) # Cleanup temporary files

    def summarize(data):
        return statistics.mean(data) * 1000, statistics.stdev(data) * 1000

    results = {
        'name': name,
        'parse': summarize(metrics['parse']),
        'star_ext': summarize(metrics['star_ext']),
        'star_pt': summarize(metrics['star_pt']),
        'star_nnz': (statistics.mean(metrics['star_nnz']), statistics.stdev(metrics['star_nnz'])),
        'clique_ext': summarize(metrics['clique_ext']),
        'clique_pt': summarize(metrics['clique_pt']),
        'clique_nnz': (statistics.mean(metrics['clique_nnz']), statistics.stdev(metrics['clique_nnz']))
    }

    # High-precision stdout report
    print(f"\nSTATISTICAL SUMMARY: {name} (N={iterations})")
    print(f"{'-'*60}")
    print(f"Parse Time        : {results['parse'][0]:>8.3f} ms ± {results['parse'][1]:>6.3f} ms")
    print(f"Star NNZ (Avg)    : {results['star_nnz'][0]:>8.0f} ± {results['star_nnz'][1]:>6.0f}")
    print(f"Star Extract      : {results['star_ext'][0]:>8.3f} ms ± {results['star_ext'][1]:>6.3f} ms")
    print(f"Clique NNZ (Avg)  : {results['clique_nnz'][0]:>8.0f} ± {results['clique_nnz'][1]:>6.0f}")
    print(f"Clique Extract    : {results['clique_ext'][0]:>8.3f} ms ± {results['clique_ext'][1]:>6.3f} ms")
    print(f"{'-'*60}\n")

    return results

def visualize_rigorous_comparison(results):
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

    # Execution Times with Error Bars (± 1 Std Dev)
    ax1.bar(x - 1.5*width, [r['star_ext'][0] for r in results], width, yerr=[r['star_ext'][1] for r in results],
           label='Star Extract (Rust)', color='#4c72b0', capsize=5)
    ax1.bar(x - 0.5*width, [r['star_pt'][0] for r in results], width, yerr=[r['star_pt'][1] for r in results],
           label='Star Tensor (PyTorch)', color='#81a1c1', capsize=5)
    ax1.bar(x + 0.5*width, [r['clique_ext'][0] for r in results], width, yerr=[r['clique_ext'][1] for r in results],
           label='Clique Extract (Rust)', color='#dd8452', capsize=5)
    ax1.bar(x + 1.5*width, [r['clique_pt'][0] for r in results], width, yerr=[r['clique_pt'][1] for r in results],
           label='Clique Tensor (PyTorch)', color='#ebcb8b', capsize=5)

    ax1.set_yscale('log')
    ax1.set_ylabel('Execution Time (ms) [Log Scale]', fontweight='bold')
    ax1.set_title(f'Performance Distribution across N=50 Independent Trials', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="-", alpha=0.1)

    # Topology Scaling (NNZ variance)
    ax2.bar(x - 0.2, [r['star_nnz'][0] for r in results], 0.4, yerr=[r['star_nnz'][1] for r in results],
           label='Avg Star NNZ', color='#4c72b0', alpha=0.7, capsize=5)
    ax2.bar(x + 0.2, [r['clique_nnz'][0] for r in results], 0.4, yerr=[r['clique_nnz'][1] for r in results],
           label='Avg Clique NNZ', color='#dd8452', alpha=0.7, capsize=5)

    ax2.set_yscale('log')
    ax2.set_ylabel('Non-Zero Elements [Log]', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_configs = [
        (200, 100, 0.05, "sparse_200"),
        (200, 100, 0.20, "dense_200"),
        (1000, 300, 0.02, "large_sparse"),
        (500, 100, 0.3, "dense_500"),
        (500, 100, 0.50, "extreme_dense_500"),
        (1500, 400, 0.02, "sparse_1500"),
    ]

    final_stats = [benchmark_expansions(v, e, d, n, iterations=50) for v, e, d, n in test_configs]
    visualize_rigorous_comparison(final_stats)