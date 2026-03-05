import torch
import pytest
import numpy as np
from hymeko import PyHypergraphEngine
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def visualize_clique_matrix(clique_tensor):
    """
    Visualizes the V x V adjacency matrix.
    """
    matrix_2d = clique_tensor.to_dense().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(matrix_2d, cmap='viridis')
    fig.colorbar(cax, label='Shared Hyperedges')

    ax.set_title("V x V Clique Expansion (Adjacency Matrix)", pad=20)
    ax.set_xlabel("Target Node (v)")
    ax.set_ylabel("Source Node (u)")

    plt.show()

def coo_to_dense_3d_spatial_first(tensor_3d):
    """
    Converts the coalesced 3D PyTorch sparse tensor into a dense NumPy array
    with spatial dimensions prioritized.
    """
    # tensor_3d is natively shape (K, I, J)
    # .to_dense() resolves it safely
    # .permute(1, 2, 0) shifts the axes to (I, J, K) to create the spatial-first view
    return tensor_3d.to_dense().permute(1, 2, 0).numpy()

def coo_to_dense_3d(k, i, j, v, dims):
    """
    Converts 3D sparse COO arrays into a dense NumPy array.

    Args:
        k (np.ndarray): Slice indices (k).
        i (np.ndarray): Row indices (i).
        j (np.ndarray): Column indices (j).
        v (np.ndarray): Values (v).
        dims (tuple): The (num_slices, dim_i, dim_j) shape metadata.

    Returns:
        np.ndarray: A dense 3D array of shape `dims`.
    """
    # Initialize a dense array of zeros with the explicit dimensions
    # returned by the engine's compile_epoch
    dense_view = np.zeros(dims, dtype=np.float64)

    # Scatter values using the k, i, j coordinate structure
    # NumPy handles the mapping of these index arrays to the dense grid
    dense_view[k, i, j] = v

    return dense_view

def visualize_2d_matrix_projection(k, i, j, v, dims, num_nodes):
    """
    Projects the 3D tensor into a 2D square matrix and marks the
    partition boundary between Nodes and Edges.
    """
    if len(i) == 0:
        print("The tensor is empty. Nothing to project.")
        return

    # Use the explicit dimensions (dim_i, dim_j) from Rust
    num_rows, num_cols = dims[1], dims[2]

    # Aggregation happens automatically during coo_matrix construction via summation
    matrix_2d = coo_matrix((v, (i, j)), shape=(num_rows, num_cols)).toarray()

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix_2d, cmap='magma')
    fig.colorbar(cax, label='Aggregated Weight')

    # Draw the Bipartite Partition Boundary
    # This separates Nodes (0..N-1) from Edges (N..N+E-1)
    ax.axvline(x=num_nodes - 0.5, color='cyan', linestyle='--', alpha=0.6, label='Partition Boundary')
    ax.axhline(y=num_nodes - 0.5, color='cyan', linestyle='--', alpha=0.6)

    ax.set_title("Aggregated 2D Star Expansion Matrix", pad=20)
    ax.set_xlabel("Target (j)")
    ax.set_ylabel("Source (i)")

    # Label the quadrants
    ax.text(num_nodes/2, -1, "Nodes (V)", color='blue', ha='center')
    ax.text(num_nodes + (num_cols-num_nodes)/2, -1, "Edges (E)", color='green', ha='center')

    plt.show()

def visualize_3d_topology(k, i, j, v):
    if len(k) == 0:
        print("The tensor is empty. Nothing to visualize.")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Map COO arrays to 3D scatter plot coordinates
    # k: Slice (Z-axis), i: Row (X-axis), j: Col (Y-axis)
    scatter = ax.scatter(i, j, k, c=v, cmap='plasma', s=100, depthshade=True, alpha=0.8)

    ax.set_title("3D Sparse Hypergraph Topology")
    ax.set_xlabel('Nodes (i)')
    ax.set_ylabel('Edges (j)')
    ax.set_zlabel('Slices (k)')

    # Attach a colorbar to read the weight values
    fig.colorbar(scatter, ax=ax, label='Arc Weight (v)')

    plt.show()

def test_fano_sparse_tensor_generation():
    engine = PyHypergraphEngine()

    # Load your minimal 2-node, 1-edge hypergraph directly
    file_path = "data/typical_graphs/fano_graph.hymeko"

    ir = engine.load_file(file_path)

    # Extract the 4 distinct arrays
    k, i, j, v, dims = engine.compile_star_expansion(ir)
    indices = torch.tensor([k, i, j], dtype=torch.int64)
    values = torch.tensor(v, dtype=torch.float32)

    tensor_3d = torch.sparse_coo_tensor(indices, values, size=dims).coalesce()



    print(f"Test passed successfully.")
    print(f"3D Tensor shape (k, i, j): {tensor_3d.shape}")
    print(f"Non-zero elements (NNZ): {tensor_3d._nnz()}")
    # Print the tensor
    print(tensor_3d)
    dense_tensor = coo_to_dense_3d_spatial_first(tensor_3d)
    V = engine.get_node_count()
    slice_0 = dense_tensor[0]

    assert np.all(slice_0[:V, :V] == 0), "Logic Error: Node-to-Node leakage detected!"
    assert np.all(slice_0[V:, V:] == 0), "Logic Error: Edge-to-Edge leakage detected!"

    print("Bipartite structure verified.")

    print(dense_tensor.shape)
    print(dense_tensor)
    visualize_3d_topology(k, i, j, v)
    num_nodes = engine.get_node_count()
    num_nodes = engine.get_node_count()
    visualize_2d_matrix_projection(k, i, j, v, dims, num_nodes)

def test_fano_clique_tensor_generation():
    engine = PyHypergraphEngine()
    file_path = "data/typical_graphs/fano_graph.hymeko"
    py_ir = engine.load_file(file_path)

    # 1. Extract the native 3D Clique Tensor
    k, i, j, v, dims = engine.compile_clique_tensor_expansion(py_ir)

    indices = torch.tensor([k, i, j], dtype=torch.int64)
    values = torch.tensor(v, dtype=torch.float32)

    # Coalesce the E x V x V tensor
    clique_tensor_3d = torch.sparse_coo_tensor(indices, values, size=dims).coalesce()

    print("=== 3D CLIQUE TENSOR ===")
    print(f"Shape: {clique_tensor_3d.shape}")
    print(f"Non-zero elements (NNZ): {clique_tensor_3d._nnz()}")
    # For Fano: 7 edges * (3 nodes * 2 directed connections) = exactly 42 NNZ.
    print("------------------------\n")

    # 2. Mathematically recover the 2D Adjacency Matrix by summing across the edges (dim=0)
    clique_matrix_2d = torch.sparse.sum(clique_tensor_3d, dim=0).coalesce()

    print("=== RECOVERED 2D CLIQUE MATRIX ===")
    print(f"Shape: {clique_matrix_2d.shape}")
    print(f"Non-zero elements (NNZ): {clique_matrix_2d._nnz()}")
    print("----------------------------------\n")

    # Visualize the recovered 2D projection
    visualize_clique_matrix(clique_matrix_2d)


if __name__ == "__main__":
    test_fano_sparse_tensor_generation()
    test_fano_clique_tensor_generation()