import torch
import pytest
from hymeko import PyHypergraphEngine

def test_3d_sparse_tensor_generation():
    engine = PyHypergraphEngine()

    # Load your minimal 2-node, 1-edge hypergraph directly
    file_path = "data/minimal_examples/testing_edges/minimal_test_tensor_values_2nodes_1_edge.hymeko"

    ir = engine.load_file(file_path)
    engine.apply_ir(ir)

    # Extract the 4 distinct arrays
    V = engine.get_node_count()
    k, i, j, v = engine.compile_epoch()
    slice_0 = dense[0]

    assert np.all(slice_0[:V, :V] == 0), "Logic Error: Node-to-Node leakage detected!"
    assert np.all(slice_0[V:, V:] == 0), "Logic Error: Edge-to-Edge leakage detected!"

    # 1. Validate mathematical invariants
    assert len(k) == len(i) == len(j) == len(v), "COO coordinate arrays must be identical in length."
    assert len(k) > 0, "Tensor is empty; parser or IR lowering failed."

    # 2. Construct the 3D representation
    indices = torch.tensor([k, i, j], dtype=torch.int64)
    values = torch.tensor(v, dtype=torch.float32)

    dim_k, dim_i, dim_j = max(k) + 1, max(i) + 1, max(j) + 1
    tensor_3d = torch.sparse_coo_tensor(indices, values, size=(dim_k, dim_i, dim_j))

    # 3. Force evaluation and memory coalescing
    # This ensures overlapping indices (e.g., multiple weights on the same arc) are correctly summed.
    tensor_3d = tensor_3d.coalesce()

    print(f"Test passed successfully.")
    print(f"3D Tensor shape (k, i, j): {tensor_3d.shape}")
    print(f"Non-zero elements (NNZ): {tensor_3d._nnz()}")

if __name__ == "__main__":
    test_3d_sparse_tensor_generation()