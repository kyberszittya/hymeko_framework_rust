import torch
import hymeko # Your compiled Rust extension

def build_sparse_tensor(rust_coo_tensor: hymeko.PyTensorCoo3D) -> torch.Tensor:
    """
    Converts the Rust 3D COO tensor into a PyTorch sparse tensor
    using zero-copy memory transfers.
    """
    # 1. Extract the raw NumPy arrays from the Rust FFI
    # indices_np is shape (3, NNZ) of type int64
    # values_np is shape (NNZ,) of type float32
    indices_np, values_np = rust_coo_tensor.export_to_pytorch()

    # 2. Map NumPy memory directly to PyTorch (Zero-Copy)
    indices_pt = torch.from_numpy(indices_np)
    values_pt = torch.from_numpy(values_np)

    # 3. Fetch the spatial dimensions from the Rust getter
    shape_k, shape_i, shape_j = rust_coo_tensor.shape

    # 4. Construct the mathematically pristine Sparse Tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=indices_pt,
        values=values_pt,
        size=(shape_k, shape_i, shape_j),
        dtype=torch.float32,
        device=torch.device('cpu') # Move to 'cuda' later if using Triton
    )

    # 5. Coalesce the tensor
    # Even though we mathematically sorted and coalesced in Rust,
    # PyTorch's internal C++ autograd engine requires this flag to be explicitly set.
    return sparse_tensor.coalesce()

# --- Example Execution ---
# Assuming 'engine' is your Rust class that holds the compiled graph
def main():
    engine = PyHypergraphEngine()

    # Load your minimal 2-node, 1-edge hypergraph directly
    file_path = "data/typical_graphs/fano_graph.hymeko"

    ir = engine.load_file(file_path)
    rust_tensor = engine.get_coo_tensor()
    hypergraph_tensor = build_sparse_tensor(rust_tensor)
    print(hypergraph_tensor)


if __name__ == "__main__":
    main()
