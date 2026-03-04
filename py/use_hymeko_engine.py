import torch
# Replace 'hymeko_core' with the actual name of your compiled Rust module
import hymeko

def test_zero_copy_bridge():
    print("[INFO] Initializing Rust Hypergraph Engine...")
    engine = hymeko.PyHypergraphEngine()

    # 1. Build a simple topology (2 nodes, 1 edge)
    n0 = engine.add_node()
    n1 = engine.add_node()
    e0 = engine.add_edge()

    # 2. Add arcs with a deliberate duplicate to test the Rust coalesce function
    engine.add_arc(n0, e0, 1.5)
    engine.add_arc(n1, e0, 2.5)
    engine.add_arc(n1, e0, 1.0) # Duplicate! Should sum to 3.5

    print("[INFO] Compiling epoch in Rust...")
    row_ptr, col_ind, val = engine.compile_epoch()

    print("[INFO] Zero-copy transfer complete. Building PyTorch CSR Tensor...")
    
    # 3. Create PyTorch sparse tensor directly from the referenced memory
    # PyTorch requires 32-bit or 64-bit integers for indices. 
    # torch.as_tensor avoids copying if the underlying numpy array is compatible.
    sparse_tensor = torch.sparse_csr_tensor(
        torch.as_tensor(row_ptr, dtype=torch.int64),
        torch.as_tensor(col_ind, dtype=torch.int64),
        torch.as_tensor(val, dtype=torch.float64),
        size=(2, 1) # dim_i = 2, dim_j = 1
    )

    print("\n[SUCCESS] PyTorch Tensor successfully created without memory copies!")
    print(f"Row Pointers:   {sparse_tensor.crow_indices().tolist()}")
    print(f"Column Indices: {sparse_tensor.col_indices().tolist()}")
    print(f"Values:         {sparse_tensor.values().tolist()}")

if __name__ == "__main__":
    test_zero_copy_bridge()
