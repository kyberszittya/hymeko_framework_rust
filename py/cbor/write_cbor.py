import numpy as np
from hymeko import PyHypergraphEngine
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import zlib
import base64

def test_write_fano_cbor():
    engine = PyHypergraphEngine()

    # Load your minimal 2-node, 1-edge hypergraph directly
    file_path = "data/typical_graphs/fano_graph.hymeko"

    ir = engine.load_file(file_path)
    cbor_data = ir.to_cbor()
    print("CBOR representation of the Fano graph:")
    print(cbor_data)
    # Write how many bytes the CBOR representation takes
    print(f"CBOR size in bytes: {len(cbor_data)}")
    compressed = zlib.compress(cbor_data, level=9)
    print(f"Compressed size     : {len(compressed)} bytes")


if __name__ == "__main__":
    test_write_fano_cbor()