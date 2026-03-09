package HymekoProcess {

    action def 'Process Hypergraph' {
        first start;
        // Actions within the Rust AST Producer        
        then action 'Load Index' {
            doc /* Load into BTreeMap for deterministic order */
        }

        then action 'Compute Blake3 Hash' {
            doc /* Generate Topology ID from BTreeMap keys */
        }

        // The logical fork based on Hashing
        then action 'Check Topology ID';

        then decide 'Topology Changed?';
            if true then 'Perform Expansion';
            if false then 'Update Weights Only';
        

        then action 'Perform Expansion' {
            doc /* O(N) Star/Clique expansion + Arrow Schema Definition */
        }

        then action 'Update Weights Only' {
            doc /* Direct memory copy to existing shared segment */
        }

        // Actions within the Daemon
        then action 'Host Memory Segment' {
            doc /* iceoryx2 Publisher remains active */
        }

        // Actions within the Python Subscriber
        then action 'Map via DLPack' {
            doc /* Zero-copy ingestion into PyTorch Tensors */
        }
        
        // Successions        
        succession flow 'Load Index' to 'Compute Blake3 Hash';
        succession flow 'Compute Blake3 Hash' to 'Check Topology ID';
        succession flow 'Check Topology ID' to 'Topology Changed?';

    }
}