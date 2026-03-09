# Processing Flow

The flow diagrams in this folder show how star/clique expansions move from the Rust compiler into shared memory and onward to PyTorch.

## Mermaid Flowchart

Source: `flow.mermaid`

```mermaid
flowchart TD
    subgraph "Phase 1: AST Producer (Rust)"
        A[Load Index into BTreeMap] --> B[Stream into Blake3 Hasher]
        B --> C{Topology ID Changed?}
        C -- YES --> D[Structural Expansion<br/>k, i, j Indices]
        D --> E[Define Arrow Schema]
        C -- NO --> F[Compute Weights<br/>Fast State]
    end

    subgraph "Phase 2: HyMeKo Daemon (Anchor)"
        G[iceoryx2 Service]
        H[[/dev/shm Segment A: Mapping]]
        I[[/dev/shm Segment B: Weights]]
        E -.-> H
        F -.-> I
    end

    subgraph "Phase 3: Subscriber (PyTorch)"
        J[Event Listener]
        K{Event Type?}
        J --> K
        K -- Structural Update --> L[Re-map Arrow Memory]
        K -- Weight Stream --> M[Update Tensor Pointers]
        L --> N[DLPack Zero-Copy Ingestion]
        M --> N
        N --> O[GPU Computation]
    end

    E -- "Control Event" --> J
    F -- "Data Event" --> J
```

## SysML Activity

Source: `flow.sysml`

```sysml
package HymekoProcess {
    action def 'Process Hypergraph' {
        first start;
        then action 'Load Index';
        then action 'Compute Blake3 Hash';
        then action 'Check Topology ID';
        then decide 'Topology Changed?';
            if true then 'Perform Expansion';
            if false then 'Update Weights Only';
        then action 'Perform Expansion';
        then action 'Update Weights Only';
        then action 'Host Memory Segment';
        then action 'Map via DLPack';
    }
}
```

Use the SysML model when you need explicit control-flow semantics (e.g., for tooling that generates traces or validates pre/post conditions).

