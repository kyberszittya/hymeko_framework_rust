# Hymeko Architecture Diagrams

This index documents every architecture diagram that lives under `architecture/`, pairing the rendered Mermaid views (GitHub-friendly) with their canonical SysML2 definitions. Use it as the single reference point for how the compiler, daemon, and subscribers coordinate.

## Contents

- [How to View the Diagrams](#how-to-view-the-diagrams)
- [Diagram Catalog](#diagram-catalog)
  - [System Overview](#system-overview)
  - [Crate Dependency Overview](#crate-dependency-overview)
  - [Layered Architecture View](#layered-architecture-view)
  - [Core Components](#core-components)
  - [Communication Sequence](#communication-sequence)
  - [Memory-State Transitions](#memory-state-transitions)
  - [Processing Flow](#processing-flow)
  - [Daemon Use Cases](#daemon-use-cases)

## How to View the Diagrams

| Format | Recommended Viewer | Notes |
| --- | --- | --- |
| Mermaid (`*.mermaid`) | GitHub Markdown, VS Code Mermaid preview, JetBrains Mermaid plug-in | Rendered inline below so you can track changes without external tools. |
| SysML (`*.sysml`) | [OMG SysML v2 Playground](https://sysml-v2.github.io/playground.html), Eclipse Papyrus, Modelix | Copy the snippets below (or open the files) into your SysML tool of choice to inspect ports, flows, and includes. |

## Diagram Catalog

### System Overview

High-level control-plane vs. data-plane responsibilities. Source: `architecture/overview.mermaid`

```mermaid
graph TD
%% Actors
    Comp((Compiler))
    Orch((Orchestrator))
    Pers((Persistence))
    Tele((Telemetry))
    Sub((Subscriber))

    subgraph "HyMeKo Control Plane"
        UC_Load(Load/Save)
        UC_Disc(Service Discovery)
        UC_Recover(Memory Recovery)
    end

    subgraph "High-Frequency Data Plane"
        UC_Update(Update Hypergraph)
        UC_Stream(Weight Stream)
        UC_Map(Structure Mapping)

        %% Internal requirements
        UC_Val[[Validate Schema]]
        UC_Align[[Align Tensors]]
    end

%% Key Operational Flows
    Comp --> UC_Update
    UC_Update --> UC_Val
    UC_Update --> UC_Align

    UC_Update -.-> UC_Stream
    UC_Stream -.-> UC_Map

    Orch --> UC_Recover
    Pers --> UC_Load
    Tele --> UC_Heart(Heartbeat)
```

> No SysML companion yet—add one here when we capture the same relationships in a formal model.

### Crate Dependency Overview

Workspace-level view of the Rust crates and how they depend on each other after the `hymeko_hre` extraction (2026-04-18). Source: `architecture/overview_crates.mermaid`.

```mermaid
graph TD
    classDef rust fill:#B7410E,stroke:#8B0000,stroke-width:2px,color:#fff;
    classDef hre  fill:#D2691E,stroke:#8B4513,stroke-width:2px,color:#fff;
    classDef query fill:#CC7722,stroke:#8B4513,stroke-width:2px,color:#fff;
    classDef daemon fill:#4CAF50,stroke:#2E8B57,stroke-width:2px,color:#fff;
    classDef ffi fill:#7B68EE,stroke:#483D8B,stroke-width:2px,color:#fff;
    classDef python fill:#306998,stroke:#FFD43B,stroke-width:2px,color:#fff;
    classDef cli fill:#555,stroke:#222,stroke-width:2px,color:#fff;

    subgraph "Foundations"
        Parser[parser<br/>LALRPOP grammar + lexer]:::rust
        Core[hymeko_core<br/>IR • resolution • module_store<br/>tensor primitives + HGNN/mesh ops<br/>traversal / HyperGraphView<br/>writers]:::rust
    end

    subgraph "Engine Layer"
        HRE[hymeko_hre<br/>HypergraphEngine orchestrator<br/>IR -> TensorCoo compilation<br/>star / clique expansions<br/>iceoryx2 subscriber - ipc feature]:::hre
    end

    subgraph "Query and Codegen"
        Query[hymeko_query<br/>predicate • engine • interpret<br/>rewrite • formats • kinematics<br/>URDF / SDF / MJCF / DOT / ROS2]:::query
    end

    subgraph "Runtime"
        Daemon[hymeko_daemon<br/>worker • IR-CBOR pipeline<br/>shared-memory gates]:::daemon
        Client[hymeko_client<br/>subscriber shell]:::daemon
    end

    subgraph "Surfaces"
        CLI[hymeko_cli<br/>compile + emit transforms]:::cli
        PyBind[hymeko_py<br/>PyO3 bindings]:::ffi
    end

    subgraph "External Consumers"
        PyTorch[PyTorch / DLPack]:::python
        Export[URDF • SDF • MJCF • DOT<br/>ROS2 launch]:::python
    end

    Parser --> Core
    Core --> HRE
    Core --> Query
    Core --> Daemon
    HRE --> Daemon
    HRE --> CLI
    HRE --> PyBind
    HRE --> Client
    Query --> CLI
    Query --> Export
    Daemon --> Client
    Core --> PyBind
    PyBind --> PyTorch
```

See `docs/plans/05_hre_extraction/plan.md` for the extraction rationale and why the split is engine-only rather than also pulling `traversal/`.

### Layered Architecture View

Source: `architecture/layers.mermaid`

```mermaid
graph TD
%% Styling
    classDef python fill:#306998,stroke:#FFD43B,stroke-width:2px,color:#fff;
    classDef ffi fill:#7B68EE,stroke:#483D8B,stroke-width:2px,color:#fff;
    classDef daemon fill:#4CAF50,stroke:#2E8B57,stroke-width:2px,color:#fff;
    classDef rust fill:#B7410E,stroke:#8B0000,stroke-width:2px,color:#fff;

    subgraph "Layer 4: Execution (Python / PyTorch)"
        Reactor[FFI Reactor<br/>Event Loop]:::python
        GPU[GPU Training Loop]:::python
        Reactor -- Yields Zero-Copy Tensor --> GPU
    end

    subgraph "Layer 3: Contract Boundary (FFI / Schema)"
        Registry[Schema Registry<br/>Version Handshake]:::ffi
        Reactor -- 1. Requests Mapping --> Registry
        Registry -- 2. Validates Layout --> Reactor
    end

    subgraph "Layer 2: Synchronization (HyMeKo Daemon)"
        StateGate{Atomic State Gate<br/>u8 Toggle}:::daemon
        Buffer[(Shared Memory<br/>Front & Back Buffers)]:::daemon
        StateGate -- Enforces Read-Locks --> Buffer
        Registry -. Subscribes to Events .-> StateGate
    end

    subgraph "Layer 1: Source of Truth (Rust Core)"
        Dispatcher[The Dispatcher<br/>Event Router]:::rust
        AST[(BTreeMap AST<br/>Index)]:::rust
        AST -- Hashing --> Dispatcher
        Dispatcher -- Emits TopologyShift / WeightStream --> StateGate
    end
```

> SysML companion for this layered view is not added yet; if needed, create `architecture/layers.sysml` and link it here.

### Core Components

Files:
- Mermaid: `architecture/components/components.mermaid`
- SysML: `architecture/components/components.sysml`

**Mermaid rendering**

```mermaid
graph TD
    subgraph "Rust Domain (Safe/Heavy)"
        Core[hymeko_core<br/>AST, BTreeMap, Blake3]
        Daemon[hymeko_daemon<br/>iceoryx2 Anchor]
    end

    subgraph "Shared Memory (Zero-Copy Plane)"
        direction LR
        SHM_Mapping[[Segment A: Arrow Schema<br/>k, i, j indices]]
        SHM_Weights[[Segment B: Fast State<br/>f32 weight tensors]]
    end

    subgraph "FFI Bridge"
        PyAPI[hymeko_py<br/>PyO3 + Arrow-rs]
    end

    subgraph "Python Domain (User/Compute)"
        Torch[PyTorch / LibTorch<br/>GPU Execution]
    end

    Core -->|Publish| SHM_Mapping
    Core -->|Stream| SHM_Weights

    Daemon ---|Owns/Anchors| SHM_Mapping
    Daemon ---|Owns/Anchors| SHM_Weights

    PyAPI -->|Map Pointer| SHM_Mapping
    PyAPI -->|Map Pointer| SHM_Weights

    PyAPI -->|DLPack| Torch
    Torch -->|Compute| GPU((GPU))

    Daemon -.->|Event Notification| PyAPI
```

**SysML source** (`components.sysml`)

```sysml
package HymekoArchitecture {
    part def HymekoSystem {
        part compiler : CoreEngine;
        part daemon : ServiceAnchor;
        part pythonBridge : FFIBridge;

        part sharedMemoryPool {
            part topologySegment : ArrowMemory;
            part weightSegment : RawTensorMemory;
        }

        connection c1 connect compiler.outPort to topologySegment.inPort;
        connection c2 connect daemon.anchorPort to sharedMemoryPool.mgmtPort;
        connection c3 connect pythonBridge.mapPort to sharedMemoryPool.outPort;
    }
}
```

### Communication Sequence

Files:
- Mermaid: `architecture/communication/communication.mermaid`
- SysML: `architecture/communication/communication.sysml`

**Mermaid rendering**

```mermaid
sequenceDiagram
    participant C as AST Producer (Rust)
    participant D as HyMeKo Daemon
    participant SHM as Shared Memory (SHM)
    participant P as Subscriber (PyTorch)

    Note over D: Daemon Bootstraps
    D->>SHM: Create/Open Service "HymekoFastState"
    D->>SHM: Anchor Segment (Hold Reference)

    Note over C: Compilation Phase
    C->>C: Generate BTreeMap Index
    C->>C: Compute Blake3 Topology Hash

    alt Topology Changed
        C->>SHM: Write New Mapping (k, i, j)
        C->>D: Signal: StructuralUpdate(SchemaRef)
        D->>P: Event: ReMapRequired
        P->>SHM: Map Arrow Array (Zero-Copy)
    else Only Weights Changed
        C->>SHM: Write Raw f32 Weights
        C->>D: Signal: WeightStream(PointerOffset)
        D->>P: Event: NewDataAvailable
    end

    Note over P: Ingestion Phase
    P->>P: DLPack Handshake
    P->>P: GPU Execution
```

**SysML source** (`communication.sysml`)

```sysml
package HymekoCommunication {
    item def WeightSignal {
        attribute timestamp : ScalarValues::DateTime;
        attribute offset : ScalarValues::Integer;
    }

    item def MappingSignal {
        attribute topologyHash : ScalarValues::String;
        attribute schemaRef : ArrowSchema;
    }

    flow def TensorStream {
        end : CoreEngine;
        end : FFIBridge;
        item : WeightSignal;
    }

    interface def SharedMemoryInterface {
        flow tensorFlow : TensorStream;
        doc /* High-frequency zero-copy data exchange */
    }
}
```

### Memory-State Transitions

File: `architecture/communication/memory_communication/memory_communication.mermaid`

```mermaid
stateDiagram-v2
    [*] --> Idle : Initialize Client

    state Idle {
        [*] --> Polling : Receive()
        Polling --> [*] : Timeout / None
    }

    Idle --> EvaluatingHash : Sample Received

    state EvaluatingHash {
        state if_state <<choice>>
        [*] --> if_state : Compare (Incoming == Current?)

        if_state --> StructuralShift : False (Topology Changed)
        if_state --> FastPath : True (Weights Only)
    }

    StructuralShift --> EmitMappingUpdate : Update Current Hash
    FastPath --> EmitWeightStream : Keep Current Hash

    EmitMappingUpdate --> Idle : Release to FFI
    EmitWeightStream --> Idle : Release to FFI
```

> A SysML state machine for this flow has not been authored yet—add `memory_communication.sysml` alongside the Mermaid file if you need a formal model.

### Processing Flow

Files:
- Mermaid: `architecture/flow/flow.mermaid`
- SysML: `architecture/flow/flow.sysml`

**Mermaid rendering**

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

**SysML source** (`flow.sysml`)

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

### Daemon Use Cases

Files:
- Mermaid: `architecture/daemon/use_case.mermaid`
- SysML: `architecture/daemon/use_cases.sysml`

**Mermaid rendering**

```mermaid
graph TB
    subgraph Actors
        C[Compiler]
        O[Orchestrator]
        P[Persistence]
        T[Telemetry]
        S[System/Subscriber]
    end

    subgraph "HymekoDaemon (Subject)"
        subgraph "Data Lifecycle"
            UC_Load(Load Hypergraph)
            UC_Save(Save Hypergraph)
            UC_Update(Update Hypergraph)
            UC_Validate(Validate Schema)
            UC_Align(Enforce Tensor Alignment)
            UC_Persist(Persist Hypergraph)
        end

        subgraph "Zero-Copy Data Plane"
            UC_Stream(Publish Hypergraph Stream)
            UC_Map(Publish Hypergraph Mapping)
            UC_Struct(Handle Structural Change)
            UC_SubStream(Subscribe to Stream)
            UC_SubMap(Subscribe to Mapping)
        end

        subgraph "Health & Orchestration"
            UC_Recover(Recover Shared Memory)
            UC_Heart(Heartbeat Health Check)
            UC_Disc(Discover Service Instance)
            UC_Ack(Acknowledge Mapping)
        end
    end

    C --> UC_Load
    C --> UC_Update
    P --> UC_Save
    P --> UC_Persist
    O --> UC_Recover
    T --> UC_Heart
    S --> UC_Validate
    S --> UC_SubStream
    S --> UC_SubMap

    UC_Update -- "<<include>>" --> UC_Validate
    UC_Update -- "<<include>>" --> UC_Align
    UC_Map -- "<<include>>" --> UC_Struct
    UC_Stream -. "<<extend>>" .-> UC_Map
```

**SysML source** (`use_cases.sysml` excerpt)

```sysml
package UseCases {
    part hymekoDaemon;
    part system;
    part orchestrator;
    part compiler;
    part telemetry;
    part persistence;

    use case def 'Update Hypergraph' {
        subject hymekoDaemon;
        actor :>> compiler;
        include use case 'validate' : 'Validate Hypergraph Schema';
        include use case 'align' : 'Enforce Tensor Alignment';
    }

    use case def 'Publish Hypergraph Mapping' {
        subject hymekoDaemon;
        actor :>> hymekoDaemon;
        include use case 'onStructuralChange' : 'Handle Structural Change';
    }

    use case def 'Recover Shared Memory' {
        subject hymekoDaemon;
        actor :>> orchestrator;
    }
}
```

---

Need another diagram documented? Drop it into `architecture/`, add a Mermaid/SysML pair, and extend this README so everyone knows where to find it.
