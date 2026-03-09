# Components Diagram

This folder captures how the Rust engine, daemon, shared-memory segments, and Python bridge fit together.

## Mermaid View

Source: `components.mermaid`

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

## SysML Definition

Source: `components.sysml`

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

Use the Mermaid view for quick reviews inside GitHub, and load the SysML snippet into your modeling tool when you need formal semantics (ports, parts, connections).

