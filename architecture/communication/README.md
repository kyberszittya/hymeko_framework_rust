# Communication Diagrams

These diagrams describe how expansion data moves from the Rust compiler, through the daemon, into shared memory, and finally into the PyTorch subscriber.

## Sequence Diagram (Mermaid)

Source: `communication.mermaid`

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

## SysML Interface Model

Source: `communication.sysml`

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

### Memory-State Subdiagram

For the state machine governing zero-copy handling, see `memory_communication/README.md` inside this folder.

