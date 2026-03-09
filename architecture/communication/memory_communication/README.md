# Memory Communication State Machine

This subdiagram tracks how the subscriber reacts to shared-memory samples by comparing topology hashes and deciding whether to remap schemas or just ingest weights.

## Mermaid State Machine

Source: `memory_communication.mermaid`

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

> SysML version: _not yet modeled_. When you need a formal statechart, add `memory_communication.sysml` alongside the Mermaid file and mirror the transitions above.

