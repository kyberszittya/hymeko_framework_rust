# Hymeko Architecture Overview

This document collects the rendered Mermaid diagrams that describe the control-plane and data-plane flows for Hymeko.

> Source diagram: [`overview.mermaid`](overview.mermaid)

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

## Viewing Tips

- GitHub renders Mermaid automatically. For local previews use VS Code or JetBrains Mermaid preview plugins.
- Keep the source `.mermaid` files alongside these markdown renderings so diagram diffs stay readable.

