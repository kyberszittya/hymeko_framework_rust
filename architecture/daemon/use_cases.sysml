package UseCases {    
    // --- Parts (Actor Definitions) ---
    part hymekoDaemon;
    part system;
    part orchestrator;
    part compiler;
    part telemetry;
    part persistence;
    
    // --- Use Case Definitions ---

    use case def 'Load Hypergraph' {
        subject hymekoDaemon;
        actor :>> compiler; // Specific actor assignment
        objective { doc /* Load hypergraph definition from disk or network */ }
    }
    
    use case def 'Save Hypergraph' {
        subject hymekoDaemon;
        actor :>> persistence;
        objective { doc /* Save hypergraph definition to disk or network */ }
    }

    use case def 'Update Hypergraph' {
        subject hymekoDaemon;
        actor :>> compiler;        
        include use case 'validate' : 'Validate Hypergraph Schema';
        include use case 'align' : 'Enforce Tensor Alignment'; // Added for data integrity
        objective { doc /* Update hypergraph structure and metadata */ }
    }

    use case def 'Validate Hypergraph Schema' {
        subject hymekoDaemon;
        actor :>> system;
        objective { doc /* Ensure hypergraph schema adheres to format */ }
    }

    use case def 'Publish Hypergraph Stream' {
        subject hymekoDaemon;
        actor :>> hymekoDaemon; // The daemon is the primary actor here
        objective { doc /* Publish updates to hypergraph structure */ }
    }

    use case def 'Publish Hypergraph Mapping' {
        subject hymekoDaemon;
        actor :>> hymekoDaemon;
        // Logic: Mapping publication includes structural change handling
        include use case 'onStructuralChange' : 'Handle Structural Change';
        objective { doc /* Publish mapping to interested subscribers */ }
    }

    use case def 'Handle Structural Change' {
        subject hymekoDaemon;
        actor :>> system;
        objective { doc /* Update mappings based on structural changes */ }
    }

    // --- Subscriptions ---

    use case def 'Subscribe to Hypergraph Stream' {
        subject hymekoDaemon;
        actor :>> system; // This would be the ComputingClient/PyTorch actor
        objective { doc /* Receive updates to hypergraph stream in real-time */ }
    }

    use case def 'Subscribe to Hypergraph Mapping' {
        subject hymekoDaemon;
        actor :>> system;
        objective { doc /* Adjust internal state according to mapping updates */ }
    }

    // --- Maintenance and Health ---

    use case def 'Recover Shared Memory' {
        subject hymekoDaemon;
        actor :>> orchestrator; // Orchestrator handles recovery
        objective { doc /* Attempt to recover shared memory state after failure */ }      
    }

    use case def 'Heartbeat Health Check' {
        subject hymekoDaemon;
        actor :>> telemetry; // Telemetry handles health monitoring
        objective { doc /* Perform regular checks for system responsiveness */ }
    }

    use case def 'Enforce Tensor Alignment' {
        subject hymekoDaemon;
        actor :>> system;
        objective { doc /* Ensure tensors are aligned for efficient computation */ }
    }

    use case def 'Discover Service Instance' {
        subject hymekoDaemon;
        actor :>> system;
        objective { doc /* Discover available service instances in the network */ }
    }

    use case def 'Acknowledge Mapping Success' {
        subject hymekoDaemon;
        actor :>> system;
        objective { doc /* Acknowledge successful structure mapping */ }
    }

    use case def 'Persist Hypergraph' {
        subject hymekoDaemon;
        actor :>> persistence;
        objective { doc /* Persist state to storage */ }
    }
}