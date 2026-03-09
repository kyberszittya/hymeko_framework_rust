package HymekoArchitecture {

    part def HymekoSystem {

        // Components
        part compiler : CoreEngine;
        part daemon : ServiceAnchor;
        part pythonBridge : FFIBridge;

        // Shared Memory Resources
        part sharedMemoryPool {
            part topologySegment : ArrowMemory;
            part weightSegment : RawTensorMemory;
        }

        // Connections
        connection c1 connect compiler.outPort to topologySegment.inPort;
        connection c2 connect daemon.anchorPort to sharedMemoryPool.mgmtPort;
        connection c3 connect pythonBridge.mapPort to sharedMemoryPool.outPort;
    }

    part def CoreEngine {
        port outPort;
        doc /* The Rust compiler producing indices and weights */
    }

    part def ServiceAnchor {
        port anchorPort;
        doc /* iceoryx2 node that keeps SHM segments alive */
    }
}