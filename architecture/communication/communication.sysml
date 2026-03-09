package HymekoCommunication {

    item def WeightSignal {
        attribute timestamp : ScalarValues::DateTime;
        attribute offset : ScalarValues::Integer;
    }

    item def MappingSignal {
        attribute topologyHash : ScalarValues::String;
        attribute schemaRef : ArrowSchema;
    }

    // Defining the high-speed data flow
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