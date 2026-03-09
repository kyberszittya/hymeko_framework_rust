// In hymeko_core (or hymeko_client), purely in Rust.
use iceoryx2::prelude::*;
use crate::tensor::shared_state::HypergraphWeights;
use std::error::Error;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::sample::Sample;

/// The mathematically strict representation of what happened in the shared memory.
pub enum MemoryEvent<'a> {
    /// Topology changed. Re-allocation and mapping required.
    MappingUpdate(&'a Sample<ipc::Service, HypergraphWeights, ()>),
    /// Only weights changed. Fast pointer update.
    WeightStream(&'a Sample<ipc::Service, HypergraphWeights, ()>),
}

pub struct HymekoSubscriber {
    subscriber: Subscriber<ipc::Service, HypergraphWeights, ()>,
    // We track the state in Rust, NOT in Python
    current_topology_hash: u64,
}

impl HymekoSubscriber {
    pub fn new(service_name: &str) -> Result<Self, Box<dyn Error>> {
        let node = NodeBuilder::new().create::<ipc::Service>()?;
        let service = node.service_builder(&service_name.try_into()?)
            .publish_subscribe::<HypergraphWeights>()
            .open_or_create()?;

        let subscriber = service.subscriber_builder().create()?;

        Ok(Self {
            subscriber,
            current_topology_hash: 0, // Initialize to 0 or None
        })
    }

    /// The pure Rust event loop.
    pub fn poll_memory(&mut self) -> Result<Option<MemoryEvent>, Box<dyn Error>> {
        if let Ok(Some(sample)) = self.subscriber.receive() {
            // Assume we added a `topology_hash` to the HypergraphWeights struct
            /*
            let incoming_hash = sample.topology_hash;

            if incoming_hash != self.current_topology_hash {
                self.current_topology_hash = incoming_hash;
                // Emit the structural event
                return Ok(Some(MemoryEvent::MappingUpdate(sample)));
            } else {
                // Emit the fast-path event
                return Ok(Some(MemoryEvent::WeightStream(sample)));
            }
            
             */
        }
        Ok(None)
    }
}