use std::error::Error;
use std::time::Duration;
use iceoryx2::prelude::*;
use hymeko::tensor::shared_state::HypergraphWeights;

/// Configuration injected via CLI (e.g., clap) or config files.
pub struct DaemonConfig {
    pub service_name: String,
    pub tick_rate: Duration,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            service_name: "HymekoFastState".to_string(),
            tick_rate: Duration::from_secs(1),
        }
    }
}

/// The main Daemon service object.
pub struct HymekoDaemon {
    config: DaemonConfig,
}

impl HymekoDaemon {
    pub fn new(config: DaemonConfig) -> Self {
        Self { config }
    }

    /// Takes ownership of the thread and drives the IPC lifecycle.
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
        let service_name = ServiceName::new(&self.config.service_name)?;

        // 1. Initialize the zero-copy Node
        let node = NodeBuilder::new().create::<ipc::Service>()?;

        // 2. Define the Publish-Subscribe Service
        let service = node.service_builder(&service_name)
            .publish_subscribe::<HypergraphWeights>()
            .open_or_create()?;

        // 3. Create the Publisher instance
        let publisher = service.publisher_builder().create()?;

        println!("Hymeko Daemon: Zero-copy service '{}' is live.", service_name);
        println!("Waiting for PyTorch subscriber to attach...");

        let mut had_subscribers = false;

        // 4. The Physics Loop
        while node.wait(self.config.tick_rate).is_ok() {
            let currently_has_subscribers = service.dynamic_config().number_of_subscribers() > 0;

            if currently_has_subscribers && !had_subscribers {
                println!("⚡ PyTorch subscriber connected! Shared memory mapped.");
                had_subscribers = true;
            } else if !currently_has_subscribers && had_subscribers {
                println!("PyTorch subscriber disconnected. Waiting...");
                had_subscribers = false;
            }
        }

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Later, you parse CLI args here and map them to DaemonConfig
    let config = DaemonConfig::default();

    let daemon = HymekoDaemon::new(config);
    daemon.run()
}