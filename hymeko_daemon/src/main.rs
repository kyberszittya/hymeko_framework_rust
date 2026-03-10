use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use clap::Parser;
use iceoryx2::prelude::*;
use iceoryx2::port::publisher::Publisher;
use iceoryx2::port::{LoanError, SendError};
use moka::future::Cache;
use tokio::time::interval;
use hymeko::engine::hypergraphengine::HypergraphEngine;
use hymeko::ir::ir::Ir;
use hymeko::tensor::representations::tensor_coo::TensorCoo;
use hymeko::tensor::shared_state::{ExpansionHeader, ExpansionKind, ExpansionOffsets};

/// CLI Configuration for the Hymeko Daemon.
#[derive(Parser, Debug)]
#[command(author, version, about = "Hymeko Zero-Copy Tensor Daemon")]
struct Args {
    /// Name of the iceoryx2 service segment.
    #[arg(short, long, default_value = "HymekoFastState")]
    service: String,

    /// Maximum number of cached expansions in the LRU.
    #[arg(short, long, default_value_t = 1024)]
    cache_size: u64,

    /// Heartbeat interval in milliseconds.
    #[arg(short, long, default_value_t = 1000)]
    tick: u64,
}


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
    args: Args,
    cache: Cache<[u8; 32], bool>,
}

impl HymekoDaemon {
    pub fn new(args: Args) -> Self {
        let cache = Cache::builder()
            .max_capacity(args.cache_size)
            .build();
        Self { args, cache }
    }

    /// Takes ownership of the thread and drives the IPC lifecycle.
    pub async fn run(self: Arc<Self>) -> Result<(), Box<dyn Error>> {
        let service_name = ServiceName::new(&self.args.service)?;
        let node = NodeBuilder::new().create::<ipc::Service>()?;

        let engine = HypergraphEngine::new();
        let ir = Self::build_stub_ir();
        let (header, offsets, coo) = Self::prepare_star_payload(&engine, &ir);

        // 1. Initialize the zero-copy Node
        let node = NodeBuilder::new().create::<ipc::Service>()?;

        // 2. Define the Publish-Subscribe Service
        let service = node.service_builder(&service_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;

        // 3. Create the Publisher instance
        let publisher = service.publisher_builder().create()?;

        println!("⚡ Hymeko Daemon '{}' is active.", self.args.service);
        println!("🚀 Cache capacity: {} expansions.", self.args.cache_size);
        let mut heartbeat = interval(Duration::from_millis(self.args.tick));
        let mut had_subscribers = false;

        // 4. The Physics Loop
        /*
        while node.wait(self.config.tick_rate).is_ok() {
            let currently_has_subscribers = service.dynamic_config().number_of_subscribers() > 0;

            if currently_has_subscribers {
                if !had_subscribers {
                    println!("⚡ PyTorch subscriber connected! Streaming star expansion frames...");
                }
                if let Err(err) = Self::publish_star_expansion(&publisher, &coo, &header, &offsets) {
                    eprintln!("Failed to publish star expansion: {err}");
                }
            } else if had_subscribers {
                println!("PyTorch subscriber disconnected. Waiting...");
            }

            had_subscribers = currently_has_subscribers;
        }

         */
        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    let currently_has_subscribers = service.dynamic_config().number_of_subscribers() > 0;

                    if currently_has_subscribers {
                        if !had_subscribers {
                            println!("⚡ PyTorch subscriber connected! Streaming star expansion frames...");
                        }
                        if let Err(err) = Self::publish_star_expansion(&publisher, &coo, &header, &offsets) {
                            eprintln!("Failed to publish star expansion: {err}");
                        }
                    } else if had_subscribers {
                        println!("PyTorch subscriber disconnected. Waiting...");
                    }

                    had_subscribers = currently_has_subscribers;
                }
            }
        }

        Ok(())
    }

    fn build_stub_ir() -> Ir {
        Ir::default()
    }

    fn prepare_star_payload(engine: &HypergraphEngine, ir: &Ir) -> (ExpansionHeader, ExpansionOffsets, TensorCoo<f32>) {
        let coo = engine.compile_star_expansion_core::<f32>(ir);
        let header = ExpansionHeader::new(ExpansionKind::Star3D, coo.len(), coo.num_slices, coo.dim_i, coo.dim_j);
        let offsets = header.contiguous_offsets();
        (header, offsets, coo)
    }

    fn publish_star_expansion(
        publisher: &Publisher<ipc::Service, [u8], ()>,
        coo: &TensorCoo<f32>,
        header: &ExpansionHeader,
        offsets: &ExpansionOffsets,
    ) -> Result<(), PublishError> {
        let mut sample = publisher
            .loan_slice_uninit(offsets.total_bytes)
            .map_err(PublishError::Loan)?;

        let payload = sample.payload_mut();
        debug_assert_eq!(payload.len(), offsets.total_bytes);
        let base_ptr = payload.as_mut_ptr() as *mut u8;

        let header_ptr = base_ptr as *mut ExpansionHeader;
        let k_ptr = unsafe { base_ptr.add(offsets.k_offset) as *mut i64 };
        let i_ptr = unsafe { base_ptr.add(offsets.i_offset) as *mut i64 };
        let j_ptr = unsafe { base_ptr.add(offsets.j_offset) as *mut i64 };
        let values_ptr = unsafe { base_ptr.add(offsets.values_offset) as *mut f32 };

        unsafe {
            HypergraphEngine::write_tensor_into_raw(header, coo, header_ptr, k_ptr, i_ptr, j_ptr, values_ptr, coo.len())
        }
        .map_err(PublishError::Tensor)?;

        let total_bytes = offsets.total_bytes;
        let nnz = coo.len();

        let sample = unsafe { sample.assume_init() };
        sample.send().map_err(PublishError::Send)?;
        println!("Sent star expansion frame (nnz={nnz}, bytes={total_bytes})");

        Ok(())
    }
}

#[derive(Debug)]
enum PublishError {
    Loan(LoanError),
    Send(SendError),
    Tensor(&'static str),
}

impl fmt::Display for PublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PublishError::Loan(err) => write!(f, "loan error: {err}"),
            PublishError::Send(err) => write!(f, "send error: {err}"),
            PublishError::Tensor(msg) => write!(f, "tensor copy error: {msg}"),
        }
    }
}

impl Error for PublishError {}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let daemon = Arc::new(HymekoDaemon::new(args));
    // Later, you parse CLI args here and map them to DaemonConfig
    let config = DaemonConfig::default();
    // Start the daemon with the provided configuration
    daemon.run().await
}