use std::error::Error;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use iceoryx2::prelude::*;
use moka::future::Cache;
use tokio::sync::mpsc;
use dashmap::DashMap;
use tokio::time::interval;
use tracing::{error, info, warn};
use hymeko::ir::ir::Ir;
use crate::config::DaemonConfig;
use hymeko::resolution::interner::Interner;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::shared_state::{calculate_required_bytes, ExpansionHeader};
use crate::common::{ExecutableQuery, IngressFormat, IngressPayload};
use crate::iox_ingress::{IoxIngressWorker};

pub struct PublishRequest {
    pub etag: [u8; 32],
    pub nnz: u64,
    pub tensor_data: Vec<u8>,
}

pub struct HymekoDaemon {
    pub config: DaemonConfig,
    pub cache: Cache<[u8; 32], bool>,
    pub graph_memory: Arc<DashMap<[u8; 32], Arc<Ir>>>,
    pub interner: Arc<RwLock<Interner>>,
    pub agg_cfg: Arc<AggCfg>,
}

impl HymekoDaemon {
    pub fn new(config: DaemonConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.cache_size)
            .build();
        let interner = Arc::new(RwLock::new(Interner::new()));
        let graph_memory = Arc::new(DashMap::new());
        // Define the permanent aggregation rules
        let agg_cfg = Arc::new(AggCfg {
            sign: SignAgg::PreferNonNeutral,
            weight: WeightAgg::Sum,
            clamp01: false,
        });
        Self { config, cache, graph_memory, interner, agg_cfg }
    }


    // The main execution loop of the daemon.
    // It initializes both the Iceoryx data plane and the Zenoh control plane,
    // then enters an event-driven loop.
    pub async fn run(self: Arc<Self>) -> Result<(), Box<dyn Error>> {
        // 1. Setup the Unified Execution Funnel
        let (tx, mut rx) = mpsc::channel::<ExecutableQuery>(100);
        let is_running = Arc::new(AtomicBool::new(true));

        // 2. Initialize the Shared Memory Data Plane (Egress)
        let node = NodeBuilder::new().create::<ipc::Service>()?;
        let egress_name = ServiceName::new(&self.config.service_name)?;
        let egress_service = node.service_builder(&egress_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;

        let publisher = egress_service
            .publisher_builder()
            .initial_max_slice_len(1024 * 1024)
            .create()?;

        // 3. Spawn Parallel Iceoryx Workers
        // These now use the synchronous compiler logic on their own OS threads
        let _handle_src = IoxIngressWorker::new(
            self.config.service_name.clone() + "/query/src",
            IngressFormat::RawUtf8,
            tx.clone(),
            Arc::clone(&is_running),
            Arc::clone(&self),
        ).spawn();

        let _handle_src_cbor = IoxIngressWorker::new(
            self.config.service_name.clone() + "/query/cbor_src",
            IngressFormat::CborEncoded,
            tx.clone(),
            Arc::clone(&is_running),
            Arc::clone(&self),
        ).spawn();

        let _handle_ir = IoxIngressWorker::new(
            self.config.service_name.clone() + "/query/ir",
            IngressFormat::CompiledIr,
            tx.clone(),
            Arc::clone(&is_running),
            Arc::clone(&self),
        ).spawn();

        // 4. Initialize Zenoh Control Plane
        // We bridge Zenoh to the Fan-In channel by spawning a compilation task
        let z_session = zenoh::open(zenoh::Config::default()).await.map_err(|e| e.to_string())?;
        let sub_utf8 = z_session
            .declare_subscriber(format!("{}/query/utf8", self.config.service_name))
            .await.map_err(|e| e.to_string())?;

        let tx_zenoh = tx.clone();
        let self_zenoh = Arc::clone(&self);
        tokio::spawn(async move {
            while let Ok(msg) = sub_utf8.recv_async().await {
                let payload = msg.payload().to_bytes().to_vec();
                if let Ok(query_str) = String::from_utf8(payload) {
                    // Perform compilation in the background task
                    if let Ok(ir) = self_zenoh.compile_to_ir_only(query_str) {
                        let _ = tx_zenoh.send(ExecutableQuery { ir }).await;
                    }
                }
            }
        });

        info!(marker = "[>]", service = %self.config.service_name, "Hymeko Fan-In Engine Active");

        let mut heartbeat = interval(self.config.tick_rate);
        let mut had_subscribers = false;

        // --- THE CORE DISPATCHER LOOP ---
        loop {
            tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!(marker = "[x]", "Graceful shutdown initiated.");
                is_running.store(false, Ordering::Relaxed);
                break;
            }

            // The Unified Execution Pipe: Everything here is already compiled IR
            Some(query) = rx.recv() => {
                let start = std::time::Instant::now();

                // 1. Execute Math (Star Expansion)
                let result_tensor = self.expand_graph(&query.ir);

                // 2. Dispatch Result (Zero-Copy)
                if let Ok(mut sample) = publisher.loan_slice_uninit(result_tensor.len()) {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            result_tensor.as_ptr(),
                            sample.payload_mut().as_mut_ptr() as *mut u8,
                            result_tensor.len(),
                        );
                        sample.assume_init().send().ok();
                    }
                    info!(
                        marker = "[+]",
                        elapsed = ?start.elapsed(),
                        "Tensor dispatched."
                    );
                }
            }

            _ = heartbeat.tick() => {
                let currently_active = egress_service.dynamic_config().number_of_subscribers() > 0;
                if currently_active && !had_subscribers {
                    info!(marker = "[+]", "Subscriber connected");
                } else if !currently_active && had_subscribers {
                    warn!(marker = "[-]", "Subscriber disconnected");
                }
                had_subscribers = currently_active;
            }
        }
        }
        Ok(())
    }
}