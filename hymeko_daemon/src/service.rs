use std::error::Error;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::iox_ingress::IoxIngressWorker;

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
        let node = NodeBuilder::new().create::<ipc::Service>()?;
        // 1. Iceoryx2 Data Plane (Inline to avoid generic type hell)
        // 1. Iceoryx2 Egress Publisher
        let egress_name = ServiceName::new(&self.config.service_name)?;
        let egress_service = node.service_builder(&egress_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;
        let publisher = Arc::new(egress_service
            .publisher_builder()
            .initial_max_slice_len(1024 * 1024)
            .create()?);

        // 2. Iceoryx2 Ingress Subscriber (CBOR)
        let ingress_name = ServiceName::new(&(self.config.service_name.clone() + "_query_cbor"))?;
        let ingress_service = node.service_builder(&ingress_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;
        let iox_subscriber = ingress_service.subscriber_builder().create()?;

        // 3. The Graceful Thread Bridge
        let (tx_iox_src, mut rx_iox_src) = mpsc::channel::<Vec<u8>>(100);
        let is_running = Arc::new(AtomicBool::new(true));

        let ingress_worker_src = IoxIngressWorker::new(
            self.config.service_name.clone() + "/query/src",
            tx_iox_src,
            Arc::clone(&is_running)
        );
        let ingress_handle_src = ingress_worker_src.spawn(); // Spin up the OS thread
        // IceOryx2 Pre-compiled ID (Fast path)
        let (tx_iox_ir, mut rx_iox_ir) = mpsc::channel::<Vec<u8>>(100);

        let ingress_worker_ir = IoxIngressWorker::new(
            self.config.service_name.clone() + "/query/ir",
            tx_iox_ir,
            Arc::clone(&is_running) // Share the exact same atomic shutdown flag!
        );
        let ingress_handle_ir = ingress_worker_ir.spawn();

        // 4. Zenoh Control Plane
        info!(marker = "[*]", "Initializing Zenoh session...");
        let z_session = zenoh::open(zenoh::Config::default()).await.map_err(|e| e.to_string())?;
        let sub_utf8 = z_session.declare_subscriber(format!("{}/query/utf8", self.config.service_name)).await.map_err(|e| e.to_string())?;
        let sub_cbor = z_session.declare_subscriber(format!("{}/query/cbor", self.config.service_name)).await.map_err(|e| e.to_string())?;

        info!(marker = "[>]", service = %self.config.service_name, "tri-channel daemon active");

        let mut heartbeat = interval(self.config.tick_rate);
        let mut had_subscribers = false;
        let (tx_pub, mut rx_pub) = mpsc::channel::<PublishRequest>(100);

        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    info!(marker = "[x]", "shutdown signal received, initiating graceful teardown...");
                    // Atomically command the Iceoryx2 thread to halt
                    is_running.store(false, Ordering::Relaxed);
                    break;
                }

                // 3. The Zenoh 1.x Async Reactor Catch-Block
                sample = sub_utf8.recv_async() => {
                    match sample {
                        Ok(msg) => {
                            let payload = msg.payload().to_bytes().into_owned();
                            info!(marker = "[<]", source = "zenoh_utf8", bytes = payload.len(), "received query payload");

                            let self_clone = Arc::clone(&self);
                            let tx_clone = tx_pub.clone();

                            tokio::spawn(async move {
                                if let Err(e) = self_clone.handle_utf8_query(payload, tx_clone).await {
                                    error!(source = "zenoh_utf8", "query processing failed: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!(source = "zenoh_utf8", "subscriber receive failed: {}", e);
                        }
                    }
                }
                sample = sub_cbor.recv_async() => {
                    match sample {
                        Ok(msg) => {
                            let payload = msg.payload().to_bytes().into_owned();
                            info!(marker = "[<]", source = "zenoh_cbor", bytes = payload.len(), "received query payload");
                            let self_clone = Arc::clone(&self);
                            let tx_clone = tx_pub.clone();
                            tokio::spawn(async move {
                                if let Err(e) = self_clone.handle_cbor_query(payload, tx_clone).await {
                                    error!(source = "zenoh_cbor", "query processing failed: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!(source = "zenoh_cbor", "subscriber receive failed: {}", e);
                        }
                    }
                }

                // Iceoryx2 Ingress Bridge
                Some(payload) = rx_iox_src.recv() => {
                    info!(marker = "[<]", source="iceoryx2_src", bytes = payload.len(), "received query");
                    let self_clone = Arc::clone(&self);
                    let tx_clone = tx_pub.clone();
                    tokio::spawn(async move {
                        if let Err(e) = self_clone.handle_utf8_query(payload, tx_clone).await {
                            error!(source = "iceoryx2_src", "query processing failed: {}", e);
                        }
                    });
                }
                // Iceoryx2 Ingress Bridge (IR)
                Some(payload) = rx_iox_ir.recv() => {
                    info!(marker = "[<]", source="iox_ir", bytes = payload.len(), "received compiled IR");
                    let self_clone = Arc::clone(&self);
                    let tx_clone = tx_pub.clone();
                    tokio::spawn(async move {
                        // Directly to the tensor builder bypass
                        if let Err(e) = self_clone.handle_fast_path_ir(payload, tx_clone).await {
                            error!(source = "iox_ir", "IR query failed: {}", e);
                        }
                    });
                }

                Some(req) = rx_pub.recv() => {
                    info!(marker = "[>]", nnz = req.nnz, bytes = req.tensor_data.len(), "publishing COO tensor to shared memory");

                    // Loan the exact required memory size from the zero-copy pool
                    match publisher.loan_slice_uninit(req.tensor_data.len()) {
                        Ok(mut sample) => {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    req.tensor_data.as_ptr(),
                                    sample.payload_mut().as_mut_ptr() as *mut u8,
                                    req.tensor_data.len()
                                );

                                // Mathematically declare the memory as initialized
                                let initialized_sample = sample.assume_init();

                                // Commit and instantly release to PyTorch clients
                                if let Err(e) = initialized_sample.send() {
                                    error!(marker = "[x]", "Iceoryx2 publisher failed to send sample: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            error!(marker = "[x]", "Iceoryx2 failed to loan memory: {}", e);
                        }
                    }
                }

                _ = heartbeat.tick() => {
                    let currently_has_subscribers = egress_service.dynamic_config().number_of_subscribers() > 0;
                    if currently_has_subscribers && !had_subscribers {
                        info!(marker = "[+]", "subscriber connected");
                    } else if !currently_has_subscribers && had_subscribers {
                        warn!(marker = "[-]", "subscriber disconnected");
                    }
                    had_subscribers = currently_has_subscribers;
                }
            }
        }
        let _ = ingress_handle_src.join();
        let _ = ingress_handle_ir.join();
        Ok(())
    }


}