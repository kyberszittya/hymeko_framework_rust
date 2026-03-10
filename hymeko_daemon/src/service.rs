use std::error::Error;
use std::sync::{Arc, RwLock};
use iceoryx2::prelude::*;
use moka::future::Cache;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{error, info, warn};


use crate::config::DaemonConfig;
use hymeko::resolution::interner::Interner;
use hymeko::tensor::shared_state::{calculate_required_bytes, ExpansionHeader};

pub struct PublishRequest {
    pub etag: [u8; 32],
    pub nnz: u64,
    pub tensor_data: Vec<u8>,
}

pub struct HymekoDaemon {
    pub config: DaemonConfig,
    pub cache: Cache<[u8; 32], bool>,
    pub interner: Arc<RwLock<Interner>>,
}

impl HymekoDaemon {
    pub fn new(config: DaemonConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.cache_size)
            .build();
        let interner = Arc::new(RwLock::new(Interner::new()));
        Self { config, cache, interner }
    }



    // The main execution loop of the daemon. It initializes both the Iceoryx data plane and the Zenoh control plane, then enters an event-driven loop.

    pub async fn run(self: Arc<Self>) -> Result<(), Box<dyn Error>> {
        // 1. Iceoryx2 Data Plane (Inline to avoid generic type hell)
        let service_name = ServiceName::new(&self.config.service_name)?;
        let node = NodeBuilder::new().create::<ipc::Service>()?;
        let service = node.service_builder(&service_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;
        let publisher = Arc::new(service.publisher_builder().create()?);

        // 2. Zenoh 1.x Control Plane (No .res() needed anymore)
        info!(marker = "[*]", "Initializing Zenoh session...");
        let zenoh_config = zenoh::Config::default();
        let z_session = zenoh::open(zenoh_config).await.map_err(|e| e.to_string())?;

        let query_topic = format!("{}/query", self.config.service_name);
        let subscriber = z_session.declare_subscriber(&query_topic).await.map_err(|e| e.to_string())?;

        info!(marker = "[>]", service = %self.config.service_name, topic = %query_topic, "daemon active");

        let mut heartbeat = interval(self.config.tick_rate);
        let mut had_subscribers = false;
        let (tx_pub, mut rx_pub) = mpsc::channel::<PublishRequest>(100);

        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    info!(marker = "[x]", "shutdown signal received");
                    break;
                }

                // 3. The Zenoh 1.x Async Reactor Catch-Block
                sample = subscriber.recv_async() => {
                    if let Ok(msg) = sample {
                        let payload = msg.payload().to_bytes().into_owned();
                        info!(marker = "[<]", bytes = payload.len(), "received query payload");

                        let self_clone = Arc::clone(&self);
                        let tx_clone = tx_pub.clone();

                        tokio::spawn(async move {
                            if let Err(e) = self_clone.handle_query(payload, tx_clone).await {
                                error!("[X] Query processing failed: {}", e);
                            }
                        });
                    }
                }

                // 2. ICEORYX2 EGRESS ACTOR
                Some(req) = rx_pub.recv() => {
                    let required_bytes = calculate_required_bytes(req.nnz as usize);
                    match publisher.loan_slice_uninit(required_bytes) {
                        Ok(mut sample) => {
                            let payload_ptr = sample.payload_mut().as_mut_ptr() as *mut u8;
                            unsafe {
                                // Write Header
                                let header_ptr = payload_ptr as *mut ExpansionHeader;
                                (*header_ptr).nnz = req.nnz;
                                (*header_ptr).etag = req.etag;

                                // Fast memcpy the tensor data immediately following the header
                                let data_ptr = payload_ptr.add(std::mem::size_of::<ExpansionHeader>());
                                if !req.tensor_data.is_empty() {
                                    std::ptr::copy_nonoverlapping(
                                        req.tensor_data.as_ptr(),
                                        data_ptr,
                                        req.tensor_data.len()
                                    );
                                }

                                let initialized_sample = sample.assume_init();
                                if let Err(e) = initialized_sample.send() {
                                    error!("[X] Iceoryx2 broadcast failed: {:?}", e);
                                } else {
                                    info!(marker = "[^]", nnz = req.nnz, "broadcasted tensor");
                                }
                            }
                        }
                        Err(e) => error!("[X] Failed to loan shared memory: {:?}", e),
                    }
                }

                _ = heartbeat.tick() => {
                    let currently_has_subscribers = service.dynamic_config().number_of_subscribers() > 0;
                    if currently_has_subscribers && !had_subscribers {
                        info!(marker = "[+]", "subscriber connected");
                    } else if !currently_has_subscribers && had_subscribers {
                        warn!(marker = "[-]", "subscriber disconnected");
                    }
                    had_subscribers = currently_has_subscribers;
                }
            }
        }
        Ok(())
    }


}