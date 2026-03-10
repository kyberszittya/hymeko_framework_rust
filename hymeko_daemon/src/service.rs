use std::error::Error;
use std::sync::{Arc, RwLock};
use iceoryx2::prelude::*;
use moka::future::Cache;
use tokio::time::interval;
use tracing::{error, info, warn};

use crate::config::DaemonConfig;
use hymeko::resolution::interner::Interner;

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

    pub async fn run(self: Arc<Self>) -> Result<(), Box<dyn Error>> {
        let service_name = ServiceName::new(&self.config.service_name)?;
        let node = NodeBuilder::new().create::<ipc::Service>()?;
        let service = node.service_builder(&service_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;

        let publisher = Arc::new(service.publisher_builder().create()?);

        info!(marker = "[>]", service = %self.config.service_name, "daemon active");

        let mut heartbeat = interval(self.config.tick_rate);
        let mut had_subscribers = false;

        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    info!(marker = "[x]", "shutdown signal received");
                    break;
                }
                _ = heartbeat.tick() => {
                    let currently_has_subscribers = service.dynamic_config().number_of_subscribers() > 0;

                    if currently_has_subscribers && !had_subscribers {
                        info!(marker = "[+]", "subscriber connected");

                        let current_etag: [u8; 32] = blake3::hash(b"hymeko_empty_baseline").into();

                        if !self.cache.contains_key(&current_etag) {
                            if let Err(e) = self.compute_expansion(current_etag, Arc::clone(&publisher)).await {
                                error!("[X] Expansion failed: {}", e);
                            } else {
                                self.cache.insert(current_etag, true).await;
                            }
                        }
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