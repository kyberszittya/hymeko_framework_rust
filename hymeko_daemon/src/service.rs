use std::error::Error;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::*;
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use iceoryx2::service::port_factory::PortFactory as PortFactoryTrait;
use moka::future::Cache;
use tokio::sync::mpsc;
use dashmap::DashMap;
use tokio::time::interval;
use tracing::{info, warn};
use hymeko::ir::ir::Ir;
use crate::config::DaemonConfig;
use hymeko::resolution::interner::Interner;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use crate::common::{ExecutableQuery, IngressFormat};
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

/// Concrete alias for the Iceoryx publish/subscribe pub-port used for
/// every byte-slice channel the daemon exposes. All three publishers
/// (star, clique, raw IR) are the same instantiation.
type BytePublisher = Publisher<ipc::Service, [u8], ()>;
/// Factory handle for the byte-slice service — we only hold one
/// long-lived reference (the egress service), used for subscriber-count
/// introspection in the heartbeat.
type ByteServiceFactory = PortFactory<ipc::Service, [u8], ()>;

impl HymekoDaemon {
    pub fn new(config: DaemonConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.cache_size)
            .build();
        let interner = Arc::new(RwLock::new(Interner::new()));
        let graph_memory = Arc::new(DashMap::new());
        let agg_cfg = Arc::new(AggCfg {
            sign: SignAgg::PreferNonNeutral,
            weight: WeightAgg::Sum,
            clamp01: false,
        });
        Self { config, cache, graph_memory, interner, agg_cfg }
    }

    /// The main execution loop of the daemon. Initializes the Iceoryx
    /// data plane + Zenoh control plane, spawns ingress workers, then
    /// enters an event-driven `select!` over shutdown / query / heartbeat.
    pub async fn run(self: Arc<Self>) -> Result<(), Box<dyn Error>> {
        let (tx, mut rx) = mpsc::channel::<ExecutableQuery>(100);
        let is_running = Arc::new(AtomicBool::new(true));

        let node = NodeBuilder::new().create::<ipc::Service>()?;
        let (publisher, pub_clique, pub_ir, egress_service) =
            self.setup_publishers(&node)?;

        self.spawn_ingress_workers(&tx, &is_running);
        self.setup_zenoh_bridge(tx.clone()).await?;

        info!(
            marker = "[>]",
            service = %self.config.service_name,
            "Hymeko Fan-In Engine Active",
        );

        let mut heartbeat = interval(self.config.tick_rate);
        let mut had_subscribers = false;

        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    info!(marker = "[x]", "Graceful shutdown initiated.");
                    is_running.store(false, Ordering::Relaxed);
                    break;
                }
                Some(query) = rx.recv() => {
                    self.dispatch_query(query, &publisher, &pub_clique, &pub_ir);
                }
                _ = heartbeat.tick() => {
                    had_subscribers = update_subscriber_state(&egress_service, had_subscribers);
                }
            }
        }
        Ok(())
    }

    /// Build the three byte-slice publishers (star expansion, clique
    /// expansion, raw compiled IR) plus the egress service factory the
    /// heartbeat introspects for subscriber count. All share the same
    /// Iceoryx node so their lifetime is tied to `node`.
    fn setup_publishers(
        &self,
        node: &Node<ipc::Service>,
    ) -> Result<(BytePublisher, BytePublisher, BytePublisher, ByteServiceFactory), Box<dyn Error>> {
        let egress_name = ServiceName::new(&self.config.service_name)?;
        let egress_service = node.service_builder(&egress_name)
            .publish_subscribe::<[u8]>()
            .open_or_create()?;
        let publisher = egress_service
            .publisher_builder()
            .initial_max_slice_len(1024 * 1024)
            .create()?;

        let name_clique = ServiceName::new(&(self.config.service_name.clone() + "/tensor/clique"))?;
        let pub_clique = node.service_builder(&name_clique)
            .publish_subscribe::<[u8]>().open_or_create()?
            .publisher_builder().initial_max_slice_len(10 * 1024 * 1024).create()?;

        let name_ir = ServiceName::new(&(self.config.service_name.clone() + "/ir/cbor"))?;
        let pub_ir = node.service_builder(&name_ir)
            .publish_subscribe::<[u8]>().open_or_create()?
            .publisher_builder().initial_max_slice_len(1024 * 1024).create()?;

        Ok((publisher, pub_clique, pub_ir, egress_service))
    }

    /// Spawn the three Iceoryx ingress workers — raw UTF-8, CBOR query
    /// source, and pre-compiled IR. Each runs on its own OS thread and
    /// feeds the unified execution funnel (`tx`).
    fn spawn_ingress_workers(
        self: &Arc<Self>,
        tx: &mpsc::Sender<ExecutableQuery>,
        is_running: &Arc<AtomicBool>,
    ) {
        for (suffix, format) in [
            ("/query/src", IngressFormat::RawUtf8),
            ("/query/cbor_src", IngressFormat::CborEncoded),
            ("/query/ir", IngressFormat::CompiledIr),
        ] {
            let _handle = IoxIngressWorker::new(
                self.config.service_name.clone() + suffix,
                format,
                tx.clone(),
                Arc::clone(is_running),
                Arc::clone(self),
            ).spawn();
        }
    }

    /// Bridge Zenoh UTF-8 query subscription → the execution funnel.
    /// Compilation happens inside the spawned task so the Zenoh recv
    /// loop stays non-blocking.
    async fn setup_zenoh_bridge(
        self: &Arc<Self>,
        tx: mpsc::Sender<ExecutableQuery>,
    ) -> Result<(), Box<dyn Error>> {
        let z_session = zenoh::open(zenoh::Config::default())
            .await.map_err(|e| e.to_string())?;
        let sub_utf8 = z_session
            .declare_subscriber(format!("{}/query/utf8", self.config.service_name))
            .await.map_err(|e| e.to_string())?;

        let self_zenoh = Arc::clone(self);
        tokio::spawn(async move {
            while let Ok(msg) = sub_utf8.recv_async().await {
                let payload = msg.payload().to_bytes().to_vec();
                if let Ok(query_str) = String::from_utf8(payload) {
                    if let Ok(ir) = self_zenoh.compile_to_ir_only(query_str) {
                        let _ = tx.send(ExecutableQuery { ir }).await;
                    }
                }
            }
        });
        Ok(())
    }

    /// Execute one compiled query: broadcast the IR, compute + dispatch
    /// the star expansion, then the clique expansion. Each leg logs its
    /// own elapsed-time after dispatch so a straggler is attributable to
    /// the right tensor kind.
    fn dispatch_query(
        &self,
        query: ExecutableQuery,
        pub_star: &BytePublisher,
        pub_clique: &BytePublisher,
        pub_ir: &BytePublisher,
    ) {
        let start = std::time::Instant::now();

        if let Ok(ir_bytes) = self.serialize_ir_to_cbor(&query.ir) {
            publish_bytes(pub_ir, &ir_bytes);
        }

        let star_bytes = self.expand_graph(&query.ir);
        publish_bytes(pub_star, &star_bytes);
        info!(marker = "[+]", elapsed = ?start.elapsed(), "Star Tensor dispatched.");

        let clique_bytes = self.expand_graph_clique(&query.ir);
        publish_bytes(pub_clique, &clique_bytes);
        info!(marker = "[+]", elapsed = ?start.elapsed(), "Clique Tensor dispatched.");

        info!(
            marker = "[=]",
            elapsed = ?start.elapsed(),
            "Multiplexing complete: IR, Star, and Clique dispatched.",
        );
    }
}

/// Zero-copy dispatch: loan a sample of the right size, memcpy, send.
/// Falls silent if the loan fails — the publisher is non-blocking and
/// the sample gets dropped if send fails; both are expected under
/// back-pressure and not fatal.
fn publish_bytes(publisher: &BytePublisher, bytes: &[u8]) {
    let Ok(mut sample) = publisher.loan_slice_uninit(bytes.len()) else { return };
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            sample.payload_mut().as_mut_ptr() as *mut u8,
            bytes.len(),
        );
        sample.assume_init().send().ok();
    }
}

/// One heartbeat tick: log subscriber-count transitions (connect /
/// disconnect) and return the new state for the caller to track.
fn update_subscriber_state(
    egress_service: &ByteServiceFactory,
    had_subscribers: bool,
) -> bool {
    let currently_active = egress_service.dynamic_config().number_of_subscribers() > 0;
    if currently_active && !had_subscribers {
        info!(marker = "[+]", "Subscriber connected");
    } else if !currently_active && had_subscribers {
        warn!(marker = "[-]", "Subscriber disconnected");
    }
    currently_active
}
