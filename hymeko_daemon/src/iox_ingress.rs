use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use iceoryx2::prelude::*;
use tokio::sync::mpsc;
use tracing::{error, info};
use hymeko::ir::ir::Ir;
use hymeko::resolution::interner::Interner;
use crate::common::{ExecutableQuery, IngressFormat, IngressPayload};
use crate::service::HymekoDaemon;

pub struct IoxIngressWorker {
    service_name: String,
    format: IngressFormat,
    tx_out: mpsc::Sender<ExecutableQuery>,
    is_running: Arc<AtomicBool>,
    daemon: Arc<HymekoDaemon>,
}



impl IoxIngressWorker {
    pub fn new(
        service_name: String,
        format: IngressFormat,
        tx_out: mpsc::Sender<ExecutableQuery>,
        is_running: Arc<AtomicBool>,
        daemon: Arc<HymekoDaemon>, // The real reference to your engine
    ) -> Self {
        Self { service_name, format, tx_out, is_running, daemon }
    }

    pub fn spawn(self) -> thread::JoinHandle<()> {
        let daemon = Arc::clone(&self.daemon);
        thread::spawn(move || {
            info!(marker = "[*]", service = %self.service_name, "Iceoryx2 Ingress Worker thread started");

            // 1. Construct the Node and Subscriber strictly inside this OS thread
            let node = NodeBuilder::new().create::<ipc::Service>().expect("Failed to create Iox node");
            info!(marker = "[*]", service = %self.service_name, "iox node created");

            let ingress_name = ServiceName::new(&self.service_name).unwrap();
            let ingress_service = node.service_builder(&ingress_name)
                .publish_subscribe::<[u8]>()
                .open_or_create()
                .expect("Failed to open Iox service");
            info!(
                marker = "[*]",
                service = %self.service_name,
                topic = %ingress_name,
                "ingress publish_subscribe topic opened or created"
            );

            let subscriber = ingress_service.subscriber_builder()
                .create()
                .expect("Failed to create Iox subscriber");
            info!(marker = "[*]", service = %self.service_name, topic = %ingress_name, "ingress subscriber created");

            // 2. The Control Plane (Event Interrupt)
            let event_name = ServiceName::new(&(self.service_name.clone() + "/query/event")).unwrap();
            let event_service = node.service_builder(&event_name)
                .event()
                .open_or_create()
                .expect("Failed to open Iox event service");
            info!(
                marker = "[*]",
                service = %self.service_name,
                topic = %event_name,
                "event topic opened or created"
            );

            let mut listener = event_service.listener_builder().create().expect("Failed to create listener");

            info!(marker = "[*]", service = %self.service_name, topic = %event_name, "event listener created");

            // 3. The WaitSet Multiplexer
            let waitset = WaitSetBuilder::new().create::<ipc::Service>().expect("Failed to create WaitSet");
            info!(marker = "[*]", service = %self.service_name, "waitset created");
            let _guard = waitset.attach_notification(&listener).expect("Failed to attach listener");
            info!(marker = "[*]", service = %self.service_name, topic = %event_name, "listener attached to waitset");

            // 2. The Polling Loop
            while self.is_running.load(Ordering::Relaxed) {
                let _ = waitset.wait_and_process_once_with_timeout(
                    |_attachment_id| {
                        while let Ok(Some(sample)) = subscriber.receive() {
                            let raw_bytes = sample.payload().to_vec();

                            // Parallel Compilation Step
                            let ir_result = match self.format {
                                IngressFormat::RawUtf8 => {
                                    let dsl = String::from_utf8_lossy(&raw_bytes).to_string();
                                    // Call the synchronous compiler helper in worker.rs
                                    daemon.compile_to_ir_only(dsl)
                                },
                                IngressFormat::CompiledIr  => {
                                    daemon.deserialize_cbor_ir(&raw_bytes)
                                }
                                IngressFormat::CborEncoded => {
                                    // Manual extraction to avoid using '?' or 'return' incorrectly
                                    match serde_cbor::from_slice::<serde_cbor::Value>(&raw_bytes) {
                                        Ok(serde_cbor::Value::Text(dsl)) => {
                                            self.daemon.compile_to_ir_only(dsl)
                                        }
                                        Ok(_) => Err("Expected Text in CBOR".into()),
                                        Err(e) => Err(e.into()),
                                    }
                                }
                            };

                            match ir_result {
                                Ok(ir) => {
                                    let payload = ExecutableQuery { ir };
                                    if self.tx_out.blocking_send(payload).is_err() {
                                        error!(service = %self.service_name, "Main channel closed");
                                        return CallbackProgression::Stop;
                                    }
                                }
                                Err(e) => {
                                    error!(service = %self.service_name, "Compilation error: {:?}", e);
                                    // We continue to the next message instead of crashing
                                }
                            }
                        }
                        CallbackProgression::Continue
                    },
                    Duration::from_millis(100)
                );
            }
            info!(marker = "[*]", service = %self.service_name, "Iceoryx2 Ingress Worker thread safely terminated");
        })
    }
}