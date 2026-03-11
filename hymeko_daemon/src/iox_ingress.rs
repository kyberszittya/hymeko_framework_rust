use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use iceoryx2::prelude::*;
use tokio::sync::mpsc;
use tracing::{error, info};

pub struct IoxIngressWorker {
    service_name: String,
    tx_out: mpsc::Sender<Vec<u8>>,
    is_running: Arc<AtomicBool>,
}

impl IoxIngressWorker {
    pub fn new(
        service_name: String,
        tx_out: mpsc::Sender<Vec<u8>>,
        is_running: Arc<AtomicBool>,
    ) -> Self {
        Self { service_name, tx_out, is_running }
    }

    pub fn spawn(self) -> thread::JoinHandle<()> {
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

            let listener = event_service.listener_builder().create().expect("Failed to create listener");
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
                        // WOKEN UP BY PYTORCH! Drain all pending samples from the subscriber.
                        while let Ok(Some(sample)) = subscriber.receive() {
                            let payload: Vec<u8> = sample.payload().to_vec();
                            if self.tx_out.blocking_send(payload).is_err() {
                                info!(marker = "[x]", service = %self.service_name, "Main channel closed");
                                // Tell the WaitSet to abort processing this cycle
                                return CallbackProgression::Stop;
                            }
                        }
                        // Tell the WaitSet it is safe to check other attachments
                        CallbackProgression::Continue
                    },
                    Duration::from_millis(100)
                );
            }
            info!(marker = "[*]", service = %self.service_name, "Iceoryx2 Ingress Worker thread safely terminated");
        })
    }
}