use std::error::Error;
use std::thread;
use std::time::Duration;

use clap::Parser;
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about = "Send UTF-8 query payloads to the daemon Iceoryx2 src ingress topic")]
struct Args {
    /// Base daemon service name (same value as hymeko_daemon --service)
    #[arg(long, default_value = "HymekoFastState")]
    service: String,

    /// UTF-8 payload to publish to <service>_query_src
    #[arg(long, default_value = "BenchmarkCase\n{}\ncontext\n{\n    n0{}\n}")]
    message: String,

    /// Number of payloads to send
    #[arg(long, default_value_t = 1)]
    repeat: u32,

    /// Delay between payloads in milliseconds
    #[arg(long, default_value_t = 0)]
    interval_ms: u64,

    /// Requested max slice len when opening/creating the src topic
    #[arg(long, default_value_t = 65536)]
    max_slice_len: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let src_topic_name = format!("{}/query/src", args.service);
    let event_topic_name = format!("{}/query/src/query/event", args.service);

    let src_topic = ServiceName::new(&src_topic_name)?;
    let event_topic = ServiceName::new(&event_topic_name)?;

    let node = NodeBuilder::new().create::<ipc::Service>()?;

    let src_service = node
        .service_builder(&src_topic)
        .publish_subscribe::<[u8]>()
        .open_or_create()?;
    let publisher: Publisher<ipc::Service, [u8], ()> = src_service.publisher_builder()
        .initial_max_slice_len(65535)
        .create()?;

    let event_service = node
        .service_builder(&event_topic)
        .event()
        .open_or_create()?;
    let notifier = event_service.notifier_builder().create()?;

    // Setup the Ingress (Listening for the Tensor)
    let tensor_topic_name = ServiceName::new(&args.service).expect("Invalid service name");
    let tensor_service = node.service_builder(&tensor_topic_name)
        .publish_subscribe::<[u8]>()
        .open_or_create()?;
    let subscriber = tensor_service.subscriber_builder().create()?;

    println!(
        "sending to topic '{}' with event topic '{}' (requested_max_slice_len={})",
        src_topic_name, event_topic_name, args.max_slice_len
    );

    for i in 1..=args.repeat {
        let payload = args.message.as_bytes();

        let mut sample = match publisher.loan_slice_uninit(payload.len()) {
            Ok(sample) => sample,
            Err(e) => {
                eprintln!(
                    "error: loan_slice_uninit failed for {} bytes ({e}). Try a larger --max-slice-len and restart daemon+client.",
                    payload.len()
                );
                return Err(Box::new(e));
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                sample.payload_mut().as_mut_ptr() as *mut u8,
                payload.len(),
            );
            sample.assume_init().send()?;
        }

        // Wake the daemon waitset so it drains src samples immediately.
        if let Err(e) = notifier.notify() {
            eprintln!("warn: event notify failed after send #{i}: {e}");
        }

        println!("sent #{i}: {} bytes", payload.len());

        // 4. Synchronous Polling Loop for the Response Tensor
        let start_wait = std::time::Instant::now();
        loop {
            if let Some(received_sample) = subscriber.receive()? {
                let elapsed_us = start_wait.elapsed().as_micros();
                println!(
                    "    [+] SUCCESS: Received {} byte tensor in {} µs!",
                    received_sample.payload().len(),
                    elapsed_us
                );
                break;
            }

            if start_wait.elapsed() > Duration::from_secs(2) {
                eprintln!("    [-] Timeout: Daemon failed to respond within 2 seconds.");
                break;
            }

            // Yield the thread to prevent 100% CPU starvation while spinning
            thread::sleep(Duration::from_millis(1));
        }

        if args.interval_ms > 0 && i < args.repeat {
            thread::sleep(Duration::from_millis(args.interval_ms));
        }
    }

    Ok(())
}
