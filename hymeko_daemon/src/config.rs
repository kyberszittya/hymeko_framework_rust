use clap::Parser;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about = "Hymeko Zero-Copy Tensor Daemon")]
pub struct Args {
    #[arg(short, long, default_value = "HymekoFastState")]
    pub service: String,

    #[arg(short, long, default_value_t = 1024)]
    pub cache_size: u64,

    #[arg(short, long, default_value_t = 1000)]
    pub tick: u64,
}

pub struct DaemonConfig {
    pub service_name: String,
    pub tick_rate: Duration,
    pub cache_size: u64,
}

impl From<Args> for DaemonConfig {
    fn from(args: Args) -> Self {
        Self {
            service_name: args.service,
            tick_rate: Duration::from_millis(args.tick),
            cache_size: args.cache_size,
        }
    }
}