// Declare the module tree so the compiler stitches the pieces together
pub mod config;
pub mod service;
pub mod worker;

use std::error::Error;
use std::sync::Arc;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::config::{Args, DaemonConfig};
use crate::service::HymekoDaemon;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the logger just like in your original file
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    // Parse the CLI arguments
    let args = Args::parse();
    let config = DaemonConfig::from(args);

    // Instantiate and run the modular daemon
    let daemon = Arc::new(HymekoDaemon::new(config));
    daemon.run().await
}