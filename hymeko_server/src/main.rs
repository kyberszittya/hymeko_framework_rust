//! Binary entry point for `hymeko_server`. Binds Axum to
//! `127.0.0.1:3000` by default; override via `HYMEKO_SERVER_ADDR`.

use std::net::SocketAddr;

use hymeko_server::{AppState, app};

#[tokio::main]
async fn main() {
    let addr: SocketAddr = std::env::var("HYMEKO_SERVER_ADDR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 3000)));

    let workspace_root = std::env::var("HYMEKO_WORKSPACE")
        .unwrap_or_else(|_| ".".to_string());

    let state = AppState::new(workspace_root);
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    println!("hymeko_server listening on http://{addr}");
    axum::serve(listener, app(state)).await.expect("serve");
}
