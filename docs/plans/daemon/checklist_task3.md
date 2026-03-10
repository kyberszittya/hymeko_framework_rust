# Checklist for HyMeKo Daemon Refactor: Phase 3

## Phase 3: The Control Plane (Concurrency & Networking)

- [ ] **Task 3.1: Concurrent Cache Integration (`moka`)**
  - [x] Add `moka = { version = "0.12", features = ["future"] }` to `hymeko_daemon/Cargo.toml`.
  - [x] Initialize `moka::future::Cache` in `HymekoDaemon` struct mapping `[u8; 32]` (Blake3 ETag) to compute state.
  - [ ] Implement cache lookup logic: if ETag exists, skip computation and signal the subscriber to reuse the existing `iceoryx2` segment.

- [x] **Task 3.2: Tokio & Zenoh Reactor (The I/O Hub)**
  - [x] Initialize the `tokio` multi-threaded runtime in `hymeko_daemon/src/main.rs`. *(`#[tokio::main]` is active.)*
  - [x] Modularize daemon bootstrap flow in `hymeko_daemon/src/main.rs` so startup runs `config::{Args, DaemonConfig}` -> `service::HymekoDaemon::new(config).run().await` with a thin orchestrator `main`.
  - [x] Add `zenoh` dependency and initialize a session to listen for incoming CBOR-encoded query objects. *(Session setup now lives in `hymeko_daemon/src/service.rs::run` via `zenoh::open(...)` and `declare_subscriber(...)`.)*
  - [x] Implement the main `tokio::select!` loop to multiplex network requests and the `iceoryx2` heartbeat. *(Current loop handles Zenoh receive, heartbeat tick, and subscriber-gated publishing path.)*
  - [x] Replace ad-hoc console prints with structured runtime logging (`tracing` + `tracing-subscriber`) and geometric/ascii markers in daemon status messages.

- [ ] **Task 3.3: The Async-to-Sync Bridge (Tokio-to-Rayon)**
  - [x] Add `rayon = "1.10"` to `hymeko_daemon/Cargo.toml`. *(Using `1.11.0`.)*
  - [ ] Implement `tokio::sync::oneshot` channels to send hypergraph ASTs from the async reactor to the Rayon thread pool. *(A scaffold exists in `hymeko_daemon/src/worker.rs::compute_expansion`, but the Rayon branch is still commented and not active.)*
  - [ ] Define the worker closure that executes the `hypergraphengine` math and writes directly to the loaned `iceoryx2` slice. *(Present only as commented pseudo-flow in `worker.rs`; needs live execution path.)*
  - [ ] Ensure the Rayon worker completion path signals back into the async runtime for publish/ack.
