# Project Changelog — 2026-03-11

## Daemon Logging Context Enrichment
- Extended worker-path tracing in `hymeko_daemon/src/worker.rs` so structured events include `service`, `request_id`, and per-stage context (`source`, payload sizes, compile timing, NNZ/tensor metadata, and enqueue outcomes).
- Added compact ETag correlation in debug logs via an `etag_prefix` field (first 8 bytes) to improve cross-event traceability without emitting full hashes.
- Ensured Rayon closure logs keep service context by cloning `self.config.service_name` into closure scope before async-to-sync handoff.

## Ingress Thread Observability
- Updated `hymeko_daemon/src/iox_ingress.rs` worker lifecycle logs to include `service` for thread start, channel-close, and graceful termination events.
- Improved control-plane ingress logs in `hymeko_daemon/src/service.rs` with explicit source labeling (`zenoh_utf8`, `zenoh_cbor`, `iceoryx2_src`, `iox_ir`) and branch-specific receive/processing failure logs.

## Iceoryx Ingress Query Normalization
- Refactored `hymeko_daemon/src/iox_ingress.rs` so each ingress worker now binds a concrete `IngressFormat` and `Arc<HymekoDaemon>` reference, then emits `ExecutableQuery` payloads instead of forwarding raw bytes.
- Added in-thread normalization branches for `RawUtf8`, `CompiledIr`, and `CborEncoded` ingress payloads, using `compile_to_ir_only`, `deserialize_cbor_ir`, and CBOR text extraction before channel handoff.
- Routed normalized IR units into the Tokio side through `mpsc::Sender<ExecutableQuery>`, establishing a clearer sync-ingress to async-execution contract.
- Added compile/deserialization error logging in the ingress callback so malformed frames are dropped without crashing the worker loop.

## Standalone Iceoryx Client Scaffold
- Added a dedicated `hymeko_client` crate with its own `Cargo.toml` and a runnable CLI at `hymeko_client/src/main.rs` for manual Iceoryx ingress testing.
- Wired the client to publish UTF-8 payloads on a src query channel, emit the paired event notification, and then synchronously poll the daemon tensor egress subscriber for a response/timeout result.
- Exposed CLI knobs for `--service`, `--message`, `--repeat`, and `--interval-ms` so query bursts and latency smoke checks can be run without touching daemon internals.
- Captured the publisher-side slice sizing experiment (`publisher_builder().initial_max_slice_len(65535)`) and kept the operational note that `ExceedsMaxLoanSize` depends on the live service configuration footprint.

## Checklist Trace Update
- Updated `docs/plans/daemon/checklist_task3.md` to record that Task 3.2 structured runtime logging now includes service-aware worker and ingress context fields for operational debugging.
- Added Task 3.2 checklist evidence for the new external `hymeko_client` ingress test harness path.
- Added Task 3.3 evidence that ingress workers now perform format-aware query normalization to `ExecutableQuery` prior to async execution handoff (`hymeko_daemon/src/iox_ingress.rs`).
