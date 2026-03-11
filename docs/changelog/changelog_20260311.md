# Project Changelog — 2026-03-11

## Daemon Logging Context Enrichment
- Extended worker-path tracing in `hymeko_daemon/src/worker.rs` so structured events include `service`, `request_id`, and per-stage context (`source`, payload sizes, compile timing, NNZ/tensor metadata, and enqueue outcomes).
- Added compact ETag correlation in debug logs via an `etag_prefix` field (first 8 bytes) to improve cross-event traceability without emitting full hashes.
- Ensured Rayon closure logs keep service context by cloning `self.config.service_name` into closure scope before async-to-sync handoff.

## Ingress Thread Observability
- Updated `hymeko_daemon/src/iox_ingress.rs` worker lifecycle logs to include `service` for thread start, channel-close, and graceful termination events.
- Improved control-plane ingress logs in `hymeko_daemon/src/service.rs` with explicit source labeling (`zenoh_utf8`, `zenoh_cbor`, `iceoryx2_src`, `iox_ir`) and branch-specific receive/processing failure logs.

## Checklist Trace Update
- Updated `docs/plans/daemon/checklist_task3.md` to record that Task 3.2 structured runtime logging now includes service-aware worker and ingress context fields for operational debugging.

