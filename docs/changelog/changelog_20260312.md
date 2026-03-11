# Project Changelog — 2026-03-12

## Daemon Multiplexed Egress Channels
- Extended `hymeko_daemon/src/service.rs` to publish three coordinated outputs per query cycle: raw compiled IR (`/ir/cbor`), star expansion tensor (default tensor channel), and clique expansion tensor (`/tensor/clique`).
- Added dedicated Iceoryx publishers for clique and IR streams with explicit slice-capacity configuration to reduce allocation churn during dispatch.
- Updated per-cycle runtime logs to reflect multiplexed dispatch completion and separate star/clique send events.

## Worker Serialization Refactor and Type Fixes
- Extracted graph-name derivation into a standalone helper `graph_name_from_ir` in `hymeko_daemon/src/worker.rs` so Arrow metadata naming is centralized.
- Updated Arrow IPC serialization call sites to pass the full function signature (`tensor`, `etag`, `graph_name`) consistently.
- Fixed invalid hash usage and type mismatches in worker flow (`etag` now remains `[u8; 32]` through publish request construction and log correlation).
- Removed reliance on non-existent `Meta::name` field by introducing a deterministic fallback naming strategy based on `doc_hash` prefix.

## Tasklist Synchronization
- Updated `docs/plans/daemon/checklist_task3.md` to record the active multiplexed IR + Star + Clique dispatch path in `hymeko_daemon/src/service.rs`.
- Marked Tokio-to-Rayon oneshot completion signaling as implemented in `hymeko_daemon/src/worker.rs` (`execute_compilation` and `handle_fast_path_ir`).
- Clarified remaining gap for Task 3.3: Rayon worker currently computes and enqueues publish payloads, while final Iceoryx loan/write still occurs in the async service loop.

