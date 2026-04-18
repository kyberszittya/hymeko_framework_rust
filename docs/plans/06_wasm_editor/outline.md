# Plan 06 — WASM Editor + MCP Server (future)

**Status:** indexed / not started.
**Source spec:** `steps/20260418/hymeko_claude_code_spec.md` (902 lines) — full technical specification dropped in by the user on 2026-04-18.
**Reference bundle:** `steps/20260418/files (5).zip`.

## Scope summary

The spec targets a WASM-based visual editor layered on top of the Rust workspace, together with an MCP server for Claude Code integration and a wire-format crate for P2P-ready CBOR packets. It proposes the following new crates (none of which exist yet):

| Crate | Purpose | Notes |
|-------|---------|-------|
| `hymeko_ir` | Standalone IR crate (slotmap-backed `HyMeKoIR`, `IRDelta` mutation units) | Spec §2. Extract from `hymeko_core::ir` but with a slotmap-based shape rather than the current arena. |
| `hymeko_emitter` | M2T emitters: `emit_hymeko`, `emit_sysml`, `emit_rust_stubs`, `emit_lean4` | Spec §3. Deterministic text emitters feeding both the WASM editor and MCP tools. |
| `hymeko_wasm` | `wasm-bindgen` exposure of IR (`EditorSession`) | Spec §4. Targets `wasm-pack build --target web`. |
| `hymeko_server` | Axum static server + REST file I/O | Spec §5. Serves the WASM app at `localhost:3000`. |
| `hymeko_mcp` | MCP server via `rmcp` | Spec §7. stdio transport, tools: `add_vertex`, `add_hyperedge`, `emit_hymeko`, `emit_sysml`, `snapshot`, `reset`. |
| `hymeko_wire` | CBOR + zstd + xxhash3 packet layer | Spec §8. Magic `0x484D4B4F` ("HMKO"), `PacketHeader` with delta sequence + checksum. |
| `hymeko_p2p` | iroh-based P2P gossip (future) | Spec §1 (listed, not designed). |

## Implementation order (from spec §12)

1. `hymeko_ir` extraction & `IRDelta::apply` tests
2. `hymeko_emitter` with `emit_hymeko` + `emit_sysml` working
3. `hymeko_wasm` — `wasm-pack build` succeeds, `snapshot()` works in-browser
4. `hymeko_server` — serves static WASM, REST file I/O
5. `hymeko_mcp` — Claude Code calls `add_vertex`, `emit_sysml`
6. `hymeko_wire` — encode/decode round-trip
7. React canvas — SVG hyperedge rendering (convex hull for N-ary edges)
8. T2M — wire existing `parser` to `EditorSession::parse_text`

## Relationship to the current workspace

- The spec's `hymeko_ir` is **not** a rename of the existing `hymeko_core::ir`. The spec proposes a slotmap-shaped IR with `IRDelta` mutation events, intended to drive a CRDT-friendly WASM editor. The existing arena IR in `hymeko_core::ir` remains the compilation-path source of truth. Both may coexist, or the spec's IR may be re-based on the existing one during extraction — design decision deferred to Step 1.
- The existing `hymeko_query::rewrite` engine already provides one M2T pathway (template-driven URDF/SDF/MJCF/DOT/ROS2-launch). The spec's `hymeko_emitter` is a **different** M2T stack aimed at editor + MCP tooling (HyMeKo text, SysML v2, Rust stubs, Lean 4 obligations). See `docs/examples/hymeko_to_sysmlv2.md` for a worked SysML v2 example.
- The spec's `hymeko_mcp` is the entry point for Claude Code integration; the existing `hymeko_cli` remains the non-MCP CLI.

## Design invariants (spec §13)

1. IR is the single source of truth — canvas and text are views only.
2. `emit_hymeko()` is deterministic — same IR always same text.
3. `IRDelta` is the only mutation path — never mutate IR fields directly.
4. WASM has no file I/O — all file ops go through `hymeko_server` REST.
5. MCP server is stateful per session — `Arc<Mutex<HyMeKoIR>>` is correct.
6. CBOR encoding is deterministic mode — required for CRDT checksumming.

## Intermediate deliverables already landed today (2026-04-18)

- `docs/examples/visualizations.md` — DOT / URDF / Mermaid hypergraph renders of the existing robotics fixtures (`mini_arm.hymeko`, `anthropomorphic_arm.hymeko`), to seed the editor's visual language before `hymeko_wasm` exists.
- `docs/examples/hymeko_to_sysmlv2.md` — end-to-end workflow from a `.hymeko` source to SysML v2 text, with a hand-authored ground-truth output for `mini_arm.hymeko` that the eventual `hymeko_emitter::emit_sysml` must match.

These two documents are standalone useful and also anchor later work: the SysML v2 walkthrough defines the output contract that Step 2 of the spec must satisfy.
