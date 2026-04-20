# Plan 06 — Step 1 Design: IR Strategy for the WASM Editor

**Status:** accepted 2026-04-18
**Supersedes:** `steps/20260418/hymeko_claude_code_spec.md` § 2 ("Step 0 — IR Extraction") which proposed a dedicated `hymeko_ir` crate.

## Question

The spec proposes extracting a **slotmap-shaped** `hymeko_ir` crate with
`SlotMap<VertexId, Vertex>` / `SlotMap<EdgeId, HyperEdge>` / `IRDelta` —
explicitly aimed at interactive editing and CRDT gossip. But the workspace
already has a **full, working IR** inside `hymeko_core::ir`:
arena-backed, content-hashed via Blake3, lowered from AST by
`ModuleStore::compile()`, consumed by the query/rewrite engine, and
projected to `TensorCoo` by `hymeko_hre::HypergraphEngine`.

So: do we (A) extract the existing IR into a new crate, (B) build the
spec's slotmap IR side-by-side, or (C) rebase the existing IR to match the
spec?

## Options considered

| # | Option | Shape | Pro | Con |
|---|--------|-------|-----|-----|
| A | **Extract existing `hymeko_core::ir` → new `hymeko_ir` crate** | Arena + decl-tree (unchanged) | Zero semantics churn, just packaging. | Arena IR is awkward for atomic mutations — `IRDelta::MoveVertex` doesn't map cleanly onto `ir.decl_nodes`. |
| B | **Dual IR: keep arena in core, add editor IR in `hymeko_emitter`** | Arena (compile) + slotmap (editor) with a bridge | Each IR is optimised for its phase; compiler pipeline unchanged; spec's editor semantics get the shape they want. | Two structures to maintain; a bridge layer is needed. |
| C | **Rebase `hymeko_core::ir` onto slotmap** | Slotmap everywhere | Single IR. | Massive blast radius: lowering, hashing, tensor expansion, module_store, daemon, ~500 call-sites. |

## Decision — **Option B**

Build the editor-facing IR as a **module inside `hymeko_emitter`** rather than a separate crate. Rationale:

1. **Phase separation is the right axis.** The arena IR is a *compilation* IR (lowering / hashing / tensor projection); the editor IR is a *mutation* IR (small atomic deltas, undo/redo, P2P gossip). Squeezing both into one structure compromises both.
2. **Industry precedent.** rustc ships HIR + MIR + Ty + ThirBody; Swift has AST + SIL + IRGen; GCC has GENERIC + GIMPLE + RTL. Multiple phase-specific IRs with bridges is the normative compiler pattern, not the exception.
3. **Two crates (arena + slotmap) is an *over*-split right now.** `hymeko_emitter` already needs the editor IR to feed the WASM editor's `apply_delta` / `snapshot` API. Hosting it inside the emitter keeps the blast radius to one new crate. If the editor IR later needs to stand alone (e.g. because `hymeko_mcp` wants it without the emitter), we promote it then.
4. **Bridge is small.** Roundtripping is `editor_ir.to_compiler_ir(interner)` (replays deltas into a fresh arena `Ir`) and `editor_ir.from_compiler_ir(ir, interner)` (walks the arena and inserts slotmap entries). Both are ~100 lines.

## Crate layout after Step 1

```
hymeko_core/                    (unchanged)
├── src/ir/                     canonical compile-time IR
│   ├── ir.rs                   arena (DeclId, NodeRec, EdgeRec, Arc)
│   ├── hash.rs                 Blake3 canonical hash
│   └── lower.rs                AST → arena IR
└── ...

hymeko_emitter/                 (NEW, created in Step 2)
├── Cargo.toml
└── src/
    ├── lib.rs                  re-exports
    ├── editor_ir.rs            HyMeKoIR (slotmap) + IRDelta + apply
    ├── bridge.rs               to_compiler_ir / from_compiler_ir
    ├── emit_hymeko.rs          arena IR → .hymeko text (deterministic)
    ├── emit_sysml.rs           arena IR → SysML v2 (matches docs/examples/hymeko_to_sysmlv2.md)
    ├── emit_rust_stubs.rs      arena IR → Rust trait skeletons
    └── emit_lean4.rs           arena IR → Lean 4 theorem templates
```

**Key point:** emitters take `&hymeko::ir::ir::Ir` (the arena) as input, not the editor IR. Editor IR is what the WASM canvas mutates; when you ask the editor to *emit*, it calls `to_compiler_ir()` first and then hands off to the emitter. This keeps emitters testable against fixtures loaded through the existing `ModuleStore` pipeline — no WASM in the loop.

## Editor IR shape (inside `hymeko_emitter::editor_ir`)

Follows the spec closely, with one clarification: the slotmap keys are **independent** of `hymeko_core`'s `NodeId` / `EdgeId`, because editor-IR mutation pre-dates lowering. The bridge maps them when converting.

```rust
// hymeko_emitter/src/editor_ir.rs
use slotmap::{SlotMap, new_key_type};
use serde::{Serialize, Deserialize};

new_key_type! {
    pub struct VertexKey;
    pub struct EdgeKey;
    pub struct PatchKey;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub name: String,
    pub level: i8,                       // G-SPHF level: -2..=8
    pub attributes: Vec<Attribute>,
    pub position: Option<Position>,      // canvas layout hint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperEdge {
    pub incident: Vec<(VertexKey, Sign)>, // sign per arc (+,-,~)
    pub weight: f64,
    pub patch_id: Option<PatchKey>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Sign { Plus, Minus, Neutral }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRDelta {
    AddVertex { data: Vertex },
    RemoveVertex { key: VertexKey },
    AddHyperEdge { data: HyperEdge },
    RemoveEdge { key: EdgeKey },
    MoveVertex { key: VertexKey, position: Position },
    UpdateWeight { key: EdgeKey, weight: f64 },
    UpdateSign { key: EdgeKey, arc_index: usize, sign: Sign },
    AttachAttribute { key: VertexKey, attr: Attribute },
    DetachAttribute { key: VertexKey, name: String },
    AddPatch { data: Patch },
    Batch { deltas: Vec<IRDelta> },     // one-shot bulk apply
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyMeKoEditorIR {
    pub vertices: SlotMap<VertexKey, Vertex>,
    pub hyperedges: SlotMap<EdgeKey, HyperEdge>,
    pub patches: SlotMap<PatchKey, Patch>,
}

impl HyMeKoEditorIR {
    pub fn apply(&mut self, delta: IRDelta) -> Result<(), IRError> { /* ... */ }
}
```

The spec's `AttributeValue`, `Position`, `Patch` copy over verbatim. Two
changes from the spec:

- **`incident` is `Vec<(VertexKey, Sign)>`** not `Vec<VertexKey>` — we need
  to preserve the `+ / - / ~` sign discipline the arena IR already
  encodes, otherwise the bridge is lossy.
- **`IRDelta::Batch`** — batching many deltas into one atomic application
  is the natural unit for CRDT gossip and for the bridge's "rebuild arena
  IR after n edits" pattern.

## Bridge sketch

```rust
// hymeko_emitter/src/bridge.rs  (skeleton)
pub fn to_compiler_ir(
    editor: &HyMeKoEditorIR,
    interner: &mut hymeko::resolution::interner::Interner,
) -> hymeko::ir::ir::Ir {
    // 1. Walk editor.vertices → NodeDecl + NodeRec in a fresh Ir.
    // 2. Walk editor.hyperedges → EdgeDecl + EdgeRec, build Arc from
    //    (VertexKey, Sign) tuples, inserting SignedRefR::{Plus/Minus/Neutral}.
    // 3. Run intern_pass + resolve on the fresh AST-equivalent surface.
    // 4. Return the lowered `Ir`.
    todo!("implement in Step 2")
}

pub fn from_compiler_ir(
    ir: &hymeko::ir::ir::Ir,
    interner: &hymeko::resolution::interner::Interner,
) -> HyMeKoEditorIR {
    // Inverse walk: each ir.nodes[i] → insert Vertex; each ir.edges[i] +
    // ir.arcs[...] → insert HyperEdge with signs from SignedRefR variant.
    todo!("implement in Step 2")
}
```

The bridge is **reversible-by-construction** once round-trip tests pin
down the invariants. That test is the first thing Step 2 builds.

## Implications for the spec's downstream steps

| Spec step | Impact of this decision |
|-----------|-------------------------|
| §3 `hymeko_emitter` (M2T) | Unchanged scope; now owns `editor_ir` too. |
| §4 `hymeko_wasm` | Imports `hymeko_emitter::editor_ir` instead of a standalone `hymeko_ir` crate. Depends on `hymeko_emitter`. |
| §5 `hymeko_server` | Unchanged; serves WASM + REST. |
| §7 `hymeko_mcp` | Imports `hymeko_emitter::editor_ir` + `hymeko_emitter::emit_sysml` et al. One dependency instead of two. |
| §8 `hymeko_wire` | CBOR serialisation of `IRDelta` — derives from `editor_ir`'s `#[derive(Serialize, Deserialize)]`. |
| §9 Workspace `Cargo.toml` | One fewer new crate (`hymeko_ir` dropped from members list). |

## Risks

- **Drift between the two IRs.** Mitigated by a round-trip property test:
  `from_compiler_ir(to_compiler_ir(editor_ir)) == editor_ir` and the
  reverse. Ship this with Step 2.
- **Bridge cost on large models.** The editor is expected to operate on
  models authored interactively (≤ thousands of vertices); the arena
  compile IR already handles that. For the robotics fixtures in
  `data/robotics/` (dozens of nodes) the bridge cost is irrelevant.
- **Spec author's original reason for a separate crate.** The spec
  implicitly wanted `hymeko_ir` usable without depending on
  `hymeko_emitter` or `hymeko_core`. If that becomes a real requirement
  (e.g. `hymeko_wire` wants to decode IR deltas without pulling in the
  emitter), we promote `editor_ir` to its own crate in a follow-up. Until
  then, YAGNI.

## What Step 2 does with this decision

Scaffolds `hymeko_emitter`:

1. `cargo new --lib hymeko_emitter`, register in workspace members.
2. Add `editor_ir.rs` per the shape above, with `apply(delta)` matching
   spec §2 including the new `Batch` variant.
3. Add `bridge.rs` with `to_compiler_ir` / `from_compiler_ir` stubs,
   followed by a round-trip property test over the existing
   `data/robotics/mini_arm.hymeko` fixture.
4. Add `emit_hymeko.rs` producing deterministic `.hymeko` text from an
   arena `Ir`. Round-trip test: `parse_description(emit_hymeko(ir)) ==
   ir` modulo canonical-hash equality.
5. Add `emit_sysml.rs` reproducing the ground truth in
   `docs/examples/hymeko_to_sysmlv2.md` for `mini_arm`. Use that doc as
   the golden fixture.
6. Stub `emit_rust_stubs` and `emit_lean4` with minimal output so the
   public surface is complete; full bodies can land in a later slice.

Exit criterion for Step 2: `cargo test -p hymeko_emitter` green with
bridge round-trip + emit_hymeko round-trip + emit_sysml golden match.

## Follow-up: what changes upstream in `hymeko_core`?

**Nothing.** That's the whole point of Option B. The existing arena IR,
`ModuleStore::compile()`, canonical hashing, and the
`hymeko_hre::HypergraphEngine` projection pipeline stay exactly as they
are. The editor is purely additive.
