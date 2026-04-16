# T01 — Code Review & SignedRefR Optimization

**Status:** ✅ DONE  
**File:** `hymeko_core/src/ir/ir.rs`  
**Lines changed:** +25

---

## Problem

The `SignedRefR` enum required a triple-match pattern every time any code needed the inner `RefAtomR`, target `DeclId`, or sign value:

```rust
// This pattern appeared 10+ times across the codebase
let atom = match sref {
    SignedRefR::Plus(a) | SignedRefR::Minus(a) | SignedRefR::Neutral(a) => a,
};
```

Free functions `ref_target()` and `ref_sign()` in `ir/common.rs` partially addressed this but were not discoverable through IDE autocomplete and separated the logic from the type.

## Solution

Added inherent methods to `SignedRefR`:

```rust
impl SignedRefR {
    pub fn atom(&self) -> &RefAtomR { ... }
    pub fn target(&self) -> DeclId { ... }
    pub fn sign(&self) -> i8 { ... }  // +1, -1, 0
}
```

## Impact

- All new query code uses `sref.target()` and `sref.sign()` directly
- Existing `ir::common::ref_target` / `ref_sign` still work (no breaking change)
- `kinematic.rs` joint topology extraction uses `sref.atom().weights` to read origin transforms

## Other Review Findings (not acted on)

1. **`resolve_ref_to_declid` clones `fq_buffer` on ambiguity path** — acceptable for query sizes, note for future optimization
2. **`Ir::preallocate_from_index` fills with `DeclKind::Node` default** — harmless but confusing for debugging; could add `DeclKind::Uninitialized` sentinel later
3. **`edge_rec.arcs: Vec<HyperArcId>` is the authoritative arc-per-edge list** — confirmed correct, used by query engine's `check_arc_ref`
