# [CRITICAL] Nagare → HSMM Compiler — FROZEN, AWAITING FPGA

**Status: FROZEN. Do not implement in current sessions.**

This directory holds a critical plan for a future session. It describes
the Nagare → HSMM (Hypergraph Storage Modification Machine, Hajdu &
Csapó 2026) compiler — the systems counterpart to the Nagare
algorithmic framework.

## Why frozen

The HSMM FPGA implementation is a prerequisite, and not in this repo
yet. See the prerequisites checklist in `plan.tex` Section
"Prerequisites checklist." All four items must be true before
reopening this plan:

1. HSMM Zynq bitstream or public reference impl available
2. Nagare Stage 3 (GPU backend) shipped
3. Catmull-Rom spline ported to pure Rust (Nagare Stage 4)
4. Slashdot 5-seed AUC parity Nagare-CPU vs PyTorch+Triton

## Reopen criteria — one of:

- HSMM bitstream lands in `hymeko_monitor/` (the existing
  `hsmm_bridge.rs` scaffold becomes a working bridge)
- Hajdu & Csapó companion paper publishes a public HSMM reference
  implementation
- Explicit user decision to prioritise this over Nagare Stage 1-4

## Why critical

This plan completes a three-paper arc:

1. **Hajdu & Csapó 2026 (Zenodo, in review):** HSMM theory
2. **HymeKo-Nagare 2026 (planned, MLSys 2027):** Cycle-additive
   ML framework with closed-form Clifford backward; structural
   correspondence to HSMM
3. **This plan, 2027 work:** Compiler that realises the
   correspondence empirically — Nagare programs on Zynq UltraScale+
   HSMM kernel, with throughput + energy measurements

Without the compiler, the three-paper arc is two papers with a
theoretical bridge. With the compiler, it's a complete system from
abstract machine through programming model to deployed hardware.

Venue (when shipped): **FPGA 2028 / FCCM 2028 / ASPLOS 2028**.

## Files

- `plan.tex` — full plan (~10 pages compiled)
- `plan.pdf` — built from plan.tex
- `plan.tikz` — compiler-stages structural diagram (TBD when reopened)
- `plan.mmd` — Mermaid flowchart (TBD when reopened)
- `README.md` — this file

The TikZ and Mermaid artefacts are deferred to the reopen session;
the tex/pdf carry the load for now since this is a frozen plan.

## Pointers to current Nagare state

- `docs/plans/2026-05-11-hymeko-nagare-flow/plan.{tex,pdf}` — Nagare
  algorithmic plan (active, Stages 1-2 shipped today)
- `docs/plans/2026-05-11-hymeko-nagare-flow/math.{tex,pdf}` —
  Mathematical foundations of Nagare; Theorem 1 (closed-form Clifford
  backward) is the key result that makes this compiler structurally
  clean
- `hymeko_nagare/src/` — Nagare crate (8 ops tests pass today)
- `hymeko_graph/src/spine.rs` — Clifford-FIR forward + backward
  (8/8 tests pass, numerical-grad verified)
- `docs/plans/plans_20260429/hymeko_monitor_plan.md` — existing
  HSMM integration scaffolding (`hsmm_bridge.rs` mentioned but
  not yet implemented)
