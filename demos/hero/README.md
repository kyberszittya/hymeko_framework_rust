# HyMeKo hero demo — *one accountable model, many faithful targets*

**Thesis:** one authored `.hymeko` model → signed-typed hypergraph IR →
**validation gate** → many faithful emitters. *We make AI systems structurally
accountable.* The gate is the point: a malformed model is **rejected before any
target is trusted**.

This is a thin orchestrator over the real `hymeko` CLI — it does not
re-implement parse / query / transform (CLAUDE.md §6.5 #3).

## Run

```bash
cargo build -p hymeko_cli         # builds target/debug/hymeko[.exe]
python demos/hero/hero_demo.py    # validate + emit into demos/hero/out/
```

Each scenario is validated, then (if the gate trusts it) fanned out:

- **robot** scenarios → URDF, SDF, MJCF, DOT, Mermaid;
- **learner** scenarios → `torch_dataflow` (a runnable `torch.nn.Module`, the
  Gömb cascade) + DOT — *the same accountable IR generates the learner.*

The run ends with the **accountability act**: a self-contained broken model (a
dangling joint endpoint) is rejected by the gate. (The demo emits + gates the
torch module *source*; it never imports torch, so it stays dependency-free.)

Example output:

```
## FANUC LR Mate 200iD arm  [fanuc_arm]
   gate: warnings — … compiled with 1 warnings:
   [ok ] urdf       4903 B
   [ok ] sdf        5824 B
   [ok ] mjcf       1369 B
   [ok ] dot        1128 B
   [ok ] mermaid     938 B

## Accountability gate (broken twin: dangling joint endpoint)
   gate: rejected — ❌ … failed: UnresolvedRef { … target: "ghost_link" }
   trusted for emission? False
```

## Layout

| file | role |
|---|---|
| `hero_demo.py` | orchestrator: `GateVerdict` + pure `parse_validate_output`, the `HeroDemo` CLI driver, the `SCENARIOS` catalog, and `main()` |
| `test_hero.py` | pytest — unit (gate parsing, catalog) + integration (real emit/validate; skipped if the CLI isn't built) |
| `out/` | generated artifacts (git-ignored by convention; regenerate any time) |

## Adding a scenario

Append one `HeroScenario(...)` to `SCENARIOS` in `hero_demo.py` — a source path,
a model name, and the target list. No new code.

## Roadmap (see `docs/plans/2026-06-15-hero-demo-phase1/`)

- **Phase 1.5 — resolved** — the CLI already exits non-zero on failure; the
  orchestrator uses the exit code as the authoritative gate signal
  (`verdict_from_run`).
- **Phase 2 — hybrid — DONE 2026-06-15** — paired neural models (`data/nn/*`) →
  `torch_dataflow` (the Gömb cascade) behind the *same* gate: one accountable IR
  generates both the robot and its learner.
- **Phase 3 — Gömb/Soma perception** — a signed-hypergraph / HSiKAN model →
  Gömb cascade + gate (needs Soma vision round-tripped through `.hymeko`).
- **Editor** — load a hero cell as an editor profile (the multi-file FANUC cell
  loads via the imports feature).
