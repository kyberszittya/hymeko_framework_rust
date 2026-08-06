# Hero demo — Phase 1 (robotics spine)

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-hero-demo-phase1/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Scaffolded the hero demo Phase 1 — *one authored `.hymeko` model → validation
gate → many faithful targets* — as a thin orchestrator (`demos/hero/`) over the
real `hymeko` CLI (no parse/transform reimplementation, CLAUDE.md §6.5 #3). The
framing is **"structurally accountable AI": the gate, not the speed, is the
star** — a malformed model is rejected before any target is trusted.

Verified end-to-end: each shipped arm model emits **5 targets** (URDF, SDF, MJCF,
DOT, Mermaid), and a self-contained **broken twin** (dangling joint endpoint) is
**rejected** by the gate.

The user chose "go with #1 (FANUC robotics spine), but plan the others" — so the
other tracks (hybrid robots+learner, Gömb/Soma perception, CLI exit-code
hardening, editor integration) are written up as the roadmap in the plan + README.

## What runs today (measured)

```
## FANUC LR Mate 200iD arm  [fanuc_arm]      gate: warnings
   urdf 4903 B · sdf 5824 B · mjcf 1369 B · dot 1128 B · mermaid 938 B
## Anthropomorphic 6-DoF arm  [anthropomorphic_arm]   gate: warnings
   urdf 4784 B · sdf 5592 B · mjcf 1336 B · dot 1009 B · mermaid 819 B
## Accountability gate (broken twin)         gate: rejected
   ❌ … failed: UnresolvedRef { target: "ghost_link" } ; trusted? False
```

(Also confirmed in discovery: `data/nn/simple_net.hymeko` → `torch_dataflow`
emits a runnable `torch.nn.Module` — the Phase 2 hybrid path is real.)

## Files touched

New (all under `demos/hero/`):
- `hero_demo.py` — orchestrator: `GateStatus`/`GateVerdict` + pure
  `parse_validate_output`; `Target`/`HeroScenario`/`EmitResult`/`HeroReport`
  dataclasses; `HeroDemo` (CLI driver); `SCENARIOS` catalog; `BROKEN_TWIN`;
  `main()`.
- `test_hero.py` — pytest: 5 unit (gate parsing CLEAN/WARNINGS/REJECTED/
  fail-closed, catalog integrity) + 2 integration (emit-all-targets,
  broken-twin-rejected; skip if the CLI isn't built).
- `README.md`, `.gitignore` (`out/`).

No crates, `data/`, or `CORE.YAML` items touched. Built `hymeko_cli` (already
non-core) to run the demo; no source change to it.

## CORE.YAML items touched

None. `demos/**` is non-core; the orchestrator only invokes the built `hymeko`
binary. No new dependency (stdlib `subprocess` only).

## Test results

- **pytest (`-p no:randomly`):** 7 passed / 0 failed (2.2 s) — 5 unit + 2
  integration (the binary was built, so integration ran).
- **Demo run:** both arms emit all 5 targets; broken twin rejected.

## Static analysis / health

- `ruff check demos/hero/`: clean. `mypy --strict` on both files: clean.
  `radon cc -a -nc`: no function above rank B.
- **No §6.5 anti-patterns:** thin orchestrator over the CLI (no
  parse/transform duplication, #3); data-driven `SCENARIOS` (adding one = one
  entry, no code); pure gate parser split from the subprocess driver; fail-closed
  verdict (unrecognised output ⇒ rejected).

## Key finding (corrected 2026-06-15)

**Correction:** an earlier draft claimed `hymeko validate`/`emit` "exit 0 even on
failure." That was wrong — it came from reading `$?` *after a pipe* (`… | head`),
which returns `head`'s exit code, not `hymeko`'s. Re-checked directly: the CLI
**already exits non-zero** on a hard failure (`validate` broken → exit 1, `emit`
broken → exit 1; warnings keep exit 0). So Phase 1.5 (CLI hardening) was a
non-issue; the real work was making the orchestrator use the exit code as the
authoritative gate signal — done in
`reports/2026-06-15-hero-demo-gate-exitcode.md`.

## Open issues / follow-ups

- Phase 1.5 (CLI exit codes), Phase 2 (hybrid robot+learner via `torch_dataflow`),
  Phase 3 (Gömb/Soma perception — needs Soma↔`.hymeko` wiring), editor profile
  for a hero cell. All detailed in the plan + README roadmap.
- The FANUC *handover task* layer (`handover_task.hymeko`) validates but its
  task-graph emitters (BehaviorTree/PDDL/ROS2-action) are an upstream open
  follow-up; Phase 1 emits from the kinematic arm model.

## Experiment provenance

Not a measurement experiment (a demo orchestrator). Toolchain: rustc/cargo
stable (`hymeko_cli` built), Python 3 + pytest/ruff/mypy/radon, MiKTeX pdflatex
(plan). Working tree dirty from prior session work unrelated to this change.
