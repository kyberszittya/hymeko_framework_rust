# Hero demo — Phase 2 (hybrid: robots and learners)

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-hero-demo-phase2/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Extended the hero orchestrator so the **same gated pipeline** emits both robot
descriptions and the **learner**: a neural model authored in `.hymeko` →
`torch_dataflow` (the Gömb cascade) → a runnable `torch.nn.Module`. This lands
the full thesis — *one accountable IR generates both the robot and its learner*,
each behind the same A1–A5 gate.

Additive to `demos/hero/` only (Phase 1's structure held): a `Target.via` field
("emit" vs template-driven "transform"), a pure `emit_args` builder, a scenario
`kind` ("robot"/"learner"), `NEURAL_TARGETS`, two learner scenarios, and
`cwd=REPO_ROOT` so `--transforms-dir` + relative includes resolve.

## What runs today (measured)

```
2 robot + 2 learner scenarios, one A1–A5 gate

[robot]   FANUC LR Mate 200iD arm     gate: warnings
          urdf 4903 · sdf 5824 · mjcf 1369 · dot 1128 · mermaid 938  (B)
[robot]   Anthropomorphic 6-DoF arm   gate: warnings
          urdf 4784 · sdf 5592 · mjcf 1336 · dot 1009 · mermaid 819  (B)
[learner] Simple MLP                  gate: clean
          torch_dataflow 4205 · dot 309  (B)
[learner] MNIST ResMLP                gate: clean
          torch_dataflow 6224 · dot 311  (B)
[gate]    broken twin                 rejected (UnresolvedRef ghost_link) — not trusted
```

The learner artifacts are runnable `torch.nn.Module` sources (`import torch …`).

## Files touched

- `demos/hero/hero_demo.py` — `Target.via`; pure `emit_args`; `HeroScenario.kind`;
  `NEURAL_TARGETS` + two learner scenarios; `emit` uses `emit_args`; `_run` sets
  `cwd=REPO_ROOT`; summary shows kind + a robot/learner count line.
- `demos/hero/test_hero.py` — +4 tests (`emit_args` emit vs transform, `via`
  default, robot+learner catalog, learner emits a torch module).
- `demos/hero/README.md` — learner track + Phase 2 marked done.

## CORE.YAML items touched

None. `demos/**`; orchestrator invokes the built `hymeko` binary. **No torch
dependency** — the demo emits + gates the module *source* but never imports
torch, so it stays dependency-free / CI-safe (running the module is the user's
step; torch is a §1 CORE-pinned dep).

## Test results

- **pytest (`-p no:randomly`):** 11 passed / 0 failed (2.1 s) — 9 unit (incl. the
  new `emit_args`/catalog units) + 2 robot integration + 1 learner integration.
- **Demo run:** 2 robots × 5 targets + 2 learners × 2 targets emitted; broken
  twin rejected.

## Static analysis / health

- `ruff check demos/hero/`: clean. `mypy --strict` (both files): clean.
- **No §6.5 anti-patterns:** the emit/transform split is isolated in the pure,
  unit-tested `emit_args` (no duplicated CLI surgery); scenarios stay
  data-driven (a learner is one entry + a target tuple); still a thin
  orchestrator over the CLI (#3).

## Open issues / follow-ups

- Phase 1.5 (CLI exit-code hardening) turned out to be a non-issue — the CLI
  already exits non-zero on failure; the orchestrator now uses the exit code as
  the authoritative gate signal (`reports/2026-06-15-hero-demo-gate-exitcode.md`).
- Phase 3 (Gömb/Soma perception) — needs Soma vision round-tripped through
  `.hymeko` (currently Python-only in `signedkan_wip`).
- Optional: a single scenario that pairs a robot *and* its perception net in one
  source (today robots and learners are separate scenarios under one gate).

## Experiment provenance

Not a measurement experiment. Toolchain: `hymeko_cli` (cargo, stable), Python 3 +
pytest/ruff/mypy, MiKTeX pdflatex (plan). Working tree dirty from prior session
work unrelated to this change.
