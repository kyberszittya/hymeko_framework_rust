# Hero demo — Phase 3 (Gömb structural parity + minimal Soma)

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-hero-demo-phase3/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Brought the cognitive-stack **learner** into the gated hero pipeline with
**structural parity**: the emitted `torch.nn.Module` faithfully realises the
architecture declared in the `.hymeko` IR — proven text-only, torch-free.

Per the user's decisions: **Gömb/HSiKAN cascade first, then minimal Soma;
structural parity (gate-framed); Soma included; torch-free.**

- **Gömb/HSiKAN cascade** (`data/nn/hsikan_mixed.hymeko`, already authored) added
  as a gated learner scenario → emitted module is **faithful, 6/6 layers**
  (sk2/sk3/sk4/sk5 + arity_mixer + signed_classifier).
- **Minimal Soma vision** (`data/nn/soma_vision.hymeko`, new): patch-graph
  `hypergraph_conv` → `walk_layer` (open-walk conv) → `signed_classifier` →
  faithful **3/3**.
- **Parity checker** (`demos/hero/learner_parity.py`, pure): `parse_hymeko_layers`
  / `parse_torch_attrs` / `structural_parity` → `ParityReport`; faithful iff
  every declared layer is realised as a `self.<name>` sub-module.

## Verified (all run 2026-06-15)

```
[learner] Gömb HSiKAN cascade   gate: clean   parity: faithful (6/6 layers)
[learner] Soma vision           gate: clean   parity: faithful (3/3 layers)
[learner] Simple MLP            gate: clean   parity: faithful (2/2)
[learner] MNIST ResMLP          gate: clean   parity: faithful (5/5)
[gate]    broken twin           rejected (UnresolvedRef ghost_link)
```

- `hymeko validate` on `hsikan_mixed.hymeko` + `soma_vision.hymeko`: both ✅.
- `pytest -p no:randomly demos/hero`: **15 passed** (3 new: parse-layers,
  parity faithful/missing, Gömb+Soma integration).
- `ruff check` + `mypy --strict` (3 files): clean.
- Plan PDF compiles.

## Files touched

- `demos/hero/learner_parity.py` — **new, pure** (no torch import).
- `data/nn/soma_vision.hymeko` — **new** (uses only emittable `meta_nn` kinds).
- `demos/hero/hero_demo.py` — Gömb + Soma learner scenarios; parity line per
  learner in `main`.
- `demos/hero/test_hero.py` — parity units + Gömb/Soma integration test.

## CORE.YAML items touched

None. `demos/**`, `data/**`. **No torch import** — parity is text-only over the
emitted source, so the demo stays CI-safe (torch is a §1 CORE-pinned dep;
running the module remains the user's step).

## Static analysis / health

- ruff + mypy --strict clean. **No §6.5 anti-patterns:** pure parity core split
  from the demo driver; data-driven scenarios (Gömb/Soma are one entry each);
  Soma uses only template-covered layer kinds so parity is honest (an
  unsupported kind would show *missing*, not pass silently).

## Honesty note

"Structural" parity asserts the **layer set + names** (the emit realises every
declared layer), **not** trained weights or numeric behaviour — the full
runnable round-trip was the explicitly-rejected option (it pulls torch). The
faithful Soma internals (Hodge/stim/patch) live in `hymeko_neuro` and remain a
later round-trip item; `soma_vision.hymeko` is the minimal patch→walk→classify
skeleton in the Soma family.

## Process note

Most of this phase was authored while the harness command-safety classifier was
temporarily unavailable (all shell execution blocked). Code was written via
file tools and **held as unverified** until the classifier recovered, then
verified in one batch — no success was reported before it ran.

## Open issues / follow-ups

- Faithful Soma (Hodge/stim/patch round-trip emitter) and a numeric round-trip
  vs `cascade.py` remain (larger; the latter pulls torch).

## Experiment provenance

Not a measurement experiment. Toolchain: `hymeko_cli` (cargo, stable), Python 3 +
pytest/ruff/mypy, MiKTeX pdflatex. Working tree dirty from prior session work.
