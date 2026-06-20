# WASM editor: wire in the Galambos RL state + group the gallery into projects

*2026-06-21 · Aiko (Claude Code) for Dr. Csaba Hajdu*

## Summary

Two changes to the in-browser HyMeKo editor (`docs/editor/`): (1) the **Galambos
grasper RL state** — the two-arm signed kinematic hypergraph the HSiKAN actor/critic
message-pass over — is now a gallery example that renders in Hypergraph 3D; (2) the
example **gallery is grouped into projects** via `<optgroup>`. JS-only; no WASM rebuild.

## Changes

- `docs/editor/views/examples.js`:
  - New `GALAMBOS_STATE` source — **self-contained** (inline `kit` vocabulary, no `@"…"`
    import, since the browser can't fetch repo files): six link vertices
    (base/upper/lower × left/right) + four revolute hyperedges `(+ parent, − child,
    − AXIS_Z)`. Validated through the CLI (same compiler as the WASM) — compiles clean.
  - Each `EXAMPLES` entry gained a `group` field; new `examplesByGroup()` helper buckets
    the catalog by project, first-seen order.
  - Groups: **Robotics** (kinematic arm), **Galambos (RL grasp)** (the RL state),
    **Systems engineering** (SysML trace), **Combinatorial hypergraphs** (Fano / sunflower
    / K₄ / generic).
- `docs/editor/editor.js`: the gallery dropdown now renders `<optgroup>`s from
  `examplesByGroup()`; import bumped to `?v=21` (cache-bust).
- `docs/editor/views/examples.test.mjs`: +2 tests (Galambos state present + self-contained
  + renders in 3D; `examplesByGroup` buckets every entry into exactly one group).

## How to use

Serve `docs/editor/` (README "Build + serve"), open the gallery dropdown — it's now grouped
by project; pick **Galambos (RL grasp) → "Galambos grasper — RL state"** to load the two-arm
hypergraph and jump to the 3D view. The embedded source compiles in the existing WASM build
(no `wasm-pack` rebuild needed).

## Test results

- `node --test docs/editor/views/*.test.mjs` — **68 passed** (incl. the 2 new). The
  embed≡fixture consistency test is unaffected (the Galambos entry is inline-only, not in
  `FIXTURE_OF`).
- `hymeko validate` on the standalone Galambos state source — ✅ (clean, no warnings).

## CORE.YAML / dependencies

**None.** `docs/editor` + `hymeko_*` untouched at the Rust level; no WASM rebuild, no
dependency change.

## Open / follow-up

- "Project" is currently a gallery group of single-file entries. A richer **multi-file
  project** (load `galambos_planar` + `galambos_env` + `galambos_task` + `galambos_strategy`
  together via the editor's compile `space`) is the natural next step if you want the whole
  MDP description editable as one project, not just the state hypergraph.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). Node v24; editor served statically.
