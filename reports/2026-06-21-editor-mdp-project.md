# WASM editor: the Galambos MDP as one multi-file project

*2026-06-21 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plan: [docs/plans/2026-06-21-editor-mdp-project/](../docs/plans/2026-06-21-editor-mdp-project/)*

## Summary

The whole Galambos MDP is now loadable in the browser editor as one project: the four instance
files (robot / env / reward / strategy), switchable, with the meta vocabulary they import placed
in the editor's compile `space`. The embedded sources are **generated** from the real `.hymeko`
files, so they don't drift by hand. JS + data only — no WASM rebuild.

## Design

`parse_and_compile_files(ROOT_NAME, {root, ...space})` (already in the WASM) + the existing
profile pattern do the multi-file work. A *project* extends it: several editable instance files
sharing a meta space, with a file switcher that sets the compile `space` to exactly the metas the
selected file imports.

- **`scripts/gen_editor_projects.py`** (NEW) — reads each instance file under `data/robotics/`,
  parses its `@"…"` imports, resolves + reads those meta files, and emits
  `docs/editor/views/projects_data.js` (`PROJECTS` catalog with embedded sources). Re-run after
  editing a project file. JS escaping handles backticks/`${` in the sources.
- **`docs/editor/views/projects.js`** (NEW) — re-exports `PROJECTS` + `projectById`, `projectFile`,
  `spaceForFile` (the meta subset a file imports).
- **`docs/editor/editor.js`** — the left palette is a **two-tab** panel (`.palette-tabs` +
  `.palette-tabpane`): **Outline** (a collapsible tree of projects → files) and **Tools** (the
  element palette / selection / stats / view / query). Clicking a file in the tree sets the textarea
  to that file's source and `space` to its imported metas, recompiles, and jumps to the file's view
  (active file highlighted). Deep-links `?project=<id>`, `?view=<v>`, `?tab=<outline|tools>` set the
  initial project / view / tab — shareable editor links.
- **`docs/editor/index.html` / `editor.css`** — the `.palette-tabs` + `.palette-tabpane` structure
  and the `.proj-node` / `.proj-file` tree styling.

The Galambos project: files `galambos_planar` (robot, → 3D view), `galambos_env`, `galambos_task`,
`galambos_strategy`; meta `meta_kinematics` / `meta_env` / `meta_reward` / `meta_strategy`.

## How to use

Serve `docs/editor/` (e.g. `python -m http.server`); on the left, the **Project** tab shows the
**Galambos MDP (RL grasp)** tree — click **robot / environment / reward / strategy** to edit each as
the compile root (its meta imports go into the `space`). The **Tools** tab holds the element palette.
Or deep-link directly: `…/index.html?project=galambos&view=graph`. Editing any file recompiles live.

## Test results

- `node --test docs/editor/views/*.test.mjs` — **72 passed** (4 new: project bundles the 4 files;
  every file's `@"…"` imports resolve in the meta map; `spaceForFile` returns exactly the file's
  metas; listed imports actually appear in the source). `node --check editor.js` — valid.
- Generator + import-completeness verified; all four instance files independently `hymeko validate`
  ✅ (so each compiles with its meta space — the same compiler the WASM uses).
- **Not directly tested:** in-browser render (no headless browser here) — but the multi-file
  compile path is the one the profiles already use in the browser, and every source validates.

## CORE.YAML / dependencies

**None.** `docs/editor` + `scripts/` (non-core); the WASM compiler is unchanged (multi-file entry
already existed) — no `wasm-pack` rebuild.

## Open / follow-up

- A "save project" round-trip (edit in-browser → write back to the repo files) is not wired; the
  editor remains read-from-generated-snapshot. Regenerate with the script to refresh after repo edits.
- **Stale-cache gotcha:** `editor.css` / `editor.js` are loaded without a `?v=` query, so a browser
  may serve a cached copy after edits (Chrome did during this session; Firefox showed the new UI).
  Hard-refresh (Ctrl-F5), or add cache-busting versions to those two references in `index.html`.
- Verified working in Firefox (the tab switch + tree + deep-links). Tabs confirmed via per-tab
  headless screenshots (`?tab=outline` / `?tab=tools`).

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). Node v24; editor served statically.
