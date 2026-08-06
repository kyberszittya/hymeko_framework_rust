---
name: project-editor-mdp-project
description: WASM editor loads the Galambos MDP as a multi-file project; left palette is a TabView (Outline/Tools); DONE + verified in Firefox
metadata: 
  node_type: memory
  type: project
  originSessionId: e049ea12-7387-4a59-87f4-051966d7cfcb
---

The browser editor (`docs/editor/`) now loads the **Galambos MDP as one multi-file project** and the
left palette is a real **TabView**. State as of 2026-06-21 (DONE, verified working in Firefox):

- **Tabs:** `.palette-tabs` + `.palette-tabpane` in `index.html`; switch logic `showPaletteTab()` in
  `editor.js`. **Outline** tab = collapsible project tree (`#projectTree`, `.proj-node`/`.proj-file`)
  → files robot/env/reward/strategy; **Tools** tab = Add palette + selection/stats/view/query.
- **Project data is GENERATED**, not hand-embedded: `scripts/gen_editor_projects.py` reads the real
  `data/robotics/galambos_*.hymeko` + the meta vocab they `@"…"`-import → `docs/editor/views/projects_data.js`.
  Re-run the script after editing any project `.hymeko`. Wrapper `views/projects.js` (`projectById`,
  `spaceForFile`). The editor compiles the selected file as root with its metas in the WASM `space`
  via the existing `parse_and_compile_files`.
- **Deep-links:** `?project=galambos&view=graph&tab=outline` (also `tab=tools`). Reuses the existing
  `?profile=`/`?view=`/`?layout=` query convention.
- **Tests:** `node --test docs/editor/views/*.test.mjs` = 72 pass (incl. 4 in `projects.test.mjs`
  guarding import-completeness). No WASM rebuild — JS/data/CSS only, non-core.

**Gotcha:** `editor.css`/`editor.js` load WITHOUT `?v=` cache-busting, so Chrome served a stale copy
during this session (looked like the tabs weren't switching); Firefox/hard-refresh showed the new UI.
Follow-up worth doing first tomorrow: add `?v=` to those two refs in `index.html`.

**Open follow-up:** no save-back round-trip (in-browser edit → write repo files); editor is
read-from-generated-snapshot. Report: `reports/2026-06-21-editor-mdp-project.md`. Plan:
`docs/plans/2026-06-21-editor-mdp-project/`. Editor served on `localhost:8731` during the session.

Related: [[project-mdsd-reuse-and-docs]], [[project-seminar-demos-and-hymeyolo-plan]],
[[project-galambos-reward-shaping]].
