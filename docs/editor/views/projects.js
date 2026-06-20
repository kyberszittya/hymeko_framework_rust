// Multi-file project loading for the editor. A *project* is a HyMeKo description spread over several
// editable instance files (robot / env / reward / strategy) that share a meta vocabulary; the editor
// edits one file at a time as the compile root and places the meta files that file `@"…"`-imports into
// the compile `space`. The embedded sources are GENERATED from the repo files by
// scripts/gen_editor_projects.py (the browser cannot fetch data/), so they never drift by hand.
import { PROJECTS } from "./projects_data.js?v=1";

export { PROJECTS };

/** Look up a project by id, or null. */
export function projectById(id) {
  return PROJECTS.find((p) => p.id === id) ?? null;
}

/** Look up a project file entry by name, or null. */
export function projectFile(project, fileName) {
  return project?.files.find((f) => f.name === fileName) ?? null;
}

/**
 * The compile `space` for editing ``fileName`` of ``project``: exactly the meta files that file
 * `@"…"`-imports (keyed by their import filename), so the WASM resolves the root's imports and
 * nothing unrelated is compiled.
 * Preconditions: every name in the file's `imports` is a key of `project.meta` (the generator +
 * projects.test.mjs guarantee this).
 * Postconditions: a `{ filename: source }` map; `{}` if the file is unknown.
 */
export function spaceForFile(project, fileName) {
  const f = projectFile(project, fileName);
  if (!f) return {};
  const space = {};
  for (const imp of f.imports) {
    if (project.meta[imp] !== undefined) space[imp] = project.meta[imp];
  }
  return space;
}
