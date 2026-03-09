# Project Changelog — 2026-03-09

## Architecture Catalog & Folder READMEs
- Rebuilt `architecture/README.md` into a full index covering every diagram (overview, components, communication, flow, daemon use-cases) with inline Mermaid renderings plus linked SysML2 snippets and viewer guidance.
- Added per-subfolder READMEs (`architecture/components`, `architecture/communication`, `architecture/communication/memory_communication`, `architecture/flow`, `architecture/daemon`) so each diagram now documents itself in-place; missing SysML assets are called out for future contributors.
- Ensured the catalog points to the new documentation so navigation from root → diagram → SysML model is one click.

## Repository README Polish
- Surfaced `logo.png` under the main title of `README.md` and introduced a dedicated **Architecture** section that links to the new diagram catalog, improving first impressions and discoverability for design docs.
- Updated the table of contents and descriptive text around the architecture anchor so readers can jump straight into the control/data-plane overview from the landing page.

## Follow-ups
- If additional diagrams are added under `architecture/`, mirror the pattern: drop the Mermaid/SysML files alongside a README snippet, then extend the catalog entry in `architecture/README.md` and the next dated changelog.

