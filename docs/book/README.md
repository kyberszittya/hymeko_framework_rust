# HyMeKo book

User-facing documentation for the framework, built with [mdBook](https://rust-lang.github.io/mdBook/).

## Build & serve locally

```bash
cargo install mdbook        # one-time
mdbook serve docs/book      # serves http://localhost:3000
mdbook build docs/book      # static site -> docs/book/book/
```

## Structure

```
docs/book/
├── book.toml          mdbook config
└── src/
    ├── SUMMARY.md     master TOC (mdbook reads this)
    ├── intro.md       landing page
    ├── quickstart/    13 use-case tutorials
    ├── architecture/  crate map, data flow, layers, extension points
    ├── concepts/      IR, queries, templates, tier system, tensor decomp
    ├── recipes/       add-a-format, add-a-layer-kind, add-a-query, debug
    ├── research/      signedkan_wip orientation, HSiKAN, HyMeKo-driven training
    ├── reference-cli.md
    └── reference-env-vars.md
```

## Adding content

- New page: create `<path>.md` in `src/` and add a line to `SUMMARY.md`
- Cross-links: relative paths, e.g. `[Add a new format](../recipes/add-a-format.md)`
- Code blocks: ```` ```bash / ```rust / ```python ```` etc. work; mdBook adds copy buttons for runnable blocks
- Math: `book.toml` has `mathjax-support = true`; use `\\( ... \\)` inline or `\\[ ... \\]` block

## Publishing to GitHub Pages

```bash
mdbook build docs/book
# copy docs/book/book/ to gh-pages branch (or use a CI action)
```

A future GitHub Action could auto-deploy on push to master.

## Source-of-truth pointers

Per project memory: do not duplicate research-state from `signedkan_wip/HSIKAN_*.md` — link instead. The book cites concrete file paths so source updates flow naturally.
