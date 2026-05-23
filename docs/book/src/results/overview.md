# Results & evidence hub

This part of the book is for **humans and agents**: one place for **abbreviations**, **mathematics**, **committed benchmark numbers**, and **where files live** so scores are not argued from memory.

> **Warning:** before stating an AUC or “underperformance”, open the cited artifact or the evidence contract. Joint HSiKAN mix on Bitcoin OTC (**≈ 0.98**) and Phase‑8 **lean** panel HSiKAN (**≈ 0.85**) are **different protocols** — same family name, different configuration.

## How to read this part

| Chapter | Use when |
|---------|----------|
| [Abbreviations & symbols](./abbreviations.md) | You hit `ABB`, `OTC`, `edge_cr`, tuple labels, … |
| [Mathematics](./mathematics.md) | You need definitions: AUC, paired Δ, spectral entropy sketch |
| [NN variants & layer geometry](../research/nn-architectures-and-layer-geometry.md) | You need to know which `nn.Module` maps to which benchmark row |
| [SOTA snapshot & diagrams](./sota-snapshot.md) | **One-page architecture × dataset table** (HyMeKo ★ vs baselines) + bar charts for Bitcoin / Slashdot / Epinions |
| [Evidence contract](./evidence-contract.md) | You need the canonical path list + anchored numbers |
| [Artifact index](./artifact-index.md) | You need the full experiment tree inventory |
| [Cold start](./cold-start.md) | You are new to the repo (build, test, PYTHONPATH) |

## Source files on disk (edit these first)

| Role | Repository path |
|------|-------------------|
| Evidence rules + anchors | `docs/RESULTS_DISCIPLINE.md` |
| SOTA charts (duplicate of book chapter source) | `docs/SOTA_RESULTS.md` |
| Repo cold start | `COLD_START.md` (repo root) |
| Experiment file manifest | `signedkan_wip/experiments/results/AGGREGATE_index.md` |
| Report / overnight manifest | `reports/AGGREGATE_index.md` |

The book chapters **include** those files where possible so agents editing the repo update a **single** markdown source; rebuild mdBook after edits.

## Build the book

From `docs/book/`:

```bash
mdbook build --open   # optional --open
```

HTML output: `docs/book/book/` (gitignored in many setups — check `.gitignore`).

**Public deploy:** this book is built by **GitHub Actions** into the same **`_site/`** tree as the WASM editor and Rust API docs — see [`docs/DEPLOY_GITHUB_PAGES.md`](../../../DEPLOY_GITHUB_PAGES.md).

Math is rendered with **MathJax** (`mathjax-support = true` in `book.toml`). **Mermaid** blocks in the SOTA chapter are turned into SVG by `theme/mermaid-init.js`, which loads **Mermaid 10** from jsDelivr (needs network once per session for diagrams).
