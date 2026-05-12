# Cold start — HyMeKo repo

Minimal path from **zero context** to **building, testing, and reading results** without burning time or re-arguing settled numbers.

---

## 0. Read first (order)

| File | Why |
|------|-----|
| `CORE.YAML` | Core lockdown list — know before you edit. |
| `tools.yaml` | Pinned tool majors (repo contract). |
| `CLAUDE.md` | Full workflow if you use Claude Code here. |
| **`docs/RESULTS_DISCIPLINE.md`** | **Evidence contract:** canonical AUC paths, joint vs phase‑8 vs strict — **read before any score claim.** |
| **`docs/SOTA_RESULTS.md`** | **SOTA snapshot:** Mermaid bar charts + tables (Bitcoin, Slashdot, Epinions, master table excerpt). |
| `signedkan_wip/experiments/results/AGGREGATE_index.md` | What lives under experiment results (150+ artifacts). |
| `reports/AGGREGATE_index.md` | Reports, overnight JSON/err, narrative `reports/*.md`. |

Deeper product overview: `README.md` (TOC, architecture, CI).

### Handbook (mdBook)

```bash
cd docs/book && mdbook build
```

Open `docs/book/book/index.html` — **Results & evidence** part mirrors `docs/SOTA_RESULTS.md`, `docs/RESULTS_DISCIPLINE.md`, and root `COLD_START.md` via includes.

---

## 1. Rust workspace

From repo root:

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings   # gate if you touch Rust
```

Crates are `hymeko_*`, `parser`, `hymeko_cli`, etc. — see root `Cargo.toml` / `README.md` “Project structure”.

---

## 2. Python (`signedkan_wip`, benches)

```bash
cd /path/to/hymeko_framework_rust
uv sync --group dev --group ml --all-packages   # or your venv equivalent
export PYTHONPATH="$PWD"
pytest -p no:randomly signedkan_wip/tests -q     # scope as needed
```

If global pytest plugins break collection: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

---

## 3. Where the numbers live (do not guess)

| Topic | Location |
|-------|----------|
| **Charts + curated SOTA tables** | **`docs/SOTA_RESULTS.md`** |
| Joint mix Bitcoin (e.g. **~0.98** OTC/Alpha, tuples `c3,c4,w2,w3`) | `signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl` |
| Phase‑8 Bitcoin panel (**lean** HSiKAN vs baselines) | `signedkan_wip/experiments/results/phase8_bitcoin_5seed.json` |
| Multi-arch summary table | `signedkan_wip/experiments/results/master_table.md` |
| Thesis IV entropy regulariser suite (111 rows) | `RESULTS_VIEWS_SUITE.md` (repo root) |
| Gömb / HSiKAN harness narratives | `reports/*.md` |

**Rule:** same name ≠ same experiment. Always tie a number to a **file + protocol** (see `docs/RESULTS_DISCIPLINE.md`).

---

## 4. Optional: Cursor

Repo rule for assistants: `.cursor/rules/results-discipline.mdc` (always applies here).

---

## 5. If something is wrong

Update **`docs/RESULTS_DISCIPLINE.md`** when a new run **supersedes** an anchored table, or add a pointer to the new artifact. Refresh **`docs/SOTA_RESULTS.md`** charts/tables when headline metrics change. Keep `AGGREGATE_index.md` files in sync when you add large new result trees.
