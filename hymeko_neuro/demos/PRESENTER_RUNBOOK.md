# HyMeKo Seminar — Presenter Runbook

The on-stage operational layer for the demo program. The *build specs* live in
[SEMINAR_DEMO_OUTLINE.md](../../SEMINAR_DEMO_OUTLINE.md) and its three
companions; **this file is what you actually do at the podium**: the exact
command, what to click, the number to point at, and the honest caveat to say.

Status legend: **READY** (built + tested) · **PENDING** (spec'd, not yet built).

---

## 0. Pre-flight (do this once, before the talk — not on stage)

The HSiKAN/engine demos need the `ml`/`dev`/`demo` packages **and** the built
`hymeko` PyO3 module.

```powershell
# from repo root
uv sync --group ml --group dev --group demo            # torch, sklearn, matplotlib, yaml
.venv/Scripts/maturin.exe develop --manifest-path hymeko_py/Cargo.toml   # builds `hymeko`
```

**Operational gotchas — read these:**
- **Run demos with the venv interpreter, not `uv run --group …`.** `uv run
  --group` re-syncs the environment and *drops the editable `hymeko` install*.
  Use `PYTHONPATH=. .venv/Scripts/python.exe -m …`.
- **Cold start is ~40 s** (torch + triton + matplotlib import) on Windows. The
  compute itself is ~1.6 s. **Pre-warm every demo once** before the audience is
  watching, or run with `--quick`, or keep the precomputed figures open.
- Everything is CPU-runnable; `--device auto` picks CUDA if present. Determinism
  is fixed at `--seed 0`. Each demo prints **peak RSS + wall time** on exit and
  asserts < 16 GB.
- Outputs land in `demo_out/<demo>/`; every printed number names its checkpoint/
  fixture.

---

## 1. Run order (≈ 50–60 min) — mirrors the umbrella outline

| Slot | Demo | Slide | Status |
|---|---|---|---|
| Open | **Demo 1 — balance** (frustrated cycles) | 12 | READY |
| Part I | **Demo 2 — HIVE compile** + **star-expansion viewer** + **live editor** | 6–7 | viewer READY · Demo 2 READY · editor READY (committed bundle) |
| Part II | **Demo 3 — HSiKAN forward** + **latency bench** | 17–18 | Demo 3 READY · bench READY |
| Transfer | **Demo 4 — mesh / chiral** (+ MuJoCo video already in deck) | 18–19 | PENDING |
| Close | **Demo 5 — sim→perception** | 23 | PENDING |

---

## 2. Per-demo script

### Demo 3 — Signed-graph link prediction  ·  slide 17  ·  **READY**
```powershell
PYTHONPATH=. .venv/Scripts/python.exe -m hymeko_neuro.demos.seminar link `
    --dataset bitcoin_otc --device auto --seed 0
```
- **Do:** run it; let the results table print; open
  [demo_out/link/ROC_bitcoin_otc.png](../../demo_out/link/) and `alpha_k_*.png`.
- **Point at:** `AUC 0.9957` and the `PASS` line (matches the checkpoint's own
  recorded AUC to ±0.000); the αₖ regime bars.
- **Say (caveat):** "These are the `optuna_best` weights — a *transductive*
  convention; the strict protocol is the architectural baseline. The win we
  claim is Epinions + accuracy-per-parameter, not flat SOTA."
- Source: `checkpoints/hsikan/bitcoin_otc_optuna_best.pt`.

### Star-expansion 3D viewer  ·  slide 7  ·  **READY (browser render not auto-verified)**
- **Do:** double-click [demo_web/star_expansion_viewer.html](../../demo_web/star_expansion_viewer.html)
  (loads `star_expansion_data.js`, no server). Use the **Star ↔ Clique** toggle;
  the **HyMeKo source** preset shows the engine-fed graph; "Sandbox" presets are
  synthetic.
- **Point at:** the live `star incidences` vs `clique edges` counts and the
  `clique is N× the star` ratio; the `source` + `canonical_hash` provenance line
  with the green ✓ (JS-derived counts agree with the engine).
- **To show the big gap** (1,498 vs 10,991 talking point), regenerate for the
  robot first (pre-flight):
  ```powershell
  PYTHONPATH=. .venv/Scripts/python.exe demo_web/export_star_expansion.py `
      --src data/robotics/robot_4wh.hymeko --out demo_web/star_expansion_data.json
  ```
- **Say (caveat):** "The 3D layout is **force-directed for legibility, not
  geometric ground truth**. The edge-count arithmetic is exact and engine-sourced."

### HyMeKo Editor (WASM, in-browser)  ·  slides 6–7  ·  **READY (committed bundle; browser render not auto-verified)**
```bash
cd docs/editor && python3 -m http.server 8000   # then open http://localhost:8000/
```
- **Do:** click **Example** (loads a 2-link continuous-joint robot) → click a
  node → set its mass → **Download** as URDF. Every UI edit is a string mutation
  on the `.hymeko` source pane (single source of truth); the Cytoscape canvas is
  a live read via `parse_and_compile` → `snapshot_json`.
- **Point at:** the round-trip — the canvas, the source text, and the URDF/SDF/
  DOT export are all the same IR; the live predicate **Query** box
  (`INHERITS(link)`, `KIND(...)`).
- **View tabs (2026-06-12):** the canvas pane has three tabs, all live off the
  same `snapshot_json()`/`to_urdf()`: **Graph** (Cytoscape; `<isa>` inheritance
  shown as dashed lines), **Hypergraph 3D** (three.js star/clique, Star↔Clique
  toggle), and **Kinematic** (a true robot render from the emitted URDF + the αₖ
  regime compass and signed-topology ring). Switch tabs while editing — each
  re-renders on the 400 ms debounce. See
  `reports/2026-06-12-editor-stereotype-views.md`.
- **Say (caveat):** "This is the **same WASM engine** as the rest of the
  toolchain — source-text-as-truth, so what you edit is exactly what `hymeko
  emit` consumes. MVP limits: edits reflow whitespace (structural round-trip,
  not byte-identical), joints are made via a modal not drag-to-connect, no
  structural undo yet."
- **Pre-flight gotcha:** the committed bundle `docs/editor/pkg/` is from
  2026-05-07 and runs the editor's stable surface fine. A clean rebuild
  (`wasm-pack build --target web --release --out-dir ../docs/editor/pkg` from
  `hymeko_wasm/`) needs `wasm-pack` + the `wasm32-unknown-unknown` target —
  **not installed**; install is a toolchain decision, not a stage action.

### Demo 2 — HIVE compilation (canonicalisation)  ·  slides 6–7  ·  **READY**
```powershell
PYTHONPATH=. .venv/Scripts/python.exe -m hymeko_neuro.demos.seminar hive `
    --src data/typical_graphs/fano_graph.hymeko --device cpu
```
- **Do:** run it; point at `canonicalisation: PASS` and the star/clique COO nnz.
  Surfaces `.hymeko` → IR → COO/CSR side by side; the canonical hash is **equal**
  for two declaration-order permutations and **different** after a one-edge change.
- **Say (caveat, corrected):** the hash is invariant to *how you write* the
  model (node/edge declaration order), **not** to relabeling or within-edge
  member order — node identity is semantic in HyMeKo. (Full isomorphism
  invariance is a deferred plan: `docs/plans/2026-06-10-canonical-hash-iso-invariance/`.)

### Latency bench  ·  slide 18  ·  **READY (CLI mode 2026-06-12)**
```powershell
PYTHONPATH=. .venv/Scripts/python.exe -m hymeko_neuro.demos.seminar latency `
    --device cpu --seed 0
```
- **Do:** run it; it reads the committed `inference_bench.json` and writes
  [demo_out/latency/latency_{cpu,cuda}.png](../../demo_out/latency/) (~1.5 s, no
  torch import). **Point at:** `mean joint/lean (cpu) 3.58x`, `(cuda) 1.14x`.
- SGCN vs HSiKAN-lean (h=4) vs HSiKAN-joint (h=16) bars; ≥5 repeats (20 timed +
  5 warmup), median/IQR/worst. Artifacts: `inference_bench.json` +
  `inference_bench_{cpu,cuda}.png`. Re-measure (heavy, pre-flight only): `python
  -m hymeko_neuro.experiments.runs.run_inference_bench` then `… .bench_to_png`.
- **Say (caveat):** "HSiKAN's *absolute* forward is **heavier** than SGCN. The
  measured within-family width gap (h4 vs h16, **same device**) is **~3.5× on
  CPU, ≈1× on CUDA** — shown side by side so it's honest, not 'faster than
  SGCN'. (The old 11× was the optuna_best_otc-vs-joint number: OTC-specific and
  tuple-set-driven, not a general width claim.)"
- **Accuracy-vs-cost Pareto** (`inference_bench_pareto_<dataset>.png`, table
  `inference_bench_table_<dataset>.md`): SGCN/SiGAT/SGT/MLP/GCN-blind + HSiKAN.
  **Say (caveat):** "optuna-HSiKAN reaches the top-left — best AUC at ~⅛ the
  params — **but it is tuned and the baselines are not**. The untuned HSiKAN in
  the same panel sits below them; both points are shown so the tuning gap is
  visible, not hidden." Cost axis = params; latency is the bars.

### Demo 1 — Affective balance  ·  slide 12  ·  **READY**
```powershell
PYTHONPATH=. .venv/Scripts/python.exe -m hymeko_neuro.demos.seminar balance `
    --graph planted --device cpu --seed 0
```
- **Do:** run it (default `--graph planted`); open
  [demo_out/balance/frustration_planted.png](../../demo_out/balance/) — the
  frustrated triad {0,1,2} is bold-black, amber vertices.
- **Point at:** `balance 0.5`, `frustration 0.5`, `planted_cycle_surfaced True`
  — the planted unbalanced triad is found by sign-product, not by label.
- **Contrast (optional):** `--graph camps` → `balance 1.0` (Harary-balanced two
  camps); `--graph karate` → `balance 1.0` over 45 triangles (a real faction
  split is structurally balanced by construction — a nice aside).
- **Say (caveat):** "Balance/frustration here is the **cycle sign-product
  statistic**, not a learned quantity — it sets up Demo 3 where the same cycles
  drive prediction. A 2-negative-edge cycle is still *balanced*; we classify on
  the product, never on how many edges are negative."
- Source: `planted`/`camps` hand-built fixtures · `karate_faction_signed()`;
  cycles via `hymeko.enumerate_top_k_cycles_rs`.

### Demo 4 — Mesh recognition + chiral ablation  ·  slide 18  ·  **PENDING**
- Planned: WL+sign → HSiKAN → Sinkhorn correspondence, signed vs unsigned.
- **Say (caveat):** "The ~1.0 vs ~0.5 gap is **measured live**, not assumed — if
  it doesn't reproduce we quote the real numbers."

### Demo 5 — Sim→perception bridge  ·  slide 23  ·  **PENDING**
- Planned: a scene as `.hymeko` IR feeding the perception stage.
- **Say (caveat):** "This is **direction of travel**, not a finished capability."

### HyMeYOLO (deck video)  ·  slide 19
- The MuJoCo/detector clip plays from the deck. **Quote the corrected mAP
  0.903 ± 0.009, never the bug-inflated 0.723.**

---

## 3. If a demo fails on stage
- Cold-import hang → it's not frozen, it's importing torch (~40 s). Wait, or use
  the pre-warmed terminal.
- `ModuleNotFoundError: hymeko` → you ran via `uv run --group`; switch to
  `.venv/Scripts/python.exe`.
- `ModuleNotFoundError: triton` → `triton-windows` not installed; `uv sync
  --group ml` (Windows wheel, win32-gated).
- Figures missing → re-run without `--no-figures`; check `demo_out/<demo>/`.
- Fall back to the committed figures/precomputed outputs in `demo_out/` and the
  deck — never retrain in the room.
