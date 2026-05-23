# GÖMB key art — image prompt with repo-backed numbers

Use the **block quote** below as the positive prompt (plus your usual negative prompt).  
Numbers are **fixed from in-repo tables or pinned artifacts** so the art reads as “instrumented lab poster,” not invented science.

**Sources**

| Claim | Value | Where |
|--------|--------|--------|
| HSiKAN test AUC (published mean ± std) | **Bitcoin Alpha 0.939 ± 0.011**, **OTC 0.930 ± 0.008** | `signedkan_wip/src/benchmarks/sota_reference.json` |
| SiGAT competitor bar (mean) | Alpha **0.899**, OTC **0.934** | same file |
| Gömb OTC 5-seed (val_auc_best) | **mean 0.9118 ± 0.0089** (n=5) | `reports/2026-05-11-hymeko-gomb-sphere.md` |
| Gömb Slashdot 5-seed (test) | **0.9031 ± 0.0008** vs edge_cr SOTA **0.9067 ± 0.0034** | `reports/2026-05-11-hymeko-gomb-slashdot-sota-attempt.md` |
| Gömb external AUC tuning (best test_auroc, one batch) | Alpha joint **0.9058** / vanilla **0.8910**; OTC joint **0.9214** / vanilla **0.9228** | `reports/2026-05-12-gomb-external-auc-tuning-results.md` |
| Full Gömb OTC (ablation table, seed 0) | val_auc_best **0.9246**, **n_params 266 321** | `reports/2026-05-11-hymeko-gomb-sphere.md` |
| HSiKAN Optuna chase OTC (`run_final_cell` JSON **`auc`**, test split, **seed 0 only**) | trial **1**: **0.9945** (**99.45%**) — chase vs SiGAT bar **0.934**; **not** the 5-seed headline row | Study `signedkan_wip/experiments/results/optuna_otc_sigat_chase_2026_05_12.db`; default **transductive** tuple protocol (`HSIKAN_STRICT_PROTOCOL` off). Re-verify from study / chase stdout before print. |

---

## Positive prompt (paste)

Ultra-aggressive neo-Tokyo cyberpunk key art, true black background. Orthogonal XYZ construction axes through the frame—laser-cyan, hot magenta, acid lime—isometric grid and vanishing-point perspective guides left visible like a HUD blueprint.

Center: hardened glass / armoured sphere, semi-transparent shell; nested wireframe polyhedra inside; signed edge energy as opposing red vs cyan vector ticks; faint triangular cycle motifs (k=3) locked in the lattice.

Above the sphere, an entropy / telemetry layer: scanline moiré, datamoshing speckle, heat-map turbulence, checksum glyphs—reads as measurement and information bleed, not decorative particles.

Typography: **FUTURA** (Futura Bold / Futura PT) for Latin **GÖMB**—razor-sharp, slightly glitched, chromatic aberration, micro-outline, industrial spec-plate. Beside it, vertical neon katakana **ゴーム** with the same treatment.

**Diegetic metric chrome (short fragments, mono / industrial, no long prose):**

- `HSiKAN | α test AUC μ=0.939 σ=0.011` · `OTC μ=0.930 σ=0.008`
- `SiGAT bar | α 0.899 · OTC 0.934`
- `HSiKAN Optuna chase OTC | trial1 auc=0.9945 (99.45%) | seed0 | vs bar 0.934 | single-seed | default tuple proto`
- `Gömb OTC 5-seed | val μ=0.9118 σ=0.0089`
- `Gömb OTC full seed0 | val 0.9246 | n_params 266321`
- `Gömb Slashdot 5-seed | test μ=0.9031 σ=0.0008` · `vs edge_cr 0.9067±0.0034`
- `Gömb ext tune (batch) | α test 0.9058 joint / 0.8910 van` · `OTC 0.9214 j / 0.9228 v`
- `Rust cycles | ABB | P-graph MSG→SSG driver`
- `attn_entropy −λH` · `spectral node_embed reg`

Japanese texture: vertical neon kanji/katakana fragments (abstract signage, not a readable sentence), brutalist warning stripes, corrupted barcode blocks, ink bleed.

Lighting: hard speculars, bloody rim light, aggressive contrast, subtle film grain. Mood: hostile, fast, engineered.

No people, no faces, no cute mascots, no generic AI brain blob. 16:9, 8k, poster composition.

---

## Negative prompt (optional)

soft pastel, cottagecore, watercolor, bokeh portrait, readable long Japanese paragraph, serif headline, Times New Roman, 3D cartoon, cute robot, stock photo, low contrast, daylight, white background, random Greek letters, omega symbol Ω ω

---

## Note

If headline numbers move after new runs, **edit this file** and keep the table in sync—image generators will not verify AUCs.

The **0.9945 / 99.45%** line is **Optuna chase décor**: one completed trial’s `run_final_cell` test `auc` at seed 0, **not** interchangeable with the **5-seed mean** in `sota_reference.json` without multi-seed gate + protocol footnotes. Refresh or drop that HUD line if the study is rerun and the value changes.
