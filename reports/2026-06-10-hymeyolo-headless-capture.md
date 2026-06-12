# HyMeYOLO Headless Slide-Capture Mode (Demo 4)

**Date:** 2026-06-10
**Plan:** `docs/plans/2026-06-10-hymeyolo-headless-capture/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan`

## Summary

Added a non-interactive (headless) capture path to the HyMeYOLO
detection demo so the seminar deck gets static GT/prediction panels and
one honest, reproducible `mAP_50 + latency + RSS` line — neither of
which the live tkinter GUI produces. The data choice (user, 2026-06-10)
is synthetic Cluttered MNIST (strong, turnkey); real-photo VOC was
explicitly rejected as not detector-presentable (0.109 mAP).

The backend-agnostic inference/render functions were extracted from the
GUI file into a new `demo_hymeyolo_core` module (CLAUDE.md §6.5 #4,
decompose-by-concern). This keeps the capture path tkinter-free
(runs without a display) and removes a duplicated rendering scaffold.
**No new mAP/IoU code** — metrics come from the existing corrected
COCO matcher `train_circles_ricci.compute_detection_metrics`.

### Plan deviation (declared)
The plan proposed bolting `--headless` onto `demo_hymeyolo_tk.py`.
Implementation instead split the module: the GUI class forces tkinter
at import, so a `--headless` flag there would still require a display.
The core extraction is the cleaner structure and satisfies the same
goal. The GUI module re-exports `load_or_train` / `predict` /
`render_axes`, so the two existing callers
(`test_demo_checkpoint_loader`, `test_train_voc_stagec`) are unchanged.

## Files touched

| File | Change | Lines |
|---|---|---|
| `signedkan_wip/src/vision/demo_hymeyolo_core.py` | **new** — moved inference/render core + headless capture | +544 |
| `signedkan_wip/src/vision/demo_hymeyolo_tk.py` | edit — import core, drop moved fns + unused imports | 432→213 (-240/+17) |
| `signedkan_wip/tests/test_demo_headless.py` | **new** — unit/integ/perf/regression tests | +159 |
| `signedkan_wip/src/vision/DEMO_README.md` | edit — document headless mode | +27 |
| `docs/plans/2026-06-10-hymeyolo-headless-capture/{plan.tex,pdf,tikz,mmd}` | **new** — plan artifacts | — |

Net algorithm code is roughly flat: the 544-line core is ~310 lines
moved verbatim from the GUI + ~230 new (headless capture + helpers).

## CORE.YAML items touched

**None.** `CORE.YAML` freezes only the Rust core (parser, query LALR
layer, HSMM ISA); nothing under `signedkan_wip/` is listed.

## New / removed dependencies

**None.** Peak RSS is read via stdlib `ctypes`
(`GetProcessMemoryInfo.PeakWorkingSetSize`) on Windows, with
`resource.getrusage` fallback on POSIX. `psutil` was deliberately
**not** added (a dependency would be a Section 1 core change).

## Test results

Runner: `pytest -p no:randomly`. Seed fixed at 0.

| Suite | Result | Duration |
|---|---|---|
| `test_demo_headless.py` (new) | 9 passed | ~40 s |
| `test_demo_checkpoint_loader.py` (regression) | 3 passed | — |
| `test_train_voc_stagec.py` (regression) | 3 passed, 1 skipped | — |
| **Combined** | **15 passed, 1 skipped** | 24.6 s |

Layers covered:
- **Unit:** `_peak_rss_mb` > 0; `_measure_fwd_ms` median/IQR/worst; the
  `<5-rep` failure case raises; `render_headless` checkpoint-required and
  `n_panels >= 1` guards raise.
- **Integration:** synthetic checkpoint writes 2 non-empty panels;
  committed checkpoint lands `mAP_50 ≥ 0.80` on the demo split.
- **Performance:** forward median < 250 ms; peak RSS < 16 GB (asserted).
- **Regression:** the Tk module re-exports the three moved functions
  (identity check) — guards the decompose.

Coverage (core module, from the headless suite): **73 %**. Uncovered:
the legacy `load_or_train` quick-train branch (moved verbatim, covered
by the existing GUI tests) and `main()` (the CLI shell — exercised by
the end-to-end run below, not via pytest).

## Performance results (end-to-end CLI, eval=200)

```
mAP_50=0.904  mAP_50_95=0.793  mean_iou=0.792
fwd_ms median=49.6  iqr=1.1  worst=50.3   (7 reps, 2 warmup)
peak_rss=1040 MB    wall=12.7 s           (6 panels + 200-image eval)
```

| Quantity | Measured | Budget (plan) | Status |
|---|---|---|---|
| Peak RSS | 1040 MB | < 1.5 GB | ✅ (66× under 16 GB cap) |
| Forward latency | 49.6 ms median | < 250 ms | ✅ |
| Wall (6 panels, eval=200) | 12.7 s | < 60 s | ✅ |
| mAP_50 (demo split) | 0.904 | ≥ 0.80 | ✅ |

`mAP_50 = 0.904` reproduces the published corrected metric
(0.903 ± 0.009) — the consumed-GT COCO matcher, **not** the
pre-2026-05-16 bug-inflated 0.723. No prior baseline existed for the
capture path itself (new code); no >10 % regression to investigate.

## Static analysis

- `ruff check` — clean (one carried-over `E702` semicolon fixed).
- `mypy --explicit-package-bases` on the core module — no issues. (A
  bare `mypy` hits a pre-existing repo dual-path module-resolution
  collision unrelated to this change.)
- `radon cc -nc` — no block ranked C or worse. Longest new function
  `render_headless` is a flat orchestrator (load → eval → panels →
  latency → RSS), helpers extracted (`_save_panel`, `_measure_fwd_ms`,
  `_peak_rss_mb`).

## §6.5 anti-patterns

None introduced. Positively addressed: #4 (decomposed the 432-line
GUI file by concern), #1/#9 (reused the existing COCO matcher rather
than re-introducing a metric path), #13 (one capture entry with a mode
arg, no `_v2` files), #11 (no globals — RSS/latency state passed
explicitly, no env-var feature flags).

## Error-handling waivers

None. `render_headless` raises `FileNotFoundError` / `ValueError` on
bad preconditions and `RuntimeError` on RSS-cap breach. Two narrow,
commented `# noqa: BLE001` on best-effort paths (warm-start skip; RSS
query fallback) — both diagnostic, never silent failures of the main
path.

## Experiment provenance

- Git SHA: `af803ee` (working tree dirty — this change set + the
  pre-existing staged seminar artifacts listed in `git status`).
- Env: Python 3.12.13, torch 2.12.0+cu132 (CPU inference), matplotlib
  3.10.9, numpy 2.4.6. Windows 11 Pro 26200.
- Seed: 0 (stimulus + torch).
- Checkpoint: `checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt`
  (label `+ricci-mod`, epochs 100).
- Artifacts: `demo_out/yolo/panel_{00..05}_seed0.png` (gitignored —
  reproducible from the CLI above).

## Open issues / follow-up

- The unified `python -m signedkan_wip.src.demo <mode>` dispatcher
  (Demo 1–6 in `docs/INFERENCE_DEMOS_OUTLINE.md`) is still unbuilt; this
  delivers Demo 4 standalone. Wiring `yolo` into that dispatcher is a
  later step.
- `main()` is covered by the end-to-end smoke, not a pytest case (CLI
  argparse shell). A `subprocess`-based CLI test could close the gap if
  desired.
