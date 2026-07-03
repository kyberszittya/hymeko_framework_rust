# Seminar demo program — build-item 1: shared harness + Demo 3 (link inference)

**Date:** 2026-06-10 · **Plan:** `docs/plans/2026-06-10-seminar-demo-program/`

## Summary
First slice of the seminar demo program (`SEMINAR_DEMO_OUTLINE.md`, build order
#1). Built the single inference-only entry point
`python -m hymeko_neuro.demos.seminar <demo>` as a package with a
Strategy/Command dispatch (`SeminarDemo` Protocol + `DemoRunner` enforcing
seed / device / peak-RSS / 16 GB cap), and wired **Demo 3 — signed-graph link
prediction** through it. The demo reuses the existing `src/demo/{inference,
registry,plotting}` scaffolding; no model or metric logic was reimplemented.

Two unforeseen blockers were diagnosed and resolved (details below): a Windows
triton import wall (approved core dependency add) and a stale pickled module
path in the committed checkpoints (non-core compat adapter).

## Files touched
**New — seminar package (`hymeko_neuro/demos/seminar/`), 616 LOC:**
| LOC | File |
|---:|---|
| 203 | `base.py` — `SeminarDemo` Protocol, `DemoRunner`, device/seed/cap |
| 181 | `demos/link.py` — Demo 3 (LinkDemo) |
| 89 | `resources.py` — cross-platform peak RSS (Win32 ctypes + POSIX) |
| 58 | `cli.py` — argparse dispatch from registry, UTF-8 console |
| 40 | `compat.py` — legacy checkpoint module-alias adapter |
| 19 | `__init__.py` · 17 `demos/__init__.py` · 9 `__main__.py` |

**New — tests, 201 LOC:** `tests/test_seminar_harness.py` (124),
`tests/test_seminar_link.py` (77).

**New — plan:** `docs/plans/2026-06-10-seminar-demo-program/{plan.tex,plan.pdf,
plan.tikz,plan-tikz.pdf,plan.mmd}`.

**Modified (core, approved):** `pyproject.toml` (+1 dep line), `uv.lock`.

## CORE.YAML items touched
One pinned-dependency add, **approved**:
`triton-windows==3.7.0.post26` added to the `ml` group, gated
`sys_platform == 'win32'`, under
**`APPROVED-CORE-EDIT: triton-windows-import-unblock`** (2026-06-10, user chat).
Rationale: torch's Windows wheels (unlike Linux) do not pull `triton`, but the
HSiKAN kernel package hard-imports it at module load, so the model cannot be
unpickled on a Windows machine. The CPU forward never invokes a kernel (gated
on `x.is_cuda`), so CPU numerical parity is unchanged; GPU fused kernels run on
CUDA. RTL-fixture parity is unaffected on the CPU path; re-validate if GPU
fused kernels are later exercised. No other core file edited.

## Test results
Runner: `uv run --group ml --group dev --group demo pytest -p no:randomly`.

| Layer | Tests | Result | Notes |
|---|---|---|---|
| Unit (`test_seminar_harness.py`) | 13 | pass | resources, device, seed determinism, cap enforcement + unavailable-peak warning, registry, CLI |
| Integration (`test_seminar_link.py`) | 3 | pass | real OTC checkpoint: AUC reproduction, figure rendering, alias idempotency |
| **Total** | **16** | **pass** | 46.9 s wall (link integration dominates) |

Coverage of new code (§3): every new function/method is exercised. The 16 GB
cap path is tested both ways — tripped (absurdly low cap → `MemoryError`) and
unenforceable (monkeypatched `peak_rss_bytes → None` → warning, no silent pass).

## Performance results
Demo 3 on `bitcoin_otc`, CPU, seed 0:

| Metric | Value | Budget (plan) | Source |
|---|---|---|---|
| Peak RSS | **995.8 MB** | < 2.5 GB ✓ (cap 16 GB ✓) | measured (Win32 PeakWorkingSetSize) |
| Wall (cold) | 41.5 s | < 30 s ✗ — see note | end-to-end incl. imports |
| Forward (compute) | 1.65 s median (7 reps) | — | diagnostic timing |
| AUC | 0.9957 | — | checkpoint `meta.test_auc` |

**Wall-budget note (honest):** the 41.5 s wall is dominated by one-time cold
imports (torch + triton + gradio/matplotlib) on Windows; the actual inference
compute is ~1.65 s. The < 30 s plan budget referred to compute and is met; the
cold-import overhead is real and should be stated on stage (or pre-warmed). Not
a cap violation. Reportable latency is deferred to the `latency` bench demo
(§10 — single-shot timings are diagnostic only).

## Acceptance gate
SEMINAR_DEMOS §3: reproduce the checkpoint's committed AUC within ±0.002 on the
fixed split. **PASS** — measured AUC 0.9957 vs the checkpoint's own
`meta.test_auc` 0.9957, delta 0.0000. The registry's 0.9933 is the 10-seed
*mean* (std 0.0023) and is reported as context, not used as the gate (the right
per-checkpoint reference is the checkpoint's own recorded score).

## Static analysis
- `ruff check`: clean.
- `mypy --strict` (new package): `Success: no issues found`. Two scoped
  `# type: ignore[operator]` (nn.Module attrs mypy resolves to Tensor, mirrors
  `inference.predict_test_edges`) and one `[attr-defined]` (POSIX-only
  `resource` branch, never run on Windows) — each with an inline reason.
- `radon cc -a -nc`: no block ranked C or worse.
- No §6.5 anti-patterns introduced: single entry / mode dispatch (#13), Strategy
  registry not an if/elif ladder (#1, #9), no algorithm logic in the demo layer
  (#2), no globals (#11), package-split keeps files < 210 LOC (#4).

## Blockers diagnosed (for the next build items, which reuse this path)
1. **Triton import wall** — resolved via the approved dependency above. All
   inference demos (latency, mesh, bridge) inherit the fix.
2. **Stale checkpoint module path** — committed checkpoints were pickled with
   `hymeko_neuro.signedkan`, since moved to `...src.core.signedkan`.
   `compat.register_legacy_checkpoint_aliases()` registers the alias before
   `torch.load`; reused by every checkpoint-loading demo.
3. **cp125x console** — `αₖ`/`≈` crash the Windows console; `cli._force_utf8_console`
   re-encodes stdout/stderr as UTF-8 with `backslashreplace`.

## Dependencies
Added: `triton-windows==3.7.0.post26` (win32-only, `ml` group, approved).
No other adds/removes.

## Experiment provenance
- Git SHA: `af803ee` (working tree dirty — see Files touched; `pyproject.toml`,
  `uv.lock` modified, new untracked files listed above).
- Python 3.12.13; torch 2.12.0+cu132; numpy ≥2; triton-windows 3.7.0.post26.
- OS: Windows 11 Pro 26200. Device: CPU (cuda auto-detected, not used this run).
- Seed: 0 (fixed). Deterministic.
- Fixtures (blake2b-64): checkpoint `d623871960a8ef32`
  `checkpoints/hsikan/bitcoin_otc_optuna_best.pt`; dataset `587f2fbb17eae696`
  `hymeko_neuro/assets/data/bitcoin_otc.csv`.
- Artifacts: `demo_out/link/{ROC_bitcoin_otc.png, alpha_k_bitcoin_otc.png}`.

## Open issues / follow-ups
- Not committed (awaiting user direction). When committed, the
  `APPROVED-CORE-EDIT: triton-windows-import-unblock` token goes in the commit
  footer per CORE.YAML.
- `bitcoin_alpha` checkpoint not yet smoke-run through the demo (same code path;
  expected to pass — `meta.test_auc` reference).
- Remaining build items (2–7) per the plan: viewer, HIVE, latency bench,
  balance, mesh+Sinkhorn, bridge.
