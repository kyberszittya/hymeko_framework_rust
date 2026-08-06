# Report — PyTorch 2.4.1 → 2.12.0+cu132 (CUDA 13.2) migration

**Date:** 2026-05-27
**Plan:** `docs/plans/2026-05-27-torch-cuda13-upgrade/`
**CORE approval:** `APPROVED-CORE-EDIT: torch-cuda13` (user chat, 2026-05-27)

## Summary

Upgraded the uv `ml` group + CORE pins for native CUDA-13.2 GPU support. GPU is
live; the Python suite shows **no torch/numpy regressions**. The definitive
RTL-golden numerical-parity re-validation (the property the pin protects) is
**not** runnable on this Windows host and remains a follow-up.

## GPU smoke — PASS

```
torch 2.12.0+cu132 · numpy 2.4.6 · torch.version.cuda 13.2 · cuda available True
device: NVIDIA GeForce RTX 3070 Laptop GPU
```

## Files touched (CORE — under the approved token)

- `CORE.YAML` `dependencies.pinned.python`: `torch "==2.4.1" → "==2.12.0"`;
  `numpy ">=1.26,<2.0" → ">=2,<3"` (required cascade — torch 2.12 wheels target
  NumPy 2). Token cited in-file.
- `pyproject.toml`: `ml` group → `torch==2.12.0`, `torchvision==0.27.0`,
  `numpy>=2,<3`; added `[[tool.uv.index]] pytorch-cu132`
  (`download.pytorch.org/whl/cu132`, `explicit=true`) + `[tool.uv.sources]`
  routing `torch`/`torchvision` to it.
- `uv.lock` regenerated (`uv sync --group ml --python 3.12`).
- `hymeko` PyO3 binding built into `.venv` (`uv pip install -e hymeko_py`) so the
  binding-dependent tests could run.

## Parity / regression gate

`PYTHONPATH=. uv run pytest -p no:randomly --continue-on-collection-errors`
(Python 3.12, torch 2.12.0+cu132, numpy 2.4.6):

- **971 tests collected; ≈947 passed; 24 failed; 11 modules un-collectable.**
- **Every failure/error is environmental — none is a torch/numpy regression:**
  | cause | count | nature |
  |:--|--:|:--|
  | `triton` not installed | ~12+ | optional GPU-kernel lib; limited on Windows |
  | `cl` (MSVC) not found | 2 | `torch.compile` JIT needs MSVC on Windows |
  | `matplotlib` / `yaml` / `optuna` | ~8 | optional deps absent from `ml`/`dev` |
  | `fcntl` | 1 | Unix-only stdlib; Windows-incompatible test |
  | subprocess runners (gomb/hsikan/konect) | ~5 | their child procs hit the above / dataset downloads |
- **No `cannot import name … from torch`, no numpy `AttributeError`, no
  numerical `assert allclose` divergence** in any test that actually ran. The
  `test_cycle_cache`, `test_cpml` (torch.compile) failures trace to `triton`/`cl`,
  not numerics. Verified the ABB-compare failure is `import hymeko` inside a
  subprocess (pre-existing binding gap), not an ABB regression.

Conclusion: the 2.4→2.12 + NumPy-2 jump is **import- and behaviour-clean** across
the runnable Python suite. The remaining failures would fail identically on torch
2.4.1 in this environment (they are missing-dependency / Windows-toolchain gaps).

## Gaps — NOT fully verified (honest)

1. **RTL golden parity (the pin's actual purpose) not run.** `rtl/golden/**` is a
   SystemVerilog/Rust golden-vector harness, not the Python suite; it was not
   executed here. So the literal "numerical parity with RTL fixtures" property is
   **not** re-validated — it must run on the RTL harness. If it diverges,
   regenerating the locked golden fixtures is a **separate** `APPROVED-CORE-EDIT`.
2. **Triton-kernel numerical paths unverified** on this host (triton absent /
   Windows-limited) — the `signedkan_inner_triton` and `torch.compile` paths.
3. **§4 RSS cap** not enforceable on Windows → heavier training runs (HSiKAN AUC)
   should run on Linux/WSL under `systemd-run --user -p MemoryMax=16G`.
4. Optional test deps (`yaml`, `optuna`, `matplotlib`) could be added to a test
   group to close the ~8 collection gaps; `triton` only meaningfully on Linux.

## Rollback

Revert `CORE.YAML`, `pyproject.toml`, `uv.lock` (+ the index block); `uv sync
--group ml` restores torch 2.4.1 (CPU). Nothing outside the Python env changed.

## Provenance

- **Git SHA:** `9abfc3435f55f7443cb07bde4583a17126ac3fc1` (branch
  `feature/pgraph_engine`); working tree uncommitted (this migration + the whole
  P-graph/regime session).
- **Host:** Windows 11, NVIDIA GeForce RTX 3070 Laptop GPU, CUDA 13.2 driver.
- **Env:** Python 3.12.13; torch 2.12.0+cu132; torchvision 0.27.0+cu132;
  numpy 2.4.6; uv 0.11.7.
- **Approval:** `APPROVED-CORE-EDIT: torch-cuda13` — to be quoted in the commit
  footer when committed.
