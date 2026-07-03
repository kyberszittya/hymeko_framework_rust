# Fix: GPU `torch.compile` CUDA-graph crash in off-policy training (`ddpg.py`)

**Date:** 2026-07-03 14:00 (+09:00 JST) · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**Git SHA:** `09e8894` (working tree dirty — see *Files touched*) · **torch** 2.12.0+cu132, CUDA 13.2, RTX 3070 Laptop

## Summary

The quadruped-standing run (`reports/2026-07-02-quadruped-standing-scenario.md`) fell back to CPU because the
GPU path crashed with

> `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`
> … at `ddpg.py:_critic_loss` during `a_loss.backward()`.

This restores the GPU `torch.compile(mode="reduce-overhead")` path by inserting
`torch.compiler.cudagraph_mark_step_begin()` at the start of each gradient update. GPU training now runs clean at
**~185 steps/s vs ~41 steps/s on CPU (~4.5×)** on the quad-standing config, widening further past the compile
warmup at longer budgets.

## Root cause (reproduced, not inferred)

`train_offpolicy` compiles two closures — `_critic_loss` and `_actor_loss` — each as a `reduce-overhead` CUDA
graph. In `reduce-overhead` these two graphs **share one static CUDA-graph memory pool**. Within a single
`_update_once` the code runs the critic graph (forward + backward + detach), then the actor graph
(`_actor_loss` re-invokes `critics[0]` on `actor(s)`). Because the two graphs alias the same pool and no
step-boundary marker separated them, the actor graph's replay overwrote buffers that the just-run critic graph's
autograd tape still referenced → the overwrite error fired at `a_loss.backward()`.

**Reproduced** with a minimal script (`scratchpad/repro_quad_gpu.py`): quad-standing setup, `device="cuda"`,
`compile=True`, pure-TD3 (`bc_coef=0`), which crashes at the first actor update (update 2, `critic_warmup=0`,
`policy_delay=2`). torch's own error text names the two sanctioned fixes: clone the outputs, or call
`cudagraph_mark_step_begin()` before each invocation.

**Correction to the prior note.** The 2026-07-02 report attributed the crash to "the quad differs in obs shape
and the pure-TD3 no-offline path." The obs shape is **not** load-bearing: the new regression test reproduces the
identical crash on **cartpole** under the same pure-TD3 + CUDA + `compile` combo. The real trigger is
*two interleaved reduce-overhead graphs with the actor re-calling the critic* — env-agnostic. Galambos/cartpole
appeared to "work" only because those runs used CPU or the BC/warm-start path, not this exact combination.

## The fix

`hymeko_rl/ddpg.py` — one call, guarded by a `cudagraph_step` flag (true only when `compile and dev==cuda`):

```python
def _update_once() -> None:
    nonlocal updates, last_c, last_a
    if cudagraph_step:
        torch.compiler.cudagraph_mark_step_begin()   # root a fresh CUDA-graph step per update
    ...
```

`mark_step_begin()` roots a new cudagraph-tree step so each update's graphs get non-aliasing memory. It is safe
here because every per-step tensor is fully consumed inside the step (backward + `.detach()→float()` host copy);
nothing is held across the boundary. On CPU (`cudagraph_step=False`) the call is never made — the CPU path is
byte-for-byte unchanged (confirmed: `test_ddpg.py` + the CPU tests in `test_offpolicy_framework.py` pass).

## Tests

- **New regression test** `test_cuda_compile_interleaved_graphs_no_overwrite` (CUDA-gated,
  `pytest.skip` without a GPU): builds pure-TD3 cartpole with `compile=True`, runs past the first actor update,
  asserts no raise + finite params.
  - **Verified it fails against the prior implementation**: temporarily reverted the fix → the test raised the
    exact `accessing tensor output of CUDAGraphs … overwritten` error (`1 failed`). Restored → passes.
- **`test_offpolicy_framework.py`: 17 passed** (incl. the new test, on GPU) in 45 s.
- **`test_ddpg.py`: 7 passed** (CPU) in 16 s.
- **Production-scale smoke (§3):** the real quad-standing config (`base=free, task=stand`, vec-8, `compile=True`)
  ran 4000 steps on GPU with no crash and healthy losses (crit 0.1–0.7, act finite, no NaN/inf).

## Performance (diagnostic wall-clock, same quad config, 4000 steps)

| device | steady steps/s | note |
|---|---|---|
| CUDA (RTX 3070 Laptop, compile) | ~140 → **185** | after cudagraph capture warmup |
| CPU | **~41** | |

~4.5× at this short budget; the CPU baseline for the 60k standing run was ~36 effective steps/s (1675 s wall).
This is a throughput *diagnostic*, not a `pytest-benchmark` measurement — the deliverable is the correctness fix.
Peak RSS well under the 16 GB cap (tiny nets; GPU VRAM negligible on a 3070).

## Files touched (all non-core, uncommitted)

- `hymeko_rl/ddpg.py` — `cudagraph_step` flag + one `cudagraph_mark_step_begin()` call (+8 / −1, plus the
  flag rename of the existing `if cfg.compile and dev.type == "cuda":` guard).
- `hymeko_rl/tests/test_offpolicy_framework.py` — new CUDA-gated regression test (+20).

**CORE.YAML items touched: none** (`hymeko_rl` is not in `CORE.YAML`).

## Static analysis / gates

- `ruff check` on both files: **clean**.
- `mypy --strict hymeko_rl/ddpg.py`: the change adds **one** `no-untyped-call` on `cudagraph_mark_step_begin`
  — the *same* torch-stub-gap class as the file's pre-existing unsuppressed `.backward()` calls (lines 403/412/423).
  Left unsuppressed for consistency with that convention; **no new `# type: ignore` introduced**. All other mypy
  errors on the file (mujoco stubs, `no-any-return`, the two `type: ignore[assignment]` on the compile lines) are
  pre-existing and untouched.
- No §6.5 anti-patterns introduced: the fix is a guarded one-liner on the existing shared update path (not a new
  code path, not a Cartesian variant, no global state — the `cudagraph_step` flag is a local closure capture).
- No plan dir: this is a single-file localised bug-fix with its regression test (§2 single-file-local-fix
  carve-out), not a feature change.

## Standing rerun on the restored GPU path (150k × 3 seeds)

With the GPU path fixed, the standing campaign was rerun at **150k vec-8 steps × 3 seeds** (`device="auto"`,
`compile=True`) — artifacts in `experiments/2026_07_03_14_57_quadruped_standing/`. Wall **~16 min/seed**
(932–984 s), confirming the ~5× over the 60k CPU run's ~28 min/seed. GPU throughput held **157–189 steps/s**.

**Result — a negative one, and informative.** More steps did **not** improve standing:

| budget | stand-rate (seeds) | median | note |
|---|---|---|---|
| 60k (CPU, prior) | 0.24 / 0.0 / 0.0 | 0.000 | seed-0 foothold |
| **150k (GPU, this run)** | **0.0 / 0.02 / 0.0** | **0.000** | seed-0's 0.24 did **not** reproduce |

- **[measured]** The actor loss climbs **monotonically** across the whole run — `-7 → -14 → -22 → -30 → -44`
  (i.e. `critics[0](s, μ(s))` inflating to ~44) — with late critic-loss spikes (37.4, 16.0). Survival stays high
  (216–245/250) but does not discriminate (the free base rarely fully inverts).
- **[inferred]** This is the **Q-overestimation** signature: the actor chases an inflating critic value instead
  of converging to a standing controller. The LayerNorm twin critics bound it only partially. The earlier 60k
  seed-0 0.24 was a **pre-divergence** point, not a foothold — extending the budget let the divergence develop.
- **[hypothesis, not yet tested]** A **PD-hold-q0 BC warm-start → TD3+BC anchor** (the galambos/FANUC
  anti-collapse recipe, memories `project-fanuc-offpolicy-collapse` / `project-ditch-ppo-offpolicy-sahsikan`)
  would bound the actor to a standing demo distribution and is the motivated next lever — the one this rerun's
  divergence now argues for over "more steps."

This is a clean separation of concerns: **the GPU fix (this report's deliverable) is confirmed working**; the
standing *task* is not solved by pure-TD3 at any budget tried, and the divergence diagnosis redirects the next
step to the BC-anchored recipe.

## Open / next

- **BC warm-start → TD3+BC on `quadruped_stand`** is now the motivated lever (measured divergence, not a guess).
  Needs a PD-hold-q0 demo source for the standing pose. *Not launched* — a separate compute + design decision (§11).
- `quad_stand_campaign_gpu.py` (scratchpad) already runs `device="auto"`; the original scratchpad
  `quad_stand_campaign.py` still carries the stale `device="cpu"` "GPU crashes" comment — supersede it with the
  GPU variant rather than editing the stale copy.
- Untouched from the prior report: the GPU 5–6× is what makes the BC-anchored rerun cheap, so fixing this first
  paid off even though the pure-TD3 rerun was negative.
