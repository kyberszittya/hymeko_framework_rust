# BUG REPORT — HSiKAN is dispatch/launch-bound; training never runs it at speed

**Filed:** 2026-06-28 (JST) · **By:** Aiko (Claude Code) for Dr. Cs. Hajdu · **Severity:** High (confounds a
scientific conclusion + makes multi-seed sweeps infeasible) · **Status:** confirmed by profile, fix not yet wired

## Summary
The HSiKAN backbone forward is **dispatch-bound** (the CPU analogue of GPU launch-bound): one forward is ~200
tiny tensor ops, of which only ~24 % is real linear-algebra compute; the rest is elementwise math, dtype copies,
and reshapes whose cost is per-op overhead, not FLOPs. The codebase **already documents** this (`policy.py`
`build_policy`, lines 211–215: *"these models are tiny and launch-bound on GPU, so `torch.compile(mode=
"reduce-overhead")` is a large win (~10× measured on a 3070)"*) — but **no training path applies it**
(`train_sac`, `train_ppo`, `diag_contact`, `exp_pernode_actor_ab`). HSiKAN therefore always trains in slow eager
mode. Consequence: the controller sweep ran **~2 h with 0 / 8 cells finished**, and — more seriously — our finding
that *"HSiKAN floors on the coin-toss (0 % at 15k and 40k)"* is **confounded with undertraining**: at feasible
wall-clock HSiKAN simply cannot take enough steps.

## Evidence (profiled this machine: CPU, Windows, 1 BLAS thread = the sweep's per-worker pin)
Model: galambos HSiKAN actor, 15 368 params, obs `(6, 8)`. `verification`-style benchmark
(`scratchpad/prof_hsikan_launch.py`).

| regime | eager forward | per-sample | note |
|---|---|---|---|
| **B = 1** (rollout per-step) | **1.80 ms** | 1.80 ms | the bottleneck — dispatch overhead dominates |
| **B = 256** (gradient update) | 16.2 ms | **0.063 ms** | **28× cheaper/sample** → overhead is fixed per-op, amortised by batch |

Op breakdown of one B=1 forward (top, per forward): `mul` ×35, `_to_copy` ×32, `copy_` ×39, `add` ×24,
`unsqueeze` ×20, `permute` ×20, `clamp` ×10, `gather` ×8 — **vs only `einsum` ×4 + `addmm` ×7 of actual compute.**
~**76 % of the time is elementwise + dtype-copy + reshape overhead**; the Catmull-Rom spline is evaluated as many
small ops, and ~16 % of total time is *dtype copies alone* (`_to_copy`/`copy_`).

`torch.compile(mode="reduce-overhead")` — the documented 10× fix — **errors here**:
`InductorError: Compiler: cl is not found` (the Inductor C++ backend needs MSVC `cl`, absent on this box; it works
on the Linux/GPU machine where the 10× was measured).

## Root causes (three, independent)
1. **Eager training.** The documented `torch.compile` win is applied *nowhere* in the training/rollout loops. The
   model trains and rolls out eager. (One-line-per-call-site fix, gated on the toolchain — see #3.)
2. **Dispatch-bound forward.** Even eager, the forward is ~200 micro-ops because the signed spline message-pass is
   not vectorised: per-edge / per-step elementwise spline basis + repeated dtype casts. This is fixable **without
   any compiler** by batching the spline eval and the message-pass into a few large ops (gather + batched
   Catmull-Rom + sparse/dense matmul).
3. **Toolchain gap.** `torch.compile` (Inductor) needs `cl`/MSVC on Windows (or run on Linux/GPU). Not installed,
   so even if wired in, it would no-op here.

## Impact
- **Sweeps infeasible:** 8 cells × 30k × HSiKAN × 1-thread ≈ 7–8 h; observed 0/8 in 2 h. Killed.
- **A scientific conclusion is confounded:** "HSiKAN can't learn the coin-toss" is entangled with "HSiKAN can't be
  trained enough." The launch-bound fix is a *prerequisite* to honestly testing HSiKAN's learnability on hard RL.

## The sell (why this is an asset, not just a chore)
This is a **performance-engineering contribution with a clean structural story**, and it *rescues* the HSiKAN
science. The forward is dispatch-bound *because* HSiKAN does fine-grained signed-spline message-passing over a
hypergraph — and the same structure that makes it expensive makes it **accelerable four ways**, tiered from
drop-in to principled:
1. **Op-vectorisation** (eager, no toolchain): batch the spline + message-pass → few large ops. Recovers most of
   the 76 % overhead anywhere, Windows included.
2. **Fused Triton kernel:** `hymeko_neuro/triton_kernels` already has a `cr_kernel`; one launch for the whole
   signed-spline-message-pass (GPU).
3. **`torch.compile` / CUDA-graphs:** ~10× drop-in where `cl`/Inductor is available (Linux/GPU box).
4. **SA-HSiKAN (B^L collapse):** for a *fixed* topology the entire iterated message-pass precomputes to one matmul
   (the StructuralActor) — ~10× more, ~30× fewer params. **Launch-bound → matmul, enabled by the declared
   structure.** This is the HyMeKo control-substrate thesis paying off in wall-clock.

Together: "HSiKAN's expressiveness is dispatch-bound by construction; the hypergraph structure that causes it also
admits a precompute that removes it." That is a paper-grade narrative *and* the reason the floors-conclusion must
be re-run post-acceleration.

## Next
Acceleration plan with the four tiers, the profiling protocol (py-spy flamegraph + this op-breakdown as the
regression gate), and the re-test of HSiKAN learnability post-speedup: `docs/plans/2026-06-28-hsikan-acceleration/`.
