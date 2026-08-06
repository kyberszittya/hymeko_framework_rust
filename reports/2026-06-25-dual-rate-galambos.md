# Rate-asymmetric dual-loop controller — prototype + discriminating experiment (2026-06-25)

## Summary

Built the **rate-asymmetric dual-loop controller** (Kato's reflexive/deliberative split, made compute-honest by
the measured ~15× latency asymmetry: HSiKAN ~420 Hz vs MLP ~6400 Hz, single-thread CPU). A fast MLP **reflex**
runs every step; a slow HSiKAN **deliberation** runs every *N* steps, its output held as a conditioning signal
the reflex consumes: `action = head(reflex(obs) ⊕ held_deliberation)`.

Ran the discriminating experiment on Galambos (behaviour cloning of the scripted demonstrator): does the dual
beat MLP-alone and HSiKAN-alone, and does deliberating every *N>1* steps preserve delivery?

**Result: a tie on the architecture (all within noise), with a weak but sensible rate-asymmetry signal.** This
*reproduces* the known Galambos no-structural-leverage root-cause — Galambos cannot discriminate this
architecture, so the fair test needs a structural-leverage task.

## Files touched

- `hymeko_rl/dual_rate.py` (+145) — `DualRateController` (reflex ⊕ held-deliberation, context-optional so it is
  drop-in for `behaviour_clone`/PPO), `RateAsymmetricLoop` (the N-step cadence), `build_dual_rate`.
- `hymeko_rl/tests/test_dual_rate.py` (+69) — 5 tests: fused forward shapes; gradient to both streams; held
  context conditions the reflex; the cadence fires exactly every N; context-optional path.
- `hymeko_rl/exp_dual_rate.py` (+99) — the experiment harness (reuses `collect_galambos_demos`,
  `behaviour_clone`, `eval_delivery`; adds `eval_delivery_rate_asym` for N>1 with per-episode reset).
- `hymeko_rl/policy.py` (−10/+8) — reverted the broken `compile_backbone` flag (it prefixed state-dict keys with
  `_orig_mod.`, breaking checkpoint round-trip — verified); docstring now points to call-site compile.
- `docs/plans/2026-06-24-kato-collab-dual-discriminator/plan.{tex,pdf}` — folded in the rate-asymmetry (measured
  evidence, revised fusion-topology decision, new deliberation-cadence decision N). PDF rebuilt.

CORE.YAML items touched: none.

## Results (delivery rate, 24 eval episodes/seed, 3 seeds)

| policy        | s0    | s1    | s2    | mean  | std   | params |
|---------------|-------|-------|-------|-------|-------|--------|
| MLP-alone     | 0.042 | 0.083 | 0.292 | 0.139 | 0.109 | 14921  |
| HSiKAN-alone  | 0.167 | 0.042 | 0.125 | 0.111 | 0.052 | 30025  |
| dual (N=1)    | 0.167 | 0.167 | 0.083 | 0.139 | 0.039 | 28105  |
| dual (N=4)    | 0.250 | 0.083 | 0.042 | 0.125 | 0.090 | —      |
| dual (N=8)    | 0.167 | 0.000 | 0.042 | 0.069 | 0.071 | —      |

**Architecture: tie.** Means 0.07–0.14 with stds 0.05–0.11 — the spread swamps every difference. The dual does
not beat either component. Consistent with `project-galambos-hsikan-tie-rootcause`: on Galambos the coin/zone are
not hypergraph nodes, so HSiKAN deliberation carries no structural information the MLP lacks; fusing a
no-information slow stream with the reflex cannot help. **Measured, not inferred:** the tie. **Inferred:** the
cause (no structural leverage) — already isolated by the prior A/B, reproduced here.

**Rate-asymmetry: holds to N=4, degrades by N=8.** dual N=4 (0.125) ≈ N=1 (0.139); N=8 drops to 0.069. The
compute saving (deliberate ¼ as often) is roughly free; holding the context 8 steps is too stale. This is the
cadence knob behaving as designed — but on a task where deliberation barely matters, so it is a soundness check,
not a benefit demonstration.

## Test results

- `pytest hymeko_rl/tests/test_dual_rate.py`: 5 passed (~6 s). ruff + mypy --strict: clean.
- Smoke (40 demos/40 epochs/8 eval) ran the full pipeline end-to-end before the 3-seed run.

## Performance

CPU-only (`CUDA_VISIBLE_DEVICES=-1`, 2 threads). 3 seeds × (BC 200 epochs × 3 architectures + 5 delivery evals)
in ~one background run, well under the 16 GB cap (tiny models, ≤30 k params). No OOM.

## Interpretation & next step

The prototype is **validated as working** (built, tested, drives the comparison + the rate-asymmetric eval). The
experiment's real finding is methodological: **Galambos cannot discriminate the dual-discriminator** — by our own
prior root-cause, it gives the deliberative stream no structural leverage, so a null result here is *expected* and
says nothing against Kato's idea. To fairly test it we need a task where structural deliberation matters:

1. **Coin/zone as a grasp/goal hyperedge** — the discriminating A/B already defined in
   `project-galambos-hsikan-tie-rootcause`; this is the minimal change that gives HSiKAN something to reason about.
2. **A multi-step / FSM-structured task** (`project-fsm-structured-rl`) where the slow loop's role is the
   phase/sub-goal context the fast reflex executes within — the natural home for a two-timescale controller.

Until a leverage task is in place, the dual-loop's *architecture* benefit is untested; only its *mechanics*
(fusion, cadence, compute saving) are confirmed.

## Provenance

Git: branch `fix-hsikan`, working tree dirty (this change + the session's hymeko_neuro/core/dual_rate work). Seeds 0,1,2;
demo seed = run seed; eval seed = 9000. Demonstrator = `GalambosDemonstrator`, `only_success=True`. Log:
`reports/2026-06-25-dual-rate-galambos.log`. Host: Windows 11, RTX 3070 (unused — CPU run).

## §6.5 anti-patterns

None introduced. The controller reuses `mlp_backbone`/`hsikan_backbone` (no new backbone), the harness reuses the
Galambos BC machinery (no scaffold duplication), the cadence is one config (`deliberate_every`), not per-N
functions, and `compile_backbone` (a flag that should have been a call-site concern) was removed, not patched.
