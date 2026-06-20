# Geometric attention head — gate inspection: the score is dead, not the value

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-geometric-attention-head](../docs/plans/2026-06-17-geometric-attention-head/) — this is the "inspect the learned gate σ(γ)" next-step the plan/report already named; a diagnostic within existing scope, no new plan dir.
**Status:** ✅ diagnostic complete + tested. **Discriminating finding: score collapse.** The geometric attention pool is "attention" in name only — every signed weight is ~0, so the pool is an unweighted mean of values. This is the blocking element behind the 5-seed negative result.

## Why this ran
The 5-seed A/B ([2026-06-17-geometric-attention-head.md](2026-06-17-geometric-attention-head.md)) found `geom_attn` ties bilinear on alpha and **regresses** on otc — adds nothing net. The report's stated next lever was *sign-aware values*, asserted as "the most likely missing ingredient." Per the operating contract (analyze, don't declare; run the discriminating test before concluding), the cheap test the report flagged but had not run: **is either geometric channel actually used, or is the pool swamped by the residual?** Two failure modes to separate:
- **score collapse** — `w_abs_mean → 0`: the `tanh` scores saturate to 0, the pool degrades to an unweighted value mean (no triad selectivity).
- **residual swamp** — `pool_to_hv → 0`: the pool is informative but tiny next to `h_v` in `node = h_v + pool`.

## Result (seed 0, tuned recipe = the regressing config)
`reports/geom_gate_inspect_20260617.jsonl`

| dataset | AUROC | gate σ(γ) | \|w\|mean | %dead (\|w\|<0.05) | %sat (\|w\|>0.9) | pool/h_v | ‖W_q‖ | ‖W_k‖ | ‖W_v‖ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bitcoin_alpha | 0.8614 | 0.4998 | 0.0019 | **1.00** | 0.00 | 0.144 | 0.156 | 0.133 | 3.239 |
| bitcoin_otc | 0.8779 | 0.4999 | 0.0023 | **1.00** | 0.00 | 0.141 | 0.255 | 0.239 | 3.174 |

### Measured / inferred / hypothesis (per the contract)
**Measured:** 100 % of the 50.8k (alpha) / 75.0k (otc) signed weights are dead (`|w|<0.05`); none saturate. The gate parameter never moved off its init (σ=0.500 → equal quaternion/Clifford). `W_q`/`W_k` stayed at their `0.1×` init scale (Frobenius 0.13–0.26 over a 32×32 matrix ≈ the init), while `W_v` grew to ≈3.2. `pool_to_hv ≈ 0.14`.

**Inferred:** the geometric score collapsed to ≈0 everywhere, so `Σ w·v / Σ|w|` with uniform-tiny `w` is an **unweighted mean of `W_v(triad)`** — the pool carries a vector (hence `pool/h_v ≈ 0.14`, *not* residual swamp) but no triad-specific selectivity. The gate is irrelevant because both channels feed a dead score. Mechanism: the score branch is born small (`W_q,W_k ×0.1` init × `scale = n_blocks^-0.5 = 1/√8` × `tanh`), and the value+residual path (`h_v` + value mean) already minimises the loss, so the score receives no gradient pressure to become selective — classic dead-attention.

**This is score collapse, decisively — not residual swamp** (`pool/h_v` is non-vanishing; `w_frac_dead` is the pinned variable, exactly 1.00 on both).

**Hypothesis (untested, for the next step):** waking the score will require (a) removing the `0.1×` `W_q`/`W_k` init suppression or adding a learnable score scale (give the score dynamic range), and (b) injecting the triad sign σ into the **score/key** path so attention can vote by balance. The report's *sign-aware values* (σ into the value) is still expected to help — an unweighted mean of σ-weighted values carries the sign — but the diagnostic shows it is not the primary cause; the dead component is the score, not the value content.

Seed 0 only, but the signal is structural (gate exactly 0.500, dead fraction exactly 1.00, identical on both datasets), not a noisy metric — a 5-seed sweep is unnecessary to establish "the attention is dead."

## What was built
A pure, testable analysis function + a generic Observer hook, reusing the existing driver setup (no duplication):
- `signedkan_wip/src/core/geometric_triad_attention.py` — `summarise_gate(pool, h_node, h_triad, inc_vertex, inc_triad) -> dict` (~45 LOC): `@torch.no_grad`, read-only; reports σ(gate), signed-weight saturation (`w_abs_mean/std`, `w_frac_dead/saturated`), pool-vs-h_v residual magnitude, and projection norms.
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` — added `GeomTrainedState` (frozen dataclass) + an optional `on_trained: Callable | None = None` Observer hook to `run()` (default `None` → zero behaviour change; fires only for `head='geom_attn'`). Reuses all training/setup — the diagnostic does not re-implement the driver.
- `signedkan_wip/experiments/runs/inspect_geom_gate.py` (new, ~70 LOC) — drives `run()` with the tuned recipe over the small bitcoin fixtures, captures `summarise_gate` via the callback, emits one merged jsonl row per cell + a stderr table.

## Files touched
- `signedkan_wip/src/core/geometric_triad_attention.py` (+~50 LOC: `summarise_gate`)
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` (+~25 LOC: `GeomTrainedState`, `on_trained` hook, `Callable` import)
- `signedkan_wip/experiments/runs/inspect_geom_gate.py` (new)
- `signedkan_wip/tests/test_geometric_triad_attention.py` (+3 tests: ranges/init, zero-query-kills-weights, gate σ tracks param)
- `signedkan_wip/tests/test_hsikan_rotor.py` (+2 tests: on_trained fires+aligned for geom_attn, ignored for bilinear)
- CORE.YAML items touched: **none** (CORE.YAML protects only `hymeko_core/` Rust).

## Tests
- `ruff check` (5 changed files): **PASS**.
- `pytest -p no:randomly` (both suites): **29 passed in 27.9 s**. 2 warnings are the pre-existing torch sparse-CSR-beta notices from `signedkan.py:753`, not this change.
- Coverage (§3): `summarise_gate` driven by 3 new unit tests incl. the zero-query failure case; the `on_trained` hook driven by 2 new integration tests (fires-for-geom / ignored-for-bilinear) — the latter is a regression test that would fail had the hook not been gated on `geom`.

## Performance
- Wall: ~25 s total for the 2-cell diagnostic (120 epochs, early-stop each).
- Peak RSS not separately polled this run; same scale/config as the 5-seed run measured at **1724 MB** (10.5 % of the 16 GB cap). No new tensors of graph size are allocated by the diagnostic.

## §6.5 anti-patterns
None introduced. The diagnostic reuses `run()` via an Observer callback rather than duplicating the driver (avoids #3 scaffold duplication); `summarise_gate` lives next to the module it analyses (cohesion); no new function-name-per-axis (#1/#5); no globals (#11); no `_v2` file (#13). The `on_trained` hook is a generic callback param, not a string-flag branch.

## Next step (the fork, now evidence-based)
The score is the dead component. The decision is whether to **wake the readout** with a focused structural revision of the head — (a) score-scale fix (drop `0.1×` init / learnable scale), (b) σ into the score/key path (and value), gated + A/B'd — *or* **pivot to Berge cycles** now. The revision is a model change → its own plan (`docs/plans/...`). Recommendation: one woken-score A/B first; if a score with real dynamic range still adds nothing, pivot to Berge with confidence rather than on the report's untested guess.

Memory: `project-hsikan-geometric-attention-berge` (updated).
