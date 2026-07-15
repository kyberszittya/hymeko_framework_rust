---
title: "Compositional-spec benchmark — the non-trivial regime MetaWorld cannot provide"
date: 2026-07-15
author: Aiko (Opus 4.8)
branch: feat/spec-reward-close-loop
stage: thread-A-compositional-benchmark
status: COMPLETE — non-trivial by construction; arbiter recovers; pgraph ties greedy on F1 (payoff is ABB scale)
tags: [spec_bench, hymeko_pgraph, ssg, abb, benchmark, compositional]
---

# Compositional-spec benchmark

**[2026-07-15 19:44 JST]**

## Summary

The close-the-loop arc established that MetaWorld cannot exercise the arbiter's real work: `in_place_reward` is a
monotone success proxy, so *every* task's success spec is trivially single-signal (`F(in_place>=0.6)` AUC ≈ 1.0 on
coffee-push/push/pick-place/door-open/reach — measured). This benchmark supplies the missing regime: a controlled
**K-way conjunction over K true signals among D distractors**, where **no single signal separates success**, so
the arbiter must drop the distractors and calibrate. We sweep D at K=3 and measure non-triviality, the arbiter
lift, a greedy baseline, and the ABB branch-and-bound.

**Result (K=3, n=100, one seed; test F1):**

| D | #cands | best-single AUC | raw F1 | greedy F1 | **pgraph F1** | ceiling | ABB explored / pruned |
|--:|--:|--:|--:|--:|--:|--:|--|
| 2 | 5 | 0.691 | 0.361 | 1.000 | 1.000 | 0.947 | 220765 / 110294 |
| 4 | 7 | 0.729 | 0.214 | 0.947 | 0.947 | 0.913 | 180615 / 90214 |
| 6 | 9 | 0.688 | 0.182 | 0.971 | 0.971 | 0.889 | 175611 / 87706 |

Three findings, each stated honestly:

1. **Non-trivial by construction (the point).** Best single-signal AUC 0.69–0.73 across all D — *no one signal
   separates success*, unlike every MetaWorld task. This is the regime the arbiter's conjunct-pruning +
   calibration exists for, which MetaWorld's single-proxy interface structurally cannot present.
2. **The arbiter recovers, and its lift grows with D.** The raw over-constrained spec (`F(∧ all K+D)`) fails and
   **fails harder as D grows** (0.36 → 0.21 → 0.18 — more distractors reject more positives); the arbiter
   (`refine_via_pgraph`) prunes back to ~ceiling (1.0 / 0.95 / 0.97). The raw→arbiter gap *widens* with the
   distractor pool — the arbiter earns its keep more as the problem gets harder.
3. **pgraph ties greedy on F1 — reported, not hidden.** A 3-line forward-`greedy` conjunct-selector reaches the
   *same* F1 as the `hymeko_pgraph` SSG on every cell (1.0 / 0.947 / 0.971). On pure conjunctions each true
   conjunct improves F1 marginally, so greedy does not stall — the SSG's value here is **not** an F1 win. The
   P-graph's genuine payoff is **ABB branch-and-bound** doing real combinatorial work: it prunes ~50 % of a
   175k–220k-node search tree. (Its explored/pruned is roughly constant across D here only because the
   temporal-coverage aspect count is **capped at 5** — logged in the JSON as `abb_aspects_capped`; the
   aspect-count scaling where ABB pruning grows is `reports/2026-07-13-pgraph-scaled-temporal-refinement.md`:
   2→4 aspects, explored 67→527.)

**Honest net.** This rebuts the *"trivial task"* caveat decisively (the benchmark is provably non-trivial, and
the arbiter recovers where a naive over-constrained spec fails). It only **partially** rebuts *"the P-graph is
over-engineered for toys"*: the ABB branch-and-bound genuinely prunes, but on *conjunct-selection* a greedy
baseline ties it on F1 — the SSG/ABB earns its keep on search feasibility at scale, not on a small-pool accuracy
win. Figure: `reports/figures/compositional_bench/compositional_spec_benchmark.png`.

## Unification note — why greedy ties the SSG here but loses in the causal setting (the calibration escape)

I tried to build the regime where greedy *provably stalls* and the SSG wins on **accuracy**: a **luring
distractor** engineered to be the best single predictor (so forward-greedy picks it first). It is the best single
in **8/8 seeds** — yet greedy still ties the SSG (median F1 **0.982 vs 1.000**). The reason is diagnostic: greedy's
recovered spec is `F(true_0≥0.653 AND true_1≥0.658 AND lure≥0)` — threshold **calibration drives the lure conjunct
to a vacuous `≥0`**, neutralising it. In the spec-conjunction language a bad conjunct can *always* be calibrated
into harmlessness, so greedy never truly stalls; there is no local optimum to trap it.

This gives **one** explanation for both threads: the SSG beats a greedy/marginal method on **accuracy only where
there is no neutralisation escape** — i.e. a genuine **causal interaction** mechanism (`x_a·x_b`, which no
threshold can neutralise), exactly Thread B (`SignedHyperLiNGAM`: recall 1.0 vs pairwise 0.73). In
spec-conjunction-selection the escape exists, so greedy+calibration matches the SSG and the SSG/ABB's value is
*scale/feasibility*, not accuracy. Captured by `test_calibration_escape_greedy_matches_ssg_on_lure`
(with `synth_compositional_lure`).

## Files (new/extended; worktree `feat/spec-reward-close-loop`)

| file | note |
|---|---|
| `hymeko_rl/eval/spec_bench/scale.py` (+64) | `synth_compositional(K,D)` + `compositional_ground_truth`/`_raw_spec`/`_signals` + `_compositional_trace` |
| `hymeko_rl/eval/spec_bench/pgraph_refine.py` (+31) | `greedy_conjunct_select` (the baseline the SSG is measured against) |
| `hymeko_rl/experiments/exp_compositional_spec_benchmark.py` (+140) | the D-sweep runner; JSON + scaling plot |
| `hymeko_rl/tests/test_compositional_benchmark.py` (+95) | 9 tests |

Reuses `propose_and_gate`/`refine_via_pgraph`/`refine_scaled_abb`/`coverage_pgraph_hymeko`/`spec_reward_separation`
— no re-implemented solver or metric (§6.1). No §6.5 anti-patterns.

## CORE.YAML items touched

**None.** No new dependency (`pgraph` binary already built).

## Test results

`test_compositional_benchmark.py`: **9 passed** (131 s — `refine_via_pgraph` calibration is the cost). Covers:
generator validation, balance/determinism, **non-triviality** (best single AUC < 0.8), **compositional necessity**
((K-1)-subset F1 < full), **arbiter recovery** (raw < 0.6 → pgraph ≥ 0.85), greedy recovers + pgraph ≥ greedy,
single-predicate calibration, spec formats, and the runner smoke. `ruff`: clean. `mypy --strict`: my additions
clean (a pre-existing `solve_pgraph` `dict|None` type-arg note is baseline, untouched). `radon`: my additions A/B
(pre-existing `refine_scaled` C(16) untouched).

## Performance

- Per cell dominated by `refine_via_pgraph` calibration (2^(K+D) subsets × coordinate-ascent) + the capped ABB
  solve: D=2 12 s, D=4 64 s, D=6 363 s. Peak RSS < 0.5 GB. The ABB aspect cap (5) keeps a single solve under the
  60 s `pgraph` timeout.

## Experiment provenance

- Worktree `hymeko_spec_reward_wt` on `feat/spec-reward-close-loop`; Apple-Silicon Mac, `.venv` cpython-3.11,
  numpy 2.x, built `target/debug/pgraph` (symlinked). K=3, D∈{2,4,6}, n=100, seed 0.

## Open issues / follow-ups

1. **Show ABB scaling directly:** lift the aspect cap and sweep aspect count (not distractor count) to plot ABB
   explored/pruned growing — the P-graph's real payoff — instead of citing the prior report.
2. **A regime where greedy stalls:** greedy ties the SSG on pure conjunctions. A conjunct-selection landscape with
   interaction (a conjunct useful only *in combination*) would separate them — that is Thread B's joint-mechanism
   territory (`SignedHyperLiNGAM`), and unifying the two is the natural next step.
