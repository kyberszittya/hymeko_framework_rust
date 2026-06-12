# Claude Code Brief — Experiment E1: Spectral CR Filter vs Mixed-Arity HSiKAN

**Paste-target:** Claude Code session in `hymeko_framework_rust/`.
**Spec:** `D:\hakiko_ai_ws\01_analysis\Math\cr_spectral_bridge_2026-06-11.pdf` §6 (E1). Read it first, fully.
**Contract:** `CLAUDE.md` applies in full. This brief restates the binding points; the contract wins on any conflict.

---

## Task

Implement and run Experiment E1: a Catmull–Rom-parametrized spectral filter on the signed normalized Laplacian (Chebyshev-evaluated, K=8, G=8, strict protocol), compared against HSiKAN-strict on Bitcoin-Alpha. Falsifiable accept/reject criteria below. Either verdict is a deliverable.

## Step 0 — Read + discovery (mandatory, §0/§6.1/§6.5 #12)

1. Read `CORE.YAML`, `tools.yaml`, the E1 spec PDF (or its `.tex` next to it).
2. Discovery pass before creating ANY artifact:
   - `grep -rin "cheby\|spectral\|laplacian\|polynomial filter" signedkan_wip/src/ hymeko_graph/src/`
   - `ls signedkan_wip/experiments/results/ | grep -i "spectral\|cheby"`
   - Locate the existing strict-protocol masked-matrix path (M_tr assembly) used by the Gömb-strict benchmark — REUSE it, do not rewrite.
   - Locate the existing shared train/eval harness. The repo has a documented anti-pattern (#3) of 98 `run_*.py` scripts re-implementing train loops; you add a **model class + ~20-line config**, never a new training loop.
3. If discovery finds existing spectral-filter scaffolding: extend it and say so in the report. Do not duplicate.

## Step 1 — Plan (§2, non-negotiable, before any code)

`docs/plans/2026-06-12-e1-spectral-cr-filter/` with all four artifacts (`plan.tex` compiling → `plan.pdf`, `plan.tikz`, `plan.mmd`). Must state: affected files; CORE.YAML items touched = **empty list**; interface of the new model class; test strategy; performance budget (≤ 2 GPU-h total, peak RSS ≤ 16 GiB enforced via `systemd-run --user -p MemoryMax=16G`); worst-case input (full Bitcoin-Alpha, 24 186 edges); rollback (delete class + config, no other code references it).

## Step 2 — Implementation constraints

- **New model class** `SpectralCRFilter` (structural variant → class, not forward-time flags; anti-pattern #8):
  - CR spline `g` with G=8 learnable control points on λ ∈ [0,2].
  - Map control points → K=8 Chebyshev coefficients via interpolation at Chebyshev nodes (a fixed linear map), OR learn θ directly with CR-derived init. Pick one in the plan; justify in one sentence.
  - Evaluate `g(L_tr)X` by the Chebyshev recursion: K sparse mat-vecs on `L_tr = I − D̄^{-1/2} M_tr D̄^{-1/2}`, with `M_tr` from **training edges only** (strict protocol holds at the operator level — same argument as the masked-matrix paragraph in the audit work).
  - Per-edge readout consistent with the existing strict benchmark (vertex-adjacency M_e, no edge-in-cycle incidence).
- **No new dependencies** (torch sparse + existing stack suffice). A dependency add is a §1 CORE event → halt and ask.
- Config dataclass, typed; no string-typed modes (#7); no env-var flags read at depth (#11); no `_v2` files (#13).

## Step 3 — Tests BEFORE queuing (§3)

- Unit: CR spline evaluation vs closed-form Keys-kernel reference (1e-6); Chebyshev recursion vs dense `U g(Λ) Uᵀ` on a 50-node toy graph (1e-5); one failure case per public fn.
- **Strictness test (the one that matters):** rebuild features with all test-edge signs flipped/shuffled; assert the feature tensors are byte-identical. This is E1's leakage proof.
- Determinism: fixed seeds, order-independent (`pytest -p no:randomly`).
- 1-seed production-scale smoke (real dataset, real caps) before the 5-seed run; smoke wall must be ≤ 10% of queued budget (§3).

## Step 4 — Runs

- Dataset: Bitcoin-Alpha primary. Splits/seeds: **identical** to the Gömb-strict benchmark (reuse its split code + seeds 0–4).
- Comparators from existing JSONLs (`gomb_strict_benchmark_tuned_20260514T010516Z/`, HSiKAN-strict cells) — do NOT re-run them.
- Optuna: 30 trials, TPE, validation AUC, strict features only.
- Label-shuffle audit: **train-only shuffle** definition exactly as `reports/2026-05-17-hsikan-rescore-and-shuffle-audit.md` §3.1.
- Every cell → JSONL with git SHA, seed, dataset hash, config dict, peak RSS, wall (under `signedkan_wip/experiments/results/e1_spectral_cr_<UTC>/`).

## Step 5 — Verdict (pre-registered, from the spec)

- **ACCEPT** iff: AUC within seed noise of HSiKAN-strict AND wall ≤ HSiKAN-strict (enumeration + train) AND shuffled AUC ≤ 0.55.
- **REJECT** iff AUC deficit > 2 pp → write the negative result (simple cycles carry signal walks cannot see); flag E2 (signed Hashimoto operator) as follow-up. A rejection is a successful experiment.
- Either way: symbolically distill the learned g(λ) (existing distillation tooling — discovery pass will find it) and record the fitted form.

## Step 6 — Report (§9)

`reports/2026-06-XX-e1-spectral-cr-filter.md`: summary, files touched with line counts, CORE.YAML = none, test results per layer, perf vs budget, verdict against the pre-registered criteria, provenance block, "no §6.5 anti-patterns introduced" (or waivers), follow-ups.

## Halt conditions (verbatim §11 spirit)

Halt and ask the user if: any CORE.YAML item would be touched; a needed dependency is missing; the strictness test cannot be written; a test fails for an ununderstood reason; the wall estimate disagrees >2× with the smoke; RSS approaches the 16 GiB cap.

---

*Related, ready when this lands: `docs/plans/2026-06-11-gsphf-rust-crate/` is a complete four-artifact plan for the `gsphf` crate (Phase P1 starts at the parity fixtures). Separate session, separate plan — do not mix.*
