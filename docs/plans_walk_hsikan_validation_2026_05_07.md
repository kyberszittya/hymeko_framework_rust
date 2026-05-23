# Walk-HSiKAN 5-seed validation plan (2026-05-07)

Goal: promote Walk-HSiKAN from a §V Future-Work bullet in `paper/smc2026_hsikan_wip/main.tex` to an honest table row, with 5-seed evidence that it beats cycle-HSiKAN under the same protocol.

The Walk-HSiKAN enumerator is shipped (Rust + PyO3, 12-case parity suite green; `signedkan_wip/src/walks.py`, env-var `HSIKAN_WALK_LENS`). Single-seed exploratory training already exists; what remains is the same statistical rigour that locks in the cycle-HSiKAN Table I numbers.

## Numbers to beat (cycle-HSiKAN, paper Table I, 5-seed)

| dataset      | cycle-HSiKAN AUC ($h$) | best baseline | margin to baseline |
|--------------|------------------------|---------------|--------------------|
| Bitcoin Alpha | $0.939 \pm .011$ ($h{=}16$) | SGCN $0.874$ | $+0.065$ |
| Bitcoin OTC   | $0.930 \pm .008$ ($h{=}16$) | SiGAT $0.934$ | $-0.004$ (within noise) |
| Slashdot      | $0.861 \pm .002$ ($h{=}4$)  | SGT $0.897$  | $-0.036$ |
| SBM $n{=}200$ | $0.911 \pm .028$ ($h{=}16$) | SGT $0.563$  | $+0.349$ |
| SBM $n{=}400$ | $0.962 \pm .009$ ($h{=}16$) | SGT $0.690$  | $+0.273$ |

These are the bars. Walk-HSiKAN must beat its own cycle-HSiKAN row by $> 1\sigma$ on at least 3 of 5 datasets and not regress (>1σ) on any to be promotable.

## Single-seed Walk-HSiKAN snapshot

From `docs/meeting_pimentel_outline.md` (line 275). One seed each, no ± std, protocol assumed to match Table I but not verified.

| dataset       | Walk-HSiKAN | cycle-HSiKAN | apparent Δ |
|---------------|-------------|--------------|------------|
| Bitcoin Alpha | $0.973$     | $0.939$      | $+0.034$ |
| Bitcoin OTC   | $0.959$     | $0.930$      | $+0.029$ |
| Slashdot      | $0.861$     | $0.861$      | $0.000$ (parity) |
| SBM $n{=}200$ | $0.999$     | $0.911$      | $+0.088$ |
| SBM $n{=}400$ | —           | $0.962$      | not run yet |

Param count for Walk-HSiKAN was ~1.3 M vs cycle-HSiKAN at $h{=}16$ (~few-hundred-K) — **parameter parity is not yet established**. A win at 4× param budget is not a fair win.

## Required runs

For each (dataset, model) cell: 5 seeds, same protocol as Table I (edge-in-cycle leakage, `signedkan_wip/src/run_final_cell.py` driver), one isolated CUDA process per seed.

Five datasets × {Walk-HSiKAN, cycle-HSiKAN-rerun} × 5 seeds = **50 runs** at minimum. The cycle-HSiKAN re-run is for the **paired** comparison (same seed list as Walk-HSiKAN, so we can compute paired Δ rather than two independent means). If the existing Table I seed list is recoverable, skip the cycle re-run and re-use the 5 logged results.

### Configurations

| axis | value |
|---|---|
| Walk lengths $L$ | $\{2, 3, 4\}$ for Bitcoin/SBM; $\{2, 3, 4, 5\}$ for Slashdot (matches paper $\mathcal{K}$ axis) |
| Hidden $h$ | $16$ for paired comparison; $4$ as a pruning sweep on Slashdot |
| Layers $L_{\rm SKL}$ | $2$ shared (matches paper) |
| Spline grid $G$ | $3$ |
| Epochs / optimiser | match `signedkan_wip/src/run_final_cell.py` defaults exactly |
| Seeds | the same 5 used for cycle-HSiKAN Table I — **find and pin these first**; do not re-roll |

### Param-parity guard

Before the win/loss claim is honest, match parameter counts $\pm 10\%$ between Walk-HSiKAN and cycle-HSiKAN at the comparison point. The 1.3 M single-seed run was likely over-parameterised; an iso-param Walk-HSiKAN at $h{=}16$ should land in the same ballpark as cycle-HSiKAN. If the single-seed gain disappears under iso-param, that is a real result (geometry, not capacity, was carrying the win).

## Acceptance criteria

To promote Walk-HSiKAN to **Table I** (joint-headline status):
1. 5-seed mean ± std reported for every cell, paired against cycle-HSiKAN at the same seeds.
2. Paired Δ AUC $> 1\sigma$ (paired) on at least 3 of 5 datasets.
3. No paired regression $> 1\sigma$ on any dataset.
4. Latency and parameter count within $1.2\times$ of the cycle-HSiKAN entry at the same dataset.
5. The same edge-in-cycle leakage protocol applies; if Walk-HSiKAN needs a different incidence definition, document it explicitly and don't claim the rows are comparable.

To promote to a **§III.G "Walk-HSiKAN preview" subsection** only (more conservative):
- 1, 2, 3 above. Latency / params can lag — flagged as "investigative".

## Risk register

Anchored to recent precedent in this codebase, not generic optimism:

| risk | probability | precedent |
|---|---|---|
| Single-seed gain collapses to NULL at n=5 | **moderate** | `project_overnight_2026_05_06_sober.md` — entropy-reg n=3 lift $+0.006$ collapsed to NULL ($+0.0007$) at n=5. The single-seed Walk-HSiKAN $+0.034$ on BA is bigger and may survive, but treat with caution. |
| Slashdot stays flat | **high** | Single-seed already at parity (0.861); SGT 0.897 remains the walk-rich champion. The likely outcome is "no change", which still doesn't unlock the walk-rich regime. |
| Iso-param Walk-HSiKAN regresses | **moderate** | The 1.3 M single-seed had ~4× the cycle-HSiKAN budget. Capacity vs structure is the standard confound. |
| SBM near-saturation regresses | **low** | $0.999$ has no headroom but synthetic SBMs are stable; biggest risk is variance, not bias. |

If risk #1 fires (single-seed → NULL), the conservative §V softening already in the paper is the correct landing state — no further edit needed.

## Decision tree after the runs

```
┌── 3-of-5 paired wins, no regressions
│   ├── iso-param OK ───► promote to Table I; rewrite §V bullet to cite the row;
│   │                    Conclusion gets a sentence on Walk-HSiKAN as the
│   │                    cycle/walk unifier with measured backing.
│   └── iso-param fails ► §III.G "preview" subsection only; do NOT touch Table I.
│                         Note the param caveat explicitly.
├── 1-2 wins, no regressions
│   └── stay with the conservative §V softening that already shipped 2026-05-07;
│       schedule a journal-version follow-up when results harden.
└── any regression > 1σ
    └── do not touch the paper. Treat the regression as the result and document
        which regime broke. (Likely Slashdot.)
```

## Order of operations

1. **Recover the cycle-HSiKAN seed list** — search `signedkan_wip/experiments/results/` for the JSON manifests that produced the Table I rows. Hardpin the 5 seeds.
2. **Iso-param Walk-HSiKAN config** — derive $h$ such that the param count matches cycle-HSiKAN at $h{=}16$ within $\pm 10\%$. The `signedkan_wip/src/walks.py` Walk-HSiKAN module probably already supports this; if not, add a `--match_params` flag.
3. **Run the 5×5 grid** — 25 Walk-HSiKAN cells, in isolated CUDA processes, log to `signedkan_wip/experiments/results/walk_hsikan_5seed_2026_05_07.jsonl`.
4. **Compute paired Δ** — for each dataset, paired $t$-test or paired bootstrap on the 5 (cycle, walk) pairs; report mean ± std and paired-Δ ± paired-std.
5. **Apply the decision tree** — edit the paper or don't, based on outcome.

## What this plan deliberately does NOT do

- Does not chase Slashdot. The paper's two-regime story is intact and saying "Walk-HSiKAN doesn't beat SGT on Slashdot" reinforces it. Don't spend a day on hyperparameter games trying to flip 0.861 → 0.90.
- Does not extend $\mathcal{K}$ beyond what the paper's Table I covers. The walk version of $\alpha_k$ mixing is the contribution; piling on $k=6$/$k=7$ enumeration is journal-scope.
- Does not re-run baselines (SGCN, SiGAT, SGT). Those numbers are locked.

## Files this plan will touch when executed

```
signedkan_wip/src/walks.py                            (possibly: --match_params flag)
signedkan_wip/experiments/results/walk_hsikan_5seed_2026_05_07.jsonl   (new)
paper/smc2026_hsikan_wip/main.tex                     (Table I + §V if green)
docs/plans_walk_hsikan_validation_2026_05_07.md       (this file — close out with results)
```
