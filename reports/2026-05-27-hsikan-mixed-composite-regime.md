# Report — Canonical+No-Excess (Composite) P-graph regime driving the HSiKAN-mixed protocol sweep

**Date:** 2026-05-27
**Plan:** `docs/plans/2026-05-27-hsikan-mixed-composite-regime/` (tex/pdf/tikz/mmd)
**Crates / packages touched:** `signedkan_wip`, `data/hsikan` (all non-core)
**CORE.YAML items touched:** **none** (see provenance for the torch
dependency-drift caveat — pre-existing, not introduced here).

## Summary

Wired the newly-canonical P-graph MSG/ABB machinery (Pimentel fix +
`Regime` Strategy refactor, both 2026-05-27) to an **HSiKAN-mixed ×
protocol-axis** architecture sweep and validated the pipeline end-to-end
with a Bitcoin Alpha abbreviated smoke under the **Composite**
(`canonical+no-excess`) regime.

The structural primitive is **fixed** to the mixed cycles+walks family
(`c3,c4,w2,w3`, the `JOINT_BA_SLOTS` default); the P-graph ranges over
*the other protocols* layered on top — attention (`none|dot|quaternion`),
edge-gate (`scalar|edge_cr`), direct-messaging (`off|on`), hidden width,
training length — encoded as a mandatory chain so a feasible ABB solution
must select exactly one unit per axis.

**Result (wiring smoke, not a benchmark — see caveats):** the Composite
regime selects a valid, non-empty HSiKAN-mixed architecture; the
ABB-cost-minimal pick (`struct_mixed, attn_none, gate_scalar, dm_off,
model_h8, train_short`) runs on Bitcoin Alpha at **AUC 0.9337, 16 s,
1.56 GiB peak RSS** — under the 7 GiB budget and the 16 GB cap. A live
**rich** combo (attention=dot + edge_cr + direct-messaging) also runs
(**AUC 0.9171, 31 235 params**), but **only with an enumeration cap**:
uncapped it OOMs the 7.6 GiB GPU because attention disables
cycle-batching. The driver now applies that cap automatically.

### What "the canonical ABB + MSG" buys here

- **Composite ≠ Canonical, observably.** Canonical's maximal structure
  admits the wasteful `attn_quaternion` unit (a by-product producer);
  Composite (`canonical+no-excess`) prunes it. Regression-tested both
  ways. This is the No-Excess refinement composing correctly over the
  canonical base.
- **No empty-structure regression.** Composite returns a non-empty ABB
  unit set spanning all six axes — the failure mode the Pimentel fix
  addressed is guarded by a test.
- **Feasibility shaping, not quality ranking.** ABB is cost-minimal, so
  it returns the cheapest launchable protocol combo (the baseline). A
  *quality* comparison across protocols is a 5-seed job (see Follow-ups),
  deliberately **not** run here.

## Files touched

| File | Status | LOC |
|:--|:--|--:|
| `data/hsikan/sweep_msg_mixed_protocols.hymeko` | new | 107 |
| `signedkan_wip/experiments/runs/run_hsikan_mixed_composite_smoke.py` | new | 361 |
| `signedkan_wip/tests/test_mixed_composite_regime.py` | new | 232 |
| `signedkan_wip/src/hsikan_pgraph_mapping.py` | +19 (additive dict keys) | — |
| `docs/plans/2026-05-27-hsikan-mixed-composite-regime/{tex,pdf,tikz,mmd}` | new | — |
| `reports/2026-05-27-hsikan-mixed-composite-regime.md` | new | — |

No Rust edits: the Composite regime was already reachable via
`hymeko_pgraph_dump --regime canonical+no-excess`.

## Test results

Runner: `.venv/bin/python -m pytest -p no:randomly` (pytest 8.4.2,
torch 2.4.1 — the unit/integration/regime tests are torch-free).

| Layer | Test | Count | Result |
|:--|:--|--:|:--|
| Unit | mapping merge (new units, precedence, KeyError) | 4 | pass |
| Unit | knob→env translation (bools/strings/partial/attention-cap) | 4 | pass |
| Unit | budget gate (pass + 5 violation cases) | 6 | pass |
| Unit/regression | regime semantics (canonical keeps / composite prunes quaternion; no-excess agreement; unknown-regime raises) | 3 | pass |
| Integration | ABB non-empty + spans all 6 axes; dry-run solve→map→env | 2 | pass |
| — | `test_mixed_composite_regime.py` total | **19** | **pass (0.18 s)** |
| Regression | existing `test_hsikan_pgraph_mapping.py` (additive change) | 7 | pass |

Lint: `ruff check` clean on all three changed/new Python files.

### Coverage rule (CLAUDE.md §3)

Every new function is exercised: `solve_regime`, `structure_to_env`,
`check_budget`, `main` (dry-run path), plus the additive mapping keys and
the sweep fixture. `run_cell` / `_last_json_line` / `_git_sha` are driven
by the live smoke (the GPU integration path) rather than pytest; this is
declared, not hidden.

## Performance results

**Production-scale smoke** (CLAUDE.md §3) — single run, *not* a
5-iteration `criterion`/`pytest-benchmark` measurement. Enforced with
`systemd-run --user --scope -p MemoryMax=16G` (cgroups v2 RSS gate;
`ulimit -v` not used, per §4).

| Config (BA, seed 0, 3 epochs, h=8) | AUC | wall | peak RSS | vs budget |
|:--|--:|--:|--:|:--|
| ABB-selected baseline (attn=none, gate=scalar, dm=off, uncapped) | 0.9337 | 16.0 s | 1.56 GiB | ✓ (<180 s, <7 GiB) |
| Rich (attn=dot, edge_cr, dm=on, **per-vertex top-K=8**) | 0.9171 | ~30 s | <2 GiB | ✓ |
| Rich (attn=dot, edge_cr, dm=on, **uncapped**) | — | — | — | ✗ **GPU OOM** (7.6 GiB) |

`hymeko_pgraph_dump` Composite solve: <0.1 s (`explored` ≈ 4k nodes,
trivial). The two AUC numbers are **not comparable** — different epochs
effectively, a top-K=8 cap that discards cycles, single seed, 3-epoch
abbreviated. No statistical claim is made; promotion to any table
requires the 5-seed paired protocol (memory
`feedback_n_seed_before_paper_promotion`).

### Contract-preservation finding (the smoke's real payoff)

The attention axis **disables cycle-batching** (`run_final_cell.py:~504`),
so peak GPU memory scales with the full per-edge cycle/walk set and OOMs
even on small Bitcoin Alpha. The plan flagged attention as the
highest-risk axis; the smoke confirmed it. The driver's
`structure_to_env` now auto-sets `HSIKAN_TOPK_MODE=per_vertex` /
`HSIKAN_TOPK_K=8` whenever an ABB selection turns attention on, so the
attention branch inherits an enumeration cap (callers may override).
Unit-tested both directions (attention-on caps; attention=none does not).

## New / removed dependencies

None.

## Experiment provenance

- **Git SHA:** `8fd8187` (working tree **dirty**). Files changed by this
  task: `signedkan_wip/src/hsikan_pgraph_mapping.py` (M, additive) plus
  three new untracked files (sweep, driver, test). `scripts/pgraph/verify.sh`
  also shows as modified but **predates this task** (not touched here).
- **Smoke interpreter:** `/home/kyberszittya/miniconda3/bin/python` 3.13.5,
  **torch 2.11.0+cu130**. **DEPENDENCY DRIFT (flagged, user-approved):**
  CORE.YAML pins `torch==2.12.0` (cu132, bumped 2026-05-27); no local env
  has 2.12.0 (`.venv` has 2.4.1 and lacks the `hymeko` wheel; miniconda3
  has the wheel + 2.11.0). User chose to run under miniconda3 with the
  drift recorded — the AUC numbers are **wiring-smoke** values, not
  CORE-reproducible benchmarks.
- **OS/kernel:** Linux 6.17.0-29-generic. **CPU:** AMD Ryzen 7 3700X
  (8-core). **RAM:** 31 GiB. **GPU:** NVIDIA RTX 2070 SUPER, 8192 MiB,
  driver 580.126.09.
- **Seed:** 0 (fixed; `torch.manual_seed` + `np.random.seed`).
- **Sweep fixture hash:** `md5 d0fd8010cef7473cae20c2edd8a37e3b`
  (`data/hsikan/sweep_msg_mixed_protocols.hymeko`).
- **Result artifact:** `/tmp/hsikan_composite_smoke/results.jsonl`
  (provenance row), log `/tmp/hsikan_composite_smoke/cell_bitcoin_alpha_seed0.log`.

## §6.5 anti-patterns

None introduced. No Cartesian-product wrappers (one driver, config via
unit-set + env); no algorithm code behind a binding; no string-typed
config crossing into Rust (the regime spec is parsed by the existing Rust
CLI); path helpers inlined into the driver (justified inline) rather than
imported, to avoid dragging the torch+PyO3 chain into a torch-free module.

## Open issues / follow-ups

1. **Regime A/B/C quality comparison (the original question's deeper
   half).** This task validated the pipeline; whether
   `canonical+no-excess` *selects a better architecture* than canonical-
   or no-excess-alone needs the SSG Pareto set run at 5 seeds, iso-param,
   paired. The infrastructure is now in place; it is a launch, not new
   code.
2. **Construction-protocol axis (P1/P2/P3) is dormant on BA.** Bitcoin
   Alpha is a native signed graph; the tabular construction protocols
   only become a live axis on tabular datasets. The sweep reserves the
   slot conceptually but does not exercise it here.
3. **Attention top-K cap value (8) is a smoke default,** tuned to fit the
   7.6 GiB GPU at h=8. The right cap for a real comparison should be swept
   alongside `m`, not hard-pinned.
4. **`params: null` in the baseline row** — `run_final_cell` does not emit
   `n_params` on the non-attention path (the rich/attention path does:
   31 235). Cosmetic; the AUC/RSS/wall gate is unaffected.
