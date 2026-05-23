# Outer HSIKAN → Clifford-FIR → Gömb (residual) — WIN — 2026-05-21

## Summary

Yesterday's substitutive composition of the outer-HSIKAN→
Clifford-FIR architecture was null on Bitcoin Alpha and
negative on Slashdot d=2. The user (correctly) flagged
"there must be something missing." There was: the
substitutive composition silently degenerated the
architecture because Gömb's Clifford-FIR layer was tuned
to consume a learned `nn.Embedding`, and substituting an
HSIKAN-refined activation broke that contract.

The fix is a **highway-gated residual composition**:
$$
x_{\text{embed}} = (1 - g) \cdot \text{base\_node\_embed}
  + g \cdot \text{HSIKAN\_refined},
\quad g = \sigma(\text{logit}_g) \in (0, 1),
\quad \text{logit}_g \text{ init} = -3.0 \text{ per channel}.
$$

At init $g \approx 0.05$ per channel → the model starts
effectively as plain Gömb (base dominates). Training can
lift $g$ toward 1 per channel if HSIKAN's refinement helps.
Same productive pattern as the inner_skip="highway" gate
that's been the consistent lever across phase 21/22 and
the arc-weight work.

**Headline:** **Bitcoin Alpha d=4 outer-HSIKAN lifts AUC by
+0.0062 over plain Gömb at 4.04σ paired, 3/3 wins.** The
first architectural positive of the day's signed-graph work.

## Results

**Bitcoin Alpha** (vs plain Gömb baseline 0.9001 ± 0.0098):

| outer_d | mean AUC ± σ | paired Δ | σ_d | wins | wall |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9001 ± 0.0102 | −0.0000 | −0.02 | 2/3 | 7.0 s |
| 2 | 0.9046 ± 0.0106 | +0.0045 | +1.48 | 2/3 | 8.3 s |
| **4** | **0.9063 ± 0.0073** | **+0.0062** | **+4.04** | **3/3** | 11.0 s |

**Slashdot** (vs plain Gömb baseline 0.9010 ± 0.0006):

| outer_d | mean AUC ± σ | paired Δ | σ_d | wins | wall |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8984 ± 0.0003 | −0.0026 | — | 0/3 | 26.7 s |
| 2 | 0.9001 ± 0.0011 | −0.0009 | — | tied | 33.7 s |
| 4 | OOM (CR spline) | — | — | — | — |

**Residual vs substitute (the same architecture, only the
composition rule fixed):**

| dataset, outer_d | residual mean | substitute mean | Δ | σ_d | wins |
| --- | --- | --- | --- | --- | --- |
| BA, d=4 | 0.9063 | 0.8977 | **+0.0086** | **+5.77** | **3/3** |
| BA, d=2 | 0.9046 | 0.8993 | +0.0053 | +0.93 | 2/3 |
| BA, d=1 | 0.9001 | 0.8998 | +0.0003 | +0.16 | 1/3 |
| Slashdot, d=2 | 0.9001 | 0.8912 | **+0.0089** | **+11.53** | **3/3** |
| Slashdot, d=1 | 0.8984 | 0.9010 | −0.0026 | −2.94 | 0/3 |

The d=2/d=4 cases show the composition fix is doing
substantive work — recovering an order-of-magnitude paired-σ
of lift over the substitute version.

## Interpretation

**What the substitute version got wrong.** `x_embed = HSIKAN_only`
forced the Clifford-FIR layer to consume cycle-aware
features in slots that were tuned for a learned random
embedding. The model couldn't recover the learned-embedding
behavior (no path back), so adding HSIKAN was a strict
constraint, not an addition.

**What the residual version gets right.** `x_embed = (1−g)·base
+ g·HSIKAN` starts as ≈ plain Gömb (g ≈ 0.05). The model
can train HSIKAN's contribution UP per channel where it
helps, and leave it near zero where it doesn't. The
architecture is now an **additive enrichment**, not a
substitution. Same lesson as the morning's
`outer_grad_checkpoint` fix and the CR-highway arc-weight
mode: when composing two architectural pieces, the right
default is "the new piece is gated off at init, gradients
lift it where useful."

**Why Bitcoin Alpha d=4 in particular.** The
d=4 outer HSIKAN does enough cycle-aware preprocessing
that the Clifford-FIR layer benefits from richer input
features. The lift is monotonic with depth on Bitcoin
Alpha (d=1: +0, d=2: +0.0045, d=4: +0.0062), suggesting
the trend would continue if we could fit d=8 (we can't
on the 7.6 GiB GPU without gradient-checkpointing).

**Why Slashdot d=4 OOMs.** Same architectural memory issue
as the morning's stacked-middle and substitute-version d=4
runs: 4-layer HSIKAN at the Slashdot cycle-count scale
hits the CR spline eval site. Gradient checkpointing
inside the HSIKAN stack would unblock this; we deferred
that work yesterday.

## Files touched (since yesterday's substitute version)

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/hymeko_gomb/cascade.py` | extended | +28 (`base_node_embed` + `hsikan_gate_logit` + residual mix; `node_embed` property points at base) |
| `signedkan_wip/tests/test_outer_hsikan_gomb.py` | extended | +50 (3 new tests: base-embed exposure, gate-low-at-init, backward-reaches-gate-and-base) |
| `signedkan_wip/experiments/run_outer_hsikan_gomb_residual_2026_05_20.sh` | new | 128 (overnight grid with paired Δ vs plain Gömb AND vs substitute) |
| `reports/2026-05-21-outer-hsikan-gomb-residual-WIN.md` | new | this file |

Everything else (the class, the CLI, the GombConfig fields,
the smoke-runner) stays from yesterday's substitute
version. The change is **one line of math** in the forward
pass plus two small parameters
(`base_node_embed` + `hsikan_gate_logit`).

## CORE.YAML items touched

None.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_outer_hsikan_gomb.py` | **10 / 10 pass** (8 prior + 3 new − 1 updated) |
| All prior interpret / side / arity / fuzzy / stacked-middle / gomb-signature / cpml-edge-logits suites | no regression |
| Bitcoin Alpha grid: 9/9 cells complete | — |
| Slashdot grid: 6/9 cells complete (d=4 × 3 OOM at CR spline, same site as before) | — |

## §6.5 anti-pattern audit

- One-line forward change + two new parameters; no new
  function names, no Cartesian-product API surface.
- Highway gate is a learnable parameter, not a string-typed
  config — the model decides per-channel whether to use
  HSIKAN.
- The `outer_hsikan_n_layers` axis is still a structural
  config (class-per-variant via dispatch in
  `_MODELS["outer_hsikan_gomb"]`), not a forward-time toggle.

Clean.

## Open follow-ups

1. **Gradient checkpoint inside outer HSIKAN** — would
   unblock Slashdot d=4. The Bitcoin Alpha trend is
   monotonic, so d=8 on Bitcoin Alpha might lift further.
2. **Validate on Bitcoin OTC and Epinions** — same paired
   protocol. If both lift like Bitcoin Alpha, this is a
   generalisable architectural improvement, not BA-specific.
3. **Inspect the trained gates.** Across the d=4 cells, what
   does $g$ converge to per channel? If $g \to 1$ in many
   channels, HSIKAN is heavily used. If $g$ stays near
   init across most channels, the lift is from a small
   subset of channels.
4. **Combine with weighted arcs.** The morning's
   `cr_highway` arc-weight mode can plug into the outer
   HSIKAN's `inner_skip="cr_highway"`. Per-edge magnitude
   information could amplify the d=4 lift.
5. **HymeYOLO vision port.** Now that we have a positive
   architectural lever, the vision port has a real shot at
   helping.

## Experiment provenance

- **Git SHA:** uncommitted.
- **Grid:** 18 cells × 60 epochs. Bitcoin Alpha 7–11 s/cell;
  Slashdot 26–34 s/cell at d=1,2 (d=4 OOM).
- **Wall:** ~5 min total.
- **GPU:** RTX 2070 SUPER 8 GiB.
- **JSONL:**
  `signedkan_wip/experiments/results/outer_hsikan_gomb_residual_2026_05_20.jsonl`
  (15 successful cells).
- **Baselines:**
  - Plain Gömb 3-seed at the same config from
    `signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl`
    (depth=1 cells = plain `HymeKoGomb`).
  - Substitute outer-HSIKAN 3-seed from
    `signedkan_wip/experiments/results/outer_hsikan_gomb_overnight_2026_05_20.jsonl`.

## Acceptance check

- [x] Plan in 4 formats on disk (from yesterday's substitute
      version, same architecture).
- [x] CORE.YAML items touched = 0.
- [x] 10 / 10 unit tests + no regression on the broader
      suite.
- [x] Bitcoin Alpha grid complete; Slashdot d∈{1,2} complete
      (d=4 OOM same site).
- [x] **Bitcoin Alpha d=4 lifts AUC +0.0062 over plain Gömb
      at 4.04σ paired, 3/3 wins.** First architectural
      positive on signed-graph datasets in this session.
- [x] **Residual vs substitute paired Δ is +0.0086 at 5.77σ
      on BA d=4 and +0.0089 at 11.53σ on Slashdot d=2** —
      the composition fix was the missing piece.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
- [x] Memory updated.
