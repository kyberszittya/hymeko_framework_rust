# Multimodal update-0 (M1+M2) — does a K-head acceptable-set proposal recover held-out?

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-acceptable-set-proposal` · **Base:** tag
`coin-coverage-alone-insufficient` (`5fbf3c16`).

## Why a K-head proposal

M0 proved the blocker is **modality, not representation**: averaging valid modes fails (4/6 dev acceptable-set centroids
non-delivering), and the single-θ regressor collapses both held-out proposals into one wrong region. The fix — emit K
distinct legal-θ modes instead of one interpolated centre, and let the fixed budget-8 search reach a delivering mode.

## Model (M1)

- `KHeadProposalNet`: shared B0-feature trunk (42 → 128 → 128) → K bounded 6-D heads (Tanh → `ThetaBox.denorm`, always
  legal). Uniform mode prob → the budget split is exactly the even K×(8/K). Implements
  `option_rl.MultimodalProposalPolicy` (K=1 recovers the single-θ B0).
- **Permutation-invariant acceptable-set loss** (per state): bidirectional Chamfer — **recall** (every acceptable mode is
  covered by a head) + **precision** (every head lands on a real acceptable θ, no between-mode average) — plus a
  head-collapse hinge **gated** on the state's acceptable set being multimodal.
- Training targets = the DEV acceptable sets M0 harvested (frozen-K6 ∧ motion-compatible; deterministic, identical to M0).

## Deploy (M2) — fair, centre-inclusive

Total physical budget **8 for every K**. Split across modes by `option_rl.allocate_budget` (K×8/K), then the coin's
**centre-inclusive** `fixed_search_select` runs around **each** mode centre with its allocated budget (the generic
`MultimodalBudgetSearch`/`FixedBudgetSearch` never evaluate a mode centre for n≥2 — centre-inclusion is the coin's
established correctness fix and must hold per mode so the learned mode-centres are actually tried). Global argmax of the
frozen delivery score wins; the selected mode's centre is the Bellman action, θ_exec is provenance.

## Protocol (clean, dev-only selection)

- **K selected by DEV-ONLY leave-one-dev-out CV** (physical held-out-DEV K6 at budget 8). No panel / held-out
  involvement in selection. Tie → smaller K.
- Freeze K*, arch, loss, seed; retrain on all 6 dev; **one** frozen-panel `{s1,s3,s4,s7}` evaluation.
- Held-out s4/s7: eval-only, never in any training or selection.

### Hard gate

`MULTIMODAL_UPDATE_ZERO_PASS` **iff** frozen K6 = 4/4, held-out = 2/2, search budget = 8, no motion regression, no
held-out fitting, exact head/θ₀/θ_exec provenance. **Only this authorises SAC/TD3.**

## Results

Artifact `reports/2026-07-27-coin-multimodal/multimodal_update_zero.json`, figure `multimodal_update_zero.png`. Wall
338 s, peak RSS 0.33 GB. Dev acceptable-set sizes: s1 24, s3 12, 16500 4, 17750 68, 19500 10, 24000 3 (all multimodal
except 24000).

**Frozen run (K selected by DEV-only LODO-CV = K\*=2):**

- **DEV LODO-CV** (physical held-dev K6): K=1 → **0/6**, K=2 → **1/6**, K=4 → **0/6** → K\* = 2.
- Frozen-panel deploy (budget 8): dev **1/2** (only s1) · held-out s4 **0** · s7 **0** · total **1/4**.
- Proposal-only (K centres, no jitter): dev 1/2, held 0/2. **GATE FAIL**, `authorises_sac_td3 = False`.

**Diagnostic ablation (K × diversity — audit of the dev regression, `scratchpad/ablate.log`):**

| K | diversity | dev K6 | s4 | s7 | total | note |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 2 | 0.0 | 1/2 | 0 | 0 | **1/4** | s3 lost (head-to-own-acc 0.38 — no head on its delivering basin) |
| 2 | 0.5 | 1/2 | 0 | 0 | **1/4** | diversity penalty is **not** the cause |
| 3 | 0.0 | 2/2 | 0 | 0 | **2/4** | s3 recovered (head on its θ, 0.001) — **= single-θ B0** |
| 4 | 0.0 | 2/2 | 0 | 0 | **2/4** | held-out still 0/2 |

Two clean facts: (1) the K=2 dev regression is **head capacity** (s3's acceptable set spans ~8 basins per M0; 2 heads
cannot cover it — the diversity penalty is exonerated); (2) **held-out is 0/2 for every K ∈ {2,3,4} and every diversity
setting.** At best (K≥3) the K-head only *matches* the single-θ B0 (2/4, held 0/2); it never touches held-out.

## Verdict

**MULTIMODALITY_PRESENT_BUT_UPDATE_ZERO_STILL_FAILS** — gate FAIL, `authorises_sac_td3 = False`. The K-head does **not**
recover held-out at any K, and the LODO-CV (best 1/6) shows the proposal barely generalises even to a held-out *dev*
cradle.

**Decisive synthesis — the blocker is REPRESENTATION, not proposal modality.** M0 was right that the acceptable set is
multimodal and that a single MSE centre averages badly; but M1 shows that *covering* those modes (K≥3, dev 2/2) still
delivers held-out 0/2. So the update-0 held-out failure is **not** cured by emitting more modes — it is the cross-cradle
**feature→delivering-θ mapping** that does not generalise: for a held-out cradle, no dev-trained proposal (single- or
multi-head) places a mode near that cradle's delivering θ (which sits ~0.65 from the nearest dev mode, M0 — beyond the
budget-8, std-0.15 search reach). This is the pre-registered outcome `REPRESENTATION_NOT_PROPOSAL_MODALITY_IS_BLOCKER`,
now reached empirically rather than assumed.

**SAC/TD3 is NOT authorised** (the hard gate is the sole authoriser and it failed).

**Honest caveats.** (a) The frozen run's dev-LODO selected K\*=2, which under-covers s3; the diagnostic sweep shows K≥3 is
the fair capacity, and it too fails held-out — so the K choice does not change the verdict. (b) This is the first,
deliberately simple K-head with a Chamfer set-loss; a richer proposal is possible, but the ablation already shows the
ceiling (matching B0) is a *modelling* wall on held-out, i.e. a representation wall, not a proposal-capacity one.

### Next (the axis this points to)

The evidence has now excluded **coverage** (COVERAGE_ALONE_INSUFFICIENT) and **proposal modality**
(this report). The remaining axis is the **feature representation / conditioning**: the decision-time features must be
made to *discriminate* which delivering θ a novel cradle needs (e.g. richer geometric/contact conditioning, a
cradle-relative canonicalisation, or a similarity-retrieval proposal that maps a novel cradle to its nearest dev
delivering solution), so that a proposed mode lands within search reach of the held-out delivering θ. This is a
separate, gated investigation — not RL.

## Tests

`hymeko_rl/tests/test_coin_multimodal_proposal.py` — 7 fast (permutation-invariant loss; penalises the between-mode
average; diversity only on multimodal states; legal uniform modes; the even K×(8/K) split; K=2 training covers a two-mode
set) + 1 slow physical (budget-8 split [4,4], centre-inclusive, delivers on s1). All pass; ruff clean.

## Files touched

- `hymeko_rl/coin_delivery/theta_option/multimodal_proposal.py` (new).
- `hymeko_rl/experiments/coin_theta_rl_benchmark.py` (`--multimodal` mode + viz).
- `hymeko_rl/tests/test_coin_multimodal_proposal.py` (new).

**CORE.YAML items touched:** none (reuses `option_rl.allocate_budget`, does not modify `option_rl`). **Performance:**
peak RSS 0.33 GB (« 16 GB cap); wall 338 s (frozen run) + ~330 s (diagnostic ablation). Single-threaded.
