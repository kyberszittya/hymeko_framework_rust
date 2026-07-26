# R2 — HyMeKo relational update-0: does relational *organisation* extract the held-out rule?

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r2-relational-update0` · **Start commit:** `9d6fc3d5`
(R2 graph adapter) · **Parent design contract:** `8c16076c` (FROZEN R2 relational contract).

**Result in one line:** at a matched parameter budget and with *everything* else frozen, reorganising the R1 v3 canonical
information as an explicit HyMeKo typed relational graph reproduces the flat R1 result **exactly** — dev 2/2, held-out
0/2, total **2/4** — so **`RELATIONAL_ORGANISATION_ALONE_INSUFFICIENT`** (decision **Case C**). SAC/TD3 remain **blocked**.

---

## 1. Frozen baselines this run rests on (not re-measured)

**Physical feasibility (closed, `coin-physical-feasibility-closed` / `a3459629`).** A HyMeKo-structured slew-admissible 6-D
torque option with phases **PUSH → BRAKE → RELEASE** and velocity-feedback braking delivers genuine frozen-K6 on all four
cradles (dev s1=14250, s3=14750; held-out s4=15000, s7=15750). Frozen K6: zone 0.02 m, settle 0.06 m/s, dwell 6 steps.
Teacher/search/physics oracle = 4/4.

**Flat R1 learned baseline (`c70c844a`, `FLAT_R1_LEARNED_AMORTISATION_FAILS`).** `KHeadProposalNet(K=4, feat_dim=43,
h=128)` = **25 240** trainable params, dev-only LODO `{1:0, 2:0, 4:1}` → K_main=4, panel **dev 2/2, held 0/2, total 2/4**,
final set-loss 0.4516. R1 canonicalisation is load-bearing diagnostically (nearest-acceptable distance R0 0.874 → R1
0.697) but the flat learned representation stays at the 2/4 ceiling.

## 2. The single scientific question (isolated axis)

> Can HyMeKo relational **organisation** extract the held-out control rule from the *same physically correct information*
> where the capacity-matched **flat** R1 model could not?

Only one axis moved: **flat 43-D vector → typed HyMeKo relational graph**. No physics, information, action, data,
optimisation, search, or evaluation axis changed (audited in §3 and `contract_audit.json`).

## 3. Contract audit (Stage 0) — no mismatch

`contract_audit.json` confirms all 10 checks: every one of the 25 R1 v3 groups has exactly one recovery home in the graph
(43-D parity, tested both ways); the graph is built from the **verified R1 canonical extractor** (`build_graph =
build_graph_from_canonical(*canonicalise(r1_grouped_features(snap)))`), **not** the env's `node_features()`; the only
non-R1 attributes are two documented **constants** (COIN phase placeholder = 0 at t=0; TARGET zone spec = CENTER_TOL /
SETTLE_VEL / HELD_DWELL — constant across every cradle → zero per-state signal); frozen dev seeds / panel / held-out
discipline / K_main=4 / budget-8 / centre-inclusion / canonical θ decode (balance θ[2] sign) all verified. **CORE.YAML
items touched: none.**

### Information-parity map (unchanged from the frozen contract)

| node | type | attributes (all from the R1 canonical extractor) | dim |
|---|---|---|---|
| `coin` | COIN | dtz, coin_vel_along/perp (+ phase const 0) | 4 |
| `target` | TARGET | CENTER_TOL, SETTLE_VEL, HELD_DWELL (constant) | 3 |
| `tip_{L,R}` | TIP (tied) | tip_coin_along/perp | 2×2 |
| `contact_{L,R}` | CONTACT (tied) | normal_along/perp, vrel_normal/tangent, fn, friction_util | 2×6 |
| `port_{L,R}` | PORT (tied) | slew_head_up/dn, prev_tau_arm, btau_side_auth | 2×4 |
| goal | coin→target | dtz, coin_vel_along/perp | 3 |
| authority | port→contact→coin | btau_svals/summary, forward_push/reverse_reach, brake_opposed_reach, bcoin_min_attenuation, lateral | 2×11 |
| **bimanual** | {contact_L,contact_R,coin,target} | straddle, normal_force_reach_pair, balance_reach_signed, combined push/brake, lateral pair | 8 |

## 4. Encoder (Stage 1) — capacity-matched typed message passing

`hymeko_rl/coin_delivery/theta_option/relational_encoder.py` — `RelationalKHeadNet(k=4, h=25)`:

```
typed node encoders  →  EXACTLY 2 typed message-passing rounds  →  coin+target+bimanual pooling  →  K-head (Tanh→θ box)
```

- **Tied weights** for the two TIP / CONTACT / PORT sides (one MLP per node *type*, applied to both rows) — the net cannot
  re-learn the arbitrary L/R label; combined with the canonical graph this gives **mirror + side-permutation invariance**.
- Relation-type-specific linear messages: `geom` (tip→contact), `auth` (port→contact, +authority attrs), `prop`
  (contact→coin), `goal` (coin→target, +goal attrs), and the **explicit bimanual hyperedge** {contactₗ, contactᵣ, coin,
  target} carrying straddle / combined push-reverse-brake / lateral-spin / squeeze-internal-force / L/R-balance context.
- **SUM/MEAN aggregation only, no attention, no recurrence** (the two rounds carry *separate* weights), **no residual
  policy** outside the K-head, **no access to physical side identity**. The whole graph is **not** flattened before message
  passing (encoders → 2 rounds → pool). Output = Tanh → `ThetaBox.denorm` (legal canonical θ), the **same** K-head
  acceptable-set semantics and `set_loss` as flat R1. Deploy is a drop-in: `RelationalKHeadProposal.modes(graph)` +
  `relational_deploy_one` reuse the unchanged budget-8 centre-inclusive `fixed_search_select`.

## 5. Parameter budget (Stage 2) — `parameter_budget.json`

| model | width | trainable params | vs R1 |
|---|---|---|---|
| **flat R1** `KHeadProposalNet` | h=128 | **25 240** | — |
| **R2 primary** `RelationalKHeadNet` | **h=25** | **25 774** | **+2.12 %** (window [23 978, 26 502] ✓) |
| R2 secondary (same-hidden-width) | h=128 | 580 120 | +2199 % — reported, **NOT deployed** |

H=25 is the *unique* integer width inside ±5 % (H=24 = −5.20 %, H=26 = +9.70 %). Per-module: node encoders 600, the two
message-passing rounds **22 750** (88 % of the budget — genuinely relational, not a flat MLP), K-head 2 424. The primary
model is the binding capacity-matched deploy; an R2 win could not be dismissed as "just a bigger model".

## 6. Correctness tests + lint (Stage 3)

`hymeko_rl/tests/test_coin_relational_encoder.py` — **10/10 pass** (9 fast in 0.9 s + the physical deploy smoke in 22 s):
(1) information parity + only-documented-constants, (2) graph & encoder mirror-equivalence with flag flip, (3)
node-permutation invariance, (4) tied side weights, (5) θ-output equivariance incl. balance θ[2] sign reversal, (6)
parameter budget within ±5 %, (7) deterministic forward, (8) K=4 bounded-θ K-head contract (no aliasing), (9)
search-provenance (budget-8, centre inclusion, decoded centres), (10) physical deploy smoke (legal bounded θ reaches the
executor, no safety bypass). **ruff: clean.** Regression: the 62 existing canonical-frame / acceptable-set / theta-option /
relational-graph / multimodal / directional-authority tests still pass. Complexity: all new functions ≤ CC 11 (< the
fail-at-15 gate; radon).

## 7. Dev-only training + pre-panel gates (Stage 4) — `training.json`

Byte-for-byte the R1 `fit_khead` contract, graph inputs only: same 6 dev cradles, same acceptable-set harvest (`n_global=600`,
deterministic seeds → identical targets to R1), same canonical θ labels, Adam lr=1e-3, **1500 epochs**, seed 0, same
`set_loss`, output bounds, normalisation. **Held-out s4/s7 never entered training, validation, selection, or tuning.**

Pre-panel gates (all **PASS**): finite loss (0.4639, vs R1 0.4516), bounded Tanh outputs, deterministic checkpoint reload,
non-constant embedding, and a graph embedding that **responds to relation changes** (zeroing authority+bimanual moves the
heads by Δ=0.130). Dev-only LODO diagnostic (recorded, *not* a re-selection — K stays frozen at 4): **`{1:0, 4:1}`** —
identical to R1's `{1:0, 2:0, 4:1}`.

## 8. One frozen panel (Stage 5) — `frozen_panel.json`

Pipeline per state: physical state → frozen canonical R2 graph → relational encoder (reloaded `r2_khead_K4.pt`) → K=4
canonical θ centres → inverse T_θ → **unchanged budget-8 centre-inclusive search** → 6-D PUSH→BRAKE→RELEASE option →
frozen K6. Every state used exactly **8** candidate rollouts; motion within contract (peak q̇ ≤ 2.03 ≤ 3.0); peak RSS 0.32 GB.

| state | split | K6 | head | sw | dtz start→end (mm) | zone | v_term | q̇ₚₖ | nearest dev-acc | failure |
|---|---|:--:|:--:|:--:|---|:--:|:--:|:--:|:--:|---|
| **s1** | dev | ✅ | 0 | F | 76.4 → **19.4** | ✓ | 0.00 | 2.00 | 0.203 | — |
| **s3** | dev | ✅ | 0 | T | 99.9 → **17.7** | ✓ | 0.00 | 2.03 | 0.252 | — |
| **s4** | held-out | ❌ | 2 | F | 96.1 → 69.8 | ✗ | 0.00 | 2.00 | **0.404** | NEVER_REACHED_ZONE |
| **s7** | held-out | ❌ | 2 | T | 137.7 → 60.9 | ✗ | 0.00 | 1.10 | **0.423** | NEVER_REACHED_ZONE |

K=1 side-diagnostic: total 2/4 (same as R1's K=1 side).

## 9. Comparison with flat R1

| | flat R1 (h=128, 25 240 p) | relational R2 (h=25, 25 774 p) |
|---|---|---|
| dev-only LODO | {1:0, 2:0, 4:1} | {1:0, 4:1} |
| panel dev K6 | 2/2 | 2/2 |
| panel held-out K6 | **0/2** | **0/2** |
| panel total | **2/4** | **2/4** |
| held-out failure mode | NEVER_REACHED_ZONE (dtz 64.7/67.2 mm) | NEVER_REACHED_ZONE (dtz 69.8/60.9 mm) |
| `improves_over_r1` | — | **False** |

The two curves are the same curve. On the held-out cradles the relational proposal lands **far from any dev-acceptable θ**
(0.40 / 0.42 normalised) — the same out-of-distribution single-region regression signature the flat model showed. Plot:
`reports/2026-07-27-coin-r2-relational-update0/r2_panel.png`.

## 10. Honest decision-tree verdict

**Case C — `RELATIONAL_ORGANISATION_ALONE_INSUFFICIENT`.** Development is preserved (2/2), the pipeline and provenance are
valid (pre-panel gates pass, budget-8, motion ok), so this is a *genuine scientific negative*, not an implementation
failure (Case D excluded), and R2 does **not** improve over the flat R1 2/4 / held 0/2 baseline (Case B excluded — its
"improves generalisation" wording would be dishonest here).

**What the result means (measured vs inferred).** *Measured:* at matched capacity and matched everything-else, HyMeKo
relational organisation of the R1 v3 information yields the identical 2/4 panel and identical LODO to flat R1. *Inferred:*
the held-out bottleneck is therefore **not representation organisation** — the flat model already had the information and
the relational model already has the invariances; both fail the *same* held-out cradles the *same* way. *Consistent with*
the arc's prior findings `FLAT_R1_LEARNED_AMORTISATION_FAILS` and `COVERAGE_ALONE_INSUFFICIENT`: a single-θ (or K-mode)
regressor amortised from the dev acceptable sets does not transfer to held-out cradles because its proposal is OOD there
(large nearest-acceptable distance), and no amount of re-organising the *input* fixes a *decoder-that-must-extrapolate*
problem.

- **SAC/TD3 authorisation: BLOCKED.** Case A (4/4 incl. held-out 2/2) is the only thing that authorises matched SAC/TD3;
  R2 did not reach it.

## 10.5 Frozen search-basin audit (discriminating test before naming the axis) — `basin_audit.json`

Before calling this a *decoder* problem, one number had to be audited (raised by C. Hajdu): the §9 distances 0.40/0.42
were to the **dev** acceptable set, not to the **held-out working** set — so the failure could still be *search-basin
geometry* (anisotropic reach, a narrow disconnected basin, or the K×(8/4) allocation not filling the metric ball) rather
than a wrong-basin proposal. `--r2-basin-audit` runs the **training-free, frozen** discriminating test: reconstruct the
frozen R2 actor θ, harvest the **held-out** acceptable set (eval-only), measure the actor→teacher / actor→nearest-working
gap in **SEARCH_STD (0.15) units**, and sweep θ(α) = (1−α)·θ_actor + α·θ_teacher through the direct centre and the budget-8
search. It changes nothing in the frozen result.

| held-out | actor→teacher (norm) | ×SEARCH_STD | n working θ (600 harvest) | budget-8 first K6 | direct first K6 | dominant per-component gap (σ) | mechanism |
|---|--:|--:|--:|--:|--:|---|---|
| **s4** | 1.216 | **8.1** | **0** | α = **1.0** | α = 1.0 | squeeze +6.1, forward −3.7, release +2.8, ramp +2.4 | ACTOR_OUTSIDE_CAPTURE_BASIN |
| **s7** | 0.798 | **5.3** | 2 | α = **0.8** | α = 1.0 | balance −3.9, squeeze −2.8, release +1.9 | ACTOR_OUTSIDE_CAPTURE_BASIN |

**Overall: `PHYSICAL_INTENT_DECODER_AUTHORISED`.** The three alternatives are ruled out by the numbers: (a) *not* search
allocation/anisotropy — the gap is **5.3–8.1 search-stds**, orders beyond any budget-8 cloud (7 jittered samples at σ=0.15
cover ≈1 std), and it is **multi-dimensional** (squeeze, forward, balance, ramp, release all off by 2–6 σ), not one narrow
anisotropic axis; (b) *not* a narrow basin a tiny move restores — budget-8 first re-delivers only at **α = 0.8 / 1.0**, not
α ≈ 0.1; (c) even the **nearest of the 4 heads** is 5.9 σ (s4) / 5.3 σ (s7) from the teacher, so no mode points near a
working θ. The frozen R2 actor genuinely targets the **wrong basin** on held-out cradles. Plot: `basin_audit.png`.

*Honest caveat (measured):* s4's held-out harvest returned **0** motion-compatible delivering θ in 600 global samples even
though the teacher θ delivers frozen-K6 in the sweep — i.e. s4's teacher solution sits at/over the harvest motion limit and
its motion-compatible working set is essentially a point; the robust s4 number is therefore the actor→teacher distance
(8.1 σ), not a nearest-working distance. This does not change the verdict (the actor is far from even that point) and is
consistent with the arc's `REALISTIC_MOTION_CONTRACT` note that some coin solutions ride the dynamics limit.

- **Exact next action (authorised by the audit; still needs a fresh frozen contract before building):** **canonical
  structured state → physical intent → deterministic authority-aware decoder → the unchanged 6-D θ option → same budget-8
  search → frozen K6.** The *intent* should be physical and cradle-agnostic (desired forward impulse, desired peak coin
  velocity, lateral correction, squeeze/contact retention, brake-entry condition, braking impulse, release condition), and
  the *decoder* should compute physical θ from the measured B_coin, contact-authority, slew-headroom and geometry — the
  hypothesis being that the right physical *intent* generalises where the concrete torque-option parameterisation does not.
  This is a **decoder** axis, not a representation axis. **SAC/TD3 remain BLOCKED.**

## 11. Files touched (no CORE.YAML items)

| file | Δ |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/relational_encoder.py` | +249 (new) |
| `hymeko_rl/tests/test_coin_relational_encoder.py` | +204 (new) |
| `hymeko_rl/experiments/coin_theta_rl_benchmark.py` | +233 `--r2-update0` mode + helpers, +≈150 `--r2-basin-audit` mode |
| `hymeko_rl/experiments/_r2_panel_viz.py` | +55 (§9 panel plot) + ≈35 (basin plot) |
| `reports/2026-07-27-coin-r2-relational-update0/` | contract_audit / parameter_budget / training / frozen_panel / r2_update_zero / **basin_audit** .json, r2_checkpoint.pt, r2_khead_K{1,4}.pt, r2_panel.png, **basin_audit.png** |

No §6.5 anti-patterns introduced (the mode is a flag on the one benchmark harness, not a v-file; the deploy path is the
shared `relational_deploy_one`, not a copy of `_r1_deploy_one`; no globals; string→enum n/a). Plan-of-record: the frozen
contract at `8c16076c` (`reports/2026-07-27-coin-r2-relational-contract.md`); pdflatex/lualatex are absent on this host, so
the four-format docs/plans bundle was not built (recorded in `contract_audit.json`, not silently skipped).

## 12. Provenance

Start commit `9d6fc3d5` · branch `recovery/coin-r2-relational-update0`. Deploy checkpoint `r2_khead_K4.pt`
(= `r2_checkpoint.pt`). Seeds: train seed 0; deploy RNG `90000 + i·131 + K`; harvest
per-cradle seeds = dev seed. Env: `.venv` torch 2.12.0 (cu-pin), NumPy 2, mujoco; Apple-Silicon CPU (`torch.set_num_threads(1)`).
Wall 396.5 s; peak RSS 0.32 GB (≪ 16 GB cap). Working tree: only the documented pre-existing untracked `.pt`/viewer/report
artefacts plus this run's outputs.
