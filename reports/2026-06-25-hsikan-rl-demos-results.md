# HSiKAN RL: demos, results, and an honest verdict (2026-06-25)

## Executive summary
This session produced (a) three working, presentation-ready manipulation/control demos; (b) one genuine positive
result (the collaborative reframe is parameter-efficient); (c) a substantial theory (gauge-theoretic
signed-hypergraph learning + the semiring convolution); (d) a working structural diagnostic; and (e) two confirmed
*negatives* that are themselves findings (off-policy / flat-PPO divergence on structured policies and harder
tasks). **The central scientific claim — HSiKAN beats a params-matched MLP — is not established; it is open, and
the discriminating test (quadruped) is blocked by training instability, not by the architecture.** Honest verdict:
good demos + theory + tooling + one efficiency result; no "HSiKAN wins" headline yet, but a clear, well-motivated
path to one.

## 1. Demos (presentation-ready; each = numbers + plot + GIF)
| demo | metric | HSiKAN | MLP | artifact |
|---|---|---|---|---|
| Inverted pendulum (SAC) | curve-max return /200 | **200.0** (solved) | 200.0 | `reports/gifs/cartpole/cartpole_sac_hsikan.gif` |
| Collaborative coin-toss (BC) | delivery rate | **0.208** @ 20.5k params | 0.25 @ 30.0k | `reports/gifs/collaborative/coin_toss_collab_ctde.gif` |
| FANUC pick-and-place (scripted) | reach→grasp→lift→place | — | — | `reports/gifs/fanuc_pick.gif` |

The cart-pole *solved* (200/200) and the collaborative two-arm cooperation is visible and delivers; the FANUC pick
is a clean scripted reference. All three carry the §9 graphical output (numbers + plot + animation).

## 2. The central question: HSiKAN vs params-matched MLP
Measured, across tasks — **no HSiKAN win; ties or HSiKAN-worse:**
- **Cart-pole (SAC):** 200.0 vs 200.0 — tie *at the ceiling*. Expected: the 2-vertex cart-pole is the
  structure-not-load-bearing floor; both saturate it. (PPO at the generic budget reached only ~31 — a config
  floor, not a method result.)
- **Galambos, from-scratch (offpolicy_eval, curve-max return):** PPO HSiKAN −214 vs MLP −215 (tie); **SAC HSiKAN
  −248 vs MLP −222 (MLP better).** All negative = tie-at-failure (nothing delivers from scratch; the documented
  hard-exploration wall).
- **Task-graph A/B (BC delivery):** coin/zone-in-graph **helped MLP** (0.139→0.250) and **hurt HSiKAN**
  (0.111→0.042) — falsified the "structure-in-graph pays off" hypothesis *as currently wired*.
- **C3 (quadruped, branching 14-vertex topology) — the real test — BLOCKED** (see §4): SAC diverged before a
  number was obtained.

**Interpretation (honest):** every tie is *consistent* with HSiKAN being better only where topology is
load-bearing (branching/cycles/repetition), but that has not been demonstrated — the one task that would show it
(quadruped) hasn't completed. The structural prior has *not* paid off on the tasks measured.

## 3. The positive result: collaborative reframe is parameter-efficient
The two-arm task recast as cooperative CTDE (one shared HSiKAN backbone → per-arm action heads + centralized
critic, team reward) delivers **comparably to a single monolithic policy at ~⅓ fewer parameters**: BC delivery
0.208 (20,489 params) vs single 0.25 (30,025 params) — within noise on delivery, materially cheaper. This is a
defensible claim: the per-arm head decomposition over shared structural reasoning earns its keep. (5 unit tests;
the reframe is the Galambos/Kato headliner demo.)

## 4. Confirmed negatives — which are findings, not just failures
Both point to the *same* gap: standard RL updates corrupt **structured policies** on **harder tasks**.
- **Off-policy divergence:** DDPG/SAC refine diverges on some seeds/tasks into NaN/denormal arithmetic (≈100×
  slowdown). Seen on a ddpg cell (didn't reproduce in isolation → environmental there), on the controlled-LR
  demo, and — decisively — **SAC on quadruped diverged** (2.5 h, no progress), blocking C3.
- **Flat-PPO collapses the CTDE:** PPO-refine via the standard (flat-action) trainer collapsed the collaborative
  CTDE across 3 seeds (BC 0.208 → refined **0.042**, per-seed 0.0/0.125/0.0), while the single policy held
  (~0.236). The flat update treats the per-arm heads + shared backbone as one vector and corrupts them.
- **Consequence:** the BC versions are the keepers; *strengthening* structured policies needs a **structure-aware,
  stable learner** (per-agent CTDE advantages + TD3+BC anchor / gradient clipping), not flat refine. This single
  build unblocks **both** the quadruped C3 test and the collaborative strengthening.

## 5. Theory and tooling (the durable contributions)
- **Gauge-theoretic signed-hypergraph learning** (`docs/theory/gauge_holonomy_signed_hsikan.pdf`): balance =
  $\mathbb{Z}_2$ holonomy (theorem); HSiKAN = fiber, rotor = connection, spikes = timing; the spiral collects
  parallel-transported fibers at spike-walk ends = a learned continuous holonomy generalising balance — a
  falsifiable AUROC bet (transport, *not* the settled readout). Spans rate/spike/balance/fuzzy/spinor regimes.
- **Semiring convolution** (`docs/theory/hypergraph_convolution_semantics.pdf`): every HyMeKo computation is
  $B\circledast x$; well-defined iff the declared algebra is a commutative monoid (semiring for signed-weighted).
- **`hsikan_diagnose`** (built + 3 tests): a structural health map that localises a divergence to a *named*
  component (e.g. "up-chain signed-agg W- L0") — debuggability an MLP cannot offer. The recurring divergences are
  exactly what wiring this into the training loop would auto-catch.
- **Graphical-output strategy** (CLAUDE.md §9 + `campaign_viz`): every experiment emits numbers + plot + 960×720
  GIF, wired into the campaign drivers.
- **Latency:** HSiKAN single-step ~2.4 ms / ~420 Hz CPU (embeddable); ~10.8× with `torch.compile` on GPU.

## 6. Are these good results? (the honest verdict)
- **Yes** as *demos*: three work, are watchable, and are backed by theory — a strong position for the Kato meeting.
- **Yes** as one *result*: the collaborative parameter-efficiency is real and defensible.
- **Yes** as *foundations*: the gauge/semiring theory and the diagnostic are novel and durable.
- **Not yet** as the *headline*: HSiKAN-beats-MLP is unproven; the measured tasks tie or favour the MLP, and the
  one task that could show a win is blocked by instability. That is the gap to close, not paper over.

## 7. Next step (one build closes the gap)
A **stable, structure-aware learner**: a CTDE trainer with per-agent advantages + off-policy stabilisers (TD3+BC
anchor, gradient/weight clipping, flush-to-zero or NaN-guards), with `hsikan_diagnose` wired into the loop to
abort+localise on divergence. This makes the **quadruped C3 test feasible** (the real HSiKAN-vs-complex-topology
verdict) and lets the **collaborative be strengthened** without collapse.

## Artifacts
GIFs: `reports/gifs/{cartpole,collaborative,fanuc_pick.gif,campaign}`. Plots:
`reports/gifs/campaign/galambos_curvemax_hsikan_vs_mlp.png`, `reports/gifs/cartpole/campaign_curvemax.png`,
`hsikan_diagnose_demo.png`. Theory PDFs: `docs/theory/*.pdf`. Architecture: `docs/plans/2026-06-24-kato-collab-
dual-discriminator/architecture.pdf`. Code: `hymeko_rl/{collaborative,campaign_viz,hsikan_diagnose,structural_
critic,exp_collaborative,offpolicy_eval}.py`.

## Provenance
Git: branch `fix-hsikan`, dirty. CPU-only (`CUDA_VISIBLE_DEVICES=-1`); 16 GB cap respected; per-cell subprocess
isolation after MuJoCo box-box aborts. Seeds as noted per result. Logs:
`reports/2026-06-25-*.{log,json,jsonl}`. Two long runs stopped mid-flight (quadruped SAC, controlled-divergence)
as pathologically slow — diagnosed as divergence, not hung.
