# HyMeKo — Done (completed work)

**Last updated:** 2026-06-21 · Companion: [BACKLOG.md](BACKLOG.md) (open work).

Curated milestone log. The **full archive is `reports/`** (~250 reports — every
completed task has one); this file is the human-scannable highlight reel, newest
first. When a backlog item lands, move its line here with the report link.

---

## 2026-06-21 — Cart-pole HSiKAN actor-critic, vectorized PPO, editor attribute-HUD

- ✅ **PPO is the RL BASELINE (saved).** On-policy clipped-surrogate + GAE, diagonal-Gaussian actor over a
  HyMeKo-described cart-pole (`data/robotics/inverted_pendulum.hymeko`; `hg_state` from the .hymeko). The
  reference for all future RL-algorithm work (DDPG/TD3/SAC): **5-seed, vec N=16, 120 iters** — HSiKAN
  **192.0 ± 15.3** upright-steps/200 (5/5 learn). Artifact: `reports/2026-06-21-cartpole-multiseed.jsonl`.
- ⚠️ **Structure is NOT load-bearing on cart-pole (control overturned the first read).** A params-matched
  MLP (26.7k ≈ HSiKAN's 26.2k) ties HSiKAN: **195.2 ± 8.4, 5/5 learn**; over-param MLP (135k) = 200±0. The
  earlier "MLP fails 3/5 → HSiKAN robustness win" was an **under-parameterized baseline** (9k params), not
  structure — same trap/verdict as the 2026-06-18 rotor-vs-MLP-embed ablation. Caveat: a 2-vertex graph has
  no topology to exploit, so cart-pole is the wrong task to judge the architecture; a fair test needs the
  6-DOF arm or Galambos. Control artifact: `reports/2026-06-21-cartpole-controls.jsonl`.
  `reports/2026-06-21-cartpole-hsikan-wirein.md`
- ✅ **Vectorized PPO rollout — 3.1× faster wall, learning preserved.** Measured the batch-1 forward at 87%
  of the rollout (dispatch-bound, not FLOPs); `_collect_vec` batches the forward over N lock-step envs →
  iter 3.08 s→0.93 s (N=16), 120-iter run 452 s→147 s, upright 144→161. Single-env reach path untouched.
  `reports/2026-06-21-vectorized-ppo-rollout.md`
- ✅ **Provenance + `--save` + auto-labeled render (train→store→render chain).** Stored `.hymeko` carries a
  `provenance {algo,backbone,upright,seed}` block (`policy_to_hymeko(meta=…)`, `read_provenance`); every trainer
  got `--save`; `load_policy_from_hymeko` generalized to reconstruct PPO/DDPG/SAC actors bit-exact (dispatch on
  keys; `GreedyPolicy` Protocol); render self-labels (HUD + `cartpole_<algo>_<backbone>.gif`) from provenance,
  no `--algo`. The gif HUD also shows step/return/pole/cart/force/status. 46-test suite green.
  `reports/2026-06-21-sim-interface-and-t2.md`
- ✅ **SAC — max-entropy off-policy + the entropy-feedback seat.** Squashed-Gaussian actor + twin soft-Q +
  auto-temperature α (`sac.py`, reuses the off-policy scaffolding). **Strongest learner:** solves cart-pole in
  **~4k steps** (curve `196·200·200·200·122` — held 200 8k–16k; final-snapshot metric noisy under late dips,
  curve is the truer signal). The `α·H` term is the explicit seat where the user's **TD-k / entropy-feedback**
  idea lands: 3 interchangeable signals — policy-entropy (SAC, done) · critic-ensemble disagreement (TD-k =
  `n_critics` config) · **structural entropy** (HyMeKo `hymeko entropy`, the novel bet). 4 tests.
  `reports/2026-06-21-sac-and-entropy-seat.md`
- ✅ **TD3 (P3) — as 3 config axes of the DDPG core, not a fork (§6.5#1).** `n_critics=2` (clipped double-Q) +
  `policy_delay=2` + `target_noise` (smoothing); one `train_offpolicy`/`OffPolicyConfig`, DDPG = degenerate
  preset; `--algo {ddpg,td3}`. Honest non-result: on cart-pole TD3 shows **no advantage over DDPG** (single
  seed: TD3 final 114 vs DDPG 199, curve oscillates) — its robustness benefit is multi-seed/harder-task
  territory, which the easy cart-pole can't show. 7 tests + 33-test regression clean. Next: SAC.
  `reports/2026-06-21-td3.md`
- ✅ **DDPG — first off-policy actor-critic (P2) + off-policy RL survey report.** Replay buffer + Q-critic +
  deterministic actor + Polyak targets + DPG/Bellman + Gaussian exploration; same swappable backbone. mlp DDPG
  **27→199 upright, solved in ~8k env steps** (vs PPO's ~2M) — **~250× more sample-efficient** (PPO still wins
  wall, being vectorised). `eval_balance` retyped to a `GreedyPolicy` Protocol (PPO + DDPG share one eval). 11
  tests. Survey: `reports/2026-06-21-offpolicy-rl-survey.pdf` (DDPG/TD3/SAC + REDQ/TQC/DroQ/CrossQ/TD7 +
  safe-RL). Next: TD3 (P3), then SAC. `reports/2026-06-21-ddpg-offpolicy.md`
- ✅ **Runnable+visualizable sim interface + binary storage T0/T1/T2 complete.** `render_inverted_pendulum.py`
  loads a policy **from its `.hymeko`** (architecture inferred from tensor shapes), runs it, renders a GIF
  (cart-pole balancing 200/200) + trajectory PNG — the storage loop closed. Binary tiers: **T0** decimal
  332 KB / **T1** base64 142 KB / **auto** 148 KB / **T2** content-addressed npz blob (**9 KB** .hymeko +
  sha256-verified blob, tamper-rejecting). All bit-exact + valid HyMeKo. 38 tests.
  `reports/2026-06-21-sim-interface-and-t2.md`
- ✅ **`signedkan` — learned-incidence policy (the trained weights ARE the star edges).** `learn_incidence`
  flag makes `a_pos/a_neg` trainable `nn.Parameter`s (init = kinematic); new `"signedkan"` kind. Incidence
  drifts 0.026 from init (genuinely learned), round-trips bit-exact as a valid `.hymeko`. Honest: 39.9 vs
  hsikan 56.8 upright — parity, no structural gain on 2-vtx (mechanism demo, not a perf claim). +fig7. 13
  policy tests. Binary-storage **plan** (T0/T1/T2 tiers) written, not built.
  `reports/2026-06-21-signedkan-learned-incidence.md`
- ✅ **Trained policy stored AS a HyMeKo hypergraph (storage thesis → artifact).** A weight matrix is the
  star expansion of a weighted hypergraph (incidence $B_{ij}=W_{ij}$), so a trained cart-pole HSiKAN
  actor-critic round-trips `state_dict ⇄ .hymeko`: 332 KB **valid HyMeKo**
  ([data/nn/cartpole_hsikan_policy.hymeko](data/nn/cartpole_hsikan_policy.hymeko)), **bit-exact** (max |Δ|
  0.00 over 26 259 weights), reconstructed policy eval identical. 6-figure battery in
  `docs/figures/2026-06-21-policy-storage/`. `hymeko_rl/policy_store.py`, 11 tests.
  `reports/2026-06-21-policy-weight-storage.md`
- ✅ **Editor: hyperedge-on-hyperedge rendering** (strategy.hymeko bundles now visible) +
  **attribute-on-HUD folding** (leaf value-decls fold onto a node's HUD with values; `NodeDto.value` +
  WASM rebuild). `reports/2026-06-21-editor-hyperedge-on-hyperedge.md`, `reports/2026-06-21-editor-attribute-hud.md`

## 2026-06-18 — Rotor signed-link: input enrichment + inductive transfer + the rotor ablation

- ✅ **Strengthened inductive transfer — the learned increment clears σ on harder pairs.** 3-arm decomposition (real / shuffle-gate / random-init floor) on train-small (bitcoin) → eval-large (slashdot/epinions), 5-seed. The learned-from-source increment (real − max(shuffle, random-init)) is **positive in 8/8 cells and ≥ 1σ_shuffle in 7/8**, reaching **+2.90σ** on the cleanest pair (`bitcoin_otc→epinions` +0.0261, σ_sh 0.009) — promoting the bitcoin-pair's borderline "~1–1.5σ, suggestive" to a real result. Key insight: **variance, not effect size, separates the train graphs** — otc-trained shuffle arms are tight (σ_sh≈0.009 → +2.3–2.9σ) while alpha-trained are noisy (σ_sh≈0.05 → ~1.1σ despite a *larger* raw increment). Random-init stays at/below chance everywhere, so learned = real − shuffle. Engineering: consolidated the previously-ad-hoc 3-arm decomp into the canonical driver (`Arm` enum + unified `run_grid`, §6.5 #3/#13) and **fixed a latent resumption-key collision** (real↔random-init shared a key → random-init silently dropped on resume), pinned by a regression test. Peak RSS 1.9 GB (heaviest cell). `reports/2026-06-18-transfer-grid-strengthen.md`
- ✅ **Is the Cayley-rotor geometry load-bearing? — NO (foundational ablation).** `MLPEmbedSignedModel` control: identical proj/SGCN/classifier, rotor embedding swapped for a higher-capacity MLP of the same output dim (16,661 ≥ 15,761 params — generous). 5-seed deduped gated: rotor − MLP = ±0.002–0.005 (all inside σ 0.01–0.027), both leakage-clean. So the rotor's S³ geometry adds nothing measurable on signed-link prediction (consistent with the head ablation: rotor *algebra* not load-bearing at the readout either). **The line's wins (leakage-free, inductive, param-light) come from replacing the node-ID table with structural features — NOT from the rotor; the MLP-embed control has all three too.** Reframes the contribution as structural-feature signed-link prediction. `reports/2026-06-18-rotor-vs-mlp-embed-ablation.md`
- ✅ **Inductive transfer test (nuanced positive — the line's distinctiveness validated).** Cross-graph transfer (train weights on graph A, evaluate frozen model on B's strict deduped split). **Mechanism + discrimination win:** the rotor transfers (0.81–0.86 AUROC on the *unseen* graph) while transductive `dadsgnn` **cannot** (`nn.Embedding(n_A)` can't index B). The naive shuffle-A gate is confounded (B's real signed adjacency carries the signal), so a **random-init control** was added (below chance, 0.38–0.47 — the true floor); the learned-from-A increment is **+0.038–0.063** (4/4 positive, ~1–1.5σ — suggestive, not strong). Bulk of transfer AUROC is the eval graph's own structural signal, exposed by training but not A-specific. Train loop refactored into a reusable, behaviour-preserving `_train`. `reports/2026-06-18-inductive-transfer-test.md`
- ✅ **Leakage-free input enrichment — `StructuralFeature` registry + exact k-walk signed `A^k` profile (new).** Extensible Strategy registry (degree/cycle single-sourced; new k-walk profile via sparse `A^k` mat-vecs, cap-free; new clustering ratios). 5-seed deduped gated: the walk profile lifts the **non-propagated** rotor baseline (audit `cyc+walk` otc **+0.0168**, gate-clean) but is **parity on the slerp-propagated line** (+0.003 otc) — slerp propagation already performs signed-`A^k` aggregation, so the feature is redundant there. A seed-0 otc 0.9131 (apparently beating pure SiGAT) **did not replicate** (5-seed 0.882) — recorded as variance, not a win. The residual gap to pure SiGAT (0.895) is **expressivity** (learned attention), not input features. `reports/2026-06-18-leakage-free-input-enrichment.md`

## 2026-06-17 — Rotor signed-link: the input fix (computer-graphics × AI)

- ✅ **Adaptive rotor propagation (neutral / knob-removal).** Phase 1 learnable per-block self-retention `α_b = retention_floor + exp(θ_b)` (log-space — softplus is for heads, not scale-free manifold mixing; ≡ a per-block sigmoid residual gate): 5-seed A/B **parity** with fixed sw=4 (alpha +0.0002 / otc −0.0010, gates clean) → removes the per-dataset `sw` knob at no AUROC cost. Phase 2 depth scan {2,3,4,6}: **no gain** (alpha over-smooths past r2; otc flat). Third independent confirmation the val ceiling is **input-bounded** (degree-only STRUCT_DIM=6), not retention/depth/readout. Next numbers lever = leakage-free input enrichment. `reports/2026-06-17-adaptive-rotor-propagation.md`
- ✅ **Protocol-matched honest SiGAT comparison — the "0.04 gap" was a dedup artifact.** `--dedup` true-held-out filter added to `run_baseline_audit` (single-sourced into the datasets layer); 5-seed grid, 4 models × 2 Bitcoin graphs × {deduped, non-deduped}, all shuffle-gated. On the matched **deduped** protocol the gap **inverts vs `sigat_rotor`** (0.833/0.868 < our 0.850/0.879) and the real residual is vs **pure SiGAT** only (+0.036 alpha / +0.016 otc). The non-deduped protocol leaks under shuffle (gate ⚠) → deduped is the honest protocol. Deduped rotor numbers reproduce the slerp report exactly. `reports/2026-06-17-protocol-matched-sigat.md`
- ✅ **Signed slerp/nlerp rotor propagation — first confirmed lift on the leakage-free rotor line.** Diagnosed the ~0.04 SiGAT gap as **input-bounded** (rotor input was a 6-dim degree-only feature; `val` pinned across 4 readout levers). Fix = propagate rotors over the signed graph on S³ (nlerp = "interpolate normals then renormalise"; edge-signed). r2 sw4: alpha 0.8455→**0.8500** (+0.0045) / otc 0.8685→**0.8790** (+0.0105), 5-seed, gates ≈0.5. `reports/2026-06-17-signed-rotor-slerp-propagation.md`
- ✅ **Link-head ablation (fair, on the same propagated rotors)** — real bilinear > complex > geodesic > quat (5-seed, gaps ~0.13, not a param artifact). Expressivity beats algebra at the readout; the head is **not** the gap. Side finding: real-on-q alone ≈ full triad-encoder pipeline. `reports/2026-06-17-link-head-ablation.md`
- ✅ **Diagnostic chain that localised the bottleneck** — geom-attn gate-collapse, woken score, rotor-relative projection, k=4 Berge cycles (all flat, all leakage-clean). `reports/2026-06-17-{geom-gate-inspection,geom-attention-wake-score,rotor-relative-projection,berge-kcycle-rotor,hsikan-rotor-sota-levers}.md`
- ✅ **Editor 3D hyperedge labels** — were never rendered (only vertex sprites); now sign-tinted at the member centroid, toggle with Labels, all modes. `docs/editor/views/hypergraph3d.js`
- ▶ **Next:** protocol-matched honest SiGAT comparison + adaptive propagation / propagation-as-encoder — plans saved, see BACKLOG.md.

## 2026-06-16 — Param-efficiency baseline + framework maturity

- ✅ **DETR baseline fairness guard fixed** — from-scratch MiniDETR could not overfit (box IoU plateaued at 0.46 < 0.5); fix = additive `l1giou` box loss + `lr 1e-3 / grad-clip 0.1` recipe → overfit guard `mAP→1.0`, PASSES (was `xfail`). Unblocks the RicciStim-vs-DETR head-to-head. `reports/2026-06-16-detr-overfit-fix.md`
- ✅ **Generators into `hymeko_core`** (single-sourced; 28 tests) — `reports/2026-06-16-generators-into-core.md`
- ✅ **RicciStim topology cache** (made the 40-ep full vision run feasible) — `reports/2026-06-16-stim-geometry-cache.md`
- ✅ **Full vision run (cached, upgraded RicciStim)** — Cluttered-MNIST config F, 5000 img / 40 ep, **mAP50_proxy 0.228** @ 5 896 params (single seed; beats prior hypergraph-vision number, not a parity claim) — perf log in `ROADMAP.md`
- ✅ **Technology-milestone snapshot** (proven / prototype / program maturity table) — `reports/2026-06-16-technology-milestone.md`

## 2026-06-15 — Editor + hero-demo sprint

- ✅ **Editor: hypergraph examples gallery** (Fano / sunflower / K₄³ / generic) — `reports/2026-06-15-editor-hypergraph-examples.md`
- ✅ **Editor: parametric generators** (Steiner S(2,3,n), sunflower, complete Kₙ⁽ʳ⁾) + progress bar — `reports/2026-06-15-editor-hypergraph-generators.md`
- ✅ **Editor: multi-file imports + vocabulary profiles** (WASM `parse_and_compile_files`, rebuilt pkg) — `reports/2026-06-15-editor-profiles-imports.md`
- ✅ **Editor: arc-ref value editing** (sign/target/value, source-as-truth) — `reports/2026-06-15-editor-arc-values.md`
- ✅ **Editor: 2D composite (roots-centred) layout + 3D pan** (hyper3d) — `reports/2026-06-15-editor-layout-pan.md`
- ✅ **Editor: robot-arm hero-cell profile** (imported kinematics) — `reports/2026-06-15-editor-robot-arm-profile.md`
- ✅ **Editor: 2D bidirectional relations** (reciprocal-pair detection + double connector) — `reports/2026-06-15-editor-bidirectional-2d.md`
- ✅ **HRI profile fix** — relations as signed hyperedges — `reports/2026-06-15-editor-hri-hyperedges.md`
- ✅ **Hero demo Phase 1** (robotics spine: one model → URDF/SDF/MJCF/DOT/Mermaid, gated; broken-twin rejected) — `reports/2026-06-15-hero-demo-phase1.md`
- ✅ **Hero demo Phase 2** (hybrid: robots + learners via `torch_dataflow`) — `reports/2026-06-15-hero-demo-phase2.md`
- ✅ **Hero demo gate hardening** (exit-code-authoritative; corrected a wrong "exit 0" finding) — `reports/2026-06-15-hero-demo-gate-exitcode.md`
- ✅ **Hero demo Phase 3** (Gömb structural parity 6/6 + minimal Soma 3/3, torch-free) — `reports/2026-06-15-hero-demo-phase3.md`
- ✅ **Seminar deck additions** (star/clique tensor-view heatmaps + Reference & Future-work slides; python-pptx approved) — `reports/2026-06-15-seminar-deck-additions.md`
- ✅ **SysML requirements-trace lens** (SMC #5) + **editor SysML lens** — `reports/2026-06-14-sysml-requirements-trace.md`, `reports/2026-06-14-editor-sysml-lens.md`

## Soma-vision line (built; vision hypothesis falsified — see BACKLOG decision)

- ✅ **Base-Soma vs Linear control (MNIST, 2026-06-15)**: walk-conv base-Soma **0.52** vs linear **0.91** (−0.387 paired, all 5 seeds) — walks-only falsified for vision too (2.2× more param-efficient, but far lower absolute). Completes the Soma-vision vision-falsification. `reports/2026-06-15-base-soma-vs-linear-mnist.md`
- ✅ **Holonomy (sign-as-connection) re-test (MNIST, 2026-06-29)**: the 2026-06-15 base-Soma encodes sign as *routing* (dual `W±` banks + sign-blind sum); the StructuralActor/gauge result said sign is a *connection* (σ-product Z₂ holonomy, `M_v(σ⊙m)` = signed Bᴸ) — never run on vision. Added `HOLONOMY` aggregation to GömbSoma `HypergraphConv` (enum + Strategy dispatch, no CORE) and re-ran the exact 3-arm A/B. **Falsification CONFIRMED & tightened:** holonomy **0.4888 ± 0.0093** (1226p) ≤ routing **0.5186 ± 0.0204** (2010p) ≪ linear **0.9056 ± 0.0079**. Both anchors reproduced 2026-06-15 to 4 decimals. Testing the *right* operator returns the same answer — signed walk-holonomy is not load-bearing for MNIST patch-graph vision (its value stays on the control/RL side). `reports/2026-06-29-soma-holonomy-aggregation.md` + figure. 13 new tests, 62 regression green.
- ✅ **Multi-query pool = flatten-matching scalable readout (2026-06-30)**: K learned query vectors each softmax-attend over patches → K·d "content slots" (`_MultiQueryReadout`, K=1=single-query attention; grid-independent). Same GPU+resident harness, 5-seed cluttered ladder: attention(K=1) 0.467(2027p) < spatial-tree 0.553(5227p) < **multi-query(K=8) 0.605(3258p) ≈ flatten 0.605(24890p)**. **Multi-query MATCHES flatten at ~1/8 params + scale-free + variable-size** — the scalable readout we were after (spatial tree = simpler runner-up). Cost: higher seed variance (±0.045 vs flatten ±0.016). Attention arm reproduced earlier 0.466 → harness consistent. Batched + parity-tested. `reports/2026-06-29-soma-spatial-tree-readout.md` (multi-query §) + figure `soma_multiquery_ladder_20260630.png`. 4 tests.
- ✅ **GPU-resident training path + measurement-discipline correction (2026-06-30)**: added `--resident` (materialize cached dataset to device tensors, mini-batch by slicing — no DataLoader) + on-device loss accumulation (sync once/epoch, not per-batch `.item()`). **Lesson:** single-shot timings were noise (laptop RTX3070 thermal/boost; first reads said "resident slower 17.6 vs 12.1s" — wrong). Proper median-of-5 (2000×2ep): DataLoader 6.30s (IQR 0.18) vs **resident 6.01s (IQR 0.04)** → ~5% faster + ~4× lower variance (resident worst < DataLoader median). Honest verdict: after caching, training is **compute-bound (fwd+bwd), not data-bound** — data-side levers ~exhausted; resident gives a small consistent edge. Real remaining lever = mixed precision (fp16 tensor cores). Reinforced §3: single-shot wall-clock is not a measurement.
- ✅ **Batched GPU training + dataset cache (2026-06-30)**: the train/eval loop already feeds batches → with the batched model forward, training runs batched on GPU end-to-end (sparse-mm backward on CUDA verified). **Gradient-parity verified** (batched backward ≡ per-image loop grads, atol 1e-4 → batched training is numerically identical, just faster). Found training was DataLoader-bound (per-item synthetic-image gen starved the GPU); added `cache=True` to `ClutteredMNISTClassification` (precompute deterministic samples to tensors, ~46MB). End-to-end training (gomb_soma_tree cluttered, 3000×3ep): CPU 71.1s → **GPU 12.1s = 5.9×** (was 3.3× uncached). Next lever for more: pinned-memory / num_workers data pipeline. 1 grad-parity test + cache tests.
- ✅ **HolonomyClassifier BATCHED (2026-06-30)**: same `(B,N,d)` treatment applied to the holonomy-group ablation classifier (none/routing/Z2/U1). Batched conv (message einsum `bnki,bnkij`, batched aggregate via M_v column-pack), `_rotate_pairs` generalized to leading dims (U1), batched edge-diffs/signs/flatten. Per-image path kept (gated x.ndim). **Parity-verified** (batched ≡ loop, atol 1e-5, all 4 modes). Bench (canvas48, batch64, U1): CUDA(RTX3070) 250→**6.1ms 41×**; CPU 164→77ms 2.1×. 13 tests.
- ✅ **WalkConvImageClassifier end-to-end BATCHED (2026-06-30)**: whole forward runs on `(B,N,d)` in one pass (grid topology shared) — batched patchify/edge-signs/walk-conv (message einsum gains batch axis; sparse aggregation packs batch into M_v's column axis = one `sparse.mm`) + batched readouts (dim=-2). Per-image path untouched (gated on x.ndim) → RicciStim variable-anchor + holonomy ABC unaffected. **Parity-verified** (batched ≡ per-image loop, atol 1e-5, all readouts + cheby/holonomy). Bench (canvas48, 144 patches, batch64, spatial-tree): CPU 312→152ms **2.1×**; CUDA(RTX3070) 204→**7.4ms 27.7×** (GPU launch-bound→batched). `bench_walk_conv_batched.py` + figure `walk_conv_batched_bench_20260630.png`. Realises the end-to-end GPU win. Report Follow-ups §.
- ✅ **SpatialPyramidPool GPU-optimized (2026-06-30)**: the pyramid is a LINEAR operator — `cells = P @ features` (P = (n_cells,N) row-normalized cell-indicator matrix). All levels → ONE matmul; fixed grid → precompute P (`set_fixed_positions`) → single fused batchable `einsum('cn,bnd->bcd')`; no scatter/level-loop. Replaces the per-image launch-bound loop (B=1 dispatch problem). Bench (N=144,d=16,batch=128, median): CPU loop 28969µs→matmul 938µs **30.9×**; CUDA(RTX3070) 27146µs→341µs **79.6×** (loop ~27ms BOTH devices = launch-bound not compute-bound). `forward` now batched-ready. `bench_spatial_pyramid.py` + figure `spatial_pyramid_bench_20260630.png` + json. Parity test (matmul==scatter), batched==loop test. Systemic follow-up: batch the upstream walk-conv for the end-to-end win. Report Follow-ups §.
- ✅ **Spatial-pyramid = compressed flatten readout — EXPORTED (2026-06-30)**: (a) **gate ablation** — static pyramid 0.558 ≈ dynamic 0.537 (tied) → the learned per-cell gate is INERT; the multi-scale *pyramid structure* is the compressor (gate-free is the keeper, now default). (b) **RicciStim transfer** — pyramid pool over variable anchor *positions* (`_AnchorSpatialTreeReadout`) lifts RicciStim: full-tree **0.292** > mean 0.235 > attn 0.191; structure stays load-bearing (full-tree−enc-tree=+0.080). (c) **Principle + export** — flatten over a spatial field compresses to a multi-scale pyramid (~1/5 params, ~90% acc, + scale-invariance + variable-N). Extracted domain-agnostic `SpatialPyramidPool` (`signedkan_wip/src/vision/spatial_pyramid.py`, input `(features, positions∈[0,1]²)`); both soma grid + RicciStim anchor readouts now delegate to it (no duplication). Export targets: HyMeYOLO detection heads, RL vision, graph/point readouts. `reports/2026-06-29-soma-spatial-tree-readout.md` (Follow-ups §) + figure `soma_ricci_tree_readout_20260630.png`. 35 tests.
- ✅ **Dynamic spatial-tree readout (Cluttered-MNIST, 2026-06-29)** — the scalable position-aware readout we were missing. `_SpatialTreeReadout`: quadtree pyramid (1×1+2×2+4×4 = 21 cells, each mean-pools its region = multi-scale position), learned per-cell activity gate (dynamic), `out_dim=21·d` grid-independent → scales + handles variable anchors (unlike flatten). Cluttered ladder (5 seed×5 ep×5000): mean 0.314 < attention 0.466 < **spatial-tree 0.537 (5227p)** < flatten 0.617 (24890p). **Clears the single-vector plateau, ~87% of the way to flatten at ~1/5 the params** — the best *scalable* readout, and the one that can go into RicciStim's adaptive anchors. `reports/2026-06-29-soma-spatial-tree-readout.md` + figure `reports/figures/soma_readout_ladder_20260629.png`. Next: static-vs-dynamic gate ablation; tree into RicciStim. 5 tests.
- ✅ **Holonomy-GROUP ablation (Cluttered-MNIST, 2026-06-29)**: prior vision test used only Z₂ (sign); user noted holonomy generalizes to any group. Lifted the *same* brightness connection Z₂→U(1) (magnetic: per-edge phase α·tanh(Δbright), walk holonomy Σθ rotates feature pairs), all on the fair flatten readout. **No group is the lever:** none 0.493, routing(Z₂-switch) 0.529, Z₂-connection **0.399 (worst)**, **U(1) 0.510 ≈ none** — all within ~1σ except Z₂-connection which hurts. Discriminating check: U(1) flux α grew 1.0→1.114 (connection *actively used*, not ignored), yet still ≈ none → an active continuous holonomy gives no benefit. Confirms the lever is the readout (position), never the connection group. `reports/2026-06-29-soma-holonomy-group-ablation.md` + figure; `holonomy_walk.py` (9 tests). New non-core, no dep.
- ✅ **Position-aware readout Phase 2 — RicciStim structural contribution (Cluttered-MNIST, 2026-06-29)**: wired attention readout + encoder-only ablation into `RicciStimClassifier` (reuses `_AttentionReadout`; flatten correctly rejected for variable anchors). First pass was confounded (highway off → encoder-only = zero input = chance); **corrected by turning highway ON for all arms** (it carries the encoder features to the head, making encoder-only a real control). Corrected 2×2 (3 seeds×3 ep×2000, highway on): full-mean **0.235**, full-attn 0.191, enc-mean **0.152**, enc-attn 0.141; chance 0.10. **G2 PASSES — structure is load-bearing:** the walk/polygon/triangle branches add **+0.083** (mean) / +0.051 (attn) over the same encoder features (~3σ). Readout: attention ≤ mean again (no lift — consistent with Phase 1 single-vector plateau). Absolute level modest (~0.235 ≈ linear, < WalkConv+flatten 0.62) at this reduced/undertrained scale, but the *relative* structural signal is clean+positive. Figure `reports/figures/soma_phase2_ricci_highway_20260629.png`; jsonl `…_phase2_ricci_highway_20260629.jsonl`.
- ✅ **Position-aware readout Phase 1+1.5 (Cluttered-MNIST, 2026-06-29)**: built a single-digit Cluttered-MNIST classification adapter (random-position digit) + scale-free `ATTENTION` and `POS_ATTENTION` readouts. Cluttered 5-seed: linear 0.213 < mean-pool 0.314 < attention 0.466 ≈ pos-attention 0.446 < **flatten 0.617**. **Finding:** position-keeping helps (both beat mean-pool; walk-conv features matter — flatten 0.62 ≫ linear 0.21), but **single-vector pools plateau ~0.45** — the bottleneck is single-vector *compression*, not absence of position (pos-attention didn't help). Flatten's full spatial map wins on bounded grids; the scalable equivalent would need *multi-query* pooling. 28 tests. `reports/2026-06-29-soma-posreadout-phase1.md` + figure. **Phase 2 caveat:** RicciStim's adaptive quadtree has *variable* anchors → flatten N/A there; needs attention pool.
- ✅ **Chebyshev-CR cell × readout sweep (MNIST, 2026-06-29)** — **PARTIAL REHABILITATION**: the Soma-vision pipeline never used the framework's HSiKAN cell (bare `Linear`+`GELU`, global mean-pool). 2×2 factorial (cell {GELU, Chebyshev-CR} × readout {mean-pool, flatten}): base/mean **0.519**, cheby/mean **0.553**, GELU/flatten **0.945**, cheby/flatten **0.948**. **Readout main effect +0.410, cell +0.018 (~23×)** → the 0.52 ceiling was **pooling-bound (H2), a readout artifact**, NOT expressivity. **Position-preserving Soma (0.945–0.948) BEATS the linear control (0.906)** at comparable capacity — the signed walk-conv *is* net-useful once the readout stops discarding spatial layout. Opposite outcome to the holonomy re-test. Reusable: `MessageActivation.{CR,CHEBY_CR}`, `PatchEncoder.CHEBY_CR`, `Readout.FLATTEN`. Open: scalable position-aware pool (flatten doesn't scale), cluttered/non-centred re-test. `reports/2026-06-29-soma-cheby-cell-readout-sweep.md` + figure. 10 new tests.
- ✅ **Cichy-92 cortical-prediction article** (why brain-predictivity, not accuracy, is the honest test) — `docs/articles/cichy-cortical-prediction/article.pdf`
- ✅ **Seminar: Publications & submissions slide + portfolio figure** (8 venue families, status-coloured) — `docs/seminar/{make_publications_figure.py, insert_into_deck.py}` → `HyMeKo_Seminar.with_refs.pptx` (37 slides)

- ✅ **RicciStim stack**: Forman κ, Hodge Laplacians, adaptive quadtree, Bochner-coupled hg-conv, stim-graph builder, SDRF — all implemented + 168+ tests — plans `2026-05-14-gomb-soma-ricci-stim(+bench)`
- ✅ **Perf passes 1–4**: 296× speedup (8.3 s → 28 ms/image) — `reports/2026-05-15-gomb-soma-ricci-stim-sdrf-optimization.md`
- ✅ **Hodge boundary₂ vectorize + dead-code cleanup** — `reports/2026-05-16-gomb-soma-hodge-vectorize.md`
- ✅ **Rust quadtree port** (PyO3, 3.9–9.8×) — `reports/2026-05-16-gomb-soma-quadtree-rust.md`
- ✅ **Cortical infra (Slice 1)**: scorer + ResNet-tiny baseline + synthetic Cichy smoke (21 tests) — `reports/2026-05-19-gomb-soma-cortical-implementation.md`
- ✅ **Fair vision re-bench** (CNN/MLP/HSiKAN/HGNN/RicciStim × MNIST/Fashion): **negative result, well-engineered** — `reports/2026-05-28-vision-hypergraph-vs-cnn-rebench.md`

## HyMeYOLO line (Cluttered-MNIST stages landed; VOC transfer open)

- ✅ Stage A-2 (cosine+e100, +0.118), Stage B (ResNet-tiny, +0.149), Stage C (FPN), warm-start (+0.124), ricci-weight sweep — `reports/2026-05-16/17-hymeyolo-*-5seed.md`
- ✅ Stage D / D-1 **falsified with diagnostic** (from-scratch & ImageNet both ≈0.01 mAP on VOC → head is the bottleneck); Stage D-2 head-bottleneck diagnosis complete
- ✅ VOC test baseline (ep60 floor 0.0149) + `eval_voc` tool + headless capture for slides — `reports/2026-06-10-voc-test-baseline.md`, `…-hymeyolo-headless-capture.md`

## Gömb signed-link prediction (results on disk)

- ✅ Gömb-strict link prediction across Bitcoin/OTC/Slashdot/Epinions/wiki_elec (~0.90–0.94 AUC, config-dependent; leakage control → 0.540 = chance); ~30k params — artifacts under `signedkan_wip/experiments/results/*.jsonl`, `reports/gomb_tune_*`, `reports/hsikan_*5seed*`

## Framework / systems (selected)

- ✅ P-graph engine + A1–A5/MSG/SSG/ABB + Pimentel book validation; reachability-rules unification — plans `2026-05-19-pgraph-*`, `2026-06-14-reachability-rules-audit-pgraph`
- ✅ WASM editor MVP + stereotype views; canonical Blake3-hash IR; query-driven transforms (urdf/sdf/mjcf/gazebo/ros2/dot/mermaid/sysml/torch_dataflow)
- ✅ SISY 2026 control paper (review-fixed, ROS pick-and-place demo) — `[[project-sisy2026-control-paper]]`

> For anything not listed here, search `reports/<date>-<slug>.md`.
