# HyMeKo — Backlog (open work)

**Last updated:** 2026-06-29 · Companion: [DONE.md](DONE.md) (completed work).

The canonical record stays in `docs/plans/` (plans) + `reports/` (acceptance). This
file is a hand-maintained **index of what's still open**, seeded from the
2026-06-15 plan audit (Soma-vision + HyMeYOLO lines), the 2026-06-15 full
plan-dir sweep (~127 non-vision plan dirs) and the project memory.

**Convention.** One line per item: `[area] description — plan/ref — STATUS — prio`.
STATUS ∈ OPEN / PARTIAL / QUEUED / BLOCKED / DECISION. When an item finishes, move
its line to `DONE.md` with the report link. Keep newest areas on top.

**Tally (2026-06-15):** 51 open line-items across 9 areas. Of these, ~11 are **P1**
(prefilter, HSIKANConfig, 3-dataset bench, tier-2/3 audit, Friedler-quotient,
reachability-rules pgraph, Phase-B shuffle, Nature-Comm audit, Stage D-3, Stage H,
cortical Brain-Score). The May signed-link items predate the leakage audit —
re-validate any AUC against the σ-masked strict protocol before reporting.

---

## Gauge-holonomy / structural-probe decider — 2026-06-29 session

> **The swing-factor decider (gates the gauge line the LLM bet below rides on).** Two of three HyMeKo pillars
> (DTC, gauge-holonomy learning) rest on the *unproven* claim that the predictive signal on signed graphs is a
> cycle **holonomy**, not node features (C1/C3, `docs/theory/gauge_holonomy_signed_hsikan`). Existing toys circle
> it (rotor_probe = 1-cycle regression; structural_probe = B²x regression; soma-holonomy = vision) but none
> isolates holonomy as the *only* signal in a classification through the production arms. This decides it cheaply —
> win → "structure is load-bearing" becomes a measurement; tie → down-grade the gauge line.

- [ ] **Holonomy-discriminator toy** — classification where the label is *provably* a cycle holonomy and nothing else, run through 3 arms (StructuralActor walk-gather=Bᴸ / HSiKAN / params-matched MLP) on **test accuracy + forward latency**, with a confound guard (logreg-on-raw-features must be at chance) and a **zero-cost aromaticity certification** (Hückel 4n+2 vs Möbius 4n — `docs/theory/chem_bio_neuro_equivalences` test #2). T1 parity (Z₂), T2 Spin-angle (readout fixed = C1's mechanism), T3 aromaticity. New `hymeko_rl/holonomy_probe.py` + tests; reuses `structural_probe`/`structural_actor`/`rotor_probe` (no edits, §6.1). No CORE.YAML, no new dep. Plan (4 artifacts, PDF built): `docs/plans/2026-06-29-holonomy-discriminator-toy/` — OPEN — **P1** (`[[project-hsikan-loses-possible-bug]]`, `[[project-gauge-holonomy-signed-hsikan]]`)
  - **stretch (separate line, gate on the toy landing):** grid-cell path-integration probe — a StructuralActor on a hexagonal lattice trained to path-integrate should develop toroidal/grid population structure (the chem/bio/neuro "crown jewel"; converts the normative-model claim into a comp-neuro result) — OPEN — P2

## LLM architecture (Gömb / HSiKAN / Fiber-Spike-Rotor) — 2026-06-29 session

> **Big bet (user, "initiate"):** a new autoregressive LM whose sequence-mixer replaces softmax attention
> with a signed, rotor-transported, spike-gated **walk-holonomy** over a causal token graph, channel-mixer =
> HSiKAN on the **Chebyshev/CR** cell (train-CR / deploy-Chebyshev), residual stream on the hypersphere
> (**Gömb**). Prior-art checked (2026-06-29): KAN-as-FFN is crowded (KAT/ChebyKAN/PolyKAN), but the
> *signed-Z₂-holonomy + rotor-connection-generalising-RoPE + spike sparsity* synthesis as an LM is "none
> found" in a bounded search. Framed as **falsifiable**, not a foregone LLM.

- [ ] **Gömb/HSiKAN/Fiber-Spike-Rotor LM** — plan-first DONE 2026-06-29: `docs/plans/2026-06-29-gomb-hsikan-fsr-llm/` (4 artifacts, PDF compiles). New non-core package `hymeko_lm/` composing existing libs (`hymeko_neuro.core`, `hymeko_rl.structural_actor`/`spike_probe`, `cayley_rotor`/`hymeko_clifford`, `hymeko_gomb`, `hymeko_graph`) — discovery confirms **zero existing transformer/LM code** (clean slate), CORE.YAML untouched, no new dep. Staged: Ph0 skeleton+interfaces+smoke (no training); **Ph1 go/no-go** = tiny byte-LM (enwik8 default) vs matched-param transformer on val bits-per-byte + tokens/s + params (≥3 seeds); Ph2 ablations (rotor=RoPE-vs-learned-conn, sign, spike, CR-vs-Chebyshev parity) as classes not flags; Ph3 scale (separate plan). **Ph0 DONE 2026-06-29** (`reports/2026-06-29-fsr-lm-phase0.md`): `hymeko_lm/` package built (composition over the reused libs, CORE untouched, no new dep); FSR mixer **validated** on the lag-copy discriminator — converges to the irreducible floor (0.42 vs floor 0.35 vs uniform 2.77), proving offset-routing works; 15 tests pass, ruff+mypy clean, smoke 66 MB GPU peak. Two bugs + one design finding fixed at Ph0 (rotor broadcast; gate routing-mass normalization not neighbour-count; spike gate needs a positional bias term). `cr_cheby` cell already existed → no new basis code. **Memory result (same session):** added associative-recall task + `baselines.py` (matched causal transformer). First FSR FAILED content-addressed recall (plateau 2.07≈uniform 2.77 while xfmr→0.000); hypothesis "gate sharpness" FALSIFIED (softmax≈sigmoid); **root cause = no value projection** (mixer transported raw hidden not a learned `W_V` fiber → can't form the 2-layer induction circuit). **FIXED with value fiber → FSR now solves recall (0.025 vs xfmr 0.000, matched 35k params)**, slower-to-induct than attention (~step 1250 vs 500). 18 tests pass. `GateMode` softmax/sigmoid now a config axis. **Ph1 PoC DONE 2026-06-29** (`reports/2026-06-29-fsr-lm-phase1.md`): byte-level TinyShakespeare A/B vs matched-param (~90k) causal transformer. First a NO-GO (FSR 4.845≈unigram-entropy-4.779 → learned ZERO context); diagnosed = missing learnable readout scale on the normalised sphere stream (couldn't sharpen past unigram). FIXED (`_ScaledNorm` nGPT-style + `ResidualMode` config axis sphere/prenorm). **DEFINITIVE 3-seed/1500-step/seq128 (matched ~368k):** FSR-prenorm **2.479** (seeds 2.453–2.495) **BEATS** transformer **2.570** (2.563–2.607, disjoint ranges); FSR-sphere(Gömb) **2.770 LOSES**. GO for FSR-prenorm; **Gömb sphere does NOT earn its place here** (loses even with readout-scale fix) — the signed-rotor-spike mixer + HSiKAN cell carry the win, standard pre-norm residual. Honest costs: **3.1× slower/token** (dense O(T²) mixer → hard-top-k spike-sparsity lever), tiny scale, 1MB corpus. Arch diagrams `reports/2026-06-29-fsr-lm-arch-{model,mixer}.{tex,pdf,png}`. **SPEED lever DONE** (`reports/2026-06-29-fsr-lm-speed.md`): hard top-k spike gate (`spike_k` config, `_sparse_mix` O(T·k) vs dense O(T²)) — **quality preserved/improved** (k=16 bpb 2.60 beats dense 2.65), **O(T²) wall fixed** (sparse-vs-dense speedup 1×@T128 → 25×@T512 → dense OOMs@T1024 while sparse runs). Absolute speed still 3–7× behind transformer = FUSED SDPA kernel vs our eager ops → next lever is a Triton/CUDA fused gather+quat-rotate kernel (engineering not algorithm). 21 tests. NEXT = fused kernel + SCALE (enwik8 subset, lead with spike_k=16) + Gömb ablation + Chebyshev deploy parity. Builds on `2026-06-28-cr-chebyshev-cell` + `docs/theory/gauge_holonomy_signed_hsikan.pdf` — OPEN — **P1/bet**

> **Affective/interaction model = a HyMeKo model.** Idea (user, "big hit for Kato"): external **human or
> agent** input shapes the reward **at runtime** — declared in HyMeKo, not glued in. Valence `v∈[-1,1]` from a
> pluggable `AffectiveSource` (neutral/human/agent), coupled two ways (additive `affective` term + runtime
> `(1+gain·v)·Σw·term` modulator). Plan-first per §2; runtime core prototyped in `hymeko_rl/env/affect.py`
> (lint/mypy clean, **not wired in**).

- [ ] **Affective / interaction model as a HyMeKo model** — `meta_affect.hymeko` vocabulary + `AffectSpec.from_hymeko` reader + env poll-hook + `affective` reward term + tests + demo task; ties to the shared-agent-reward-model + fuzzy lines; **Kato collaboration candidate** (steerable, auditable grasping head) — plan: `docs/plans/2026-06-22-affective-interaction-model/` (tex/pdf/tikz/mmd) — OPEN — **P1** (`[[project-kato-collaboration-grasping]]`)
- [ ] **Actor–critic SHARED REASONING via HSiKAN** (idea, write-down) — share not just reward but the signed-hypergraph structural reasoning (walks / cycle connections / jumps) between actor + critic: a shared HSiKAN trunk + dual heads, so credit assignment becomes structural. Concrete mechanism for the structure-advantage that serial chains didn't show; prototype on branched morphologies w/ a params-matched shared-trunk MLP control — `[[project-actor-critic-shared-reasoning]]` — OPEN — P2
- [ ] **Little planar walker** — `QuadrupedGoalEnv(base="planar")`: NEW `planar` base mode (slide-x + slide-z + pitch-y → confined to the sagittal plane, **can't fall sideways**, HalfCheetah-style) + **no flip-termination** for planar (whole-horizon to learn) + base-agnostic `vx` (Δx/dt). Default base is now `planar`. Lint/mypy clean; builds (njnt 11 = 8 legs + 3 passive base). **DE-RISK VERDICT (150-iter HSiKAN, goal 1.5 m): PARTIAL WIN — it walks forward.** returns −269→−14, final x **+0.47 m** (closed dist 1.5→1.04), 0% full-reach, joint spd 15 (vs free-base flail 24). The planar constraint was the fix: real sustained forward motion (~0.5 m) vs the free base's ~0. Just under the `x>0.6` flag; needs a bit more. Ckpt `checkpoints/walker_planar_hsikan.pt`. **TOMORROW (office):** (1) push it over the line — more iters (300) and/or shorter goal (~1.0 m) and/or a light upright/alive term so it doesn't crawl; (2) then full train HSiKAN+MLP + decorated comparison gif (swap `scripts/compare_backbones_gif.py` quad task to `base="planar"`); (3) update sanity/quad tests for the planar default + add a `set_base_mode("planar")` test (3-joint base, nu=8). The all-Y-leg geometry walks once planar-constrained — keep it. Prior negative: free-base 0% reached, flailing — `reports/2026-06-22-quadruped-jump.md` — OPEN — P3
  - **superseded:** Quadruped 3D goal-reaching (free base) does NOT learn to walk (progress −0.03..+0.16 m, 0% reached, flail ~24 rad/s); replaced by the planar walker above.
- [ ] **Full-3D walker via staged constraints** — promote the planar walker to full 3D. Three pillars: (1) **hip-abduction DOF** (3-DOF Ant/ANYmal legs — the planar all-Y legs CAN'T balance laterally in 3D), (2) **constraint-relaxation curriculum** (soft upright/lateral support springs decaying planar→+yaw→+roll→free, gated on measured metrics not iter-counts), (3) **stability shaping** (upright/lateral-dev/heading/foot-contact/gait-symmetry reward terms). Honest framing: multi-session research, fallback = ship S2 2.5D. Plan (4 artifacts, pdf built): `docs/plans/2026-06-22-3d-walker-constraints/` — OPEN — P3
- [x] **Fast-and-smooth reward terms** — generic `time_penalty` / `joint_velocity` / `joint_acceleration` (bounded Δq̇ jerk) in `meta_reward.hymeko` + registry; coin gets time+vel+jerk (joint speed 13.9→4.9 rad/s, smoothness confirmed); `actuated_dof_addrs` helper; 46 tests green — DONE (full 300-iter coin retrain + render still pending) — P2

## Rotor signed-link (leakage-free HSiKAN) — 2026-06-17 session

> **Result:** the SiGAT gap was **input-bounded, not readout-bounded**. Signed
> slerp/nlerp rotor propagation (computer-graphics interpolation × signed-graph
> AI) is the first confirmed lift — alpha +0.0045 / otc +0.0105, 5-seed,
> leakage-clean. Fair head ablation: real bilinear is the best readout; complex/
> quaternion/geodesic are worse (expressivity > algebra at the readout). Reports:
> `2026-06-17-signed-rotor-slerp-propagation.md` (win),
> `2026-06-17-link-head-ablation.md` (fair negative).

- [x] **Protocol-matched honest SiGAT comparison** — DONE 2026-06-17: the "0.04 gap" was a dedup artifact. `--dedup` added to `run_baseline_audit`; 5-seed grid (4 models × 2 graphs × 2 protocols, shuffle-gated). On the matched **deduped** protocol the gap **inverts vs `sigat_rotor`** (deduped 0.833/0.868 < our 0.850/0.879 → +0.017/+0.012 ahead); the real residual is vs **pure SiGAT** (+0.036 alpha / +0.016 otc), smaller than alleged. Non-deduped leaks under shuffle. → DONE.md, `reports/2026-06-17-protocol-matched-sigat.md`
- [x] **Adaptive rotor propagation** — DONE 2026-06-17 (neutral): Ph1 learnable per-block self-retention `α_b=floor+exp(θ_b)` (log-space ≡ sigmoid residual gate; softplus is for heads) — 5-seed A/B **parity** with fixed sw=4 (removes the `sw` knob at no AUROC cost). Ph2 depth scan {2,3,4,6}: **no gain** (alpha over-smooths past r2, otc flat). Third confirmation the ceiling is INPUT-bounded. → DONE.md, `reports/2026-06-17-adaptive-rotor-propagation.md`
- [x] **Leakage-free input enrichment** — DONE 2026-06-18 (mixed/honest): built extensible `StructuralFeature` registry + NEW exact **k-walk signed A^k profile** (+ratios; cycle reused). 5-seed deduped gated: walk lifts the **non-propagated** rotor baseline (audit cyc+walk otc **+0.0168**) but is **parity on the slerp-propagated line** (+0.003 otc) — propagation already IS signed-A^k aggregation (redundant). Seed-0 otc 0.9131 (beat pure SiGAT) **did not replicate** (5-seed 0.882, flagged variance). Pure-SiGAT gap = expressivity, not input. → DONE.md, `reports/2026-06-18-leakage-free-input-enrichment.md`
- [x] **Inductive transfer test** — DONE 2026-06-18 (nuanced positive): cross-graph transfer (train on A, eval frozen on B). **Mechanism + discrimination win:** rotor transfers (0.81–0.86 on unseen graph); transductive `dadsgnn` **cannot** (nn.Embedding can't index B). Naive shuffle-A gate confounded (B's real adjacency carries signal) → added **random-init control** (below chance, 0.38–0.47). Decomposition: learned-from-A increment **+0.038–0.063** (4/4 positive, ~1–1.5σ — suggestive not strong); bulk is B's structural prior. Refactored `run_audit` train loop → reusable `_train` (behaviour-preserving). → DONE.md, `reports/2026-06-18-inductive-transfer-test.md`
- [x] **Rotor-geometry ablation (is the rotor load-bearing?)** — DONE 2026-06-18 (foundational NEGATIVE): `MLPEmbedSignedModel` control (same proj/SGCN/classifier, rotor embed → higher-capacity MLP, ≥ params). 5-seed deduped gated: rotor − MLP = **±0.002–0.005, inside σ** — the rotor's S³ geometry is **not load-bearing** (head ablation already showed the algebra isn't either). **Reframe: the wins are STRUCTURAL FEATURES replacing the node-ID table, not the rotor** (MLP-embed is also leakage-free/inductive/param-light). → DONE.md, `reports/2026-06-18-rotor-vs-mlp-embed-ablation.md`
- [x] **Strengthen transfer — harder-pair grid** — DONE 2026-06-18: 3-arm decomp (real/shuffle/random-init) on train-small (bitcoin)→eval-large (slashdot/epinions), 5-seed. Learned increment **8/8 positive, 7/8 ≥ 1σ_shuffle**, **otc→epinions +0.0261 / +2.90σ** (was ~1–1.5σ on the bitcoin pairs). Variance, not effect size, separates train graphs (otc σ_sh≈0.009 clean; alpha σ_sh≈0.05 noisy → ~1.1σ despite a larger raw increment). Consolidated the lost 3-arm decomp into the canonical driver + fixed a latent resumption-key collision (real↔random-init). → DONE.md, `reports/2026-06-18-transfer-grid-strengthen.md`
- [x] **Write-up draft (reframed line + algebraic entropy feedback → RL)** — DONE 2026-06-18 (draft): `hymeko_neuro/assets/paper/structural_inductive_entropy_note.tex` (compiles). Captures (1) structural features not the rotor carry the leakage-free inductive wins (MLP ties on AUROC), (2) inductive transfer +2.90σ, (3) rotor earns its place on the embedded/ANN metric, and **the synthesis**: the framework's `StructuralEntropy` (Shannon over arity/sign/degree) + the existing Lyapunov-safe KL-feedback spectral-entropy regulariser = one control loop, transplantable to RL as a **structure-aware exploration/regularisation signal** over the kinematic/contact hypergraph (PROPOSED; falsifiable test = structure-driven entropy feedback vs vanilla β·H(π), ablated vs the MLP policy). Residual: node-holdout inductive; 10-seed on the noisy alpha→* cells — P2
- [ ] **Test the algebraic-entropy-feedback hypothesis** — once `hymeko_rl` Ph1 lands: add `H_struct(state)` + the KL-feedback schedule as an intrinsic/exploration signal; ablate vs vanilla PPO entropy, fix the algorithm. The MLP policy is the natural negative control (no hypergraph to read entropy from) — `structural_inductive_entropy_note.tex` §5 — OPEN — P2
- [ ] **Rotor → embedded substrate (Nagare)** — PLANNED 2026-06-18: `docs/plans/2026-06-18-rotor-embedded-nagare/` (4 artifacts, PDF compiles). Phase 1 characterise (MAC/memory/Nagare op-coverage; inference ≈ scatter + weightless quaternion algebra + small GEMM, ~15.7k params, no node table); Phase 2 the **discriminator** — rotor vs MLP-embed on the *embedded* metric (int8 PTQ ΔAUROC + ANN recall), since the AUROC ablation tied them but the embedded objective differs (bounded S³ output quantizes calibration-free); Phase 3 (gated on Phase 2 + a locked operating point) port the chosen inference to `hymeko_nagare` ops + `parity_gate`. No CORE.YAML, no new dep (torch.ao). **EXTENDED 2026-06-30 (user "rationale for nagare"):** the FSR-LM sequence mixer (`hymeko_lm`) is a SECOND, larger client with the SAME compute shape (gather/scatter + offset-rotor + small GEMM) — and the **strongest measured motivation yet**: profiled GPU bottleneck = `aten::_index_put_impl_` (atomic scatter-add backward) at **82% of CUDA time**, torch.compile only +3% → the cost is the von-Neumann atomic-scatter penalty on a dataflow op, which the spatial substrate removes by construction (spikes=event-driven firing, rotor=Toeplitz/systolic transform, walk-holonomy=streaming reduce). Phase-3 op set should cover both clients (shared signed-two-sided-scatter + Cayley-rotor ops). New rationale section in plan.tex/pdf. — P2
- [ ] Hygiene: in-pipeline `run_hsikan_rotor --head {complex,rotate}` are confounded for ablation (kept, defaults off); the *fair* head test lives in `run_rotor_head_ablation.py`. Note for future readers — P4

## Soma-vision (Gömb perception)

> **Decision context (2026-05-28 re-bench):** hypergraph-vision is *empirically
> falsified at small scale* — RicciStim/HSiKAN/HGNN all lose to a plain MLP on
> MNIST/Fashion; RicciStim Cluttered-MNIST = 0.14 mAP₅₀ (< 0.23 baseline). Code
> is correct (~3.8 kLOC, 168+ tests); the *approach* doesn't help vision here.
> So most items below are "explain / decide", not "build more architecture."
>
> **CAVEAT (2026-06-29 cell×readout sweep):** the *walk-conv* half of the
> falsification was substantially a **readout artifact**. base-Soma used a global
> mean-pool that discards patch position; with a position-preserving readout the
> walk-conv hits **0.945–0.948 on MNIST — above the linear control (0.906)** at
> comparable capacity (readout effect +0.410 vs Chebyshev-CR cell effect +0.018).
> So "walks-only falsified for vision" holds *only for the mean-pool readout*;
> the structure is net-useful once position is kept. The honest open question is
> a **scalable** position-aware pool + cluttered/non-centred re-test, not whether
> the structure can work at all. `reports/2026-06-29-soma-cheby-cell-readout-sweep.md`.

- [ ] **Explain the Cluttered-MNIST loss** — component ablation (Hodge vs Ricci vs SDRF vs quadtree vs σ-balance), one-at-a-time; inspect learned edge signs on a planted test — `2026-05-14-gomb-soma-ricci-stim` — OPEN — **P1**
- [ ] **Cortical Brain-Score run** — infra + synthetic smoke done; fetch real Cichy-92, run V1/V2/V4 ROI scoring vs ResNet-tiny (paired bootstrap) — this is the *untested* place Soma might actually shine — `2026-05-16-gomb-soma-cortical-benchmark` / `…-implementation` — OPEN — **P1**
- [x] **Base-Soma vision falsification** — DONE 2026-06-15: walk-conv base-Soma 0.52 vs linear control 0.91 on MNIST (−0.387 paired, all seeds) → walks-only also falsified for vision (→ DONE.md, `reports/2026-06-15-base-soma-vs-linear-mnist.md`). Residual: base-Soma vs Gömb on *signed-link* never run, but deprioritized given the vision falsification — P4
- [x] **Holonomy (sign-as-connection) re-test** — DONE 2026-06-29: the 2026-06-15 falsification tested sign-as-*routing*; re-ran the exact MNIST A/B with the StructuralActor sign-as-*connection* operator (`HOLONOMY` aggregation = `M_v(σ⊙m)`, the signed Bᴸ). **Falsification confirmed & tightened:** holonomy 0.4888 ≤ routing 0.5186 ≪ linear 0.9056 (both anchors reproduced 2026-06-15). The *right* operator gives the same answer; walk-holonomy is not load-bearing for vision. `HOLONOMY` mode kept (tested, reusable) but not promoted. → DONE.md, `reports/2026-06-29-soma-holonomy-aggregation.md` — CLOSED
- [x] **Cell × readout sweep (Chebyshev-CR)** — DONE 2026-06-29: 2×2 (cell {GELU, Chebyshev-CR} × readout {mean-pool, flatten}). **H2 (pooling-bound) decisively:** readout effect +0.410 vs cell +0.018. **Position-preserving Soma 0.945–0.948 BEATS linear 0.906** — walk-conv is net-useful once the readout keeps position; the falsification was a mean-pool artifact. → DONE.md, `reports/2026-06-29-soma-cheby-cell-readout-sweep.md` — CLOSED
- [ ] **Scalable position-aware readout + harder re-test** — Phase 1.5: single-vector pools (content-attention, pos-attention) **plateau ~0.45 < flatten 0.62** on cluttered (bottleneck = single-vector compression, not lack of position) → scalable fix is a **multi-query attention pool** (K·d, K≪n_patches), untried. Phase 2 (RicciStim, highway ON for a clean control): **structure IS load-bearing** — full-mean 0.235 vs encoder-only 0.152 (+0.083, ~3σ); attention ≤ mean again. **Next:** (a) multi-query attention pool A/B on fixed-grid cluttered (the readout that might match flatten); (b) non-centred + full-scale (5ep, more train) re-test — current ~0.235 ≈ linear is undertrained. Live revival path for Soma-vision — OPEN — **P2**
- [ ] **TriangleConv clarification** — original plan specified a Cartwright–Harary balance-gated triangle layer; only `hg_conv_bochner.py` exists. Confirm superseded or build it — `2026-05-14-gomb-soma` §6 — DECISION — P3
- [ ] **Faithful Soma round-trip (.hymeko ↔ vision pipeline)** — `soma_vision.hymeko` is the minimal skeleton; the real Hodge/stim/patch internals aren't emittable yet — hero Phase 3 follow-up — OPEN — P3
- [ ] **Perf P6–P8** (topology cache, polygon vectorize, GPU SegmentedSparse batching) — only justified if cortical/vision use-case survives — `2026-05-15-ricci-stim-opt-pass-5` — BLOCKED (on the decision above) — P4
- [ ] Hygiene: archive/delete superseded `2026-05-14-gomb-soma-stim` (never implemented; subsumed by ricci-stim) — P4

## HyMeYOLO (detection stack)

- [ ] **Stage D-3 nodelet head** — per-query objectness gates (decouple is-object from which-class); code drafted, **production smoke not run** (≥0.05 mAP₅₀ gate, then 5-seed ≥0.20). The linchpin for the VOC claim — `2026-05-19-stage-d3-nodelet-head` — OPEN — **P1**
- [ ] **Stage H person-only VOC** — single-class diagnostic: settles whether the head (D-3) is the *only* bottleneck or upstream (D-4) is needed; also unlocks the rapport demo — `2026-05-19-stage-h-voc-eyes-for-rapport` — QUEUED — **P1**
- [ ] **VOC ep180 full run on HPC** — current floor is ep60 = 0.0149 (under-trained, DataLoader host-bound at 37% GPU util → needs pinned-memory rewrite); ep180 sets the honest publishable number — `2026-06-10-voc-test-baseline` — QUEUED — P2
- [ ] **Stage D-3-bis / D-3-tris** — λ_gate⁻ override + matcher gate-veto + focal-gate; run only after D-3 base clears — `2026-05-18-stage-d3-{bis,tris}-*` — QUEUED — P3
- [ ] **conv-as-hypergraph** — RF-orbit hyperedges + shift-equivariance unit-test gate + Bochner variants; H1/H2/H3 mAP comparison not run — `2026-06-11-conv-as-hypergraph-hymeyolo` — OPEN — P3
- [ ] Tidy: confirm Stage A-3 + Stage B′ attribution controls ran (no standalone report found; likely folded into Stage B) — P4

## Hero demo

- [ ] **Runnable torch round-trip** — numeric parity of the emitted module vs `cascade.py` (pulls torch = §1 dep) — beyond the current structural parity — hero Phase 3 — OPEN — P2
- [ ] **Task-layer emitters** — BehaviorTree.CPP / PDDL / ROS 2 action servers for the `.hymeko` task layer (FANUC handover task validates but can't emit) — OPEN — P3
- [ ] **Editor hero-cell** done (robot_arm profile); optional: a NL→.hymeko stub front-end behind the gate — P4

## Editor (docs/editor)

- [x] **3D pan in the Kinematic view** — DONE 2026-06-16: right-button/shift-drag pan ported from `hypergraph3d.js`; 66/66 editor tests pass (`reports/2026-06-16-editor-kinematic-pan.md`)
- [ ] More Steiner/design generators (S(2,3,9) via Bose already; add Pasch, K₅³, projective planes) — 1 registry entry each — OPEN — P4
- [ ] Optional: shareable `?src=` deep-link (load arbitrary source) — would also unblock headless visual tests (e.g. the 2D bidirectional render) — P4

## Signed-link learning (HSiKAN / cycles / AC-HSiKAN)

> From the 2026-06-15 plan sweep of the May signed-link line. Most of these
> predate the leakage audit; **re-validate the operating point against the
> σ-masked strict protocol before reporting any new AUC** (`[[project-nature-leakage-paper]]`).

- [ ] **Vertex prefilter** — cycle-enumeration prefilter; gates the 3-dataset bench; measured ~30–40 % Epinions k=5 savings — `2026-05-11-vertex-prefilter` — OPEN — **P1**
- [ ] **HSIKANConfig derivation** — collapse 25+ `HSIKAN_*` env vars into one typed `HSIKANConfig` dataclass parsed at startup (kills the §6.5 #11 deep-call-site env reads) — `2026-05-12-hsikan-config-derivation` — OPEN — **P1**
- [ ] **3-dataset HSiKAN v1/v2 bench** — 4-phase Bitcoin/Epinions/Slashdot run; blocked behind vertex-prefilter — `2026-05-12-3dataset-hsikan-v1-v2` — OPEN — **P1**
- [ ] **Cycles strategy refactor** — 16 `#[pyfunction]` variants → 1 config-driven entry (§6.5 #1/#9); pure-Rust algo to `hymeko_graph` (§6.5 #2) — `2026-05-11-cycles-strategy-refactor` — PARTIAL — P2
- [ ] **CPML-XHC architectures** — cross-head-coupled CPML variants — `2026-05-11-cpml-xhc-architectures` — OPEN — P2
- [ ] **Self-evolving cycle sampling** — adaptive reservoir/ABB sampler that reweights by yield — `2026-05-14-self-evolving-cycle-sampling` — OPEN — P2
- [ ] **CN optimization pattern** — common-neighbour fast path — `2026-05-13-cn-optimization-pattern` — OPEN — P2
- [ ] External-AUC tuning / orthogonal FIR sweeps — `gomb-external-auc-tuning` · `gomb-tune` · `hsikan-cpml-fir-orthogonal` — OPEN — P3

## Architectures / fuzzy / sequence

- [ ] **Signed-link tier-2/tier-3 audit** — ~32 h GPU SGT/SiGAT baseline audit; **blocks the signed-link paper's baseline table** — `2026-05-17-signed-link-tier2-tier3` — QUEUED — **P1**
- [ ] **Friedler-quotient param reduction** — structural param compression via the Friedler quotient; sizing claim unverified — `2026-06-06-friedler-quotient-param-reduction` — OPEN — **P1**
- [ ] Sequence/text + fuzzy layers — `sequence-multichannel-v2` · `text-encoder-decoder` · `fuzzy-signature-layer` · `cycle-induced-memory` — OPEN — P2
- [ ] Fuzzy pose / balance-as-activation — `fuzzy-pose-detection` · `balance-as-activation` — OPEN — P3
- [ ] Fuzzy-pose config audit + restart options — **BLOCKED on user decision** — `fuzzy-pose-config-audit` · `restart-options` — BLOCKED — P4
- [ ] KLP skyline · AC-HSiKAN follow-ups — `klp-skyline` · `ac-hsikan-followups` — QUEUED — P3

## P-graph / reachability / robotics demos

> **Humanoid control + collaboration (forward milestone, gated — do NOT start yet).** The escalation target past
> the arm / quadruped / collaborative line: control + collaborative manipulation on a **humanoid** (multi-limb,
> high-DOF, bimanual). Recorded now so the bar is explicit; it initiates only once the groundwork has proven out.

- [ ] **Humanoid robot control + collaboration** — **Initiation gate (ALL must hold before any work begins):**
  (1) the **collaborative** scenario done, (2) **pick-and-place** done, (3) **6-DOF control** scenarios done — i.e.
  every current control scenario complete — **AND** (4) **HSiKAN or SA-HSiKAN survives** them (validated as
  competitive, not falsified — the architecture is load-bearing, not just at parity by accident). Until the gate
  clears this is a milestone marker, not active work — BLOCKED (on the gate) — P2/milestone
  (`[[project-kato-collaboration-grasping]]`, `[[project-hsikan-launchbound-alternatives]]`, `[[project-structural-actor-walk-holonomy]]`)

- [ ] **MuJoCo RL grasping — learned control head on the tensorised hypergraph (Kato collab POC)** — PLANNED 2026-06-18: `docs/plans/2026-06-18-mujoco-rl-grasping/` (4 artifacts, PDF compiles). Responds to Kato-sensei's ask (hypergraph→tensor→learned movement). Staged: Ph0 wire the existing star-expansion→torch bridge into a state encoder; Ph1 REACHING via imitation (MVP, no reward); Ph2 GRASPING — the **architecture ablation**: HSiKAN-on-hypergraph policy vs MLP baseline under ONE shared in-repo PPO (clean: fix the algorithm, ablate the architecture — avoids the PG-vs-PPO confound; metric = sample-efficiency + success + params, ≥3 seeds, random-policy floor); new gripper+object MJCF; Ph3 deploy-tiny-policy/warm-start tie-ins. **IN PROGRESS** — token `APPROVED-CORE-EDIT: mujoco-gymnasium-robot-rl` granted 2026-06-18; deps installed (mujoco 3.9.0, gymnasium 1.3.0). **Phase 0 DONE** 2026-06-18: (inc-1) isolated `hymeko_rl/` package + actor-critic unification (`ActorCritic` = one shared backbone, two heads); (inc-2) the **kinematic-hypergraph bridge** — `HypergraphState` (mjcf→signed kinematic hypergraph) + `HSiKANBackbone` (batched signed message passing) reading it; `build_policy("hsikan"|"mlp")` is the ablation switch; **the policy reads the compiled hypergraph, not raw joints** (demonstrated). 17 tests, ruff+mypy clean. `reports/2026-06-18-hymeko-rl-{scaffold,phase0-bridge}.md`. **Phase 1 DONE** 2026-06-18: `ArmReachEnv` (Gymnasium, node-feature obs on the kinematic hypergraph) + closed-loop **DLS-IK expert** + behaviour cloning; the HSiKAN policy **learns reaching** (0.31 m vs 0.43 m untrained floor, expert 0.08 m), error-progression figure `reports/2026-06-18-hymeko-rl-phase1-reach.png`. 23 tests, ruff+mypy clean. `reports/2026-06-18-hymeko-rl-phase1-reaching-bc.md`. Honest: BC covariate-shift gap to expert (Phase-2/PPO fix); HSiKAN-vs-MLP ablation NOT yet fair (capacity confound) — deferred to Ph2. **THREE upstream bugs FILED** (`docs/BUGS.md` B-003 import resolver / B-004 emit actuator / B-005 emitter drops joint axis). **Phase 2 PARTIAL** 2026-06-18: in-repo PPO built + tested (`hymeko_rl/ppo.py`, clipped+GAE, `--task reach-ppo`), but **honest negative** — PPO degrades a BC-pretrained policy (HSiKAN 0.28→0.35, MLP 0.25→0.35), diagnosed as **critic cold-start under the dense-negative reward**; the architecture ablation is **inconclusive** (`reports/2026-06-18-hymeko-rl-phase2-ppo.md`). 25 tests. **Phase 2 PAUSED mid-debug** 2026-06-18 — critic warm-up helped (HSiKAN stops degrading) but **PPO still doesn't beat BC**; delta-action change hurt (reverted-recommended). Strongest untested lead: **time-limit truncation treated as a true terminal → wrong value bootstrap** (most episodes truncate; classic PPO bug, fits the symptom). Full trail + revised next steps in `reports/2026-06-18-hymeko-rl-phase2-debug-checkpoint.md`. **Action interface RESOLVED** (later 2026-06-18): env now parameterised by `control_mode {torque,position,velocity}` (`arm_world.make_arm_mjcf`), **torque default** (inverse-dynamics expert `τ=M·a_des+bias`); verified experts torque 0.069 / position 0.082 / velocity 0.134 — sets up the **control-mode × architecture ablation**. **PPO truncation-bootstrap bug FIXED** (later 2026-06-18) — PPO now *improves* monotonically (torque BC→PPO: 0.363→0.347→0.344 at 50/80/150 it) instead of degrading; the core Phase-2 blocker is resolved. Torque is hard (BC 0.36 vs position 0.28; slow PPO gains). `run_ppo` takes `control_mode`. Tests REPAIRED (35 pass, mypy clean, +coverage). **Control-mode ablation DONE** (`reports/2026-06-18-hymeko-rl-phase2-control-mode-ablation.md`): clean **difficulty ordering** position(BC 0.22) < velocity(0.27) < torque(0.36, hardest); but **PPO still degrades BC in 4/6 cells** even post-truncation-fix → architecture comparison inconclusive. **Separate actor/critic networks DONE** (later 2026-06-18) — **stabilized PPO**: worst BC→PPO degradation −0.095 → −0.020 (now within eval noise); PPO *preserves* BC in all 6 cells (no more catastrophic wrecking). But PPO still doesn't *beat* BC in 80 it (gains noise-level) → that's now a **compute/scale** problem, not a bug. **Phase 2 = PPO went from catastrophically broken to stable+correct** via 5 fixes this session (critic warm-up, coherent control interface, mass-matrix torque expert, truncation bootstrap, separate networks); 36 tests, clean. **Resume options: (a) multi-seed/longer runs (or DAgger) to push past BC; (b) `meta_observation.hymeko` declarative obs proposal (offered); (c) entropy-feedback test; (d) for the demo lead with position/velocity (reaches better).** — **P1/collab**
- [ ] **Declarative observation / state-space in HyMeKo (`meta_observation`)** — PROPOSED 2026-06-18: `data/robotics/meta_observation.hymeko` (vocabulary, parses) + `arm_reach_observation.hymeko` (example, **validates** via CLI). The obs/state space as a hypergraph — per-vertex feature channels + global channels → `@observation_space` whose star-expansion is the `(N,8)` obs tensor (the declarative form of `node_features()`; channel dims 1+1+3+3=8 ≡ `_NODE_FEAT`). One `.hymeko` source → MJCF scene + kinematic hypergraph + **observation**. `reports/2026-06-18-hymeko-rl-observation-proposal.md`. Next: wire `AgentSpec.from_hymeko` (via CLI now / engine snapshot once B-003 fixed); drive `node_features()` from the declared channels; extend to action+reward + a quadruped obs profile — **P1/collab**
- [ ] **Quadruped locomotion RL (walk/run/jump) — geometry in HyMeKo (templates+annotation+reuse)** — PLANNED 2026-06-18: `docs/plans/2026-06-18-quadruped-locomotion-rl/` (4 artifacts, PDF compiles). RL-locomotion sibling of the arm grasping; reuses `hymeko_rl` (ActorCritic/PPO + the HypergraphState bridge), no new scaffold, no new dep (mujoco+gymnasium already in). Re-authors the hand-expanded `quadruped_d3_t0.hymeko` fixture elegantly: one `leg` template ×4 + annotations + `<isa>`/`using as`/`->` reuse → emits MJCF (free base, contacts) + star-expands to the obs tensor (one source, two products). Branched topology + contact cycles = where the HSiKAN-vs-MLP gap should be largest (ties to `2026-05-14-legged-locomotion-contact-mode`). Ph0 verifies the composite-template construct vs the grammar — **P2**
- [ ] **Reachability-rules audit (pgraph)** — thread a `ReachabilityRule` enum through `hymeko_pgraph` + regime sweep; unifies leakage audit ↔ A1–A5/MSG/SSG/ABB; trio of publications — `2026-06-14-reachability-rules-audit-pgraph` — OPEN — **P1** (supersedes the article-only line below)
- [ ] **hymeko_pgraph Python integration** — 4 Python integration tests over the pgraph crate — `hymeko-pgraph-py` — OPEN — P2
- [ ] Robotics/demo plans — `kinematic-demo` · `comm-cliques-demo` · `gomb-np-hard-approximation` · `camera-placement` · `robotics-behavior-collaboration` · `seminar-demo-program` (+ remaining) — OPEN — P2
- [ ] Gömb-as-planner-heuristic · legged-locomotion — OPEN — P3
- [ ] Gömb belief-planning bridge — `gomb-belief-planning-bridge` — OPEN — P4
- [ ] Niitsuma dual-UR5 · MDPI Technologies live demo — `niitsuma-dual-ur5` · `mdpi-technologies-live-demo` — QUEUED — P2

## Paper / leakage / SMC (from memory — verify before acting)

- [ ] **Nature leakage paper** — 5-seed σ-masked strict grid running; finish + lock the honest operating point — `[[project-nature-leakage-paper]]` — OPEN — P2
- [ ] **Phase-B baseline shuffle audit** — 5-seed shuffle grid → Nature Table 1; harness + ~6× vectorization done 2026-06-14, grid running — `2026-06-14-phase-b-baseline-shuffle-audit` — OPEN — **P1**
- [ ] **Nature-Comm leakage audit** — baseline label-shuffles + Reddit reachability control; feeds the strict-protocol claim — `2026-05-17-nature-comm-leakage-audit` — OPEN — **P1**
- [ ] No-leakage structural benchmark E2/E3 — E1 harness smoke-clean 2026-06-11; E2/E3 still open — `[[project-no-leakage-benchmark-resume]]` — OPEN — P2
- [ ] **SMC smc_02** edits are UNCOMMITTED in the smc_02 repo (14pp, review_1/2 addressed) — commit when ready — P3
- [ ] Reachability-rules article (audit protocols ↔ P-graph A1–A5) — plan+argument+tests done; possible new article — `[[project-reachability-rules-article]]` — P3 (impl tracked under the pgraph item above)

## Infrastructure / other (from memory)

- [ ] HSMM → FPGA: Nagare dataflow substrate → HSMM abstract machine → Zynq (frozen compiler plan exists) — `CRITICAL-nagare-to-hsmm-compiler` — OPEN — P3
- [ ] **Codebase rehaul Phases 2–7** — `RuntimeConfig` + unified `Experiment` framework (kills §6.5 #3 run_*.py scaffold dup, #11 globals); Phase 1 partial — `2026-05-11-codebase-rehaul` — PARTIAL — P2
- [ ] `04_graph_query` tests T13–T21 — `hymeko_query` integration suite — OPEN — P2
- [ ] `05_hre_extraction` — HRE extraction spec tests — OPEN — P2
- [ ] `06_wasm_editor` — full 7-crate WASM editor spec (current editor is the ad-hoc subset) — OPEN — P3
- [ ] G-SPHF Rust crate — awaiting CORE-edit token (new crate = §1 dep add) — `gsphf-rust-crate` — BLOCKED — P3
- [ ] Seminar deck: open `HyMeKo_Seminar.with_refs.pptx` and eyeball the faded heatmap-background opacity (one-line tweak in `insert_into_deck.py`) — `2026-06-15-seminar-deck-additions` — OPEN — P4
- [ ] **Technical-report scaffolder** (PARKED 2026-06-22) — `scripts/new_report.py` + shared `docs/templates/technical_preamble.tex` (the reused tikz styles/booktabs/`\code`/author block, defined once). Modes: `--kind report` (one `.tex` in `reports/`) / `--kind plan` (the §2 4-artifact `docs/plans/<date>-<slug>/` skeleton, compliant from the start); `--title --for "<name>" --author` (default *Dr. Csaba Hajdu*, no agent name) `--build` `--journal <jsonl>` (embed an `offpolicy_tables` table). Motivation: the same LaTeX preamble was hand-written 3× on 2026-06-22 (Kato handout + AC-Gömb design note + this plan) — §6.5#3 scaffold dup. Needs a short §2 plan before building — OPEN — P3
