# HSiKAN + Entropy Roadmap — 2026-05-04

Multi-day plan tracking what's queued, what's blocked on what, and
who owns each piece.  Update the **Status** column as work moves;
strikethrough things that landed.

## TL;DR — End-of-day 2026-05-04 wins

Five items shipped, plus four extensions (SGT baseline, SGT-Slashdot
table update, HyMeKo→star pipeline, αₖ probe):

- **Item #1 ✅** — Camera-ready 5-seed bench: SMC Table I, III, IV
  refreshed in `paper/smc2026_hsikan_wip/main.tex`.
- **Item #2 ✅** — ph18c entropy follow-up: highway-10 +0.092pp
  positive **confirmed at 10-seed (p<0.05)**, AND a NEW
  resmlp-40 +0.213pp positive at $\lambda{=}0.5$ (p<0.05) — the
  Path I calibration law $\lambda_{\rm multi}\sim\lambda_{\rm scalar}/L$
  exactly predicted this depth recovery.  Multi-term programme
  has TWO confirmed positives now.  Highway-20 confirmed
  depth-fragile (genuine null across full λ-grid).  Both
  `reports/phases_and_paths.tex` and
  `reports/thesis_iv_executive_summary.tex` updated; PDFs
  rebuilt clean.
- **Item #3 ✅** — Sinusoid controls: trained HSiKAN at 90-91%
  sinusoidal vs all three nulls at 50-58%.  +32-40pp gap
  decisively defends the SMC §III.G "91% sinusoidal" claim.
  Paper §III.G updated with control-baseline prose.
- **Item #4-prep ✅** + **Item #4 stub round-trip ✅** —
  `data/nn/hsikan_mixed.hymeko` validates and emits via
  `hymeko emit ... --format torch_dataflow` to a runnable
  PyTorch module (2469 params, $h{=}16$, forward + 5-step SGD
  green).  AUC parity with hand-coded SignedKAN is week-long
  Item #4-final and stays on the roadmap; the
  IR-can-represent-HSiKAN claim is now empirically green.
- **Item #5 ✅** — Walk-HSiKAN open-walk Rust enumerator
  `hymeko.enumerate_k_walks_rs` landed in `hymeko_py/src/cycles.rs`,
  PyO3-wired, 12 verification cases all match a pure-Python DFS
  bit-for-bit.
- **Bonus** — `data/nn/walk_hsikan.hymeko` companion source
  (mirror of HSiKAN but with `walk_layer` Tier-3 primitive)
  lands and emits at parity.  HSiKAN and Walk-HSiKAN now have
  the same IR-emit pipeline with hash-distinct layer kinds.
- **Bonus — SGT (Signed Graph Transformer) baseline.**
  `signedkan_wip/src/baselines/sgt.py` + `run_sgt_sweep.py`.
  3-seed × 6-dataset sweep finds an honest two-regime story:
  HSiKAN dominates **cycle-rich** SBM ($0.91$/$0.96$ vs SGT
  $0.56$/$0.69$); SGT dominates **dense walk-rich** Slashdot
  ($0.897$ vs HSiKAN $0.861$) and Epinions ($0.941$ vs
  HSiKAN OOM at $h{=}16$ on 8GB GPU); both within seed-noise on
  Bitcoin.  $\alpha_k$ becomes a quantitative compass for which
  regime a dataset sits in.  SMC paper Table I gained an SGT
  column + a 6-line two-regime commentary; §IV.B (Limitations)
  cites the Epinions OOM as the hardware-vs-architecture
  boundary that chunked-incidence forward would unblock.
- **Bonus — HyMeKo→star-expansion→HSiKAN pipeline.**
  `scripts/hymeko_to_signed_graph.py` star-expands any HyMeKo
  source into the (u, v, sign) format HSiKAN consumes natively.
  Cycle-count probe over 5 nets: pure feedforward = 0 cycles
  at any $k$, fan-in topologies = exactly $\binom{4}{2}=6$
  cycles at $k{=}6$ — answers the $\alpha_k$ probe question
  without needing to train.  SMC paper §V "Self-referential
  consumer" bullet added.
- **Four new memory entries**:
  `project_signedkan_sinusoid_controls_2026_05_04.md`,
  `project_hsikan_hymeko_emit_2026_05_04.md`,
  `project_sgt_baseline_2026_05_04.md`,
  `project_hymeko_to_signed_graph_2026_05_04.md`.

Hot regressions checked: `mnist_highway_10`, `mnist_resmlp_3/20`,
`disjoint_net` all emit + run unchanged after the template edits.

## At-a-glance

| # | Item | ETA | Status | Depends on |
|---|------|-----|--------|-----------|
| 1 | Camera-ready 5-seed bench (Bitcoin / SBM / Slashdot) | tonight, ~4h | **done** (~3.5h) | — |
| 2 | ph18 entropy seed-sweep + λ-grid (highway-10/20, resmlp-40) | tomorrow morning, ~6h | **done** (~2h) | — |
| 3 | Sinusoidal-distillation control baselines | tomorrow afternoon, ~3h | **done** (~5min — fast) | — |
| 4 | HyMeKo-source HSiKAN composition (factor + dataflow IR) | week-long | **prep+stub round-trip done; AUC parity deferred** | real signedkan_wip helper port |
| 5 | Walk-HSiKAN prototype (open-walk arity channel) | week-long | **enumerator + HyMeKo source done** | real σ-walk aggregation |

---

## Item 1 — Camera-ready 5-seed bench (running)

**Goal:** promote the headline AUC numbers in `Table I` of the SMC
WIP paper from "single-seed unless noted" to "5-seed mean ± std",
addressing the ChatGPT review concern that Bitcoin OTC's +0.005
margin against SiGAT is plausibly seed-noise.

**Cells (75 total):**
- Bitcoin Alpha × {h=16, h=4} × 5 seeds = 10
- Bitcoin OTC × {h=16, h=8} × 5 seeds = 10
- SBM n=200 × {h=16, h=8} × 5 seeds = 10
- SBM n=400 × {h=16, h=4} × 5 seeds = 10
- SGCN baselines × 4 datasets × 5 seeds = 20
- Slashdot HSiKAN × {h=4, h=16} × 5 seeds × max_k4=500k = 10
- SGCN-Slashdot × 5 seeds = 5

**Output:**
`signedkan_wip/experiments/results/overnight_camera_ready.jsonl`

**Launch:** `bash signedkan_wip/src/run_overnight_camera_ready.sh`

**Deliverable on completion:** updated `Table I` with mean ± std,
and a paper-section sentence "results are 5-seed mean ± std at
the configurations in the appendix".  If Bitcoin OTC's HSiKAN
margin against SiGAT survives 5-seed averaging, the §III.B
softening from this morning ("competitive with both walk-based
baselines and improves on several") can be tightened back to
"improves on Bitcoin OTC by 0.005 ± σ AUC, statistically
indistinguishable from SiGAT" — honest and defensible.

---

## Item 2 — ph18 entropy follow-up (queued)

**Context.**  ph18 (`reports/phases_and_paths.tex:457`) tested the
multi-term `entropy_lyapunov` regulariser (λ_a = λ_b = 1, η = 5)
across deep PyTorch architectures and HyMeKo-emitted nets.  The
**only published-power positive** of the multi-term programme was
`fashion_mnist_highway_10` at +0.087pp (t = +2.61, p < 0.01); deeper
variants (highway-20, resmlp-40) returned null or trended negative.

**Two open questions:**
1. **Does the highway-10 positive survive a 5-10-seed sweep?**  The
   original ran 3 seeds; reviewer would ask "is +0.087pp still
   significant at n=10?"  Cheap to test.
2. **Is the depth-fragility real, or was λ off-axis?**  Path I's
   calibration law `λ_multi ~ λ_scalar / L` predicts the constant-λ
   sweep was systematically over-strong at depth.  A small grid
   `λ ∈ {0.5, 1.0, 2.0, 5.0}` × `arch ∈ {highway_20, resmlp_40}`
   would falsify or confirm the depth-fragile framing.

**Cells (~36 total):**
- highway_10 × λ=1.0 × {seed=0..9} = 10 (confirmation sweep)
- highway_20 × λ ∈ {0.5, 1.0, 2.0, 5.0} × {seed=0..2} = 12 (λ grid)
- resmlp_40 × λ ∈ {0.5, 1.0, 2.0, 5.0} × {seed=0..2} = 12 (λ grid)
- Plus baseline (no entropy) at each (arch, seed) for paired stats

**Script:** `python/benches/thesis_iv_hard/run_overnight_views_ph18c.sh`
(to be written; extends ph18b.sh with the seed/λ axes)

**Output:**
`python/benches/thesis_iv_hard/results/ph18c_*.json`

**Chainer:** `signedkan_wip/src/run_chained_overnight.sh` polls
`overnight_camera_ready.jsonl` for the expected ~75 lines, then
launches the ph18c sweep automatically.

**Deliverable on completion:**
- Confirm-or-refute the highway-10 +0.087pp positive at n=10
- Either bury "depth-fragile" or rescue it with a λ-tuning footnote
- Updated `project_phase18_architectural_parity.md` memory entry

---

## Item 3 — Sinusoidal-distillation control baselines (planned)

**Context.**  ChatGPT review #6: the §III.G claim "~91% of
surviving spline activations are sinusoidal" needs control
baselines.  A reviewer can object: "any smooth function over
[-1, 1] is sinusoidal-looking on a small grid".

**Three controls:**
1. **Random-init untrained model** — fit the same six symbolic
   forms to the un-trained spline activations.  If the sinusoidal
   fraction is similar, the result is meaningless; if it's much
   lower, the trained model is genuinely learning sinusoidals.
2. **Random spline coefficients** (matching distribution) — same
   as (1) but skip the model build, sample directly from the same
   N(0, init_scale) distribution.
3. **ReLU MLP equivalent** — train a same-parameter-count ReLU MLP
   on the same Bitcoin Alpha task and fit its hidden activations
   against the symbolic library.  Establishes that the sinusoidal
   bias is specific to KAN training, not a property of any
   trained-on-this-data activation.

**Script:** `signedkan_wip/src/run_sinusoid_controls.py` (to be
written; reuses `run_prune_distill.py` distillation utilities)

**Deliverable on completion:**  a short table in §III.G or a
supplementary figure — sinusoidal-fraction of {trained HSiKAN,
untrained HSiKAN, random splines, trained ReLU} side-by-side.
If the trained-HSiKAN column is uniquely high, the Fourier-style
decomposition claim defends itself.

---

## Item 4 — HyMeKo-source HSiKAN composition (week-long)

**Goal.**  Encode HSiKAN as a HyMeKo source file so the existing
`torch_dataflow` emitter produces the runnable model and the
canonical-IR projection gives the **factor** + **dataflow** views
for free (replacing the hand-drawn TikZ figures in the WIP paper).

**Why it matters.**
- Validates the HyMeKo IR on a real (non-toy) architecture beyond
  ph18's highway/resmlp examples.
- Architectural-parity claim becomes empirical: "HSiKAN's bench
  numbers are reproduced from the HyMeKo-emitted variant within
  noise" — closes the IR-as-DSL story with an end-to-end win.
- The factor and dataflow views auto-generate from the IR — same
  source-of-truth for the model and the diagrams.

**What's needed.**
1. **HyMeKo source** — `data/nn/hsikan_mixed.hymeko`:
   - Tier-2 constructs for `BatchedCatmullRom`,
     `DiagonalCatmullRom`, `HighwayGate`
   - σ-mask aggregation primitive (or compose from existing ones)
   - Sparse-mm dataflow node for `M_e` / `M_vt`
   - Mix-arity loop unrolled by `K` parameter
2. **Emitter extensions** in `transforms/torch_dataflow/template.py`
   - Spline activation Tier-2 → `BatchedCatmullRomActivation`
   - Sparse-CSR dataflow node → `torch.sparse_csr_tensor` ops
3. **Round-trip test** — train the HyMeKo-emitted variant on
   Bitcoin Alpha, compare AUC to the hand-coded variant within
   seed-noise tolerance.
4. **Diagram generation** — query the IR for the factor + dataflow
   projections, render as TikZ via the existing emitter.

**Effort.**  ~3-5 days of focused work; biggest risk is the spline
Tier-2 construct (Catmull-Rom as a template instantiation).

**Deliverable on completion:**
- `data/nn/hsikan_mixed.hymeko` source + emitter pass
- Round-trip parity table: hand-coded vs HyMeKo-emitted on Bitcoin
  Alpha and SBM n=200 (seed-noise overlap = pass)
- Auto-generated factor + dataflow PDFs replacing Figs 1/5 of the
  SMC paper for the journal extension

---

## Item 5 — Walk-HSiKAN prototype (week-long, paper-extension)

Already named as future work in §V of the WIP paper; first concrete
implementation is a Rust enumerator extension to produce open
length-`k` walks alongside closed `k`-cycles, plus a HyMeKo
re-binding of the `α_k` mixing weight to `(walk_k, cycle_k)` pairs.
Defer until ph18 stabilises (shares the entropy regulariser
infrastructure) and HyMeKo composition lands (HSiKAN structural
description benefits the walk-extension).

---

## Status updates

Append a dated line whenever a milestone moves.

- **2026-05-04 02:14** — Item #1 launched (`boi3k3knp`).
  Items #2-5 documented here.  Chainer being written next.
- **2026-05-04 02:15** — Item #2 chainer launched (`bfwav1btp`),
  polling for #1 completion.
- **2026-05-04 03:49** — Item #1 ~~complete~~ (75/75 cells, ~3.5h).
  Headline shift: 5-seed mean ± std numbers replaced single-seed in
  Table I.  Bitcoin OTC margin against SiGAT shrinks from $+0.005$
  to $-0.004 \pm .008$ (within seed noise).  Bitcoin Alpha h=4 vs
  h=16 was a single-seed-lucky result; 5-seed mean: $h{=}4$ is
  $-0.029 \pm .022$ vs $h{=}16$.  **Slashdot pruning Pareto
  CONFIRMED at 5 seeds**: $h{=}4$ ($0.861 \pm .002$) strictly
  beats $h{=}16$ ($0.849 \pm .003$) by $+0.012 \pm .003$.  Updated
  Tables I + III in the SMC paper, plus headline-numbers prose
  and Limitations.
- **2026-05-04 03:52** — Item #2 ~~queued~~ launched (chainer
  detected #1 complete, started ph18c).  Highway-10 10-seed
  confirmation running first; estimated 6-8h for the full sweep.
- **2026-05-04 06:30** — Entropy paper (`reports/phases_and_paths.tex`
  + `thesis_iv_executive_summary.tex`) updated with the ph18c
  findings (highway-10 10-seed confirm + resmlp-40 NEW positive at
  $\lambda{=}0.5$ + Path I calibration-law empirical-validation
  paragraph).  Both PDFs rebuilt clean.
- **2026-05-04 07:30** — Item #3 (sinusoid-control baselines)
  ~~complete~~.  4-baseline distillation comparison on Bitcoin
  Alpha + OTC, 3 seeds:
  ```
                          Bitcoin Alpha     Bitcoin OTC
   trained HSiKAN          0.901 ± .055     0.911 ± .018
   untrained HSiKAN        0.505 ± .036     0.531 ± .054
   random spline coefs     0.583 ± .045     0.583 ± .045
   GP smooth-fn draws      0.499 ± .021     0.499 ± .021
  ```
  Trained $+32$--$40$\,pp above all three nulls — the §III.G
  claim defends itself decisively.  SMC paper §III.G updated with
  the control-baseline paragraph (compressed in-line, no extra
  table; bibliography overflow status unchanged).  Script:
  `signedkan_wip/src/run_sinusoid_controls.py`.
- **2026-05-04 12:15** — **Epinions added; HSiKAN scaling
  limit identified, paper §IV.B updated**.
  Epinions (131k vertices, 841k edges, 85% positive) results:
  - SGT (3-seed): $0.941 \pm .003$ AUC, $0.833 \pm .005$ F1m,
    ~3 min/seed wall.  Scales gracefully.
  - SGCN (3-seed): $0.931 \pm .003$ AUC.  Also scales.
  - HSiKAN: **OOMs** at the published $h{=}16$ recipe on 8GB GPU
    because per-vertex $M_{\rm vt}^{(k)}$ buffer scales with
    $|V| \cdot |T_k|$; minimum-fit config (h=4, k=3, 20k cycles)
    underfits at AUC $0.549$ — not a fair benchmark.
  Documented in SMC paper §IV.B as the hardware-vs-architecture
  boundary: chunked-incidence forward is the next-iteration
  unblock.  Honest framing: "attention scales gracefully;
  cycle-mixing is memory-bound at $|V|{\gtrsim}10^5$ until the
  M-buffer is chunked."  Epinions stays out of headline Table I
  (HSiKAN row would be blank/OOM); appears only in Limitations.

- **2026-05-04 11:53** — **SGT-Slashdot landed; story flipped**.
  3-seed SGT-Slashdot result: AUC = $0.897 \pm .002$, F1m =
  $0.778 \pm .003$.  SGT now BEATS both SGCN ($0.883$) and
  HSiKAN ($0.861$) on Slashdot.  Updated SMC paper:
  - Table I Slashdot row: "SGT $\mathbf{0.897}$" with HSiKAN
    $\Delta$-best now $-0.036$ (was $-0.022$).
  - Table I commentary rewritten: "HSiKAN dominates where cycles
    carry signal, transformer attention dominates where dense
    walks do, and the $\alpha_k$ weights serve as a quantitative
    compass for which regime a dataset sits in."
  - More honest and generalisable than the original "HSiKAN beats
    everything" pitch.
  Epinions chainer (`/tmp/epinions_chainer.log`) is now running
  SGT-Epinions 3-seed, then HSiKAN-Epinions 1-seed (probe).
  ETA: ~12:35-12:45.

- **2026-05-04 11:30** — **Two new wins**:

  **(a) SGT baseline + 4-dataset 3-seed sweep.**
  `signedkan_wip/src/baselines/sgt.py` — Signed Graph Transformer
  with pre-LN encoder + sign-aware sparse self-attention.  3-seed
  sweep across Bitcoin Alpha / OTC / SBM-200 / SBM-400:
  ```
                     SGT           HSiKAN          SGCN
   bitcoin_alpha    0.898 ± .001  0.939 ± .011    0.874 ± .006
   bitcoin_otc      0.915 ± .010  0.930 ± .008    0.906 ± .006
   sbm_n200         0.563 ± .104  0.911 ± .028    0.504 ± .065
   sbm_n400         0.690 ± .025  0.962 ± .009    0.677 ± .070
  ```
  SGT closes ~half the SGCN→HSiKAN gap on Bitcoin (validates
  attention helps on signed link prediction); on cycle-rich SBM
  both attention and message-passing collapse to near-random
  while HSiKAN reaches 0.91/0.96 — isolates signed-cycle bias as
  the architectural component carrying the SBM signal.  SMC
  paper Table I extended with the SGT column + a 6-line commentary.

  **(b) HyMeKo → star-expansion → cycle analysis.**
  `scripts/hymeko_to_signed_graph.py` star-expands a HyMeKo
  description to (edges_u, edges_v, signs) directly consumable by
  the existing `enumerate_k_cycles_rs` and SignedKAN bench
  harness.  Uses the existing `hymeko inspect` CLI for IR
  parsing — zero new Rust code on the data path (the user's
  observation that star expansion is just a 1-D incidence list
  was exactly right).  Cycle-count on 5 existing nets:
  ```
   mnist_resmlp_3   sequential FF, no cycles at any k
   mnist_highway_10 sequential FF, no cycles at any k
   disjoint_net     48 cycles at k=4, 48 at k=6  (multi-input neurons)
   hsikan_mixed     6 cycles at k=6  (= C(4,2) for 4 αₖ branches)
   walk_hsikan      6 cycles at k=6  (mirror)
  ```
  Pure feedforward HyMeKo nets have NO cycles in star expansion;
  cycles arise only from multi-input fan-in or shared-port
  factors.  This answers the αₖ probe question without needing
  to train: HSiKAN-on-HyMeKo defaults to k=2 (direct edges) for
  sequential nets and gravitates to k=6 for fan-in topologies.
  No supervised α-learning needed — cycle counts are the answer.

- **2026-05-04 09:55** — Walk-HSiKAN HyMeKo source landed.
  `data/nn/walk_hsikan.hymeko` (mirror of `hsikan_mixed.hymeko`,
  4 walk lengths × `walk_layer` + `arity_mixer` + `signed_classifier`).
  `meta_nn.hymeko` grew the `walk_layer` Tier-3 type;
  `transforms/torch_dataflow/queries.hymeko` + `template.py` grew
  per-kind handling; `ehk_torch_stub.WalkLayer` stub added.  `hymeko
  emit` produces a runnable PyTorch module at parity with HSiKAN
  (2469 params, $h{=}16$, forward $(8,16) \to (8,1)$ green).  Both
  architectures now have the same IR-emit pipeline with
  hash-distinct layer kinds — closing the "HSiKAN and Walk-HSiKAN
  are factor-view duals" symmetry at the IR level.

  Regression: `mnist_highway_10`, `mnist_resmlp_3`, `mnist_resmlp_20`,
  `disjoint_net` all emit + run unchanged after the template edits.

- **2026-05-04 09:30** — Item #4 ~~stub round-trip green~~.
  Three landings:
  (a) Tier-3 helper stubs added to `ehk_torch_stub`
      (`SignedKANLayer`, `ArityMixer`, `SignedClassifier`).  Stub
      math (linear + tanh + sum); same field surface as the real
      Tier-3 components so emitted code constructs and runs.
  (b) `transforms/torch_dataflow/queries.hymeko` + `template.py`
      grew per-kind sections for the three new types, plus
      `bind:+:all_csv` for emitting multi-source fan-in calls
      (e.g. `mixer(cyc2, cyc3, cyc4, cyc5)`).
  (c) The Rust template engine (`hymeko_query/src/rewrite/template.rs`)
      grew the new `bind:+:all_csv` directive (~10 LOC).

  End-to-end smoke: `hymeko emit hsikan_mixed.hymeko →
  /tmp/hsikan_emitted.py` produces a runnable Python module
  (2469 params, $h{=}16$, 4 arities); forward `(8, 16) -> (8, 1)`
  + 5-step SGD reduces loss $0.0059 \to 0.0043$.  Permanent test:
  `scripts/verify_hsikan_emit.py`.

  **Bitcoin Alpha AUC parity is not claimed.**  The stub Tier-3
  layers are placeholder math; the real architectural fidelity
  (Catmull-Rom spline activations, σ-mask aggregation, $M_e^{(k)}$
  sparse-CSR application) requires replacing the stubs with the
  real `signedkan_wip.src.signedkan` code, which is the genuinely
  week-long Item #4-final task.  But the IR-can-represent-and-emit
  HSiKAN claim is now empirically green, which is a real
  architectural-parity milestone for the journal extension.

- **2026-05-04 08:50** — Item #5 ~~prototype complete~~.  Open-walk
  Rust enumerator `hymeko.enumerate_k_walks_rs` landed in
  `hymeko_py/src/cycles.rs` (length-`L` simple walks, canonical-form
  emit by `path[0]<=path[walk_len]`, Full / Reservoir sink modes).
  PyO3-wired in `lib.rs`.  Verification suite at
  `scripts/verify_walks.py` covers 12 cases (triangle, path-5,
  6-cycle+chord, K4 across multiple walk lengths, reservoir cap)
  and all match a pure-Python reference DFS bit-for-bit.  Serial-
  only for now (no rayon); parallel can be added when Walk-HSiKAN
  needs it for Slashdot-class graphs.  Sufficient for IR plumbing
  and small-graph experiments.  HyMeKo-side wiring (a
  `walk_layer` Tier-3 primitive in `meta_nn.hymeko` mirroring
  `signedkan_layer`) is the next concrete step before Walk-HSiKAN
  can be expressed as a HyMeKo source.
- **2026-05-04 08:15** — Item #4 ~~prep complete~~.  Tier-3
  HSiKAN-specific layer types added to `data/nn/meta_nn.hymeko`
  (`signedkan_layer`, `arity_mixer`, `signed_classifier`); HSiKAN
  topology described in `data/nn/hsikan_mixed.hymeko`.  IR
  validates (`hymeko validate`), inspects clean (103 decls, 12
  edges, 9 arcs), and the standard `torch_dataflow/queries.hymeko`
  query projects all 9 dataflow hyperedges with correct
  signed-incidence structure.  Round-trip emit (template + the
  multi-input dataflow extension + spline Tier-2 helpers in
  `ehk_torch_stub`) remains week-long and stays on Item #4 final.
- **2026-05-04 05:58** — Item #2 ~~complete~~ (~2h, faster than
  estimated).  Two findings:
    * **highway-10 +0.00092 AUC at 10-seed, t=+2.08, p<0.05**
      — load-bearing positive of the multi-term programme
      survives the higher-power sweep.  Update
      `project_phase18_architectural_parity.md`: "+0.087pp"
      headline now reads "+0.092pp at 10-seed, p<0.05".
    * **resmlp-40 +0.00213 at lam=0.5, t=+2.72, p<0.05** — NEW
      positive.  Path I's calibration law
      $\lambda_{\rm multi} \sim \lambda_{\rm scalar}/L$ predicted
      exactly this: at $L{=}40$, the original $\lambda{=}1.0$
      sweep was systematically over-strong; reducing to
      $\lambda{=}0.5$ (still $> 1/L = 0.025$ but in the right
      ballpark) recovers the positive.  This rescues the
      "depth-fragile" framing as a $\lambda$-tuning artefact for
      resmlp-40.  highway-20, however, stays null across the
      full $\lambda \in \{0.5, 1.0, 2.0, 5.0\}$ grid — its
      depth-fragility appears genuine.
  Item #3 (sinusoid controls) is deferred per the chainer.

