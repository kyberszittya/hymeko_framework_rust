# HyMeKo Seminar — Master Demo Outline

The umbrella plan for the Kato-lab seminar: the deck, the live demos, the
3D viewer, and how they fit together. Detailed build specs live in the three
linked documents — this file is the index + run-of-show, not a duplicate.

## The talk
HyMeKo — *A Hypergraph Model Cognition Framework*. Mixed academic audience,
English. One thesis, two threads, one bridge:

> **one hypergraph substrate, two payoffs** — a Rust framework that *represents
> and transforms* hypergraphs, and a structural-prior model family that *learns*
> from their cycles. The graph the framework compiles is the graph the model
> reasons over.

Deck: `HyMeKo_Seminar.pptx` (23 slides, white / purple-blue-green, Futura +
Bahnschrift). Story: motivation → what HyMeKo describes → goals → **Part I
framework** (DSL · compiler · star/clique · zero-copy · transforms · SysML) →
**Part II science** (structural priors · SignedKAN→HSiKAN→Gömb · results ·
efficiency · honesty · transfer) → bridge → contributions → outlook.

---

## Asset inventory
| Artifact | Path | Status |
|---|---|---|
| Seminar deck | `HyMeKo_Seminar.pptx` *(currently in the session outputs — commit into the repo, e.g. `paper/seminar/`)* | ready (23 slides) |
| Kinematics demo video (embedded in deck slide 19) | `demo_video/hsikan_mujoco_4dof.mp4` | ready |
| Star-expansion 3D viewer | `demo_web/star_expansion_viewer.html` | prototype (synthetic data) |
| Regime-compass web demo | `demo_web/index.html` + `kinematic_data.json` | ready |
| Trained weights | `checkpoints/hsikan/*`, `checkpoints/kinematic/*`, `checkpoints/hymeyolo_demo/*` | ready |
| Inference harness | `signedkan_wip/src/demo/` (`inference.py`, `checkpoint.py`, registry) | partial |
| Latency bench | `signedkan_wip/experiments/runs/run_inference_bench.py` + `inference_bench.json` | partial |

## Detailed specs (build instructions for Claude Code)
- **Five seminar demos** → `signedkan_wip/demos/SEMINAR_DEMOS.md`
- **Inference harness / per-demo gates** → `docs/INFERENCE_DEMOS_OUTLINE.md`
- **Star-expansion viewer (phased, live option)** → `demo_web/STAR_EXPANSION_VIEWER_OUTLINE.md`

---

## The demo program (inference-only — no training in the room)
| # | Demo | Backs slide | Detailed spec | Status |
|---|---|---|---|---|
| 1 | **Affective balance** on a signed hypergraph — frustrated cycles, balance score | 12 (structure→prior) | SEMINAR_DEMOS §1 | extend (cycle scorer = frustration index) |
| 2 | **HIVE compilation** — surface `.hymeko` → IR → COO/CSR; two isomorphic inputs → same Blake3 hash | 6–7 | SEMINAR_DEMOS §2 | reuse `compile_star_expansion` + `canonical_hash` |
| 3 | **HSiKAN/Gömb forward** — load checkpoint, AUROC live + per-edge confidence, αₖ regime | 17 (results) | INFERENCE_DEMOS Demo 1 + SEMINAR_DEMOS §3 | reuse `demo/inference.py` |
| 4 | **Mesh + chiral ablation** — WL+sign hash → HSiKAN → Sinkhorn; signed≈1.0 vs unsigned≈0.5 | 18 (transfer) | SEMINAR_DEMOS §4 | mesh data reuse; **Sinkhorn matcher to write** |
| 5 | **Sim→perception bridge** — scene as IR feeding perception (roadmap sketch) | 19 / 23 | SEMINAR_DEMOS §5 | thin, build last |
| + | **Star-expansion 3D viewer** — star vs clique blow-up, live counts | 7 (star vs clique) | STAR_EXPANSION_VIEWER_OUTLINE | prototype ready |
| + | **Forward-latency bench** — within-family width gap ~3.5× CPU (h4 vs h16), ≈1× CUDA; SGCN for context | 18 (efficiency) | INFERENCE_DEMOS Demo 2 | **done** (2026-06-11) |

---

## Run-of-show (≈ 50–60 min)
1. Open on **Demo 1 (balance)** — their language; frustrated cycles. *(~8 min)*
2. Deck Part I → **Demo 2 (HIVE)** + the **star-expansion viewer** at the star/clique slide. *(~12 min)*
3. Deck Part II → **Demo 3 (HSiKAN/Gömb forward)**, then the **latency bench** at the efficiency slide. *(~14 min)*
4. **Demo 4 (mesh/chiral)** at the transfer slide; the **MuJoCo video** already plays in the bridge slide. *(~10 min)*
5. Close on **Demo 5 (sim→perception)** as the roadmap. *(~6 min)*
6. Contributions + outlook + questions.

Each demo has a `--quick` smoke path for rehearsal; on stage, prefer the
precomputed/checkpointed path.

---

## Shared conventions (all demos)
- Single entry: `python -m signedkan_wip.demos.seminar <demo> --device auto --seed 0`; one file with a mode arg, not N scripts (CLAUDE.md §6.5 #13).
- Pretrained weights / fixed inputs only — **no training in the room**.
- Deterministic (fixed seed); on exit print peak RSS + wall time; assert RSS < 16 GB.
- Outputs to `demo_out/<demo>/`; every printed number names the checkpoint/fixture it came from.

## Build order (across all docs)
1. **Demo 3** (link inference) — validates the shared harness.
2. **Star-expansion viewer Phase 1–3** — visual, low-risk, reuses `export_kinematic_data.py`.
3. **Demo 2** (HIVE/canonicalization) — pure transform.
4. **Latency bench** extension — produces the slide-18 numbers.
5. **Demo 1** (balance) — reuses the cycle enumerator.
6. **Demo 4** (mesh + Sinkhorn) — only one needing genuinely new code.
7. **Demo 5** + viewer Phase 4 (live) — last, optional.

## Honest-presentation checklist (say these aloud)
- **Results:** Gömb-strict trails SGCN on Bitcoin/Slashdot; the win is **Epinions** + **accuracy-per-parameter**. Frame as "competitive-to-leading at a fraction of the cost," not flat "SOTA."
- **optuna_best 0.996/0.993:** different (transductive) evaluation convention; strict is the architectural baseline.
- **Latency:** HSiKAN's *absolute* forward is heavier than SGCN; the measured within-family width gap (h4 vs h16, **same device**) is **~3.5× on CPU and ≈1× on CUDA** — show SGCN, HSiKAN-lean, HSiKAN-joint together. (The earlier "11×" was the optuna_best_otc-vs-joint figure — OTC-specific and tuple-set-driven, *not* a general width claim; Alpha's optuna config runs ~2× *slower*. See `reports/2026-06-11-latency-bench-extension.md`.)
- **HyMeYOLO mAP:** quote the corrected 0.903 ± 0.009, never the bug-inflated 0.723.
- **Star-expansion viewer:** the 3D layout is force-directed for legibility, **not geometric ground truth**; the edge-count arithmetic is exact and engine-sourced.
- **Mesh ablation:** the ~1.0 vs ~0.5 gap must be **measured live**, not assumed.

## Fonts note (deck)
Headers are set to **Futura**; if it isn't installed on the presenting machine, PowerPoint substitutes (Century Gothic is the closest Windows-native match). Body is **Bahnschrift** (ships with Windows 10+).
