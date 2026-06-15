# HyMeKo — Roadmap & Mission Control

**Last updated:** 2026-06-16 · Maintainer: Dr. Csaba Hajdu (with Aiko / Claude Code)

This is the single index that ties the project's tracking artifacts together. It
does **not** duplicate their content — it links them and adds the layer they
lack: **strategic goals**, **tactical goals**, and a **periodic performance log**.

| Artifact | Role |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Operating contract (rules for the agent). **Not** edited from here. |
| [BACKLOG.md](BACKLOG.md) | Open work — one line per item, 9 areas (51 open as of 2026-06-15). |
| [DONE.md](DONE.md) | Completed work with report links. |
| [docs/plans/](docs/plans/) | Per-task plans (4-format: tex/pdf/tikz/mmd). |
| [reports/](reports/) | Acceptance records — one per completed task. |
| `~/.claude/.../memory/MEMORY.md` | Cross-session project memory index. |

**Convention.** Strategic goals are long-horizon (quarters–years) and rarely
change. Tactical goals are the current focus (weeks) and are re-cut often. The
performance log is append-only: one dated snapshot per milestone.

---

## Resume here (2026-06-16 EOD snapshot)

Open threads for the next session, highest-leverage first:

- **DETR baseline — finish making it FAIR (open).** `vision/detr_baseline.py`
  (`MiniDETR`, standard + param-matched tiny) + tests are written; the
  set-loss/metric are reused (`hungarian_set_loss`, `match_f1_at_iou50`). The
  **overfit fairness guard FAILS** (mAP50_proxy 0.149 @ 200 steps) and is
  `xfail`'d. **Next:** get the from-scratch DETR to actually overfit
  (tune steps/lr/box-head init; the guard now uses single-object/600 steps as a
  starting point) → remove the `xfail` → run standard + tiny at 5000/40 →
  head-to-head vs RicciStim **0.228 @ 5 896 params**. **No parity claim until the
  guard passes** (a strawman DETR would be dishonest).
- **Stacking idea (user, 2026-06-16) — promising, needs a plan + measurement.**
  "The model is tiny (5.9k) → stack K of them orthogonally or radially for more
  streamlined calculation." My initial read: **orthogonal** = K decorrelated
  experts on orthogonal subspaces (or an orthogonality penalty on their
  representations), combined by an extended αₖ mixer — diversity without
  redundant compute; **radial** = the literal **Gömb (sphere)** realisation: K
  tiny shells around a shared core, each a radial direction/scale (the cross-scale
  pyramid + arity mixer are embryonic forms of exactly this). **Streamlining:** K
  tiny models batch as a single `(B·K, …)` forward ⇒ ~one model's wall-time for
  K× structured capacity, and orthogonality/radiality stops them recomputing the
  same thing. Caveat: must measure (more capacity may just overfit, or the
  constraint may hurt) — prototype an orthogonal stack of K tiny RicciStim heads
  and test accuracy-at-fixed-tiny-budget.
- **GCP GPU quota request (blocks Gömb).** Project `shuttle-xc-637318`,
  `GPUS_ALL_REGIONS = 0` → request an increase; only the user can submit. Gömb's
  signed-link runs OOM the 8 GB laptop card — cloud (or Komondor) is required.
- **Matched bare-sum/40-ep baseline + multi-seed** for the 0.228 result (isolates
  upgrade-vs-epochs; gives an error bar) — GPU job, ~3.5 h each.
- **Slide 26 (coworker):** HSiKAN edge_cr (Komondor) numbers are fillable; Gömb
  cells stay `[pending-cloud]`. Verified numbers are in the chat / the cache report.
- Lower priority: `HSiKANConfig` migration (config built; ~2 caller modules still
  use the env side-channel), vertex-prefilter, per-epoch JSONL logging in the
  detection runner.

---

## 1. Strategic goals (long-horizon)

- **S1 — Canonical hypergraph infrastructure.** HyMeKo as the *one source of
  truth*: a signed-typed hypergraph DSL + IR that compiles to many faithful
  targets (graph/3D views, SysML, URDF/SDF/MJCF, `torch.nn`, DOT, HIVE), with
  validation gates and structural parity. *"One source, many accountable
  targets."* Venues: IEEE SMC, IEEE T-SMC.
- **S2 — Structural-prior learning.** Cycles and walks as inductive features
  (SignedKAN / HSiKAN / Gömb), under a **strict, leakage-audited** protocol.
  Venues: Nature Communications (leakage audit), Elsevier (AC-HSiKAN), SISY.
- **S3 — Dataflow → silicon.** Nagare dataflow substrate → HSMM abstract machine
  → Zynq FPGA (theory → systems → compiler). Long-horizon.
- **S4 — Honesty as method.** The leakage audit, the σ-masked strict protocol,
  and structural-parity gates are presented as a *methodological contribution*,
  not a patch. This is a load-bearing part of the research identity.
- **S5 — Open question: does hypergraph structure help perception?** Vision is
  *empirically falsified for accuracy* (RicciStim/HSiKAN/HGNN < MLP on
  MNIST/Cluttered-MNIST). The un-falsified axis is **brain-predictivity**
  (Cichy-92). Treat as a question to settle honestly, not a goal to force.

## 2. Tactical goals (current focus — 2026-06)

- **T1 — "Turn vision positive" attempt.** Topology cache (P6) → full
  5000-img/40-epoch detection run (baseline vs upgraded). The upgrade is still
  climbing at 20 epochs; the cache makes the long run affordable. *In progress.*
- **T2 — Nature leakage paper.** Finish the 5-seed σ-masked strict grid →
  Table 1; lock the honest operating point. (`[[project-nature-leakage-paper]]`)
- **T3 — Signed-link baseline table.** Tier-2/3 audit + vertex-prefilter →
  3-dataset HSiKAN bench. Gates the signed-link paper.
- **T4 — PhD seminar (Kato).** Deck reframed per the Kato review (43 slides, two
  coupled contributions, four-block arc, Q&A/demo/collaboration). Deliver.
- **T5 — Hero / overview demo.** "One source, many lenses + validation gate"
  runbook (editor-driven). ~80% exists; needs sequencing.
- **T6 — IEEE TPAMI paper (deep method).** Signed-hypergraph structural priors
  for relational learning (HSiKAN: cycle/walk tuples, mixed-arity, Gömb cascade);
  breadth + ablations + Friedler-quotient param-efficiency, under the
  leakage-audited protocol. Vision out of scope (falsified). Plan:
  `docs/plans/2026-06-16-tpami-structural-priors/`. Gated by: prefilter →
  3-dataset bench, tier-2/3 baseline audit, strict 5-seed grid.
- **T7 — Nature-family HSiKAN paper (significance + rigor).** HSiKAN + the honest
  σ-masked protocol reshape signed-graph link prediction (inflated transductive
  numbers vs the strict operating point). Plan:
  `docs/plans/2026-06-16-nature-hsikan/`. **Decision open:** unify with the
  existing "Nature-Comm leakage" entry (recommended) vs split. Shares the
  experimental backbone with T6 — do not double-run.

## 3. Plans index (recent)

Active/recent under [docs/plans/](docs/plans/):
- `2026-06-16-generators-into-core` — generators moved into `hymeko_core` (done).
- `2026-06-15-hive-generators-parity` — HIVE Steiner/sunflower/complete (done; superseded by the core move).
- `2026-06-15-soma-vision-backbone-upgrade` — learned mixer + highway + pyramid (done; A/Bs reported).
- `2026-06-15-editor-*` — editor examples/generators/profiles/arcs/layout/bidirectional (done).
- `2026-06-15-hero-demo-phase{1,2,3}` — one-model-many-targets, gate, parity (done).
- `2026-06-16-tpami-structural-priors`, `2026-06-16-nature-hsikan` — publication plans (new; T6/T7).
- `2026-06-16-stim-geometry-cache` — RicciStim topology cache (done; enables the full vision run).
- `2026-06-13-hero-demo`, `2026-06-14-reachability-rules-audit-pgraph` — open lines.

(~80 plan dirs total; this lists the live ones. Full open work in BACKLOG.md.)

## 4. Backlog & done

- **Open:** [BACKLOG.md](BACKLOG.md) — 51 line-items across 9 areas; ~11 are P1.
  Highest-leverage P1s: tier-2/3 audit, Friedler-quotient, prefilter→3-dataset
  chain (gate the signed-link table); Phase-B shuffle + Nature-Comm audit (gate
  the leakage paper); reachability-rules pgraph; Stage D-3 / Stage H / cortical.
- **Done:** [DONE.md](DONE.md).

## 5. Performance log (append-only)

One dated snapshot per milestone. Numbers are the honest measured values; see the
linked report for provenance.

### 2026-06-16

- **Generators-in-core** (`reports/2026-06-16-generators-into-core.md`): algorithm
  single-sourced in `hymeko::generators`; `hymeko_core` 10 tests + `hymeko_hive`
  18 tests pass; clippy/fmt clean both. `S(2,3,25)` builds ≪ 100 ms.
- **Soma-vision aggregator upgrade** (`reports/2026-06-15-soma-vision-backbone-upgrade.md`),
  upgrade = learned-αₖ mixer + highway + cross-scale pyramid vs bare-sum baseline:
  - MNIST classification (3 seeds, 4 ep, CPU): **0.272 ± 0.032 → 0.303 ± 0.006**
    (+11 % mean, ~5× lower variance). 4736 → 5811 params.
  - Cluttered-MNIST detection, scale (2000 img, 20 ep, GPU): **0.106 (saturated)
    → 0.135 (+27 %, still climbing)** mAP50_proxy. 4821 → 5896 params.
  - **Honest:** the upgrade consistently beats the bare sum but does **not**
    overturn the vision falsification (below the 0.174 config-F headline at
    5000 img/20 ep; MNIST Linear baseline ≈ 0.91 vs ≈ 0.30). Per-image graph
    build is CPU-bound; topology cache (T1) is the enabler for the full run.
- **Framework article**: `docs/articles/hymeko-structural-accountability/article.pdf`
  (6 pp), honest single-seed framing.
- **Full vision run (T1 done)** — upgraded + cached RicciStim, Cluttered-MNIST
  config F, **5000 img / 40 epochs**, GPU, seed 0
  (`reports/ricci_stim_detect_full_20260616.jsonl`, ~3.5 h): final
  **mAP50_proxy 0.228**, crossing the prior bare-sum config-F headline (~0.174,
  5000 img/20 ep, 2026-05-16) at ~epoch 15 and plateauing ~0.21–0.23. **The
  topology cache made this 40-epoch run feasible (~3.5 h vs ~7 h+ uncached).**
  Honest caveats: single seed; not a perfectly matched A/B (upgraded/40 ep vs the
  old bare-sum/20 ep) — the matched 2000-img A/B already showed the upgrade at
  +27 % over bare sum, so the gain is real, but a bare-sum/40-ep run would isolate
  upgrade-vs-epochs; and 0.228 beats the prior *hypergraph-vision* number, it is
  not a claim of parity with a conventional detector. **Gömb (the strict cascade)
  still has no valid signed-link numbers — its local runs OOM'd; needs Komondor/GCP.**

_(Earlier milestones — VOC HyMeYOLO baseline mAP 0.0149 @ ep60, base-Soma 0.52 vs
linear 0.91 — are in the linked reports and `[[project-*]]` memories.)_
