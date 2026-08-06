# HyMeKo — Seminar Demo Plan (Kato Laboratory)

Five **inference-only** demos — pretrained weights or fixed inputs, forward
passes only, **no training in the room**. One per ~10 minutes. The ordering is
the story:

> **balance** (their world) → **how HyMeKo represents it** → **it actually predicts** → **it transfers to perception** → **here's where it goes**

Companion spec: `docs/INFERENCE_DEMOS_OUTLINE.md` (the harness-level view).
Build on existing scaffolding — do not fork it (CLAUDE.md §6.1). Fixed seeds,
peak-RSS + wall-time on exit, ≤ 16 GB, CPU-runnable (cuda auto-detected).

Proposed single entry: `python -m hymeko_neuro.demos.seminar <demo> [...]`,
`--device auto --seed 0`, writing to `demo_out/<demo>/`. One file with a mode
arg, not five scripts (CLAUDE.md §6.5 #13).

---

## 1 — Affective balance on a signed hypergraph  *(the Kato-aligned opener)*
**Angle:** their language — structural balance of affective relations.

- **What it shows:** load a signed affective-relation hypergraph, compute its balance, and surface the **frustrated cycles** — the configurations that cannot be consistently signed. Output: a global balance/frustration score + a "where the tension is" highlight view.
- **Reuse (on disk):**
  - Stage-1 synthetic generator — `hymeko_neuro/experiments/cortical/synthetic.py` (`make_synthetic_cichy_like`, frozen `SyntheticCorticalDataset`).
  - Cycle enumeration + the **`fraction_negative` scorer** — this *is* the frustration index (negative sign-product over a cycle); see `hymeko_neuro/hyperedge/n_tuples.py` and the Rust `enumerate_top_k_cycles_rs` path exposed in `hymeko_py`.
  - `hymeko_neuro/experiments/topk_cycle_demo.py`, `src/demo/gui.py`, `src/rapport/coherence.py` for the highlight rendering.
- **Build:** a `balance` mode that (a) builds/loads the signed affective hypergraph, (b) enumerates short cycles, (c) reports frustration = fraction of cycles with negative sign-product, (d) renders the top-frustrated cycles.
- **Output / metric:** global balance score ∈ [0,1] (1 = perfectly balanced); list + view of the most frustrated cycles.
- **Acceptance:** on a hand-constructed balanced graph → score 1.0, zero frustrated cycles; on a planted-unbalanced one → score drops, the planted cycle is surfaced. Deterministic from seed.
- **Honest note:** "balance/frustration" here is the cycle sign-product statistic, not a learned quantity — say so; it sets up demo 3 where the same cycles drive prediction.
- **Slide:** opener; ties to "From structure to inductive prior" (12).

---

## 2 — HIVE compilation, live  *(how HyMeKo represents it)*
**Angle:** pure transform, very visual, no model.

- **What it shows:** one hypergraph through surface **HyMeKo language → canonical IR → tensor encodings (COO / CSR)**, side by side. Punchline: **canonicalization** — feed two isomorphic-but-differently-written inputs and show they hash to the **same structural fingerprint**.
- **Reuse (on disk):**
  - Canonical hash (WL-style ordering + Blake3) — `hymeko_core/src/ir/canonical_hash.rs` (`CanonHashCfg`), `ir/hash.rs`, `ir/hash_pass.rs`.
  - Parse + expansions via `hymeko_py` (PyO3): `.hymeko` → IR → star/clique COO/CSR (`hymeko_hre` / `hymeko_hnn`).
- **Build:** a `compile` mode (or a thin notebook) that prints, for an input pair, the surface text, the lowered IR, the COO and CSR tensors, and the two canonical hashes — asserting hash equality for the isomorphic pair and inequality for a genuinely different graph.
- **Output / metric:** two hashes shown equal; COO/CSR shown side by side; NNZ counts (star vs clique) for the same graph (reuse the 1,498 vs 10,991 talking point).
- **Acceptance:** isomorphic-but-reordered inputs → identical Blake3 fingerprint; a one-edge change → different fingerprint. No model loaded.
- **Slide:** "From source text to a hypergraph IR" (6) + "Star vs clique" (7).

---

## 3 — Hypergraph neural inference: the Gömb / HSiKAN forward pass  *(the numbers are real)*
**Angle:** HSiKAN as the **middle shell**, not a standalone net.

- **What it shows:** signed link/sign prediction on a held-out graph (Bitcoin or Epinions), **pretrained weights loaded**, AUROC reported live with **per-edge confidence**.
- **Reuse (on disk):** `hymeko_neuro/experiments/demo/inference.py` (`ModelBundle`, `predict_test_edges`), `src/demo/checkpoint.py`, checkpoints `checkpoints/hsikan/bitcoin_{alpha,otc}_optuna_best.pt`. (This is Demo 1 of `docs/INFERENCE_DEMOS_OUTLINE.md`.)
- **Build:** the `link` CLI mode → table (AUC · F1 · n_params · fwd_ms), ROC + αₖ-regime figures to `demo_out/link/`.
- **Acceptance:** reproduces the checkpoint's committed AUC within ±0.002 on the fixed split.
- **Honest note:** these are `optuna_best` (transductive convention) — label as such; the **strict** protocol is the architectural baseline. Frame HSiKAN explicitly as the FIR → **HSiKAN-CR** → CPML middle shell.
- **Slide:** "Signed-graph link prediction results" (17).

---

## 4 — Mesh recognition with a chiral ablation  *(it transfers to perception)*
**Angle:** ties the hypergraph machinery to robotics/perception.

- **What it shows:** match two meshes via WL+sign hash → HSiKAN embedding → **Sinkhorn** correspondence, run live as a **signed-vs-unsigned ablation**: signed ≈ 1.0 correspondence vs unsigned ≈ 0.5. Visualize the correspondences.
- **Reuse (on disk):** `hymeko_neuro/data/datasets/meshes.py` — polyhedral signed-graph meshes with **face-coupled signing** (every face has a fixed sign-product); cycle-HSiKAN encoder (`mixed_arity_signedkan/`).
- **Build (not yet on disk — flag honestly):**
  - a **Sinkhorn matcher** over the two embedding sets (no `sinkhorn` impl currently in the tree — write it or pull a small dependency-free version),
  - the **chiral/signed-vs-unsigned ablation** toggle (encode with edge signs vs signs stripped),
  - correspondence visualization.
- **Output / metric:** correspondence accuracy (matched-vertex fraction) signed vs unsigned.
- **Acceptance:** signed correspondence ≫ unsigned on the polyhedral set; the ~1.0 vs ~0.5 gap is **measured and reported**, not assumed — if it doesn't reproduce, report the real numbers.
- **Slide:** transfer/perception (near HyMeYOLO, 18).

---

## 5 — Simulator → perception bridge  *(the forward-looking closer)*
**Angle:** the IR as connective tissue (the idea sketched on the 9th).

- **What it shows:** a thin end-to-end — a scene expressed as a **HyMeKo hypergraph IR** feeding a perception/embedding stage — the IR sitting between a simulator's scene graph and a detection/embedding pipeline.
- **Reuse (on disk):** `hymeko_emitter` / `hymeko_py` for scene-graph → IR; the HyMeYOLO detector checkpoint `checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt` and `src/vision/demo_hymeyolo_tk.py` as the perception stage.
- **Build:** a `bridge` mode that takes a small scene as `.hymeko`, lowers it to IR, and routes the structural encoding into the perception/embedding stage — minimal, clearly labelled as a **direction-of-travel** sketch.
- **Acceptance:** runs end to end on one scene; explicitly framed as roadmap, not a benchmarked result.
- **Honest note:** lightest demo — present it as "here's where this goes," not as a finished capability.
- **Slide:** "Conclusion & outlook" (23) / "The bridge" (19).

---

## Build order
1. **Demo 3** (link inference) — smallest, validates the shared harness; reuse the existing `demo/` package.
2. **Demo 2** (HIVE/canonicalization) — pure transform, no weights, high visual payoff.
3. **Demo 1** (balance) — the opener; reuses cycle enumeration from Demo 3's path.
4. **Demo 4** (mesh + Sinkhorn) — only one needing genuinely new code (the matcher).
5. **Demo 5** (sim→perception) — thin closer, build last.

## Cross-cutting (every demo)
- No training in the room — pretrained weights or fixed inputs only.
- `python -m hymeko_neuro.demos.seminar <demo> --device auto --seed 0`.
- Deterministic; on exit print peak RSS + wall time; assert RSS < 16 GB.
- A number is printed only if it can name the checkpoint/fixture it came from.
- Each demo ≤ ~10 min of stage time; have a `--quick` smoke path for rehearsal.
