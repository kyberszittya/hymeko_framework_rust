# Reusable results — for the Kato seminar deck

Curated 2026-06-16. Two sources: (A) the PhD defense deck
(`docs/seminar/defense_final_hun.pptx`, Hungarian, 38 slides) and (B) the
`reports/` tree. Every table below names where it came from. Honest-framing notes
follow the conventions already locked in `SEMINAR_SUMMARY.md` /
`SEMINAR_DEMO_OUTLINE.md` — do not quote a number without its protocol.

---

## A. From the PhD defense — directly reusable

### A1. DSL conciseness — HyMeKo vs URDF vs MuJoCo  *(defense slide 7, Thesis I)*
Character counts for the same kinematic structure. This is the cleanest evidence
for the deck's "one source, many targets / more compact than the XML schemas"
claim and pairs naturally with the **Query-driven transforms** slide (14).

| Kinematic structure | HyMeKo | URDF (XML) | MuJoCo (XML) |
|---|---:|---:|---:|
| 2-joint connection | 71 | 164 | 91 |
| SCARA | 101 | 252 | 129 |

→ HyMeKo is ~2.3–2.5× shorter than URDF and ~1.3× shorter than MJCF. Matches the
"≥ 20 % shorter" thesis claim (defense slide 2/13).

### A2. Structured-text vs JSON  *(defense slide 15, Thesis III)*
HyMeKo source vs JSON for the same description: 304 vs 601 chars, comparable
inference time. Supporting evidence for compactness; lighter-weight than A1.

| Attribute | HyMeKo | JSON |
|---|---:|---:|
| Length (chars) | 304 | 601 |
| Mean inference time (s) | 4.379 | 4.472 |
| Median inference time (s) | 4.235 | 4.335 |

### A3. Graph-entropy MNIST accuracy  *(defense slide 19, Thesis IV)*
Hypergraph graph-entropy backpropagation on a conv classifier — accuracy with vs
without the entropy term. A small but honest "structure-as-prior helps a little"
result; candidate for the structure→prior thread if you want a second domain.

| Metric | without entropy backprop | with entropy backprop |
|---|---:|---:|
| Min accuracy | 92.0 % | 92.1 % |
| Mean accuracy | 94.39 % | 94.45 % |
| Max accuracy | 96.13 % | 96.19 % |
| Std (accuracy) | 0.0050 | 0.0045 |

### A4. Rust IPC latency  *(defense slide 33)*
"~5 ms latency IPC (compute + transport)" — complements the zero-copy bridge
slide (13) and the within-family latency framing. The Rust framework (2026) is
the successor to the Python `himeko` line shown in the defense.

### A5. Figures worth lifting
- **Fano-plane description** (defense slide 36) — already mirrored by the deck's
  generated Fano figure; the defense framing/text can enrich the gallery notes.
- **Framework overview / CogInfoCom** (slides 35, 5) — a higher-level "where this
  sits" diagram if Kato wants the cognitive-systems framing.
- The **four-thesis spine** (DSL · tensor representation · kinematics/fuzzy ·
  neural-entropy) is a clean way to answer "what is the central contribution" —
  it maps onto the deck's two-contribution arc.

---

## B. Consolidated result tables from `reports/`

### B1. Headline — Gömb-strict, 5-seed (the honest number)
*Source: `reports/2026-05-14-executive-brief.md`.*

| Dataset | AUROC mean | ± pstd | per-seed |
|---|---:|---:|---|
| Bitcoin Alpha | 0.8972 | 0.0079 | 0.8877 · 0.9087 · 0.8901 · 0.8962 · 0.9035 |
| Bitcoin OTC | 0.9145 | 0.0068 | 0.9256 · 0.9047 · 0.9125 · 0.9127 · 0.9168 |
| Slashdot | 0.9017 | 0.0008 | 0.9007 · 0.9015 · 0.9015 · 0.9016 · 0.9033 |
| **Epinions** | **0.9526** | **0.0018** | 0.9532 · 0.9520 · 0.9499 · 0.9523 · 0.9555 |

### B2. Strict result in context of published baselines
*Source: `reports/2026-05-14-executive-brief.md` (Epinions).*

| Method | Protocol | AUROC |
|---|---|---|
| **HyMeKo-Gömb (ours)** | **strict** | **0.9526 ± 0.0018** |
| SiGAT (Huang 2019) | transductive (leaky) | ~0.95 |
| SDGNN (Huang 2021) | transductive (leaky) | ~0.95–0.96 |
| SGCN (Derr 2018) | transductive (leaky) | ~0.93 |

> Framing: competitive-to-leading **at a fraction of the cost**, with the genuine
> lead on Epinions — under the *strict* protocol vs the field's *transductive*
> (leaky) convention. Not a flat SOTA claim.

### B3. Accuracy-per-parameter — HSiKAN-Optuna, 10-seed (transductive)
*Source: `reports/2026-05-13-bitcoin-optuna-best-10seed.md`.*

| Dataset | Config | n | params | fwd ms | AUROC mean ± pstd |
|---|---|---:|---:|---:|---|
| Bitcoin Alpha | optuna_best_alpha | 10 | 30 487 | 656.1 | 0.9959 ± 0.0011 |
| Bitcoin Alpha | joint_mix (ref) | 5 | 61 094 | 341.6 | 0.9845 ± 0.0025 |
| Bitcoin OTC | optuna_best_otc | 10 | 23 815 | 30.5 | 0.9933 ± 0.0023 |
| Bitcoin OTC | joint_mix (ref) | 5 | 94 662 | 342.3 | 0.9801 ± 0.0051 |

> Honest note: these are the **transductive optuna_best** numbers (tuned; the
> baselines are not). Paired Δ vs joint_mix: Alpha +0.0119 (+11.96σ, 5/5), OTC
> +0.0139 (+7.02σ, 5/5). The OTC "~11× faster" (30.5 vs 342 ms) is the
> optuna_best_otc-vs-joint figure — **OTC-specific, not a general width claim**.

### B4. Leakage audit — label-shuffle (why strict matters)
*Source: `reports/2026-05-14-executive-brief.md`.*

| Architecture | Real labels | Shuffled train labels | Reading |
|---|---:|---:|---|
| HSiKAN-Optuna (transductive) | 0.9970 | 0.9921 | massive σ-leakage |
| HSiKAN-joint_mix (transductive) | 0.9845 | 0.8902 | moderate σ-leakage |
| **Gömb (strict)** | **0.9526** | **0.5402** | **no leakage** (structural) |
| SGCN (transductive) | 0.93 | 0.5503 | no structural prior |

### B5. Efficiency / hardware footprint
*Source: `reports/2026-05-14-executive-brief.md`.*

| Item | Value |
|---|---|
| GPU | RTX 2070 SUPER (2019, 8 GB, ~$400 retail) |
| 5-seed Epinions wall time | ~33 min (~6.5 min/seed) |
| Peak GPU memory | ~5.5 GB |
| Training data (Epinions) | 132,828 vertices × 841,372 edges |

---

## Recommendation (what I'd actually put in the talk)

1. **A1 (DSL conciseness table)** — strongest reuse; drop it on/after the
   transforms slide (14). Concrete, audience-friendly, supports the framework half.
2. **B1 + B4 together** — the strict 5-seed table next to the label-shuffle table
   is the honest core of the science half; B4 is the slide that pre-empts "is this
   leakage?". The deck already has a leakage slide — B4's numbers can populate it.
3. **A4 (5 ms IPC)** — one line on the zero-copy bridge slide (13).
4. **A3 (entropy MNIST)** — only if you want a second learning domain; it is a
   modest effect, present it as such.

Open question before any of these goes on a slide: **HyMeYOLO mAP** is inconsistent
across sources — `reports/2026-05-13-hymeyolo-ricci-5seed-backfill.md` reports
+ricci-mod = 0.723 ± 0.180, while `SEMINAR_SUMMARY.md` calls 0.723 "bug-inflated"
and quotes a corrected ≈ 0.90. Tell me which is canonical and I will align the deck
and this doc; I did not pick one.
