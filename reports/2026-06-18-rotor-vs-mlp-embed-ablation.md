# Is the Cayley-rotor geometry load-bearing? — no; a plain MLP embedding matches it

**Date:** 2026-06-18
**Status:** ✅ control built + tested + 5-seed gated A/B. **Foundational negative:** the rotor's S³ geometry is not load-bearing on signed-link prediction — a generic MLP embedding of the identical structural features performs the same. The line's wins come from *replacing the node-ID table with structural features*, not from the rotor.

## Question

The rotor line is named for the **Cayley-rotor embedding** (structural features →
Rodrigues → unit quaternion → `e = R·v₀` on S³ → proj → SGCN). It *is* utilized —
the embedding layer, exercised in every cell of every experiment. But is its
**geometry** load-bearing, or is it an arbitrary nonlinear embedding that any MLP
could replace? The head ablation already showed the rotor *algebra* is not
load-bearing at the **readout** (real bilinear ≥ complex/quaternion/geodesic); this
tests it at the **embedding**.

## Method

`MLPEmbedSignedModel` — identical architecture (same `embedding_dim = 3·n_blocks`,
same proj, same SGCN, same classifier) with **only** the rotor embedding replaced by
a 2-layer Tanh MLP (Tanh bounds the output like the S³ sphere). The MLP carries
**more** params than the rotor (16,661 vs 15,761 at hidden=32) — a deliberately
generous control: the rotor must beat extra capacity to count as load-bearing.
Registered as `mlp_embed` / `mlp_embed_walk`, measured through the unified audit
harness, deduped + shuffle-gated, 5 seeds, both Bitcoin graphs.

## Result (5-seed mean ± pstdev, deduped) — `rotor_vs_mlp_embed_ab.jsonl`

| dataset | features | rotor | MLP-embed | rotor − MLP | gate r/m |
|---|---|---|---|---|---|
| bitcoin_alpha | degree | 0.8182±0.022 | 0.8236±0.027 | −0.0054 | 0.51/0.52 |
| bitcoin_alpha | +walk | 0.8217±0.020 | 0.8200±0.021 | +0.0017 | 0.50/0.50 |
| bitcoin_otc | degree | 0.8596±0.010 | 0.8648±0.005 | −0.0052 | 0.52/0.53 |
| bitcoin_otc | +walk | 0.8713±0.010 | 0.8683±0.012 | +0.0030 | 0.53/0.53 |

All four |deltas| are 0.002–0.005 — **inside the seed σ (0.01–0.027)**. The rotor and
a generic (higher-capacity) MLP embedding of the same features are statistically
indistinguishable, both leakage-clean. (Seed-0 had showed −0.016/−0.023 degree
deficits for the rotor; they collapsed to −0.005 at 5 seeds — seed noise, flagged
and confirmed.)

## Interpretation — what this re-contextualizes

- **The rotor's geometry is decorative for this task.** With the head ablation
  (readout) and now this (embedding), *neither* the rotor algebra nor the rotor
  geometry is load-bearing on signed-link prediction.
- **The line's validated wins are not the rotor's.** Leakage-freedom, inductiveness,
  and parameter-lightness all come from **replacing the transductive `nn.Embedding`
  table with train-only structural features** — the MLP-embed control has all three
  too (it embeds the same features; gates clean; param-count node-independent). The
  inductive transfer result therefore stands as a property of *structural features +
  SGCN*, not of the rotor.
- **Honest scope.** This is "not load-bearing *for signed-link prediction with these
  features + SGCN*", not "the rotor is useless." The Cayley-rotor's distinct claimed
  value (inductive leakage-free embedding ⊕ ANN-index projection — [[project-cayley-rotor-idea]])
  is a different use that this does not test.

**Measured / inferred (CLAUDE.md).** *Measured:* the table; MLP has ≥ rotor params.
*Inferred:* the rotor geometry adds no measurable signal over a generic embedding of
the same features ⇒ the line's wins are the *features*, not the rotor.

## Files touched

- `signedkan_wip/src/baselines/cayley_rotor_baseline.py` — `_MLPEmbed`,
  `MLPEmbedSignedModel` (rotor-embedding ablation control; shares proj/SGCN/
  classifier via the spec-driven base), registered `mlp_embed` / `mlp_embed_walk`.
- `signedkan_wip/tests/test_cayley_rotor_baseline.py` — control test (fair: same
  embedding_dim, ≥ rotor params, inductive, forward shape). Removed a pre-existing
  unused import.
- Artifact: `rotor_vs_mlp_embed_ab.jsonl` (80 rows).
- **CORE.YAML items touched:** none.

## Tests / gates
- `test_cayley_rotor_baseline.py` 6 ✓, `test_structural_features.py` 13 ✓.
  `ruff check`: clean on touched files. `mypy --strict`: `cayley_rotor_baseline.py`
  is **not** strict-clean pre-existing (~19 torch-`Any`/untyped-`build_model` errors
  across the file); my additions add 3 of the same classes (forward `Any`-return,
  the deliberate `self.emb` type swap, an un-annotated `build_model` matching the
  file's convention) — not held to strict for this pervasively-untyped file, noted.

## Performance
- Per cell ~3 s, GPU; MLP-embed adds ~900 params. RSS ≪ 16 GB. No regression claim.

## Open issues / follow-ups
- **Reframe the line for the write-up:** the contribution is *structural-feature
  (leakage-free, inductive) signed-link prediction with signed message passing* —
  the rotor is one (interchangeable) embedding choice, not the active ingredient.
  Either keep the rotor for its narrative/ANN-projection tie-in and say so honestly,
  or default to the simpler MLP embedding.
- The harder-pair transfer smoke (otc→slashdot 0.844 vs random-init 0.534, ~7 s/cell)
  is queued and orthogonal — it tests the *structural-feature* line either way.
