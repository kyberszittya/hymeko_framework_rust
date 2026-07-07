# SMC_02 article — state save + acceptance-targeted content additions

**Date:** 2026-07-04 15:20 +09:00
**Repo:** `d:\hakiko_ai_ws\01_analysis\articles\superweapon\smc_02\` (own git repo, branch `master`)
**Commits:** `74e7039` (state save: rebuilt PDFs + review_1 notes), `61ff24e` (content additions)

## Summary

Two-part task: (1) commit the current article state; (2) add content targeted at the
dominant acceptance risk identified for IEEE T-SMC: Systems — **venue-scope fit** —
plus the secondary **single-team-evidence** critique.

Pre-edit verification: all three red-flag items from `review_1` were already fixed in
the tree (A4 storage-overhead statement corrected; "hash decides equality" phrasing
gone; Table V "LoC/fmt" column gone). The bibliography contained **zero** SMC-society,
digital-twin, or MBSE-consistency citations — the paper read as pure software/MDE,
inviting a "why not TSE/SoSyM?" desk reaction at T-SMC:S.

## Changes (commit 61ff24e, +56/−1 lines)

| File | Change |
|---|---|
| `sources.bib` | +5 verified entries: Tao & Qi 2019 (**T-SMC:S 49(1):81–91**), Tao et al. 2019 (TII 15(4):2405–2415), Kritzinger et al. 2018 (IFAC-PapersOnLine 51(11):1016–1022), Feldmann et al. 2019 (JSS 153:105–134), Herzig et al. 2014 (Procedia CS 28:354–362) |
| `sections/01_introduction.tex` | +1 sentence in the fragmentation-cost paragraph: digital-twin literature framing of the same drift problem |
| `sections/08_related.tex` | New subsection "Consistency management and digital twins": positions HyMeKo (prevention-by-construction, views never maintained) against MBSE detect-and-repair (Herzig, Feldmann) as complementary design points; ties cross-view consistency to the virtual side of the digital-twin loop |
| `sections/07_evaluation.tex` | New "Construct and external validity" paragraph in Threats: single-team emitter+extractor risk with three mitigations (five independent concrete-syntax parsers; implementation-independent SMT entailment; public artifacts), invariant-family scope honesty, largest-fixture bound (52-link DRC-Hubo / 68-link humanoid) |
| `bare_jrnl_new_sample4.pdf` | Rebuilt |

All five citations were **verified against live search results before insertion**
(titles, venues, volumes, pages) — none invented. Search trail: Kritzinger via
ScienceDirect/IFAC (DOI 10.1016/j.ifacol.2018.08.474); Tao & Qi confirmed as a
T-SMC:S publication; Feldmann/Herzig via MBSE inconsistency-management literature.

## Build

`pdflatex → bibtex → pdflatex ×2` (MiKTeX, Windows): **clean**.
- 0 undefined citations, 0 undefined references.
- 8 pre-existing cosmetic font-shape warnings (`T1/ptm/m/scit` substitution), unchanged.
- Page count **15 → 16**. T-SMC:S mandatory overlength: $200/page past 12 — the
  additions cost at most one billed page; if that matters, the natural offset is
  moving more appendix material to the supplementary (A1 already moved).

## Tests

Not applicable (LaTeX prose + bib only; no code path touched). The "test" is the
clean compile with zero unresolved references, verified above. No §6.5 anti-patterns
introduced; no CORE.YAML items touched (different repo entirely).

## Acceptance assessment (unchanged from analysis, restated for the record)

- Journal publishes no official acceptance rate; tier-inferred 10–20% raw.
- This paper: first-round accept ≈ 0 (IEEE Trans norm), major-revision-then-accept
  now estimated **~40–50%** (up from 35–45% — the venue-fit and validity gaps were
  the two most probable reject rationales and both now have explicit counters).
- Remaining pre-submission items: (a) confirm current T-SMC:S submission page cap
  vs the 16-page build; (b) flip `\bibliographystyle{ieeetr}` → `IEEEtran` for the
  final build (comment already in the main tex); (c) one last uniform-terminology
  pass ("machine-verified" vs "proof sketch") — the conclusion still says
  "Proof sketches for Propositions 2–4" while the abstract says machine-verified;
  harmonize in the next surgical pass.

## Addendum (15:35 +09:00): trim to 14 pages (commits `b5ca8d2`, `fa424db`)

User directive: keep ~14 pages. Executed:

- **Appendices A2–A4** (content-addressability, projection-emission independence,
  storage overhead) **moved to the supplementary**, following the established A1
  pattern; proposition statements remain in the main text (Section IV). **A5
  (cross-view consistency, the central claim) stays** in the main paper.
- **A5 dedup**: its "Conventions and the data/diagram split" + "Tested by" paragraphs
  were near-verbatim duplicates of Section VII-A; compressed to pointers (−9 source
  lines). This is a §6.1-class dedup, not a content loss.
- Supplementary gained the three proofs and its own `ieeetr` bibliography (A2 cites
  BLAKE3); the two main-text `\ref{app:proof_*}` references rerouted to
  "supplementary material".
- Result: **main 16 → 14 pages** (2 billed overlength pages at T-SMC:S instead of 4),
  supplementary 3 → 5 pages. Both build with **zero undefined references/citations**.
- Repo hygiene: `.gitignore` added, LaTeX aux/log artifacts untracked (`fa424db`).

## Open issues / follow-ups

- The new related-work subsection cites detect-and-repair as complementary; if a
  reviewer asks for a *quantitative* comparison against a consistency-rule engine,
  the honest answer is scope (different failure model) — already worded that way.
- `review_1`'s closing warning stands: no further structural rebuilds; surgical
  edits only from here.
