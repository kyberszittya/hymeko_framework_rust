# Seminar deck: tensor-view heatmaps + reference & future-work slides

**Date:** 2026-06-15
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)
**Deck:** `docs/seminar/HyMeKo_Seminar.pptx` (34 slides) → `…with_refs.pptx` (36).

## Summary

Three user-requested additions to the PhD-seminar deck:

1. **Tensor-view heatmap backgrounds** — star-expanded vs clique-expanded
   materialisation of a deterministic 50/50/0.30 signed hypergraph (the example
   slide 12 cites), placed behind slide 12 ("Star vs clique: the efficiency
   argument"): the sparse signed-incidence star (±1) vs the dense co-membership
   clique, faded to ~28% opacity and sent to the back.
2. **References (selected) slide** — curated from the project's own `.bib` files
   (every entry maps to a real citation).
3. **Future-work slide** — from the abstract's "ongoing work" + this session's
   reports.

## Files touched

- `docs/seminar/make_tensor_heatmaps.py` — **new**, renders the two heatmaps
  (matplotlib; deterministic).
- `docs/seminar/figures/{star,clique}_expansion.png` (+ faded `_bg` variants).
- `docs/seminar/insert_into_deck.py` — **new**, places the backgrounds + appends
  the two slides via `python-pptx`; **writes a new file** `…with_refs.pptx`
  (original never overwritten — it is the user's artifact).
- `docs/seminar/REFERENCES_AND_FUTURE_WORK.md` — **new**, the slide content
  (dependency-free, also usable for manual insertion).

## CORE.YAML / dependency note

- **`python-pptx==1.0.2` installed** (pulls `lxml`, `xlsxwriter`) — a §1
  dependency add, **user-approved 2026-06-15** (`APPROVED-CORE-EDIT:
  pptx-seminar-tooling`; the user said "why not install python-pptx"). Installed
  into the uv venv as seminar dev-tooling; **not** added to `pyproject`/`uv.lock`
  (one-off presentation tool, analogous to the figure scripts). matplotlib/numpy
  were already present.
- No framework code touched; `data/`/crates untouched.

## Verification

- New deck opens as a valid `.pptx`; structural check (stdlib `zipfile`):
  **36 slides** (was 34); slide 12 "Star vs clique" now has **2 media refs** (the
  heatmap backgrounds); **References → slide 35**, **Future work → slide 36**;
  original `HyMeKo_Seminar.pptx` byte-unchanged.
- `make_tensor_heatmaps.py` runs clean (star 753 NNZ, clique 2438 NNZ on the
  seeded example — illustrative; the slide's 1,498/10,991 are the benchmark's own
  counting convention).

## Verification gap (honest)

The **final pixel rendering** of the modified slides was **not** captured —
no LibreOffice/PowerPoint headless renderer is available in this environment, so
I verified slide structure (correct target slide, images sent to back, new slide
titles + bullets, fonts set to the deck's Century-Gothic/Bahnschrift) but not the
visual readability of the faded backgrounds. **Open `…with_refs.pptx` in
PowerPoint to eyeball the opacity** and adjust if needed (the fade alpha is a
one-line change in `insert_into_deck.py`).

## Open issues / follow-ups

- New slides are **appended** (35–36); if you want References/Future-work
  positioned before the Conclusion, reorder in PowerPoint (python-pptx slide
  reordering is fragile XML; appending is the safe default).
- Reference list is trimmed to ~12 lines for the slide; the full themed list is
  in `REFERENCES_AND_FUTURE_WORK.md`.

## Experiment provenance

Not a measurement experiment. Toolchain: Python 3.12 + matplotlib 3.10.9 / numpy
2.4.6 / python-pptx 1.0.2 (uv venv). Working tree dirty from prior session work.

---

## Addendum — 2026-06-16: Kato seminar-review reframing

Acted on `kato_seminar_review` (a ChatGPT review of the deck). Its verdict: the
material is strong but makes too many claims at once — **control the frame**
rather than add content. Implemented as additive, non-destructive slides through
the same `insert_into_deck.py` pipeline (original `.pptx` still untouched;
`…with_refs.pptx` is now **43 slides**, conclusion still last):

**Front (new):**
- *HyMeKo: two coupled contributions* — the central contribution stated up front
  (canonical hypergraph infrastructure + structural-prior learning; one IR is
  both the engineering transform and the learning substrate).
- *Framing & terminology (presenter)* — the "Gömb" gloss ("sphere in Hungarian;
  names the strict cascade config"), one-sentence tamings of the
  Clifford/visual-cortex and HyMeYOLO/Ricci "dragon heads", and the controlling
  frame line ("a unified representation-and-learning substrate, not a bag of
  unrelated tricks").
- *The talk in four blocks* — Problem → Framework → Learning → Evidence.

**Back (new, before conclusion):**
- *Minimum working demo* — `.hymeko` → IR → star expansion → PyTorch/URDF + the
  HSiKAN graph-only kinematics demo; the star-vs-clique efficiency anchor.
- *Where we could collaborate* — robot-structure learning.
- *Anticipated questions (Kato)* — the five likely questions with crisp answers,
  including the honest one: 0.996 is the transductive number; the **strict
  baseline** is the claim.

Same verification gap as above (no headless renderer): slide structure verified
(titles/bullets, front/back positions via `move_to`), pixel rendering not
captured — eyeball in PowerPoint. No CORE.YAML items; no new dependency.

### 2026-06-16 follow-up: visibility, square tensor, compression

Three user requests on the reframed deck:

1. **Tensor view = square matrices.** `make_tensor_heatmaps.py` now renders the
   aggregated star/clique panels with `aspect="equal"` (each materialisation
   reads as a square matrix, not a stretched rectangle).
2. **More visible + as a background.** New `tensor_view_bg.png` — the aggregated
   figure composited onto white at `FADE=0.5` (a single visibility knob). It is
   placed as a **full-width, vertically-centred, sent-to-back** background on the
   "Star vs clique" slide, so it is clearly visible behind the text (the earlier
   ~28 % wash was too faint; the foreground lower-half placement is dropped).
3. **Compress the slide count.** 43 → **38 slides**: (a) the three HyMeYOLO
   slides folded to one per Kato (`consolidate_hymeyolo` keeps the *detections*
   slide with its figures, drops the convolution-intro and Ricci/Hodge
   "dragon-head" text slides — dropping the relationship cleanly, so no orphaned
   part / duplicate-name warning); (b) my Kato additions consolidated — *two
   contributions* + *four-block arc* + the Gömb gloss into **one** front slide,
   *demo* + *collaboration* into **one** closing slide, and the standalone
   "presenter framing" slide dropped (its guidance lives in this report, not the
   audience deck).

`ruff check` on both seminar scripts is now **clean** (fixed the pre-existing
E702 semicolon style while editing). Original `HyMeKo_Seminar.pptx` still
untouched.

4. **Slide-18 chart repaired by rasterizing.** "From structure to inductive
   prior" carried a native embedded **column chart** (k-cycle counts per
   mechanism) — fragile (depends on an embedded workbook; renders blank in
   non-PowerPoint viewers and through the `python-pptx` round-trip) and visually
   sparse (almost all zeros: Four-bar=1 @ k=4, Delta=3 @ k=6, Serial arm=0).
   `rasterize_chart_on` reads the chart's data directly and re-renders it as a
   clean grouped-bar PNG (`figures/kcycle_chart.png`) with the nonzero bars
   value-labelled, then swaps the chart for the PNG at the same box. Verified:
   slide 18 now has 0 charts / 1 picture. The renderer is data-driven (no
   hardcoded numbers) and currently scoped to that one slide.

`ruff check` on both seminar scripts is clean.

**Follow-up still open (needs your eye / steer):** retitle the surviving
HyMeYOLO slide to the stress-test framing in PowerPoint (a safe in-place title
edit I left to you rather than risk mangling its fonts); the other three native
charts (slides 13 "Star vs clique", 25 "link prediction results", 26 "SOTA …")
are still embedded and can be rasterized the same way on request; and if you want
deeper compression, the leakage trio (25–27) is a *keeper* per Kato, but the
intro cluster (slides 1 / 5 / 7) overlaps the new contributions slide and could
be trimmed — I left those authored slides intact pending your call.
