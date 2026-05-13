# Report: mdBook Bitcoin Optuna vs SOTA page

## Summary

Added a new mdBook results page for the current Bitcoin Optuna-vs-SOTA snapshot,
linked it in the book table of contents, and rebuilt the book output to verify
rendering and navigation.

## Files touched (this task)

- `docs/book/src/results/bitcoin-optuna-vs-sota.md` (+35 / -0)
- `docs/book/src/SUMMARY.md` (+3 / -0)
- `docs/plans/2026-05-13-book-bitcoin-optuna-sota-page/plan.tex` (+54 / -0)
- `docs/plans/2026-05-13-book-bitcoin-optuna-sota-page/plan.tikz` (+14 / -0)
- `docs/plans/2026-05-13-book-bitcoin-optuna-sota-page/plan.mmd` (+10 / -0)
- `docs/plans/2026-05-13-book-bitcoin-optuna-sota-page/plan.pdf` (generated artifact)
- `reports/2026-05-13-book-bitcoin-optuna-sota-page.md` (+this file)

## CORE.YAML items touched

None.

## Test results

- Unit tests: not applicable (documentation-only change).
- Integration tests:
  - `mdbook build docs/book` -> pass.
- Performance tests: not applicable (no runtime/algorithm change).

## Performance results

- Budget from plan: mdBook build wall time < 120 s.
- Observed:
  - `mdbook build docs/book`: completed in ~0.4 s wall time.
- Peak RSS / inference latency: not applicable for documentation-only change.

## New / removed dependencies

None.

## Open issues / follow-ups

- Current page intentionally mixes single-trial Optuna maxima with 5-seed means
  and labels this explicitly as a snapshot. A follow-up page can add
  seed-normalized comparisons once multi-seed Optuna summaries are available.

## Experiment provenance

No new experiments were launched by this task; only existing on-disk artifacts
were cited.

- Git SHA: `0c55fa8` (working tree already dirty before this task)
- Referenced artifacts:
  - `signedkan_wip/experiments/results/optuna_alpha_slashdot_20260513T010509Z.log`
  - `signedkan_wip/experiments/results/follow_optuna_20260513T003359Z.log`
  - `docs/SOTA_RESULTS.md`
