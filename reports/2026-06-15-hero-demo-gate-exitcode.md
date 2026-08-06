# Hero demo — exit-code-authoritative gate (+ a corrected finding)

**Date:** 2026-06-15
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)
**Refines:** the hero-demo orchestrator (plans `docs/plans/2026-06-15-hero-demo-phase{1,2}/`).

## Summary

Two things:

1. **Corrected finding.** Phase 1 reported that `hymeko validate`/`emit` "exit 0
   even on failure." **That was wrong** — it came from reading `$?` *after a
   pipe* (`hymeko … | head`), which returns `head`'s exit code, not `hymeko`'s.
   Re-checked directly: the CLI **already exits non-zero** on a hard failure
   (`validate` broken → exit 1, `emit` broken → exit 1; warnings keep exit 0).
   So the planned "Phase 1.5 CLI hardening" was a non-issue.

2. **Hardened the orchestrator gate.** The hero demo now uses the **process exit
   code** as the authoritative gate signal (`verdict_from_run(text, code)`): a
   non-zero code is REJECTED regardless of message; on exit 0 the message only
   splits CLEAN vs WARNINGS. This is more robust than the previous string-match
   and cannot be fooled by an odd success message. `emit` likewise requires exit
   0 (plus the root-token sanity check).

The message-only `parse_validate_output` is retained (fail-closed) as a
documented fallback and stays unit-tested.

## Files touched

- `demos/hero/hero_demo.py` — `_run` returns `(text, returncode)`; new pure
  `verdict_from_run`; `validate`/`emit` use the exit code; `parse_validate_output`
  re-documented as the message-only fallback.
- `demos/hero/test_hero.py` — `test_verdict_from_run_uses_exit_code` (non-zero ⇒
  rejected even with a clean message; exit 0 ⇒ clean/warnings only).
- Corrected the inaccurate finding in `reports/2026-06-15-hero-demo-phase1.md`,
  `…-phase2.md`, `demos/hero/README.md`, and the project memory.

## Verification (true exit codes, no pipe)

```
validate good      → 0
validate warnings  → 0   (anthropomorphic 'world' warning)
validate broken    → 1   (self-contained undefined ref)
emit broken        → 1
```

## CORE.YAML items touched

None. `demos/**` only; no CLI source change was needed (it was already correct).

## Test results

- **pytest (`-p no:randomly`):** 12 passed / 0 failed (+1 vs Phase 2).
- `ruff check` + `mypy --strict` (both files): clean.
- Demo run: verdicts unchanged in outcome (warnings/clean/rejected) but now
  exit-code-driven; broken twin still rejected.

## Lesson (process)

Never read `$?` after a pipe to judge a command's success — capture the command's
own exit status (`cmd; echo $?`, or `subprocess` `returncode`). The wrong reading
created a phantom "Phase 1.5" work item; the real fix was a small orchestrator
robustness change.

## Open issues / follow-ups

- Phase 3 (Gömb/Soma perception — needs Soma vision round-tripped through
  `.hymeko`) and an editor hero-cell profile remain the substantive next steps.

## Experiment provenance

Not a measurement experiment. Toolchain: `hymeko_cli` (cargo, stable), Python 3 +
pytest/ruff/mypy. Working tree dirty from prior session work unrelated to this
change.
