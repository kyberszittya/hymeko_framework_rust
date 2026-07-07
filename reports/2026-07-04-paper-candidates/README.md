# Paper-candidate dossier — technical reports for publication conversion

**Created:** 2026-07-04 16:14 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Purpose:** one self-contained technical report per publication candidate, each structured so it can be
converted directly into a paper draft: claim, evidence ledger (measured / inferred / hypothesis),
on-disk artifacts, prior-art delineation, and the enumerated missing work to reach submission.

**Method:** synthesis of on-disk reports (no new experiments run). Sources are cited by path in each
file. Prior-art delineations lean on the bounded 2026-06-29 literature search
(`reports/2026-06-29-novelty-and-contribution-assessment.md`); that search was **not exhaustive** and
each file flags where a fresh, targeted search is still owed (CLAUDE.md novelty policy).

## Ranking (Nature-family capability, 2026-07-04)

| # | File | Line | Verdict | Ceiling venue | Completion |
|---|------|------|---------|---------------|------------|
| 1 | [01-reward-conflict-hypergraph.md](01-reward-conflict-hypergraph.md) | Reward as signed hypergraph; conflict causes RL failure | Most Nature-shaped story: surprise + mechanism + causal test. Missing only breadth. | Nature MI | evidence strong, breadth missing |
| 2 | [02-gauge-holonomy-learning.md](02-gauge-holonomy-learning.md) | Holonomy as the learning signal (gauge line) | Highest ceiling (chemistry+neuro+ML span); still a bet. Swing experiment planned, **not run**. | Nature (flagship, conditional) | theory done, decisive experiment pending |
| 3 | [03-nagare-entropy-pool-local-learning.md](03-nagare-entropy-pool-local-learning.md) | Backprop-free local learning (NAGARE) | Nature-tier topic, toy-tier evidence. Full missing-point collection inside. | Nature MI / NeurIPS | toy results, all hard tests pending |
| 4 | [04-leakage-audit.md](04-leakage-audit.md) | Signed-link benchmark leakage audit | Closest to submittable; one scientific gap (reddit_body residual). | Nature Communications | ~90 % |
| 5 | [05-adjacent-tracks.md](05-adjacent-tracks.md) | Cross-view verified IR (T-SMC), LiNGAM-SH, DTC | Publishable, not Nature-shaped; T-SMC is publish-now. | T-SMC / NeurIPS | varies |

Items 1–3 are one program (the signed-hypergraph / holonomy invariant) instantiated three times;
the papers should cross-cite, not merge.

## Meta-report (CLAUDE.md §9)

- **Files touched:** this folder only (6 new .md files). No code, no CORE.YAML items, no dependencies.
- **Tests:** documentation-only change — declared exempt per §3 (no observable-behavior change).
- **Provenance:** synthesized from reports on branch `hymeko-neuro-migration`; the NAGARE crate is
  committed at `0211128` (verified clean during the sweep); no experiment data generated.
- **Open issue:** a 2026-07-04 fused-parity artifact set exists with **no markdown write-up**
  (`reports/2026-07-04-nagare-fused-parity-*`) — it likely closes the perf gap four NAGARE reports
  flagged; write it up before drafting (see `03-…` gap 6).
