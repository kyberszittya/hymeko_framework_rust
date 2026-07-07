# hymeko_nagare — FROZEN (superseded)

**Status (2026-07-06): frozen. Development has moved to the standalone repo.**

> ## → https://github.com/kyberszittya/nagare
> Local path: `../nagare_github` (crate `nagare-holonomy-learn`, lib `holonomy_learn`).

This in-framework crate was the incubation home of the Nagare holonomy
local-learning line. As of 2026-07-06 the **GitHub repo is the authoritative
copy**: it carries the full superset — the `project_alpha_mix` kernel
(FD-tested forward+backward), the `ProjectionBasis` gate, the frozen seed-53
fixture, the `run_stress_ablation` harness, the point-order-shuffle ablation,
and the cycle-pool runtime — and is where all further work lands.

**Do not develop here.** Changes made in this crate will be lost: it is kept
only so the framework workspace still builds, and it is scheduled for deletion
during the detachment step (when `hymeko_clifford` / `hymeko_graph` are vendored
into the standalone repo and its two path dependencies are removed).

Reconciliation record: `reports/2026-07-06-nagare-repo-reconcile.md`.
Plan: `docs/plans/2026-07-06-nagare-repo-reconcile/` (gitignored).
