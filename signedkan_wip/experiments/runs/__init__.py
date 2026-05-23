"""Experiment entrypoints (the ``run_*.py`` scripts).

These were moved here from ``signedkan_wip/src/`` on 2026-05-19
as part of Slice B of the directory-reorganisation plan
(``docs/plans/2026-05-19-signedkan-wip-organize/``). The §6.5 #3
anti-pattern named in CLAUDE.md ("Per-experiment scaffold
duplication") is targeted by this move: the script files are now
in their semantic home; a follow-up slice (H) will introduce a
shared :class:`._experiment_base.ExperimentBase` they can inherit
from to remove the duplicated argparse-train-eval-write-JSON
scaffolding.

Invocation pattern (after the move):

    python -m signedkan_wip.experiments.runs.run_gomb_smoke --device cpu
    python -m signedkan_wip.experiments.runs.run_hsikan_optuna_chase ...

The old invocation path
(``python -m signedkan_wip<dot>src<dot>run_X args``, with ``<dot>``
substituted for ``.``) no longer works. The note above uses ``<dot>``
literally to keep the auto-sed import-rewriter from touching this
docstring during the migration.
"""
