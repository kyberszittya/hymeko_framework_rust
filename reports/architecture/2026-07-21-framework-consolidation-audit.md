---
campaign: Repository architecture consolidation audit + domain-generic command layer
title: Import/ownership inventory (56 deduped / 62 raw production→experiment imports) + one canonical command layer verified on Coin/CIP/HyperSignedLINGAM
date: 2026-07-21
branch: refactor/canonical-campaign-and-final-video
source_commit: 72de355
---

# Framework consolidation audit (§1) + command contract (§2–§3)

**Created-at:** 2026-07-21 18:20 JST. Implementation audit, not a redesign. Baseline frozen at `72de355`
(exp/coin-fast-transition-ball-tip). No CORE.YAML change; no new dependency (JSON manifests, stdlib only).

## §1.1 Import & ownership inventory
`reports/architecture/module_inventory.json` (machine-readable). Counts (walking `hymeko_rl/`, excluding
`__pycache__`):

| class | count |
|---|---|
| total `.py` | 942 |
| PRODUCTION (non-experiment, non-test) | 310 |
| ENTRYPOINT (`experiments/`) | 297 |
| TEST | 335 |

## §1.4 Architecture debt — production imports FROM experiments (recomputed)
**56 deduped (module→target) / 62 raw ImportFrom nodes** production→`hymeko_rl.experiments.*` (prior reports estimated
~45; the real current number is higher). Dominant targets:

| target | count |
|---|---|
| `experiments.galambos_demo` | 20 |
| `experiments.structural_probe` | 6 |
| `experiments.exp_galambos_coord_ab` | 5 |
| `experiments.pedc_selection` | 4 |
| others (handoff_gate, learner_stabilization, gripper_pick_bc, cip_lingam_demo, …) | ≤2 each |

By source package: eval 16, agents 12, viz 12, train 9. The `galambos_demo` cluster (20) is the largest single debt.

## §1.2/§1.5 Deletion policy — conservative this pass
The working tree carries **2143 pre-existing dirty files** (ambient, unrelated to this work). Per §1.2, a production
file is deletable only after reverse-import + behavioral verification. Given the dirty tree and that the 56/62 imports
are *live* (e.g. `pedc_selection`/`exp_v3_handoff_gate` are load-bearing for the verified neutral-delivery chain — the
recovered E approach loads through `exp_v3_handoff_gate._load_e`), **no production files were deleted this pass**: the
canonical migration of 56 live imports is a multi-session effort that must not be rushed into a mass `git rm` under a
dirty tree. Instead this pass (a) records the exact debt, (b) installs a **regression guard** so it cannot grow, and
(c) builds the canonical command layer the future migration will consolidate into. Historical evidence (reports,
manifests, metrics, checkpoints) is fully preserved.

## §1.7 #7 Architecture guard (added)
`test_architecture_guard_production_import_debt_does_not_grow` fails if the production→experiments ImportFrom count
exceeds **62**. Lowering it as debt is paid is welcome; raising it is a regression.

## §2 One domain-generic command contract (`hymeko_rl/campaign/`)
The single canonical command layer (NOT a parallel framework): `spec.ExperimentSpec` (domain-neutral — `model_variant`/
`execution_strategy`, not "policy"/"strategy"), `runner` (seed/resource/provenance/artifacts/hashing/fail-loud), thin
per-domain `adapters` (Coin/CIP/HSL), CLI `python -m hymeko_rl.campaign {run|campaign|render|verify}`. **The core imports
no domain code; adapters import their domain lazily and never import one another** (verified by two ast tests).

Common artifact contract per run: `manifest.json, resolved_config.json, provenance.json, metrics.jsonl, result.json,
stdout.log, artifact_index.json` (+ domain outputs). Fail-loud on unknown domain / bad options / missing checkpoint.

## §3 Verified on three domains through the SAME runner
| domain | adapter → real implementation | smoke result | domain artifacts |
|---|---|---|---|
| coin_delivery | `coin_neutral_start.eval_composed` | P4/S1/POINT seed 1011 → **strict 3/3** | (checkpoints/traces via domain pipeline) |
| **cip** | `eval.cip.cip_augment.estimate_cip_weights` (DirectLiNGAM) | top state dim + deprioritised dims | `prioritized_candidate_table.json`, `monitor_to_cip_trace.json` |
| **hypersignedlingam** | `eval.causal.signed_hyper_lingam.SignedHyperLiNGAM.fit` | **4 signed hyperedges**, bootstrap 4.0±0.0 | `signed_adjacency.json`, `causal_hypergraph.json`, `stability_metrics.json` |

**No Coin code imported into CIP/HSL; no CIP/HSL algorithm re-implemented or copied into the RL packages** — the
adapters call the existing `eval.causal` / `eval.cip` implementations. §3.3 acceptance met.

## Files
- NEW: `hymeko_rl/campaign/{__init__,spec,runner,adapters,__main__}.py`; `configs/campaigns/{coin_delivery_final_video,
  cip_smoke,hypersignedlingam_smoke}.json`; `hymeko_rl/tests/test_campaign_command_layer.py` (8 tests);
  `reports/architecture/{module_inventory.json,2026-07-21-framework-consolidation-audit.md}`.
- No production files migrated or deleted this pass (debt recorded + guarded; see above). No CORE.YAML; no new deps.
