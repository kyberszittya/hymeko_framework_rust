"""Canonical domain-generic experiment command layer.

ONE runner + ONE spec + thin per-domain adapters. Domains (coin_delivery, cip, hypersignedlingam, …) share the runner,
provenance and artifact contract but keep their own semantics in their own packages — the orchestration core imports no
domain code, and adapters never import one another. This is the single command layer, not a parallel framework.
"""
from hymeko_rl.campaign.runner import run_campaign, run_one
from hymeko_rl.campaign.spec import CampaignSpec, ExperimentSpec

__all__ = ["ExperimentSpec", "CampaignSpec", "run_one", "run_campaign"]
