"""§4 — the option-RL run described in HyMeKo: parse ``coin_carry_option_rl_run.hymeko`` into a queryable spec and
REGENERATE the runtime config the existing `hymeko_rl/option_rl` engine consumes — no hand-written duplicate topology.

The DSL does not reimplement SAC/TD3/replay/optimizers; it declares their structural role, bindings, invariants, and
provenance. A query over the parsed graph answers the whole run (checkpoint, budget, generator/scorer, trainable/frozen
skill, handoff certificate, the Bellman action, what the selected candidate represents, whether the target uses γ^τ, the
trainer backend, the physical selection metric, the train/dev/final manifests). Load-bearing invariants are validated
fail-closed (e.g. Bellman action must be the proposal center, the selected candidate must be provenance-only, terminal must
not bootstrap).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hymeko_rl.coin_delivery.coin_carry_monitor import MonitorSpec, load_carry_monitor_spec
from hymeko_rl.coin_delivery.coin_carry_option_rl import RLConfig
from hymeko_rl.coin_delivery.coin_carry_skills import CoinSkillRouting, load_carry_skill_routing
from hymeko_rl.env._profile import read_bundle

CARRY_OPTION_RL_RUN_HYMEKO = "data/robotics/coin_carry_option_rl_run.hymeko"


def _fields(body: str) -> dict[str, str]:
    return {k: v for k, v in re.findall(r"(\w+)\s+([\w.]+)\s*;", body)}


@dataclass(frozen=True)
class OptionRLRunSpec:
    """A queryable option-RL run, parsed from HyMeKo and composed with the §3 skill routing + §2A certificate."""

    members: dict          # member name → field dict (from the run .hymeko)
    routing: CoinSkillRouting
    monitor: MonitorSpec

    # ---- queries answerable from the parsed graph alone ----
    def proposal_checkpoint(self) -> str:
        return self.members["proposal"]["checkpoint"]

    def resolve_checkpoint(self, dirpath: str) -> str:
        """The on-disk path of the active proposal checkpoint (logical name → ``<dir>/<name>.pt``)."""
        return f"{dirpath}/{self.proposal_checkpoint()}.pt"

    def search_budget(self) -> int:
        return int(self.members["search"]["budget"])

    def candidate_generator(self) -> str:
        return self.members["generator"]["kind"]

    def candidate_scorer(self) -> str:
        return self.members["scorer"]["kind"]

    def trainable_skill(self) -> str:
        return sorted(self.routing.trainable_bindings())[0]

    def frozen_skill(self) -> str:
        return sorted(self.routing.frozen_bindings())[0]

    def handoff_certificate(self) -> str:
        return self.routing.handoff_certificate()

    def bellman_action(self) -> str:
        return self.members["option"]["bellman_action"]

    def selected_action_role(self) -> str:
        return self.members["option"]["selected_action_role"]

    def uses_gamma_tau(self) -> bool:
        return self.members["target"]["bootstrap_discount"] == "gamma_tau"

    def terminal_bootstraps(self) -> bool:
        return self.members["target"]["terminal_bootstrap"] == "1"

    def gamma(self) -> float:
        return float(self.members["target"]["gamma"])

    def trainer_primary(self) -> str:
        return self.members["trainer"]["primary"]

    def trainer_control(self) -> str:
        return self.members["trainer"]["control"]

    def physical_metric(self) -> str:
        return self.members["eval"]["physical_metric"]

    def selects_on_critic_q_alone(self) -> bool:
        return self.members["eval"]["critic_q_alone"] == "1"

    def manifests(self) -> dict:
        m = self.members["manifest"]
        return {"train": (int(m["train_lo"]), int(m["train_hi"])), "dev": (int(m["dev_lo"]), int(m["dev_hi"])),
                "final": (int(m["final_lo"]), int(m["final_hi"]))}

    def query_dump(self) -> dict:
        return {"proposal_checkpoint": self.proposal_checkpoint(), "search_budget": self.search_budget(),
                "candidate_generator": self.candidate_generator(), "candidate_scorer": self.candidate_scorer(),
                "trainable_skill": self.trainable_skill(), "frozen_skill": self.frozen_skill(),
                "handoff_certificate": self.handoff_certificate(), "bellman_action": self.bellman_action(),
                "selected_action_role": self.selected_action_role(), "uses_gamma_tau": self.uses_gamma_tau(),
                "terminal_bootstraps": self.terminal_bootstraps(), "gamma": self.gamma(),
                "trainer_primary": self.trainer_primary(), "trainer_control": self.trainer_control(),
                "physical_metric": self.physical_metric(), "selects_on_critic_q_alone": self.selects_on_critic_q_alone(),
                "manifests": self.manifests(), "certificate_tolerances": vars(self.monitor)}

    def to_runtime_config(self) -> RLConfig:
        """Regenerate the engine config the existing `hymeko_rl/option_rl` runtime consumes — the load-bearing knobs come
        from the description (b, γ); the numeric backend defaults are the engine's. No duplicate topology is hand-written."""
        return RLConfig(gamma=self.gamma(), b=self.search_budget())


def _validate(spec: OptionRLRunSpec, path: str) -> None:
    """Fail-closed on any violation of the load-bearing semantics."""
    if spec.bellman_action() != "theta_center":
        raise ValueError(f"{path}: Bellman action must be theta_center, got {spec.bellman_action()!r}")
    if spec.selected_action_role() != "provenance" or spec.members["provenance"].get("is_trained_action") != "0":
        raise ValueError(f"{path}: the search-selected candidate must be provenance-only, never the trained action")
    if not spec.uses_gamma_tau() or spec.terminal_bootstraps():
        raise ValueError(f"{path}: target must use γ^τ and terminal transitions must NOT bootstrap")
    if spec.members["proposal"].get("training_state") != "trainable":
        raise ValueError(f"{path}: the upstream proposal must be trainable")
    if spec.selects_on_critic_q_alone():
        raise ValueError(f"{path}: checkpoint selection must be physical, not critic Q alone")
    # the trainable/frozen skills + certificate must agree with the §3 routing
    if spec.frozen_skill() not in spec.routing.frozen_bindings() or spec.trainable_skill() not in spec.routing.trainable_bindings():
        raise ValueError(f"{path}: proposal/skill bindings disagree with the coin_carry_option_v1.hymeko skill routing")


def load_carry_option_rl_run(path: str = CARRY_OPTION_RL_RUN_HYMEKO) -> OptionRLRunSpec:
    """Parse the run description, compose §3 routing + §2A certificate, validate fail-closed."""
    members = {name: _fields(body) for name, kind, body, _w in read_bundle(path, "run_spec")}
    spec = OptionRLRunSpec(members=members, routing=load_carry_skill_routing(), monitor=load_carry_monitor_spec())
    _validate(spec, path)
    return spec
