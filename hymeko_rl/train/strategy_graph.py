"""The training STRATEGY as a signed dataflow hypergraph — reader, topology classifier, and runner.

Reads a ``strategy_graph`` ``.hymeko`` (training stages as nodes, signed ``flow`` hyperedges as data
dependencies; see ``meta_strategy_graph.hymeko``) and **classifies the algorithm from the TOPOLOGY**: a directed
cycle through the stages (the DAgger relabel loop ``rollout → label → aggregate → retrain → rollout``) means
``dagger``; an acyclic ``source → bc → eval`` means ``bc``. The graph is *load-bearing* — a graph missing the loop
edge classifies as ``bc``, so a DAgger run on it is rejected. Dispatch reuses the existing
:class:`hymeko_rl.train.dagger.Dagger` executor: the graph is the verified spec, **not** a second training loop
(§6.5 #3). This is the strategy-side of "HyMeKo as a declarative substrate": the algorithm's identity is its
dataflow structure, not a string label.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from hymeko_rl.env._profile import parse_fields, read_arc_weights, read_bundle, read_scene_fields
from hymeko_rl.train.dagger import Dagger, DaggerConfig


@dataclass(frozen=True)
class Stage:
    """A training-stage node: its ``role`` selects the primitive, ``knobs`` are its scalar parameters."""

    name: str
    role: str
    knobs: "dict[str, float]"


@dataclass(frozen=True)
class Flow:
    """A signed dataflow hyperedge: data flows from every ``producer`` (``+``) to every ``consumer`` (``-``)."""

    name: str
    producers: "tuple[str, ...]"
    consumers: "tuple[str, ...]"


@dataclass(frozen=True)
class StrategyGraph:
    """A parsed strategy dataflow hypergraph. # Invariants every flow endpoint names a declared stage."""

    stages: "dict[str, Stage]"
    flows: "tuple[Flow, ...]"
    iterate: int
    seeds: "tuple[int, ...]"
    profile: str
    total_steps: int = 150_000   # off-policy budget (td3_bc); the graph bundle's `total_steps` field

    @classmethod
    def from_hymeko(cls, profile: "str | Path") -> "StrategyGraph":
        """Parse the ``strategy_graph`` bundle: stages (``role`` + numeric knobs), signed ``flow`` edges, and the
        graph-level ``iterate`` / ``seeds``. # Errors ``FileNotFoundError``; ``ValueError`` (no stages, a stage
        with no ``role``, an undeclared flow endpoint)."""
        profile = str(profile)
        stages: "dict[str, Stage]" = {}
        flows: "list[Flow]" = []
        for name, kind, body, _w in read_bundle(profile, "strategy_graph"):
            if kind == "stage":
                fields = parse_fields(body)
                role = fields.get("role")
                if not isinstance(role, str):
                    raise ValueError(f"{profile}: stage {name!r} has no string 'role' field")
                knobs = {k: float(v) for k, v in fields.items() if isinstance(v, (int, float))}
                stages[name] = Stage(name=name, role=role, knobs=knobs)
            elif kind == "flow":
                arcs = read_arc_weights(profile, name)
                producers = tuple(m for sign, m, _w in arcs if sign == "+")
                consumers = tuple(m for sign, m, _w in arcs if sign == "-")
                flows.append(Flow(name=name, producers=producers, consumers=consumers))
        if not stages:
            raise ValueError(f"{profile}: strategy_graph declares no stages")
        undeclared = {m for f in flows for m in (*f.producers, *f.consumers)} - stages.keys()
        if undeclared:
            raise ValueError(f"{profile}: flow endpoints {sorted(undeclared)} are not declared stages")
        g = read_scene_fields(profile, "strategy_graph")
        iterate_raw = g.get("iterate", 1.0)
        total_raw = g.get("total_steps", 150_000.0)
        seeds_raw = g.get("seeds", (0.0,))
        iterate = int(iterate_raw) if not isinstance(iterate_raw, tuple) else 1
        total_steps = int(total_raw) if not isinstance(total_raw, tuple) else 150_000
        seeds = tuple(int(s) for s in (seeds_raw if isinstance(seeds_raw, tuple) else (seeds_raw,)))
        return cls(stages=stages, flows=tuple(flows), iterate=iterate, seeds=seeds, profile=profile,
                   total_steps=total_steps)

    def _adjacency(self) -> "dict[str, set[str]]":
        """Directed stage graph: producer → consumer for every flow endpoint pair."""
        adj: "dict[str, set[str]]" = {s: set() for s in self.stages}
        for f in self.flows:
            for p in f.producers:
                adj[p].update(c for c in f.consumers if c in self.stages)
        return adj

    def _has_cycle(self) -> bool:
        """True iff the stage dataflow contains a directed cycle (the DAgger relabel loop). Iterative DFS
        3-colouring (tiny graph; the loop back-edge ``retrain → rollout`` is the only cycle in a DAgger graph)."""
        adj = self._adjacency()
        color: "dict[str, int]" = dict.fromkeys(adj, 0)   # 0 white, 1 gray, 2 black

        def visit(root: str) -> bool:
            stack: "list[tuple[str, bool]]" = [(root, False)]
            while stack:
                node, leaving = stack.pop()
                if leaving:
                    color[node] = 2
                    continue
                if color[node] == 1:
                    continue
                color[node] = 1
                stack.append((node, True))
                for nxt in adj[node]:
                    if color[nxt] == 1:
                        return True
                    if color[nxt] == 0:
                        stack.append((nxt, False))
            return False

        return any(color[s] == 0 and visit(s) for s in adj)

    def classify(self) -> str:
        """The algorithm's identity, read from the TOPOLOGY. A ``critic`` node (an off-policy value estimator with
        Q-target edges) ⇒ ``"td3_bc"``; else a relabel cycle ⇒ ``"dagger"``; else acyclic ⇒ ``"bc"``. So all three
        are distinguished by their dataflow shape — the critic is what makes TD3+BC value-based (and what let its
        value drift collapse standing), the relabel loop is what makes DAgger pure imitation."""
        roles = {s.role for s in self.stages.values()}
        if "critic" in roles:
            return "td3_bc"
        return "dagger" if self._has_cycle() else "bc"

    def _warm_start(self) -> bool:
        """Whether any stage (the ``retrain`` bc) declares ``warm_start`` — continue re-BC from the prior round's
        weights (smoother) instead of a fresh policy each round (textbook DAgger, higher variance)."""
        return any(s.knobs.get("warm_start", 0.0) > 0.5 for s in self.stages.values())

    def _stage_of_role(self, role: str) -> Stage:
        for s in self.stages.values():
            if s.role == role:
                return s
        raise ValueError(f"{self.profile}: strategy graph has no stage with role {role!r}")

    def _bc_epochs(self) -> int:
        for s in self.stages.values():
            if s.role == "bc":
                return int(s.knobs.get("bc_epochs", 200))
        return 200

    def to_dagger_config(self, *, name: str, select: str) -> DaggerConfig:
        """Extract a :class:`DaggerConfig` from the declared stage knobs (the graph is the source of truth).
        # Preconditions the graph classifies as ``dagger`` and declares source/rollout/aggregate/eval/bc roles."""
        src = self._stage_of_role("source")
        roll = self._stage_of_role("rollout")
        agg = self._stage_of_role("aggregate")
        ev = self._stage_of_role("eval")
        return DaggerConfig(
            name=name, select=select, seeds=self.seeds,
            n_demos=int(src.knobs.get("n_demos", 200)),
            bc_epochs=self._bc_epochs(),
            dagger_iters=self.iterate,
            rollouts_per_iter=int(roll.knobs.get("rollouts", 12)),
            beta=float(roll.knobs.get("beta", 0.5)),
            beta_decay=float(roll.knobs.get("beta_decay", 0.5)),
            expert_replay_ratio=float(agg.knobs.get("expert_replay", 1.0)),
            aggregate_cap_rounds=int(agg.knobs.get("aggregate_cap", 0)),
            warm_start=self._warm_start(),
            n_eval=int(ev.knobs.get("n_eval", 24)),
            device="auto", profile=self.profile)

    def to_campaign_config(self, *, name: str, select: str) -> Any:
        """Extract a :class:`hymeko_rl.train.campaign.CampaignConfig` (TD3+BC) from the declared stages: the off-policy
        knobs come from the ``critic`` (huber, lr) and ``actor`` (bc_coef) stage nodes, the budget from ``total_steps``.
        # Preconditions the graph classifies as ``td3_bc`` (a ``critic`` node is present)."""
        from hymeko_rl.train.campaign import CampaignConfig

        src, ev = self._stage_of_role("source"), self._stage_of_role("eval")
        critic, actor = self._stage_of_role("critic"), self._stage_of_role("actor")
        offpolicy: "dict[str, float]" = {}
        for stage, keys in ((critic, ("critic_huber", "critic_lr")), (actor, ("bc_coef",))):
            offpolicy.update({k: stage.knobs[k] for k in keys if k in stage.knobs})
        return CampaignConfig(
            name=name, select=select, seeds=self.seeds, total_steps=self.total_steps,
            eval_every=max(1_000, self.total_steps // 6), n_demos=int(src.knobs.get("n_demos", 200)),
            bc_epochs=self._bc_epochs(), n_eval=int(ev.knobs.get("n_eval", 24)), n_envs=8, device="auto",
            offpolicy=offpolicy or None)


def _smoke_config(cfg: DaggerConfig) -> DaggerConfig:
    """Cap the budget for a fast local path-check (1 seed, 1 round, few demos)."""
    return replace(cfg, seeds=cfg.seeds[:1], n_demos=min(24, cfg.n_demos), bc_epochs=min(40, cfg.bc_epochs),
                   dagger_iters=min(1, cfg.dagger_iters), rollouts_per_iter=min(3, cfg.rollouts_per_iter),
                   n_eval=min(6, cfg.n_eval))


def run_dagger_graph(profile: "str | Path", *, make_env: Callable[[], Any], build: Callable[[Any], Any],
                     demos: Callable[[Any, int, int], Any], measure: Callable[[Any, Any], "dict[str, float]"],
                     expert_fn: Callable[[Any], Any], name: str, select: str,
                     smoke: bool = False, seeds_override: "tuple[int, ...] | None" = None,
                     gif: bool = True, action_bounds: "tuple[Any, Any] | None" = None,
                     base: "str | Path" = "experiments") -> "dict[str, Any]":
    """Read a strategy dataflow graph, verify it IS DAgger by its topology, and run the existing DAgger executor.

    The topology is load-bearing: a graph whose relabel loop edge is absent classifies as ``bc`` and is **rejected**
    here (so the declared dataflow, not a string, decides the algorithm). Reuses :class:`Dagger` — no second loop.
    # Errors ``ValueError`` (the graph does not classify as ``dagger``).
    """
    g = StrategyGraph.from_hymeko(profile)
    algo = g.classify()
    if algo != "dagger":
        raise ValueError(f"{g.profile}: strategy graph classifies as {algo!r}, not 'dagger' — the relabel loop "
                         f"edge (retrain -> rollout) is absent. Topology is the algorithm.")
    cfg = g.to_dagger_config(name=name, select=select)
    if seeds_override is not None:
        cfg = replace(cfg, seeds=tuple(seeds_override))
    if smoke:
        cfg = _smoke_config(cfg)
    print(f"[strategy-graph] {g.profile}: classified '{algo}' from topology "
          f"(cycle present); iterate={cfg.dagger_iters}, seeds={cfg.seeds}", flush=True)
    return Dagger(cfg, make_env=make_env, build=build, demos=demos, measure=measure,
                  expert_fn=expert_fn, gif=gif, action_bounds=action_bounds).run(base)


def run_td3bc_graph(profile: "str | Path", *, make_env: Callable[[], Any], build: Callable[[Any], Any],
                    demos: Callable[[Any, int, int], Any], measure: Callable[[Any, Any], "dict[str, float]"],
                    name: str, select: str, smoke: bool = False,
                    seeds_override: "tuple[int, ...] | None" = None,
                    base: "str | Path" = "experiments") -> "dict[str, Any]":
    """Read a strategy dataflow graph, verify it IS TD3+BC by its topology (a ``critic`` node), and run the existing
    :class:`Campaign` executor. Symmetric with :func:`run_dagger_graph`: the graph is the verified spec; the critic
    node — absent from the DAgger/BC graphs — is the discriminator. # Errors ``ValueError`` (not ``td3_bc``).
    """
    from hymeko_rl.train.campaign import Campaign

    g = StrategyGraph.from_hymeko(profile)
    algo = g.classify()
    if algo != "td3_bc":
        raise ValueError(f"{g.profile}: strategy graph classifies as {algo!r}, not 'td3_bc' — no critic node "
                         f"(with Q-target edges) in the graph. Topology is the algorithm.")
    cfg = g.to_campaign_config(name=name, select=select)
    if seeds_override is not None:
        cfg = replace(cfg, seeds=tuple(seeds_override))
    if smoke:
        cfg = replace(cfg, seeds=cfg.seeds[:1], total_steps=6_000, eval_every=2_000,
                      n_demos=min(24, cfg.n_demos), bc_epochs=min(40, cfg.bc_epochs), n_eval=min(6, cfg.n_eval))
    print(f"[strategy-graph] {g.profile}: classified '{algo}' from topology "
          f"(critic node present); total_steps={cfg.total_steps}, seeds={cfg.seeds}", flush=True)
    return Campaign(cfg, make_env, build, measure, demos=demos).run(base)
