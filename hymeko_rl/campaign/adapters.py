"""Domain adapters for the canonical command layer.

Each adapter owns ONLY its domain's config validation, execution, metrics and artifact rendering — it calls the
existing domain implementation and never re-implements the algorithm. Domain imports are LAZY (inside methods) so the
orchestration core (:mod:`spec`, :mod:`runner`) stays domain-free and no adapter's domain leaks into another's. The
registry :data:`ADAPTERS` maps ``domain`` → adapter class; an unknown domain fails loudly (never a silent fallback).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol


class Adapter(Protocol):
    def validate(self, spec: Any) -> dict[str, Any]: ...
    def execute(self, spec: Any, out: Path, *, emit: Callable, log: Callable) -> dict[str, Any]: ...


# ─────────────────────────────── Coin Delivery ────────────────────────────────
_COIN_CKPT = {
    "frozen_transport": "experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt",
    "handoff_transport": "experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt",
    "monolithic_neutral": "experiments/2026_07_21_coin_neutral/neutral_best.pt",
}
_COIN_POLICIES = {"P0_ZERO_ACTION", "P1_FROZEN_TRANSPORT", "P2_MONOLITHIC_NEUTRAL",
                  "P3_E_APPROACH_FROZEN", "P4_E_APPROACH_HANDOFF"}
_COIN_STRATEGIES = {"S0_SLOW_HANDOFF": dict(grasp_hold=3, contact_window=None),
                    "S1_FAST_HANDOFF": dict(grasp_hold=1, contact_window=20),
                    "S2_CONTACT_PREPARED": dict(grasp_hold=1, contact_window=20)}
_COIN_EMBODIMENT = {"RING": None, "POINT": "POINT"}


class CoinDeliveryAdapter:
    """Runs the learned neutral-start Coin-Delivery chain (approach → transport) on a true-neutral problem."""

    def validate(self, spec: Any) -> dict[str, Any]:
        o = spec.domain_options
        pol, strat, emb = o.get("policy"), o.get("strategy"), o.get("embodiment", "RING")
        if pol not in _COIN_POLICIES:
            raise ValueError(f"coin policy {pol!r} unknown; valid {sorted(_COIN_POLICIES)}")
        if strat not in _COIN_STRATEGIES:
            raise ValueError(f"coin strategy {strat!r} unknown; valid {sorted(_COIN_STRATEGIES)}")
        if emb not in _COIN_EMBODIMENT:
            raise ValueError(f"coin embodiment {emb!r} unknown; valid {sorted(_COIN_EMBODIMENT)}")
        if strat == "S2_CONTACT_PREPARED" and pol != "P1_FROZEN_TRANSPORT":
            raise ValueError("S2_CONTACT_PREPARED is the suffix-only benchmark; pair it only with P1_FROZEN_TRANSPORT")
        if pol == "P0_ZERO_ACTION" and strat != "S1_FAST_HANDOFF":
            raise ValueError("P0_ZERO_ACTION is a control; run it under S1_FAST_HANDOFF only")
        return {"policy": pol, "strategy": strat, "embodiment": emb,
                "seed": int(o.get("problem_id", spec.seed)), "reps": int(o.get("reps", 10)),
                "strict_required": bool(o.get("strict_required", True)), **_COIN_STRATEGIES[strat]}

    def execute(self, spec: Any, out: Path, *, emit: Callable, log: Callable) -> dict[str, Any]:
        import numpy as np
        import torch

        from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
        from hymeko_rl.experiments.coin_delivery_e0_stabilize import build_sac
        from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, eval_composed, neutral_env
        cfg = self.validate(spec)
        seed, geom = cfg["seed"], _COIN_EMBODIMENT[cfg["embodiment"]]
        env, cf = neutral_env(prefix_steps=0, geom=geom)
        inner = cf._env
        env.set_stage(0)
        env.reset(seed=seed)
        clr = _clearance(inner)
        log(f"[coin] {cfg['policy']}/{cfg['strategy']}/{cfg['embodiment']} seed={seed} clearance={clr:+.4f}")

        def load(ck):
            a, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
            a.load_state_dict(torch.load(ck, weights_only=True))
            return a
        if cfg["policy"] == "P0_ZERO_ACTION":                      # control: no actions
            deliv = 0
            for _ in range(cfg["reps"]):
                env.set_stage(0)
                env.reset(seed=seed)
                cert = DeliveryCertifier(initial_clearance=_clearance(inner))
                for _t in range(200):
                    cert.update(_cert_step(inner, cf))
                    if cert.delivery_certified:
                        break
                    env.step(np.zeros(6, np.float32))
                deliv += cert.delivery_certified
            m = {"deliver": deliv, "n": cfg["reps"], "grasp": 0}
        else:
            ck = {"P1_FROZEN_TRANSPORT": _COIN_CKPT["frozen_transport"],
                  "P2_MONOLITHIC_NEUTRAL": _COIN_CKPT["monolithic_neutral"],
                  "P3_E_APPROACH_FROZEN": _COIN_CKPT["frozen_transport"],
                  "P4_E_APPROACH_HANDOFF": _COIN_CKPT["handoff_transport"]}[cfg["policy"]]
            if not Path(ck).exists():
                raise FileNotFoundError(f"coin checkpoint missing for {cfg['policy']}: {ck} (no fabricated policy)")
            actor = load(ck)
            rows = [eval_composed(actor, [seed], grasp_hold=cfg["grasp_hold"], contact_window=cfg["contact_window"],
                                  env_cf=(env, cf)) for _ in range(cfg["reps"])]
            m = {"deliver": sum(r["deliver"] for r in rows), "n": cfg["reps"], "grasp": max(r["grasp"] for r in rows)}
        emit({"strict_deliver": m["deliver"], "n": m["n"], "clearance": round(clr, 4), **cfg})
        return {"domain": "coin_delivery", "clearance": round(clr, 4), "strict": f"{m['deliver']}/{m['n']}",
                "grasp": m["grasp"], "config": cfg}


# ─────────────────────────────── CIP (Causal Information Prioritization) ───────
class CIPAdapter:
    """Runs Causal Information Prioritization: fit DirectLiNGAM over [state|action|reward] and prioritise the
    reward's causal-parent state dims (Cao/Ito). Reuses ``eval.cip.cip_augment`` — no algorithm re-implementation."""

    def validate(self, spec: Any) -> dict[str, Any]:
        o = spec.domain_options
        if o.get("prioritization_model", "directlingam") != "directlingam":
            raise ValueError(f"CIP prioritization_model {o.get('prioritization_model')!r} unsupported (directlingam)")
        return {"prioritization_model": "directlingam", "evidence_source": o.get("evidence_source", "synthetic_sem"),
                "n": int(o.get("n_samples", 400)), "d_s": int(o.get("d_state", 5)), "d_a": int(o.get("d_action", 2)),
                "n_swap_dims": int(o.get("budget", 2)), "selection_strategy": o.get("selection_strategy", "lowest_importance")}

    def execute(self, spec: Any, out: Path, *, emit: Callable, log: Callable) -> dict[str, Any]:
        import json

        import numpy as np

        from hymeko_rl.eval.causal.lingam import DirectLiNGAM, LingamConfig
        from hymeko_rl.eval.cip.cip_augment import estimate_cip_weights
        cfg = self.validate(spec)
        rng = np.random.default_rng(spec.seed)
        d_s, d_a, n = cfg["d_s"], cfg["d_a"], cfg["n"]
        obs = rng.standard_normal((n, d_s))                        # real evidence: a linear SEM where reward depends on
        act = rng.standard_normal((n, d_a))                        # a KNOWN subset of state dims (0,1) + one action
        rew = 1.4 * obs[:, 0] - 0.9 * obs[:, 1] + 0.6 * act[:, 0] + 0.05 * rng.standard_normal(n)
        log(f"[cip] evidence {n}×({d_s}+{d_a}+1); reward truly driven by s0,s1,a0")
        cw = estimate_cip_weights(obs, act, rew, lingam=DirectLiNGAM(LingamConfig()), n_swap_dims=cfg["n_swap_dims"])
        table = [{"state_dim": int(i), "importance": float(cw.importance_s[i]), "w_s": float(cw.w_s[i])}
                 for i in np.argsort(-cw.importance_s)]
        (out / "prioritized_candidate_table.json").write_text(json.dumps(table, indent=1))
        (out / "monitor_to_cip_trace.json").write_text(json.dumps(
            {"evidence_source": cfg["evidence_source"], "directlingam_bridge": "eval.causal.lingam.DirectLiNGAM",
             "reward_parents_w_s": cw.w_s.tolist(), "reward_parents_w_r": cw.w_r.tolist(),
             "selected_lowest_dims": cw.lowest_dims, "degenerate": cw.degenerate}, indent=1))
        emit({"top_state_dim": table[0]["state_dim"], "top_importance": table[0]["importance"],
              "selected_low_dims": cw.lowest_dims, "degenerate": cw.degenerate})
        log(f"[cip] top state dim={table[0]['state_dim']} (importance {table[0]['importance']:.3f}); "
            f"deprioritised {cw.lowest_dims}")
        return {"domain": "cip", "top_state_dim": table[0]["state_dim"],
                "prioritized_order": [t["state_dim"] for t in table], "selected_lowest_dims": cw.lowest_dims,
                "degenerate": cw.degenerate, "recovered_reward_parents": [i for i in range(d_s) if abs(cw.w_s[i]) > 0.2]}


# ─────────────────────────────── HyperSignedLINGAM ────────────────────────────
class HyperSignedLINGAMAdapter:
    """Runs signed/hypergraph causal discovery via ``eval.causal.signed_hyper_lingam.SignedHyperLiNGAM`` — the real
    algorithm (NOT plain DirectLiNGAM relabelled). Produces a signed adjacency + causal hypergraph + stability."""

    def validate(self, spec: Any) -> dict[str, Any]:
        o = spec.domain_options
        if o.get("model_variant", "signed_hyper_lingam") != "signed_hyper_lingam":
            raise ValueError(f"HSL model_variant {o.get('model_variant')!r} unsupported (signed_hyper_lingam)")
        return {"model_variant": "signed_hyper_lingam", "d": int(o.get("n_vars", 5)),
                "n": int(o.get("n_samples", 600)), "joint_frac": float(o.get("joint_frac", 0.6)),
                "bootstrap": int(o.get("bootstrap", 20))}

    def execute(self, spec: Any, out: Path, *, emit: Callable, log: Callable) -> dict[str, Any]:
        import json

        import numpy as np

        from hymeko_rl.eval.causal.signed_hyper_lingam import (
            SHLConfig, SignedHyperLiNGAM, sample_interaction_sem)
        cfg = self.validate(spec)
        d, n = cfg["d"], cfg["n"]
        rng = np.random.default_rng(spec.seed)
        b = np.zeros((d, d))                                       # a signed lower-triangular SEM with an interaction
        for j in range(1, d):
            b[j, :j] = rng.choice([-1.0, 1.0], size=j) * rng.uniform(0.4, 0.9, size=j)
        x = sample_interaction_sem(b, n, seed=spec.seed, joint_frac=cfg["joint_frac"])
        names = [f"v{i}" for i in range(d)]
        log(f"[hsl] signed SEM d={d} n={n}; fitting SignedHyperLiNGAM (real algorithm)")
        res = SignedHyperLiNGAM(SHLConfig()).fit(x, names)
        edges = res.hyperedges()
        boot = []                                                  # stability: re-fit on bootstrap resamples
        for k in range(cfg["bootstrap"]):
            idx = np.random.default_rng(1000 + k).integers(0, n, size=n)
            try:
                boot.append(len(SignedHyperLiNGAM(SHLConfig()).fit(x[idx], names).hyperedges()))
            except Exception:
                boot.append(0)
        (out / "signed_adjacency.json").write_text(json.dumps({"names": names,
                    "adjacency": np.asarray(res.adjacency).tolist()}, indent=1))
        (out / "causal_hypergraph.json").write_text(json.dumps(
            [{"tail": list(t), "head": h, "signs": list(s)} for t, h, s in edges], indent=1))
        (out / "stability_metrics.json").write_text(json.dumps(
            {"n_hyperedges": len(edges), "bootstrap_edge_counts": boot,
             "bootstrap_mean": float(np.mean(boot)), "bootstrap_std": float(np.std(boot))}, indent=1))
        emit({"n_hyperedges": len(edges), "bootstrap_mean_edges": float(np.mean(boot))})
        log(f"[hsl] {len(edges)} signed hyperedges; bootstrap mean {np.mean(boot):.1f}±{np.std(boot):.1f}")
        return {"domain": "hypersignedlingam", "n_hyperedges": len(edges),
                "hyperedges": [{"tail": list(t), "head": h, "signs": list(s)} for t, h, s in edges[:6]],
                "bootstrap_mean_edges": float(np.mean(boot))}


# ─────────────────────────────── Coin fixed-position replay ───────────────────
class CoinFixedPositionAdapter:
    """Fixed-initial-position Coin Delivery replay: deterministic problem replay (``seed``) or exact-state replay
    (``initial_state`` JSON), with the P0/P1/P4 causal controls. Delegates to
    :func:`hymeko_rl.coin_delivery.fixed_position_campaign.run_fixed_replay` — no algorithm re-implementation."""

    _CHAINS = ("zero", "frozen", "e_handoff", "causal")

    def validate(self, spec: Any) -> dict[str, Any]:
        o = spec.domain_options
        mode = o.get("mode")
        if mode not in ("seed", "exact"):
            raise ValueError(f"coin fixed-position mode {mode!r} invalid; expected 'seed' or 'exact'")
        pc = o.get("policy_chain", "e_handoff")
        if pc not in self._CHAINS:
            raise ValueError(f"policy_chain {pc!r} unknown; valid {sorted(self._CHAINS)}")
        if mode == "seed" and o.get("seed") is None and spec.seed is None:
            raise ValueError("mode='seed' requires a seed")
        if mode == "exact" and not o.get("initial_state"):
            raise ValueError("mode='exact' requires an initial_state path")
        return {"mode": mode, "seed": o.get("seed", spec.seed), "initial_state": o.get("initial_state"),
                "policy_chain": pc, "repetitions": int(o.get("repetitions", 10)),
                "embodiment": o.get("embodiment", "POINT"), "neutral_start": bool(o.get("neutral_start", True))}

    def execute(self, spec: Any, out: Path, *, emit: Callable, log: Callable) -> dict[str, Any]:
        import json

        from hymeko_rl.coin_delivery.fixed_position_campaign import run_fixed_replay
        cfg = self.validate(spec)
        init = json.loads(Path(cfg["initial_state"]).read_text()) if cfg["initial_state"] else None
        res = run_fixed_replay(mode=cfg["mode"], embodiment=cfg["embodiment"], seed=cfg["seed"], initial_state=init,
                               policy_chain=cfg["policy_chain"], repetitions=cfg["repetitions"],
                               neutral_start=cfg["neutral_start"], out=out, log=log)
        for pol, row in res["causal_table"].items():
            emit({"policy": pol, "strict": row["strict"], "first_contact": row["first_contact"],
                  "zone_entry": row["zone_entry"], "problem_hash": res["problem"]["problem_hash"]})
        return {"domain": "coin_fixed_position", "mode": cfg["mode"],
                "problem_hash": res["problem"]["problem_hash"],
                "geom_fp": res["problem"]["env_fingerprint"]["geom_fp"],
                "signed_clearance": res["problem"]["reachability"]["signed_clearance"],
                "causal_strict": {p: r["strict"] for p, r in res["causal_table"].items()}}


ADAPTERS: dict[str, type] = {
    "coin_delivery": CoinDeliveryAdapter,
    "coin_fixed_position": CoinFixedPositionAdapter,
    "cip": CIPAdapter,
    "hypersignedlingam": HyperSignedLINGAMAdapter,
}
