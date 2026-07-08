"""Stabilized behaviour cloning — variance-reduction recipes for the demo-mix imitation (seed_stabilized_demo_mix_v2).

The demo-mix BC fine-tune is NOT training-seed-robust (reports/2026-07-08-option-msdm-trainseed-robustness): it
robustly injects sustained contact but robustly trades delivery, with high ft_dom variance across training seeds.
This ONE configurable trainer covers the variance-reduction recipes (config, not a function per recipe — §6.5 #1):

* **anchor** (`anchor_coef>0`): add `anchor_coef · MSE(π(o), π_DAgger(o))` — regularize toward the frozen baseline
  policy so the fine-tune cannot wander far from the delivering basin (recipe C).
* **balanced_batch** (`True`): every minibatch is 50/50 delivery-completion / sustained-contact, instead of a fixed
  global ratio — removes batch-composition gradient variance (recipe D).
* **val-selected checkpointing** (`val_every>0` + a `val_fn`): periodically evaluate a held-out validation gate and
  keep the best checkpoint by a val score — tests whether good seeds/checkpoints can be picked BEFORE test (recipe
  E). Central question of v2: does the val score predict test ft_dom?

The trainer is pure: the validation evaluator is injected as `val_fn(actor)->dict` by the driver (which owns
`measure_policy`/`audit`), so this module has no experiment-layer dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from hymeko_rl.train.demo_mix import TaggedPools, mix_pools


@dataclass
class StabilizedBCConfig:
    lr: float = 1e-3
    epochs: int = 200
    batch: int = 256
    frac_sustained: float = 0.25
    mix_total: int = 40_000
    anchor_coef: float = 0.0             # recipe C
    balanced_batch: bool = False         # recipe D
    val_every: int = 0                   # recipe E: >0 → val-selected checkpointing
    device: str = "cpu"
    seed: int = 0


@dataclass
class StabilizedBCResult:
    final_state: dict
    val_selected_state: dict
    best_val_score: float
    best_epoch: int
    val_history: list[dict] = field(default_factory=list)
    final_loss: float = 0.0
    best_val_metrics: dict = field(default_factory=dict)   # the val-gate metrics at the selected checkpoint


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _cpu_state(actor: Any) -> dict:
    return {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}


def _val_score(m: dict) -> float:
    """Selection objective — sum of the two 'preserve' headline metrics (delivery + monitor). No test leak: the
    val_fn evaluates a val seed disjoint from the test eval seeds. Picks checkpoints that keep BOTH delivery and
    monitor quality (not one at the other's expense)."""
    return float(m.get("ft_dom", 0.0)) + float(m.get("monitor_score", 0.0))


def train_stabilized_bc(pools: TaggedPools, frozen_dagger: Any, cfg: StabilizedBCConfig, *,
                        fresh_actor_fn: Callable[[], Any], val_fn: "Callable[[Any], dict] | None" = None,
                        log: Callable[[str], None] = print) -> StabilizedBCResult:
    """Warm-started BC fine-tune with optional DAgger-anchor / balanced-batch / val-selected checkpointing.

    # Preconditions: ``fresh_actor_fn()`` returns a warm-started (frozen-DAgger-init) actor; ``frozen_dagger`` is
      the frozen baseline (for the anchor + it stays frozen); pools non-empty. # Postconditions: a
      :class:`StabilizedBCResult` with the final AND val-selected state dicts (both CPU); no critic-gradient, base
      policy re-fit only; ``val_selected_state == final_state`` when no val_fn/val_every."""
    torch.manual_seed(cfg.seed)
    dev = _resolve_device(cfg.device)
    frozen_dagger.to(dev).eval()
    for p in frozen_dagger.parameters():
        p.requires_grad_(False)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731

    if cfg.balanced_batch:
        so, sa = t(pools.sustained_obs), t(pools.sustained_acts)
        do, da = t(pools.deliver_obs), t(pools.deliver_acts)
        with torch.no_grad():
            s_dag = frozen_dagger(so) if so.shape[0] else so.new_zeros((0, sa.shape[-1]))
            d_dag = frozen_dagger(do) if do.shape[0] else do.new_zeros((0, da.shape[-1]))
        n_batches = max(1, cfg.mix_total // cfg.batch)
    else:
        obs_np, act_np = mix_pools(pools, cfg.frac_sustained, total=cfg.mix_total, seed=cfg.seed)
        obs_t, act_t = t(obs_np), t(act_np)
        with torch.no_grad():
            dag_t = frozen_dagger(obs_t)
        n = obs_t.shape[0]

    actor = fresh_actor_fn().to(dev)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    best_score, best_state, best_epoch, history = -1e18, None, -1, []
    best_metrics: dict = {}
    last_loss = 0.0

    def _step(ob: torch.Tensor, ac: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
        pred = actor.action_mean(ob)
        loss = F.mse_loss(pred, ac)
        if cfg.anchor_coef > 0.0:
            loss = loss + cfg.anchor_coef * F.mse_loss(pred, dg)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        return loss.detach()

    for ep in range(cfg.epochs):
        ep_loss = torch.zeros((), device=dev)
        nb = 0
        if cfg.balanced_batch:
            half = max(1, cfg.batch // 2)
            for _ in range(n_batches):
                si = torch.as_tensor(rng.integers(0, max(1, so.shape[0]), size=half), device=dev)
                di = torch.as_tensor(rng.integers(0, max(1, do.shape[0]), size=cfg.batch - half), device=dev)
                ob = torch.cat([so[si], do[di]], 0)
                ac = torch.cat([sa[si], da[di]], 0)
                dg = torch.cat([s_dag[si], d_dag[di]], 0)
                ep_loss += _step(ob, ac, dg)
                nb += 1
        else:
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch]
                ep_loss += _step(obs_t[idx], act_t[idx], dag_t[idx])
                nb += 1
        last_loss = float(ep_loss.item()) / max(1, nb)
        if val_fn is not None and cfg.val_every and ((ep + 1) % cfg.val_every == 0 or ep + 1 == cfg.epochs):
            actor.to(torch.device("cpu")).eval()
            vm = val_fn(actor)
            score = _val_score(vm)
            history.append({"epoch": ep + 1, "score": round(score, 4), **{k: round(float(v), 4)
                            for k, v in vm.items() if isinstance(v, (int, float))}})
            if score > best_score:
                best_score, best_state, best_epoch = score, _cpu_state(actor), ep + 1
                best_metrics = {k: float(v) for k, v in vm.items() if isinstance(v, (int, float))}
            log(f"  [sbc] seed={cfg.seed} ep {ep + 1}/{cfg.epochs} loss={last_loss:.4g} "
                f"val_score={score:.3f} (ft_dom={vm.get('ft_dom', 0):.3f} mon={vm.get('monitor_score', 0):.3f})")
            actor.to(dev).train()

    actor.to(torch.device("cpu")).eval()
    final_state = _cpu_state(actor)
    return StabilizedBCResult(final_state=final_state,
                              val_selected_state=best_state if best_state is not None else final_state,
                              best_val_score=best_score if best_state is not None else 0.0,
                              best_epoch=best_epoch, val_history=history, final_loss=last_loss,
                              best_val_metrics=best_metrics)
