"""Neutral-start Coin Delivery — connect APPROACH → GRASP_ACQUIRE → (proven) TRANSPORT → BRAKE_HOLD_CERTIFY.

The initial-pose audit showed the RL policy was never trained on the approach: `ContactFormationEnv.reset` restores a
contact-prepared bank snapshot and a scripted `grasp_carry` prefix grips the coin, so the policy only ever saw the
transport suffix (5/9), and delivers 0/9 from a true neutral pose. This module makes the hidden prefix EXPLICIT:

- :class:`NeutralCoinDeliveryEnv` — a delivery env whose reset restores the **canonical neutral pose** (no bank
  snapshot, pads open, no contact) and then runs a **controllable** number of scripted `grasp_carry` steps. That count
  is the reverse-curriculum knob: 0 = true neutral (N5), rising to full-handoff = contact-prepared (N0). It fails loud
  if a NEUTRAL_START run begins with contact.
- The scripted `grasp_carry` is now visible curriculum/demonstration data, not a hidden reset side-effect.

Reuses the existing env chain, `p_grasp_carry`, the transport checkpoint, and `DeliveryCertifier`. No contact-geometry /
wrist / critic / replay / n-step / force-closure change (per the task constraints).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.delivery_certificate import CertStep
from hymeko_rl.train.coin_delivery_rl import CoinDeliveryTrainEnv, DeliveryRLConfig, p_grasp_carry


class NeutralCoinDeliveryEnv(CoinDeliveryTrainEnv):
    """CoinDeliveryTrainEnv whose reset starts from the CANONICAL NEUTRAL pose (no bank snapshot) and runs exactly
    ``prefix_steps`` scripted ``grasp_carry`` steps (the reverse-curriculum knob). ``prefix_steps=0`` = true neutral.

    # Preconditions the wrapped ContactFormationEnv's ``_restore`` is neutralised (no-op) so the planar reset stands.
    # Invariants NEUTRAL_START (prefix_steps==0) begins with no fingertip contact, else it raises.
    """

    def __init__(self, cf, cfg=None, *, prefix_steps: int = 0, direct: bool = True) -> None:
        super().__init__(cf, cfg or DeliveryRLConfig())
        self.prefix_steps = int(prefix_steps)
        self.bank_frac = 0.0                                        # >0 ⇒ curriculum: fraction of starts from the
        self._bank = cf._bank                                       #     contact-prepared bank (transport rehearsal)
        self._rng = np.random.default_rng(0)
        self._train_seeds: tuple[int, ...] | None = None
        if direct:
            self._base_override = lambda inner, t: np.zeros(self.action_space.shape[0], np.float32)
            self._delta_override = 1.0

    def set_stage(self, prefix_steps: int) -> None:
        self.prefix_steps = int(prefix_steps)

    def set_curriculum(self, *, bank_frac: float, train_seeds: tuple[int, ...]) -> None:
        self.bank_frac = float(bank_frac)
        self._train_seeds = tuple(train_seeds)

    def _bank_reset(self):
        """Contact-prepared rehearsal start: restore a bank snapshot (the transport-suffix distribution)."""
        cf = self.env
        item = self._bank[int(self._rng.integers(len(self._bank)))]
        from hymeko_rl.env.planar_snapshot import restore_planar
        restore_planar(cf._env, item["snap"])
        cf._tracker.reset(cf._env)
        cf._t = 0
        cf._both_hist = []
        cf._prev_coin = np.asarray(cf._env._planar_metrics.disk_pos[:2], np.float64)
        cf._high = cf._streak = cf._of_run = 0
        cf._had_both = False
        return cf._obs(np.zeros(4, np.float32))

    def _neutral_contact_reset(self, seed):
        """Reset the ContactFormationEnv to the CANONICAL NEUTRAL planar pose (PlanarGraspEnv.reset — arm zeros, coin
        placed for the seed, pads open, no contact) WITHOUT restoring a contact-prepared bank snapshot; mirror the
        ContactFormationEnv bookkeeping the bank path would have set."""
        cf = self.env
        cf._env.reset(seed=seed)                                     # planar neutral reset (no bank snapshot)
        cf._tracker.reset(cf._env)
        cf._t = 0
        cf._both_hist = []
        cf._prev_coin = np.asarray(cf._env._planar_metrics.disk_pos[:2], np.float64)
        cf._high = cf._streak = cf._of_run = 0
        cf._had_both = False
        return cf._obs(np.zeros(4, np.float32))

    def reset(self, *, seed=None):
        # training auto-reset (no seed) with a curriculum: sample a contact-prepared bank rehearsal start OR a neutral
        # start with a random scripted-approach prefix; seeded reset (eval) is always a true neutral start.
        bank_start = seed is None and self.bank_frac > 0 and self._rng.random() < self.bank_frac
        if bank_start:
            obs = self._bank_reset()
            self.env._horizon = self.cfg.prefix_cap + self.cfg.horizon + 8
            self._reset_state()
            self._last_obs = self._start_obs = np.asarray(obs, np.float32)
            self._prev_dtz = self._start_dtz = self._dtz()
            self._prev_both = self._both()
            return self._last_obs, {"handoff_event": True, "start": "bank"}
        use_seed = seed if seed is not None else int(self._rng.choice(self._train_seeds or (1174,)))
        prefix = self.prefix_steps if seed is not None else int(self._rng.integers(0, self.prefix_steps + 1))
        obs = self._neutral_contact_reset(use_seed)                  # true canonical neutral (no bank, no hidden pre-roll)
        self.env._horizon = self.cfg.prefix_cap + self.cfg.horizon + 8
        self._reset_state()
        self._last_obs = np.asarray(obs, np.float32)
        self._start_obs = self._last_obs.copy()
        met0 = self.inner._planar_metrics
        if seed is not None and self.prefix_steps == 0 and (met0.left_contact or met0.right_contact):
            raise RuntimeError("NEUTRAL_START invariant violated: initial fingertip contact at prefix_steps=0 "
                               "(hidden pre-roll or non-neutral snapshot leaked into the reset)")
        for _ in range(prefix):                                     # explicit, curriculum-controlled scripted approach
            acquire = np.clip(p_grasp_carry(self.inner, 0), self.cfg.lo, self.cfg.hi).astype(np.float32)
            obs, _r, _t, _tr, sinfo = self.env.step(acquire)
            self._last_obs = np.asarray(obs, np.float32)
            self._had_both = self._had_both or self._both()
            if sinfo.get("handoff_ready"):
                self._handoff = True
                break
        self._prev_dtz = self._start_dtz = self._dtz()
        self._prev_both = self._both()
        return self._last_obs, {"handoff_event": self._handoff}


def neutral_env(*, prefix_steps: int = 0, geom: str | None = None, arm_mjcf_transform=None,
                coin_shape: str = "cylinder", disk_radius_override=None, disk_radius_y_override=None):
    """Build a NeutralCoinDeliveryEnv (direct-action) whose reset is a true canonical-neutral start (no bank-snapshot
    restore). ``geom=None`` = the canonical E0 CONCAVE_CLAMP ring (the successful chain's embodiment); a fingertip
    string (e.g. ``"POINT"`` = one sphere per arm) builds that embodiment through the SAME builder for transfer tests.
    ``arm_mjcf_transform`` (requires an explicit ``geom``) lets a robot VARIANT supply its whole arm MJCF — the
    BALLTIP matched-panel regression uses ``geom="POINT"`` + a ball-tip transform. Default None ⇒ E0 unchanged."""
    if geom is None:
        from hymeko_rl.experiments.coin_delivery_e0_campaign import _e0_env
        _base, cf = _e0_env()
    else:
        from hymeko_rl.coin_delivery.env_factory import (
            C1_HORIZON, c1_contact_config, coin_monitor_contract, load_contact_bank, make_coin_env)
        from hymeko_rl.env.pad_actuation import build_wristed_contact_env
        planar = make_coin_env(embodiment=geom, arm_mjcf_transform=arm_mjcf_transform,
                               coin_shape=coin_shape, disk_radius_override=disk_radius_override,
                               disk_radius_y_override=disk_radius_y_override)
        cf = build_wristed_contact_env(planar, load_contact_bank("c1_heldseed_bank.pkl", holdout=False),
                                       coin_monitor_contract(), horizon=C1_HORIZON, cfg=c1_contact_config())
    cf._restore = lambda item: None                                 # neutralise the bank-snapshot restore ⇒ neutral pose
    env = NeutralCoinDeliveryEnv(cf, DeliveryRLConfig(), prefix_steps=prefix_steps)
    return env, cf


_HEADLINE = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)
_TRANSPORT_CKPT = "experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt"


def _load_transport():
    import torch

    from hymeko_rl.train.sac import build_sac
    ac, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    ac.load_state_dict(torch.load(_TRANSPORT_CKPT, weights_only=True))
    return ac


def collect_neutral_demos(env, cf, transport_actor, seeds, *, approach_steps=120):
    """Explicit curriculum demos: (a) scripted grasp_carry APPROACH from neutral (partial), (b) learned TRANSPORT from
    bank starts. Records (obs, executed direct-action) for BC. The approach is now visible data, not a hidden pre-roll."""
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    tfn = _greedy_action_fn(transport_actor)
    obs_l, act_l = [], []
    for s in seeds:                                                  # APPROACH demos (grasp_carry from neutral)
        env.set_stage(0)
        o = env.reset(seed=s)[0]
        for _ in range(approach_steps):
            a = np.clip(p_grasp_carry(env.inner, 0), env.cfg.lo, env.cfg.hi).astype(np.float32)
            obs_l.append(np.asarray(o, np.float32))
            act_l.append(a)
            o = env.step(a)[0]
    for s in seeds:                                                  # TRANSPORT demos (transport policy from bank)
        env.bank_frac = 1.0
        env._train_seeds = (s,)
        o = env.reset()[0]
        env.bank_frac = 0.0
        for _ in range(120):
            a = np.asarray(tfn(env, o, None), np.float32)
            obs_l.append(np.asarray(o, np.float32))
            act_l.append(a)
            o = env.step(a)[0]
    return np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32)


def eval_neutral(actor, seeds, *, env_cf=None) -> dict:
    """Full-episode neutral-start eval (prefix=0): first-contact / bilateral-grasp / strict delivery + step counts."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    env, cf = env_cf or neutral_env(prefix_steps=0)
    fn = _greedy_action_fn(actor)
    fc = bi = zone = deliv = 0
    tcert = []
    for s in seeds:
        env.set_stage(0)
        o = env.reset(seed=int(s))[0]
        cert = DeliveryCertifier(initial_clearance=_clearance(cf._env))
        got_fc = got_bi = got_zone = False
        tc = None
        for t in range(300):
            st = _cert_step(cf._env, cf)
            cert.update(st)
            got_fc = got_fc or st.left_fingertip or st.right_fingertip
            got_bi = got_bi or (st.left_fingertip and st.right_fingertip)
            got_zone = got_zone or st.disk_to_zone <= 0.02
            if cert.delivery_certified:
                tc = t
                break
            o = env.step(np.asarray(fn(env, o, None), np.float32))[0]
        fc += got_fc
        bi += got_bi
        zone += got_zone
        deliv += cert.delivery_certified
        if tc is not None:
            tcert.append(tc)
    n = len(seeds)
    return {"first_contact": fc, "bilateral": bi, "zone": zone, "deliver": deliv, "n": n,
            "med_steps_cert": int(np.median(tcert)) if tcert else None}


def _clearance(inner) -> float:
    disk_r = float(inner.model.geom_size[inner._disk_geom][0])
    return float(inner.planar_metrics.disk_to_zone) - (disk_r + float(inner._zone_half))


def _cert_step(inner, cf) -> CertStep:
    from hymeko_rl.experiments.coin_wristed_delivery import _both_pad_contact
    met = inner._planar_metrics
    lg = getattr(met, "legality", None)
    lf = bool(lg.left_fingertip_contact) if lg is not None else bool(met.left_contact)
    rf = bool(lg.right_fingertip_contact) if lg is not None else bool(met.right_contact)
    body = bool(lg.arm_body_contact) if lg is not None else False
    imp = float(lg.arm_body_contact_impulse) if lg is not None else 0.0
    v = inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]
    _both, fl, fr = _both_pad_contact(cf)
    return CertStep(disk_to_zone=float(met.disk_to_zone), disk_speed=float(np.linalg.norm(v)),
                    left_fingertip=lf, right_fingertip=rf, arm_body_contact=body, arm_body_impulse=imp,
                    force_left=fl, force_right=fr,
                    body_progress=float(getattr(inner, "_body_progress", 0.0)),   # body-only-driven coin displacement
                    ever_grasped=bool(getattr(inner, "_ever_grasped", False)))    # a bilateral grasp formed (physical-contact contract)


def _e_approach_actor():
    from hymeko_rl.coin_delivery.e_approach import load_e_approach_policy
    return load_e_approach_policy()                                # canonical production loader (no experiment imports)


def collect_e_handoff_bank(seeds, *, grasp_hold=3, maxk=160):
    """§1/§8 — regenerate EXPLICIT neutral→bilateral-grasp trajectories with the recovered E_valselect approach policy;
    snapshot the stable-grasp handoff state. These are the learned handoff states the transport must be tuned on."""
    import torch

    from hymeko_rl.env.planar_snapshot import snapshot_planar
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _e0_env
    e = _e_approach_actor()
    _dev, cf = _e0_env()
    inner = cf._env
    bank = []
    for s in seeds:
        inner.reset(seed=int(s))
        bi = 0
        for _k in range(maxk):
            m = inner._planar_metrics
            bi = bi + 1 if (m.left_contact and m.right_contact) else 0
            if bi >= grasp_hold:
                bank.append({"snap": snapshot_planar(inner), "coin_y": float(m.disk_pos[1]), "high_coin_y": False})
                break
            with torch.no_grad():
                a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].numpy()
            inner.step(np.asarray(a, np.float32))
    return bank


def eval_composed(transport_actor, seeds, *, grasp_hold=3, contact_window=None, env_cf=None, approach_actor=None) -> dict:
    """§7/§9 — learned APPROACH (recovered E, or ``approach_actor``) → TRANSPORT; strict neutral delivery + grasp rate.

    Handoff fires as soon as bilateral contact is confirmed for ``grasp_hold`` steps (§3: use 1 for the accelerated
    controller) OR — for push-delivery states where a full grasp never forms — one-sided contact is held for
    ``contact_window`` steps (removes the wait-to-160-cap dead time). ``contact_window=None`` is the original behaviour.
    """
    import torch

    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    e = approach_actor or _e_approach_actor()
    tfn = _greedy_action_fn(transport_actor)
    env, cf = env_cf or neutral_env(prefix_steps=0)
    inner = cf._env
    grasp = deliv = 0
    handoff_steps = []
    for s in seeds:
        env.set_stage(0)
        env.reset(seed=int(s))
        cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        bi = cw = 0
        got = False
        t_hand = 160
        for _k in range(160):
            m = inner._planar_metrics
            cert.update(_cert_step(inner, cf))
            bi = bi + 1 if (m.left_contact and m.right_contact) else 0
            cw = cw + 1 if (m.left_contact or m.right_contact) else 0
            if bi >= grasp_hold:
                got = True
                t_hand = _k
                break
            if contact_window is not None and cw >= contact_window:  # push-delivery early handoff (no full grasp)
                t_hand = _k
                break
            with torch.no_grad():
                a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].numpy()
            inner.step(np.asarray(a, np.float32))
        cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
        cf._t = 0
        cf._both_hist = []
        env._suffix_t = 0
        env._prev_dtz = env._dtz()
        env._prev_both = env._both()
        o = cf._obs(np.zeros(4, np.float32))
        for _t in range(200):
            cert.update(_cert_step(inner, cf))
            if cert.delivery_certified:
                break
            o = env.step(np.asarray(tfn(env, o, None), np.float32))[0]
        grasp += got
        deliv += cert.delivery_certified
        handoff_steps.append(t_hand)
    return {"grasp": grasp, "deliver": deliv, "n": len(seeds), "handoff_steps": handoff_steps}


def collect_carry_demos(seeds, *, grasp_hold=3, maxk=160, carry_steps=200):
    """§6/§11 — handoff-matched demos: from each learned E-grasp state, run the validated targetward CARRY
    (``p_grasp_carry``) and record (obs, executed carry action) ONLY for trajectories that reach strict delivery. These
    teach a transport COPY to push-deliver from the E-approach grip geometry; the scripted carry never ships in the
    final rollout (it is only the demo generator)."""
    import torch

    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    e = _e_approach_actor()
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    obs_l, act_l, n_ok = [], [], 0
    for s in seeds:
        env.set_stage(0)
        env.reset(seed=int(s))
        bi = 0
        grabbed = False
        for _k in range(maxk):
            m = inner._planar_metrics
            bi = bi + 1 if (m.left_contact and m.right_contact) else 0
            if bi >= grasp_hold:
                grabbed = True
                break
            with torch.no_grad():
                a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].numpy()
            inner.step(np.asarray(a, np.float32))
        if not grabbed:
            continue
        cf._tracker.reset(inner)
        cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
        cf._t = 0
        cf._both_hist = []
        cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        o = cf._obs(np.zeros(4, np.float32))
        traj_o, traj_a = [], []
        for _t in range(carry_steps):
            cert.update(_cert_step(inner, cf))
            if cert.delivery_certified:
                break
            act = np.clip(p_grasp_carry(inner, 0), -1.0, 1.0).astype(np.float32)
            traj_o.append(np.asarray(o, np.float32))
            traj_a.append(act)
            o = cf.step(act)[0]
        if cert.delivery_certified:                                 # keep only the SUCCESSFUL handoff-matched demos
            obs_l.extend(traj_o)
            act_l.extend(traj_a)
            n_ok += 1
    return np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32), n_ok


def train_neutral(*, steps=40_000, seed=0, out="experiments/2026_07_21_coin_neutral", eval_every=4_000,
                  bank_frac=0.35, prefix_max=110):
    """Reverse-curriculum neutral-start training: init actor from the proven TRANSPORT checkpoint; train on a curriculum
    env that samples contact-prepared bank rehearsal (≥bank_frac) + neutral starts with a random scripted-approach
    prefix in [0, prefix_max]; BC-anchor on the explicit approach+transport demos. Eval on true neutral (prefix=0)."""
    import json
    from pathlib import Path

    import torch

    from hymeko_rl.experiments.coin_delivery_e0_campaign import bc_fit
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    train_seeds = tuple(s for s in range(1000, 1700) if s not in _HEADLINE)
    env, cf = neutral_env(prefix_steps=prefix_max)
    env.set_curriculum(bank_frac=bank_frac, train_seeds=train_seeds)
    eval_env = neutral_env(prefix_steps=0)
    transport = _load_transport()
    demos = collect_neutral_demos(env, cf, transport, train_seeds[:24])
    print(f"[neutral] demos={len(demos[0])} pairs | bank_frac={bank_frac} prefix_max={prefix_max}", flush=True)
    actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(transport.state_dict())                   # §4 init TRANSPORT from the proven checkpoint
    base = eval_neutral(actor, _HEADLINE, env_cf=eval_env)
    print(f"[neutral base (transport init)] first_contact={base['first_contact']}/9 bilateral={base['bilateral']}/9 "
          f"deliver={base['deliver']}/9", flush=True)
    bc_fit(actor, demos, epochs=300, seed=seed)                     # anchor approach+transport before RL
    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0, alpha_final=0.0367,
                           log_every=eval_every, eval_every=eval_every)
    best = {"deliver": -1, "first_contact": -1}
    hist = []

    def eval_fn(_e, ac):
        m = eval_neutral(ac, _HEADLINE, env_cf=eval_env)
        hist.append(m)
        print(f"  [neutral eval#{len(hist)}] fc={m['first_contact']}/9 bi={m['bilateral']}/9 zone={m['zone']}/9 "
              f"deliver={m['deliver']}/9", flush=True)
        key = (m["deliver"], m["first_contact"])
        if key > (best["deliver"], best["first_contact"]):
            best.update(deliver=m["deliver"], first_contact=m["first_contact"])
            torch.save(ac.state_dict(), outp / "neutral_best.pt")
        return float(m["deliver"] * 10 + m["first_contact"])

    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=demos,
                      bc_coef_fn=lambda _s: 1.0)
    torch.save(actor.state_dict(), outp / "neutral_final.pt")
    (outp / "run.json").write_text(json.dumps(dict(steps=steps, seed=seed, bank_frac=bank_frac, prefix_max=prefix_max,
                                    base=base, best=best, eval_history=hist, curve=curve), indent=1, default=float))
    print(f"[neutral done] base deliver={base['deliver']}/9 → best deliver={best['deliver']}/9 "
          f"first_contact {base['first_contact']}→{best['first_contact']}/9", flush=True)
    return best


def finetune_transport_on_handoff(*, steps=24_000, seed=0, out="experiments/2026_07_21_coin_neutral_ft",
                                  eval_every=3_000, rehearse_frac=0.4):
    """§8 — the recovered E approach reaches grasp 6-7/9 but the frozen transport sees a SHIFTED handoff distribution
    (0/9 delivery). Fine-tune a COPY of transport on the learned E-handoff states + rehearse the original bank; require
    retention of the original contact-prepared competence. Then re-evaluate the full neutral chain."""
    import json
    from pathlib import Path

    import torch

    from hymeko_rl.experiments.coin_delivery_e0_campaign import (
        _HEADLINE as HB, bc_fit, collect_e0_demos, direct_e0_env, evaluate_policy)
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    train_seeds = tuple(s for s in range(1000, 1700) if s not in _HEADLINE)
    ehand = collect_e_handoff_bank(train_seeds[:70])                # learned E-approach handoff states (new distribution)
    env, cf = direct_e0_env()
    c1 = list(cf._bank)
    n_reh = max(1, int(len(ehand) * rehearse_frac / (1 - rehearse_frac)))
    cf._bank = ehand + c1[:n_reh]                                   # ≥rehearse_frac original-bank rehearsal
    print(f"[ft] E-handoff states={len(ehand)} + rehearsal={min(n_reh, len(c1))} → mixed bank {len(cf._bank)}", flush=True)
    transport = _load_transport()
    actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(transport.state_dict())                  # fine-tune a COPY of the proven transport
    split = json.loads(Path("experiments/2026_07_21_coin_e0_stabilize/state_split.json").read_text())
    demos = collect_e0_demos([s for s, _c in split["train"]])
    bc_fit(actor, demos, epochs=200, seed=seed)
    eval_env = neutral_env(prefix_steps=0)
    ret_env = direct_e0_env()                                       # original bank, for retention eval
    base = eval_composed(actor, HB, env_cf=eval_env)
    ret0 = evaluate_policy(actor, HB, env_cf=(ret_env[0], ret_env[1]))
    print(f"[ft base] neutral chain deliver={base['deliver']}/9 grasp={base['grasp']}/9 | "
          f"contact-prepared retention={ret0['delivery_count']}/9", flush=True)
    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0, alpha_final=0.0367,
                           log_every=eval_every, eval_every=eval_every)
    best = {"deliver": -1}
    hist = []

    def eval_fn(_e, ac):
        m = eval_composed(ac, HB, env_cf=eval_env)
        r = evaluate_policy(ac, HB, env_cf=(ret_env[0], ret_env[1]))
        hist.append({"deliver": m["deliver"], "grasp": m["grasp"], "retention": r["delivery_count"]})
        print(f"  [ft eval#{len(hist)}] neutral deliver={m['deliver']}/9 grasp={m['grasp']}/9 "
              f"retention={r['delivery_count']}/9", flush=True)
        if m["deliver"] > best["deliver"]:
            best.update(deliver=m["deliver"], retention=r["delivery_count"])
            torch.save(ac.state_dict(), outp / "transport_ft_best.pt")
        return float(m["deliver"] * 10 + r["delivery_count"])

    train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=demos, bc_coef_fn=lambda _s: 1.0)
    torch.save(actor.state_dict(), outp / "transport_ft_final.pt")
    (outp / "run.json").write_text(json.dumps(dict(steps=steps, seed=seed, n_ehandoff=len(ehand), base=base,
                                    best=best, eval_history=hist), indent=1, default=float))
    print(f"[ft done] base neutral deliver={base['deliver']}/9 → best={best['deliver']}/9 "
          f"(retention {best.get('retention', '?')}/9)", flush=True)
    return best


def train_handoff_transport(*, steps=16_000, seed=0, out="experiments/2026_07_21_coin_neutral_handoff",
                            eval_every=2_000, rehearse_frac=0.4):
    """§6/§11 — BC a transport COPY on the SUCCESSFUL carry-from-E-grasp demos (handoff-matched) + ≥rehearse_frac
    original-bank transport demos, then a short SAC polish (α-floor, persistent BC). Compose learned E-approach → this
    learned transport and evaluate strict NEUTRAL delivery per state. The scripted carry never appears in the rollout."""
    import json
    from pathlib import Path

    import torch

    from hymeko_rl.experiments.coin_delivery_e0_campaign import (
        bc_fit, collect_e0_demos, direct_e0_env, evaluate_policy)
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    train_seeds = tuple(s for s in range(1000, 1700) if s not in _HEADLINE)
    carry = collect_carry_demos(train_seeds[:120])                  # handoff-matched (successful carry-from-E-grasp)
    split = json.loads(Path("experiments/2026_07_21_coin_e0_stabilize/state_split.json").read_text())
    bank_demos = collect_e0_demos([s for s, _c in split["train"]])  # original transport-bank rehearsal
    n_reh = max(1, int(len(carry[0]) * rehearse_frac / (1 - rehearse_frac)))
    ridx = np.random.default_rng(0).choice(len(bank_demos[0]), size=min(n_reh, len(bank_demos[0])), replace=False)
    obs = np.concatenate([carry[0], bank_demos[0][ridx]])
    act = np.concatenate([carry[1], bank_demos[1][ridx]])
    print(f"[handoff] carry-demo trajectories ok={carry[2]} ({len(carry[0])} pairs) + rehearsal {len(ridx)} "
          f"→ BC corpus {len(obs)}", flush=True)
    if carry[2] == 0:
        raise SystemExit("BLOCKED: no successful carry-from-E-grasp trajectory to BC-seed the handoff transport")
    env, cf = direct_e0_env()                                       # bank-start env for the SAC polish
    actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    actor.load_state_dict(_load_transport().state_dict())          # a COPY of the proven transport
    bc_fit(actor, (obs, act), epochs=400, seed=seed)
    eval_env = neutral_env(prefix_steps=0)
    ret_env = direct_e0_env()

    def full_eval(ac):
        m = eval_composed(ac, _HEADLINE, env_cf=eval_env)
        r = evaluate_policy(ac, _HEADLINE, env_cf=(ret_env[0], ret_env[1]))
        return m["deliver"], m["grasp"], r["delivery_count"]

    d0, g0, r0 = full_eval(actor)
    print(f"[handoff BC-init] neutral deliver={d0}/9 grasp={g0}/9 | contact-prepared retention={r0}/9", flush=True)
    torch.save(actor.state_dict(), outp / "handoff_best.pt")
    best = {"deliver": d0, "retention": r0}
    hist = [{"deliver": d0, "retention": r0, "step": 0}]
    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0, alpha_final=0.0367,
                           log_every=eval_every, eval_every=eval_every)

    def eval_fn(_e, ac):
        d, g, r = full_eval(ac)
        hist.append({"deliver": d, "grasp": g, "retention": r, "step": len(hist) * eval_every})
        print(f"  [handoff eval#{len(hist)}] neutral deliver={d}/9 grasp={g}/9 retention={r}/9", flush=True)
        if (d, r) > (best["deliver"], best["retention"]):
            best.update(deliver=d, retention=r)
            torch.save(ac.state_dict(), outp / "handoff_best.pt")
        return float(d * 10 + r)

    train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(obs, act), bc_coef_fn=lambda _s: 1.0)
    (outp / "run.json").write_text(json.dumps(dict(steps=steps, seed=seed, carry_ok=carry[2], bc_init=dict(
        deliver=d0, grasp=g0, retention=r0), best=best, eval_history=hist), indent=1, default=float))
    print(f"[handoff done] neutral deliver {d0}→{best['deliver']}/9 | retention {best['retention']}/9", flush=True)
    return best


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="handoff", choices=["neutral", "finetune", "handoff"])
    ap.add_argument("--steps", type=int, default=16_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_neutral_handoff")
    a = ap.parse_args()
    if a.cmd == "neutral":
        train_neutral(steps=a.steps, seed=a.seed, out=a.out)
    elif a.cmd == "finetune":
        finetune_transport_on_handoff(steps=a.steps, seed=a.seed, out=a.out)
    else:
        train_handoff_transport(steps=a.steps, seed=a.seed, out=a.out)


if __name__ == "__main__":
    main()
