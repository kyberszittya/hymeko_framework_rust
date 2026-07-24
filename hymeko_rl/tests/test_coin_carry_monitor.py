"""§2A gate — the delivery verdict is an online trace-monitor whose semantics come from the `.hymeko`, and it matches the
frozen env `_strict` certificate on the FULL temporal trace (not just final K6). The env counter is a shadow only."""
import copy
import json

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_monitor import (
    MonitorBackend,
    TraceSample,
    load_carry_monitor_spec,
    make_monitor,
)
from hymeko_rl.coin_delivery.coin_carry_option_rl import execute_one_option
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
THETAS = [
    np.array([2, -2, 1, -1, 0, 0, 1, -1, 0.5, -0.5, 0, 0, 6, 6, 6], np.float32),
    np.array([3, 3, -3, -3, -2, -2, 2, 2, 0, 0, 0, 0, 8, 4, 10], np.float32),
    np.array([-1, 1, 2, -2, 1, 1, -1, -1, 1, -1, 0.5, -0.5, 4, 12, 5], np.float32),
]


def test_monitor_spec_sourced_from_hymeko_matches_frozen_constants():
    # the certificate tolerances come from coin_carry_option_v1.hymeko's @certificate, and equal the frozen env constants
    sp = load_carry_monitor_spec()
    assert sp.center_tol == CENTER_TOL and sp.settle_vel == SETTLE_VEL and sp.held_dwell == HELD_DWELL
    assert sp.entry_tol == 0.05


def test_python_backend_reproduces_strict_dwell_on_known_trace():
    sp = load_carry_monitor_spec()
    mon = make_monitor("python")
    assert isinstance(mon, MonitorBackend)
    mon.reset(sp, initial_dtz=0.10, initial_touched=True)               # start uncontained, already touched
    # 5 contained+slow steps (strict→5), then one exit (reset), then 6 contained+slow (strict→6 = K6)
    seq = [(0.01, 0.01)] * 5 + [(0.10, 0.20)] + [(0.01, 0.01)] * 6
    for dtz, sp_ in seq:
        mon.observe(TraceSample(dtz=dtz, speed=sp_, touched=True, contact=True, terminated=(dtz == seq[-1][0] and sp_ == seq[-1][1])))
    v = mon.verdict()
    assert v["max_strict"] == 6 and v["k6"] == 1 and v["reached_handoff"] == 1
    assert v["contain_exit_ct"] == 1                                    # exactly one containment exit (after the first run of 5)
    evs = [e for _t, e in v["events"]]
    assert "containment_enter" in evs and "containment_exit" in evs and "handoff" in evs and "delivered" in evs
    assert evs.index("handoff") < evs.index("delivered")               # handoff precedes delivery in time
    # K6 requires touched: an untouched identical trace must NOT deliver
    mon2 = make_monitor("python"); mon2.reset(sp, 0.10, initial_touched=False)
    for dtz, sp_ in seq:
        mon2.observe(TraceSample(dtz=dtz, speed=sp_, touched=False, contact=False, terminated=False))
    assert mon2.verdict()["k6"] == 0 and mon2.verdict()["max_strict"] == 6


def test_full_trace_parity_monitor_equals_env_shadow_failclosed():
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    n_ok, k6 = 0, 0
    for r in cfg["banks"]["late_dev"]["rows"][:6]:
        ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
        rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        for theta in THETAS:
            # verify_shadow=True fail-closes (raises) at the first step the monitor strict != env _strict
            o = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=140, verify_shadow=True)
            n_ok += 1; k6 += o["k6"]
            # verdict self-consistency: k6 ⇒ handoff; delivered event ⇔ k6; handoff event present ⇔ reached_handoff
            evs = [e for _t, e in o["monitor_events"]]
            assert not (o["k6"] and not o["reached_handoff"])
            assert ("delivered" in evs) == bool(o["k6"])
            assert ("handoff" in evs) == bool(o["reached_handoff"])
    assert n_ok == 18 and k6 >= 1                                       # all rollouts passed the fail-closed shadow gate


def test_default_executor_uses_monitor_verdict_not_env_counter():
    # the returned certificate keys come from the monitor (present) + the executor exposes the monitor event trace
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    torch.manual_seed(0)
    o = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), THETAS[0], pi0, base, gamma=0.99, horizon=120)
    for k in ("k6", "reached_handoff", "contain_exit_ct", "monitor_events"):
        assert k in o
    assert isinstance(o["monitor_events"], tuple)
