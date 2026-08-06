"""§3 gates — first-class skill routing sourced from the `.hymeko`: generated route == manual route; rollout skill-binding
trace parity; data-driven binding; fail-closed on bad attributes; a frozen skill cannot be optimized; checkpoint provenance
in the manifest."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_option_rl import execute_one_option
from hymeko_rl.coin_delivery.coin_carry_skills import (
    SkillBinding,
    handoff_index,
    load_carry_skill_routing,
    optimizer_parameters,
    skill_binding_trace,
    validate_routing,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
from hymeko_rl.option_rl.hierarchy import SkillRoute

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
CARRY = "data/robotics/coin_carry_option_v1.hymeko"


def _b(phase, role, train, binding=None, cert=None, ckpt=None, sha=None):
    return SkillBinding(phase, role, train, binding, cert, ckpt, sha)


def test_generated_route_equals_manual_route():
    r = load_carry_skill_routing()
    # the manually-intended route: PUSH/BRAKE/RELEASE trainable-upstream carry_option; HANDOFF frozen-downstream settling_pi0
    assert r.upstream_phases() == ("PUSH", "BRAKE", "RELEASE")
    assert r.downstream().phase == "HANDOFF" and r.downstream().binding == "settling_pi0"
    assert r.trainable_bindings() == {"carry_option"} and r.frozen_bindings() == {"settling_pi0"}
    assert r.handoff_certificate() == "stable_entry_v1"
    # the framework SkillRoute is GENERATED from the description (Python only supplies the callables)
    route = r.to_skill_route(handed_off=lambda o: o["reached_handoff"] == 1, downstream_skill="pi0")
    assert isinstance(route, SkillRoute) and route.name == "stable_entry_v1"
    assert route.route({"reached_handoff": 1}) == "downstream_frozen_skill"
    assert route.route({"reached_handoff": 0}) == "upstream_option_redecide"


def test_rollout_skill_binding_trace_parity():
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = load_carry_skill_routing()
    theta = np.array([2, -2, 1, -1, 0, 0, 1, -1, 0.5, -0.5, 0, 0, 6, 6, 6], np.float32)
    for row in cfg["banks"]["late_dev"]["rows"][:4]:
        ls = LateStart(seed=row[0], prefix_steps=row[1], family=row[2], obs_sha=row[3], base_sha=row[4], causal_sha=row[5])
        rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        tr = []
        o = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), theta, pi0, base, gamma=0.99, horizon=140, trace=tr)
        sb = skill_binding_trace(r, tr)
        assert len(sb) == len(tr)
        for p, s in zip(tr, sb):
            if p in ("PUSH", "BRAKE", "RELEASE"):
                assert s == "carry_option"                               # upstream trainable option
            elif p == "HANDOFF":
                assert s == "settling_pi0"                               # frozen downstream skill
            else:
                assert s is None                                          # terminal marker binds no skill
        hi = handoff_index(tr)
        assert (hi is not None) == bool(o["reached_handoff"])             # handoff time defined iff handed off
        for k in ("k6", "contain_exit_ct", "R_option", "done"):
            assert k in o                                                 # metrics (K6/exit/reward/termination) intact


def test_binding_is_data_driven_change_hymeko_changes_route(tmp_path):
    # write a variant profile (in data/robotics/ so the meta import resolves) with a DIFFERENT upstream binding
    src = Path(CARRY).read_text()
    variant = Path("data/robotics/_test_carry_skill_variant.hymeko")
    variant.write_text(src.replace("binding carry_option;", "binding carry_option_v2;")
                          .replace("coin_carry_option:", "coin_carry_option_variant:")
                          .replace("coin_carry_option_description", "coin_carry_option_variant_description"))
    try:
        r = load_carry_skill_routing(str(variant))
        assert r.trainable_bindings() == {"carry_option_v2"}             # runtime binding follows the .hymeko, no Python edit
    finally:
        variant.unlink()


def test_fail_closed_on_missing_or_contradictory_attributes():
    good_ds = _b("HANDOFF", "downstream", "frozen", "settling_pi0", "c", "ck", "sha")
    import pytest
    with pytest.raises(ValueError):                                      # missing role
        validate_routing([_b("PUSH", None, "trainable", "carry_option"), good_ds])
    with pytest.raises(ValueError):                                      # contradiction: upstream must be trainable
        validate_routing([_b("PUSH", "upstream", "frozen", "carry_option"), good_ds])
    with pytest.raises(ValueError):                                      # contradiction: downstream must be frozen
        validate_routing([_b("HANDOFF", "downstream", "trainable", "settling_pi0", "c", "ck", "sha")])
    with pytest.raises(ValueError):                                      # no upstream
        validate_routing([good_ds])
    with pytest.raises(ValueError):                                      # downstream missing certificate/checkpoint
        validate_routing([_b("PUSH", "upstream", "trainable", "carry_option"), _b("HANDOFF", "downstream", "frozen", "settling_pi0")])


def test_frozen_skill_cannot_be_optimized():
    import pytest
    r = load_carry_skill_routing()
    carry = torch.nn.Linear(4, 4); settle = torch.nn.Linear(4, 4)
    params = optimizer_parameters(r, {"carry_option": carry})            # trainable skill → params collected
    assert len(params) == len(list(carry.parameters()))
    with pytest.raises(ValueError):                                      # frozen skill offered to the optimizer → fail-closed
        optimizer_parameters(r, {"carry_option": carry, "settling_pi0": settle})


def test_checkpoint_provenance_in_manifest():
    m = load_carry_skill_routing().manifest()
    assert m["frozen_skill"] == "settling_pi0" and m["frozen_checkpoint"] == "pi0_shared_clip_actor"
    assert m["frozen_checkpoint_sha"] == "1902454c" and m["handoff_certificate"] == "stable_entry_v1"
    assert m["trainable_skills"] == ["carry_option"]
