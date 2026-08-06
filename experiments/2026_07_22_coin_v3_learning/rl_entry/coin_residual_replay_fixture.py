"""§5.6 deterministic transition fixture + §5.7 regression. Builds one transition per gate-transition type, reports
stored gate_t/gate_tp1, deployed action, reference vs learner target action, maxdiff, and whether any gate FSM
function was invoked during target construction (instrumented tripwire). Machine-readable JSON.
"""
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import hymeko_rl.coin_delivery.coin_stable_engagement as cse  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_controller import ZeroInitResidualActor  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_replay import (  # noqa: E402
    ReplayControllerStateV2,
    ResidualReplayBuffer,
    ResidualTransition,
    bounded_smoothed_residual,
    controller_state_schema_hash,
    residual_target_action,
    td_target_scalar,
)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/replay_fixture.json"
GAMMA = 0.99


def fsm_tripwire():
    """Return a context that flips a flag if any StableEngagementGate method is called."""
    state = {"invoked": False}
    orig = (cse.StableEngagementGate.__init__, cse.StableEngagementGate.update, cse.StableEngagementGate.reset)

    def mk(f):
        def wrapped(*a, **k):
            state["invoked"] = True
            return f(*a, **k)
        return wrapped
    cse.StableEngagementGate.__init__ = mk(orig[0])
    cse.StableEngagementGate.update = mk(orig[1])
    cse.StableEngagementGate.reset = mk(orig[2])

    def restore():
        cse.StableEngagementGate.__init__, cse.StableEngagementGate.update, cse.StableEngagementGate.reset = orig
    return state, restore


def main():
    torch.manual_seed(0)
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    residual = ZeroInitResidualActor()
    with torch.no_grad():                                    # small nonzero residual so gate matters
        residual.net[4].weight.copy_(torch.randn_like(residual.net[4].weight) * 0.3)
        residual.net[4].bias.copy_(torch.randn_like(residual.net[4].bias) * 0.3)
    rng = np.random.default_rng(0)

    # one transition per gate-transition type: (name, mode_t, gate_t, mode_tp1, gate_tp1, done)
    types = [
        ("EARLY_CONTROL->EARLY_CONTROL", "EARLY_CONTROL", 0.0, "EARLY_CONTROL", 0.0, 0.0),
        ("EARLY_CONTROL->RESIDUAL_CONTROL", "EARLY_CONTROL", 0.0, "LATE_CONTROL_ARMED", 1.0, 0.0),
        ("RESIDUAL_CONTROL->RESIDUAL_CONTROL", "LATE_CONTROL_ARMED", 1.0, "LATE_CONTROL_ARMED", 1.0, 0.0),
        ("RESIDUAL_CONTROL->EARLY_CONTROL(REACQUIRE)", "LATE_CONTROL_ARMED", 1.0, "REACQUIRE", 0.0, 0.0),
        ("terminal", "LATE_CONTROL_ARMED", 1.0, "TERMINAL", 0.0, 1.0),
        ("truncated_reset", "LATE_CONTROL_ARMED", 1.0, "EARLY_CONTROL", 0.0, 0.0),
    ]
    buf = ResidualReplayBuffer()
    rows = []
    for name, mode_t, g_t, mode_tp1, g_tp1, done in types:
        obs_t = rng.standard_normal(48).astype(np.float32)
        obs_tp1 = rng.standard_normal(48).astype(np.float32)
        noise = torch.tensor(rng.standard_normal((1, 4)).astype(np.float32))
        cs_t = ReplayControllerStateV2(gate=g_t, mode=mode_t)
        cs_tp1 = ReplayControllerStateV2(gate=g_tp1, mode=mode_tp1)
        # deployed action at t (composite with gate_t)
        with torch.no_grad():
            base_t = torch.clamp(pi0.action_mean(torch.tensor(obs_t[None])), -4, 4)
            res_t = 0.25 * torch.tanh(residual.raw(torch.tensor(obs_t[None])))
            deployed = torch.clamp(base_t + g_t * res_t, -4, 4)[0].numpy()
        buf.add(ResidualTransition(obs_t, deployed, float(rng.standard_normal()), obs_tp1, done, cs_t, cs_tp1))
        # learner target action from STORED gate_tp1, with FSM tripwire
        sampled = buf.sample([len(buf) - 1])
        state, restore = fsm_tripwire()
        try:
            learner_tgt = residual_target_action(pi0, residual, sampled["obs2"], sampled["gate_tp1"], noise=noise)
        finally:
            restore()
        # reference target action (independent implementation)
        with torch.no_grad():
            base_tp1 = torch.clamp(pi0.action_mean(torch.tensor(obs_tp1[None])), -4, 4)
        ref_res = bounded_smoothed_residual(residual, torch.tensor(obs_tp1[None]), noise=noise)
        ref_tgt = torch.clamp(base_tp1 + g_tp1 * ref_res, -4, 4)
        maxdiff = float((learner_tgt - ref_tgt).abs().max())
        # terminal: TD target must be reward regardless of gate_tp1/residual/noise
        term_invariant = None
        if done == 1.0:
            r = sampled["reward"]
            y = td_target_scalar(r, sampled["done"], GAMMA, q_next=torch.tensor([123.0]))
            y2 = td_target_scalar(r, sampled["done"], GAMMA, q_next=torch.tensor([-999.0]))
            term_invariant = bool(torch.equal(y, r) and torch.equal(y2, r))
        rows.append({"type": name, "gate_t": g_t, "gate_tp1": g_tp1, "done": done,
                     "deployed_action": [round(x, 6) for x in deployed.tolist()],
                     "reference_target": [round(x, 6) for x in ref_tgt[0].tolist()],
                     "learner_target": [round(x, 6) for x in learner_tgt[0].tolist()],
                     "max_diff": maxdiff, "gate0_equals_base": (g_tp1 == 0.0 and
                        bool(torch.equal(learner_tgt, base_tp1))),
                     "fsm_invoked": state["invoked"], "terminal_bootstrap_masked": term_invariant})
        print(f"  {name:<40} g_t={g_t} g_tp1={g_tp1} done={done} maxdiff={maxdiff:.2e} "
              f"fsm_invoked={state['invoked']} gate0=base={rows[-1]['gate0_equals_base']}", flush=True)

    ok = (all(r["max_diff"] < 1e-6 for r in rows) and not any(r["fsm_invoked"] for r in rows)
          and all(r["gate0_equals_base"] for r in rows if r["gate_tp1"] == 0.0)
          and all(r["terminal_bootstrap_masked"] for r in rows if r["done"] == 1.0))
    out = {"schema_hash": controller_state_schema_hash(), "schema_hash_prefix": controller_state_schema_hash()[:12],
           "pi0_file_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8],
           "transitions": rows, "all_targets_match_reference": all(r["max_diff"] < 1e-6 for r in rows),
           "no_fsm_invoked": not any(r["fsm_invoked"] for r in rows), "fixture_pass": ok}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nschema hash {controller_state_schema_hash()[:12]} | all match {out['all_targets_match_reference']} "
          f"| no FSM {out['no_fsm_invoked']} | fixture_pass {ok}", flush=True)
    print("FIXTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
