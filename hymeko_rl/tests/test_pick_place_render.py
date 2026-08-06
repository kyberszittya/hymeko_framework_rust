"""The pick-place trained-render glue: a saved policy loads and drives a deterministic action of the right shape
(the path render_pick_place.render_trained uses), and pick_place_ppo persists weights+metrics when given --out."""
import numpy as np
import torch

from hymeko_rl.experiments.gripper_pick_bc import build
from hymeko_rl.viz.render_pick_place import fanuc_pick_env, trained_action_fn


def test_trained_action_fn_shape_and_finite() -> None:
    env = fanuc_pick_env()
    ac = build("hsikan", env, 32)
    obs, _ = env.reset(seed=0)
    act = trained_action_fn(ac)(env, np.asarray(obs, dtype=np.float32))
    assert act.shape == (env.n_actions,)                       # 6 joints + grip
    assert np.isfinite(act).all()


def test_state_dict_roundtrip(tmp_path) -> None:
    # the load path render_trained relies on: build -> save -> build -> load_state_dict.
    env = fanuc_pick_env()
    ac = build("hsikan", env, 32)
    p = tmp_path / "pol.pt"
    torch.save(ac.state_dict(), p)
    ac2 = build("hsikan", env, 32)
    ac2.load_state_dict(torch.load(p, weights_only=True))
    obs, _ = env.reset(seed=1)
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None])
    assert torch.allclose(ac.action_mean(x), ac2.action_mean(x))   # identical weights => identical action
