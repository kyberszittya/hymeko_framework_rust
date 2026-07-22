"""§6 prefix-replay drift diagnostic: hand off to the best C (obs+action-history) policy at progressively later phases;
if C delivers from LATE handoffs (near-distribution) but not from neutral/early, covariate shift is localized."""
import sys; sys.path.insert(0,".")
import numpy as np, torch; torch.set_num_threads(6)
from hymeko_rl.coin_delivery.full_action_bc import (FullActionBC, load_trajectory_dataset, train_bc_phase_balanced, eval_action_history_bc_delivery)
from hymeko_rl.coin_delivery.full_action_obs_history import build_history_features, ObsHistoryV1, K_OBS, K_ACT, BASE_OBS_DIM, ACTION_DIM
from hymeko_rl.coin_delivery.coin_v3_receding_horizon import capture_states, _restore
from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
import json
BASE=sys.argv[1]
# retrain best C (seed 2)
d=load_trajectory_dataset([BASE],patterns=("traj_*.npz",))
idx=[json.loads(x) for x in open(BASE+"/index.jsonl")]
fs=[int(f.split("traj_")[1].split(".npz")[0]) for f in d["files"]]
ol={r["seed"] for r in idx if r.get("delivered") and r.get("teacher")=="search"}
tids={i for i,s in enumerate(fs) if s in ol}
m=np.isin(d["traj"], list(tids)); obs,act,phase,traj=d["obs"][m],d["act"][m],d["phase"][m],d["traj"][m]
rng=np.random.default_rng(0); ids=np.unique(traj); val=set(rng.choice(ids,size=max(1,int(len(ids)*0.2)),replace=False).tolist())
vm=np.isin(traj,list(val)); tr=~vm
inp=build_history_features(obs,act,traj)
bc,_=train_bc_phase_balanced(inp[tr],act[tr],phase[tr],epochs=300,seed=2,steps_per_epoch=200,val=(inp[vm],act[vm],phase[vm]))
print("retrained C seed2; free-run headline:", eval_action_history_bc_delivery(bc,(1011,1045,1164,1174,1278,1358,1447,1568,1202))["deliver"],"/9", flush=True)
# prefix-replay: capture E-approach+handoff states per phase, seed C's history from them, roll C to horizon
def seed_hist(oh):
    h=ObsHistoryV1(); h._obs.clear(); h._act.clear()
    for j in range(K_OBS): h._obs.append(oh[j*BASE_OBS_DIM:(j+1)*BASE_OBS_DIM].copy())  # newest-first already
    base=K_OBS*BASE_OBS_DIM
    for j in range(K_ACT): h._act.append(oh[base+j*ACTION_DIM:base+(j+1)*ACTION_DIM].copy())
    return h
states=capture_states([1011,1045,1164,1174,1278,1447], per_phase=1)
env,cf=neutral_env(prefix_steps=0); inner=cf._env
from collections import defaultdict
byphase=defaultdict(lambda:[0,0])
for st in states:
    _restore(inner, st["qpos"], st["qvel"])
    cert=DeliveryCertifier(initial_clearance=st["clearance"]); cert.robot_touched=st["touched"]; cert.delivery_dwell=st["dwell"]
    h=seed_hist(st["obs_hist"]); deliv=False
    for _t in range(240):
        cert.update(_cert_step(inner,cf))
        if cert.delivery_certified: deliv=True; break
        a=np.asarray(bc.act(h.feature()),np.float32); inner.step(a)
        h.push(np.asarray(inner.node_features(),np.float32).flatten(), a)
    byphase[st["phase_name"]][0]+=int(deliv); byphase[st["phase_name"]][1]+=1
print("\n§6 prefix-replay: C policy delivery when handed off AT each phase (near-distribution takeover):")
for ph in ("TRANSPORT","TARGET_ENTRY","SETTLING","STRICT_DWELL"):
    if ph in byphase: print(f"  handoff at {ph:<14}: delivered {byphase[ph][0]}/{byphase[ph][1]}")
