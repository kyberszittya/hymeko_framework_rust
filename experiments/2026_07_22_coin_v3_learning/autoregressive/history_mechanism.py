"""§5 (action-history load-bearing?) + §6-A (cross-trajectory vs trajectory-ID) on the OPEN_LOOP_HISTORY dataset."""
import sys, numpy as np
sys.path.insert(0,".")
from hymeko_rl.coin_delivery.full_action_bc import load_trajectory_dataset
from hymeko_rl.coin_delivery.full_action_obs_history import build_history_features
from sklearn.neighbors import NearestNeighbors
PHN=["APPROACH","CONTACT_ACQ","BILATERAL","TRANSPORT","TARGET_ENTRY","SETTLING","STRICT_DWELL"]
BASE=sys.argv[1]; RNG=np.random.default_rng(0)
d=load_trajectory_dataset([BASE],patterns=("traj_*.npz",))
idx=[__import__("json").loads(x) for x in open(BASE+"/index.jsonl")]
seeds=[int(f.split("traj_")[1].split(".npz")[0]) for f in d["files"]]
ol_seeds={r["seed"] for r in idx if r.get("delivered") and r.get("teacher")=="search"}
ol_tids={i for i,s in enumerate(seeds) if s in ol_seeds}
m=np.isin(d["traj"], list(ol_tids))
obs,act,phase,traj=d["obs"][m],d["act"][m],d["phase"][m],d["traj"][m]
print(f"OPEN_LOOP_HISTORY: {len(np.unique(traj))} trajectories, {len(obs)} samples")
inst=obs                                     # 48
obs_k3=build_history_features(obs,act,traj)[:, :144]        # 3 obs, no actions
full=build_history_features(obs,act,traj)    # 152 with prev actions
def cond_disp(X, a, k=8, cross_traj=None, tr=None):
    if len(X)<k+2: return None
    if cross_traj is None:
        nn=NearestNeighbors(n_neighbors=k+1).fit(X); _,ix=nn.kneighbors(X)
        nb=a[ix[:,1:]]
    else:                                    # §6-A: neighbours must be DIFFERENT trajectory
        nn=NearestNeighbors(n_neighbors=min(len(X),40)).fit(X); _,ix=nn.kneighbors(X)
        nb=[]
        for i in range(len(X)):
            others=[j for j in ix[i] if tr[j]!=tr[i]][:k]
            if len(others)>=3: nb.append(a[others].std(0).mean())
        return float(np.mean(nb)) if nb else None
    return float(np.linalg.norm(nb-nb.mean(1,keepdims=True),axis=-1).mean())
print("\n§5 conditional action dispersion by phase (lower=more consistent): instant(48) / obs-k3(144) / FULL(152)")
for ph in (3,4,5,6):
    mm=phase==ph
    if mm.sum()<12: continue
    di=cond_disp(inst[mm],act[mm]); do=cond_disp(obs_k3[mm],act[mm]); df=cond_disp(full[mm],act[mm])
    print(f"  {PHN[ph]:<13} n={int(mm.sum()):<5} instant={di:.4f}  obs-k3={do:.4f}  FULL={df:.4f}")
print("\n§6-A cross-trajectory (neighbours from DIFFERENT trajectories) FULL-history action std by phase:")
for ph in (3,4,5,6):
    mm=phase==ph
    if mm.sum()<12: continue
    within=cond_disp(full[mm],act[mm])           # includes within-traj neighbours
    cross=cond_disp(full[mm],act[mm],cross_traj=True,tr=traj[mm])
    print(f"  {PHN[ph]:<13} within-any={within:.4f}  CROSS-TRAJ={cross if cross is None else round(cross,4)}")
