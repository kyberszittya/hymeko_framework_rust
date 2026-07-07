import numpy as np
def load(path):
    e=[]
    for ln in open(path):
        if ln[:1]=='#': continue
        p=ln.replace(',',' ').split()
        if len(p)<3: continue
        e.append((int(p[0]),int(p[1]),1 if float(p[2])>0 else 0))
    return np.array(e,dtype=np.int64)
def auroc(y,s):
    o=np.argsort(s,kind='mergesort'); ss=s[o]; ranks=np.empty(len(y))
    i=0
    while i<len(ss):
        j=i
        while j<len(ss) and ss[j]==ss[i]: j+=1
        ranks[o[i:j]]=(i+1+j)/2.0; i=j
    yb=y.astype(bool); npos=int(yb.sum()); nneg=len(y)-npos
    return (ranks[yb].sum()-npos*(npos+1)/2)/(npos*nneg)
def run(path,seed):
    E=load(path); rng=np.random.default_rng(seed)
    idx=rng.permutation(len(E)); cut=int(0.8*len(E)); tr,te=E[idx[:cut]],E[idx[cut:]]
    N=int(E[:,:2].max())+1
    po=np.zeros(N);no=np.zeros(N);pi=np.zeros(N);ni=np.zeros(N)
    tu,tv,ts=tr[:,0],tr[:,1],tr[:,2]
    np.add.at(po,tu[ts==1],1); np.add.at(no,tu[ts==0],1)
    np.add.at(pi,tv[ts==1],1); np.add.at(ni,tv[ts==0],1)
    def feat(E):
        u,v=E[:,0],E[:,1]
        return np.log1p(np.stack([po[u],no[u],pi[u],ni[u],po[v],no[v],pi[v],ni[v],
                                  po[u]-no[u]+8,pi[v]-ni[v]+8],1).clip(0,None))
    Xtr,ytr=feat(tr),tr[:,2].astype(float); Xte,yte=feat(te),te[:,2].astype(float)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=np.c_[(Xtr-mu)/sd,np.ones(len(Xtr))]; Xte=np.c_[(Xte-mu)/sd,np.ones(len(Xte))]
    w=np.zeros(Xtr.shape[1])
    for _ in range(400):
        p=1/(1+np.exp(-np.clip(Xtr@w,-30,30))); w-=0.5*(Xtr.T@(p-ytr)/len(ytr)+1e-3*w)
    print(f"  {path.split('/')[-1]:30s} seed={seed} pos={ytr.mean():.3f} test_AUROC={auroc(yte,Xte@w):.4f}",flush=True)
for ds in ["signed/soc-sign-bitcoinalpha.csv","signed/soc-sign-Slashdot090221.txt","signed/soc-sign-epinions.txt"]:
    for seed in [0,1,2]: run("/tmp/hajdu/"+ds,seed)
