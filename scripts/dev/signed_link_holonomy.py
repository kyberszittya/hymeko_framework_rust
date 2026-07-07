import numpy as np
from scipy.sparse import csr_matrix
def load(path):
    e=[]
    for ln in open(path):
        if ln[:1]=='#': continue
        p=ln.replace(',',' ').split()
        if len(p)<3: continue
        e.append((int(p[0]),int(p[1]),1 if float(p[2])>0 else 0))
    return np.array(e,dtype=np.int64)
def auroc(y,s):
    o=np.argsort(s,kind='mergesort'); ss=s[o]; r=np.empty(len(y)); i=0
    while i<len(ss):
        j=i
        while j<len(ss) and ss[j]==ss[i]: j+=1
        r[o[i:j]]=(i+1+j)/2; i=j
    yb=y.astype(bool); P=int(yb.sum()); Nn=len(y)-P
    return (r[yb].sum()-P*(P+1)/2)/(P*Nn)
def fit(X,y,Xt):
    mu,sd=X.mean(0),X.std(0)+1e-6; X=np.c_[(X-mu)/sd,np.ones(len(X))]; Xt=np.c_[(Xt-mu)/sd,np.ones(len(Xt))]
    w=np.zeros(X.shape[1])
    for _ in range(400):
        p=1/(1+np.exp(-np.clip(X@w,-30,30))); w-=0.5*(X.T@(p-y)/len(y)+1e-3*w)
    return Xt@w
def run(path,seed):
    E=load(path); rng=np.random.default_rng(seed); idx=rng.permutation(len(E))
    cut=int(0.8*len(E)); tr,te=E[idx[:cut]],E[idx[cut:]]; N=int(E[:,:2].max())+1
    u,v,s=tr[:,0],tr[:,1],(tr[:,2]*2-1).astype(float)
    A=csr_matrix((np.r_[s,s],(np.r_[u,v],np.r_[v,u])),shape=(N,N))   # signed adj (train)
    B=csr_matrix((np.ones(2*len(s)),(np.r_[u,v],np.r_[v,u])),shape=(N,N))  # binary adj
    po=np.asarray((A>0).sum(1)).ravel(); ne=np.asarray((A<0).sum(1)).ravel()
    tu,tv=te[:,0],te[:,1]
    # signed-holonomy (length-2 balance) features per test edge, leakage-free (A is train-only)
    holo=np.empty(len(tu)); supp=np.empty(len(tu))
    for i in range(len(tu)):
        au=A.getrow(tu[i]); av=A.getrow(tv[i])
        holo[i]=au.multiply(av).sum()          # sum_w sign(u,w)sign(w,v) = triad holonomy vote
        supp[i]=B.getrow(tu[i]).multiply(B.getrow(tv[i])).sum()  # #common neighbors
    deg=np.stack([np.log1p(po[tu]),np.log1p(ne[tu]),np.log1p(po[tv]),np.log1p(ne[tv])],1)
    hol=np.stack([holo,np.log1p(supp),holo/np.maximum(supp,1),np.sign(holo)],1)
    y=te[:,2].astype(float)
    # train-side features (same construction, using A for its own endpoints is fine — no self-sign leak)
    trdeg=np.stack([np.log1p(po[u]),np.log1p(ne[u]),np.log1p(po[v]),np.log1p(ne[v])],1)
    trh=np.empty(len(u)); trs=np.empty(len(u))
    for i in range(len(u)):
        trh[i]=A.getrow(u[i]).multiply(A.getrow(v[i])).sum(); trs[i]=B.getrow(u[i]).multiply(B.getrow(v[i])).sum()
    trhol=np.stack([trh,np.log1p(trs),trh/np.maximum(trs,1),np.sign(trh)],1); ytr=tr[:,2].astype(float)
    a_deg=auroc(y,fit(trdeg,ytr,deg))
    a_all=auroc(y,fit(np.c_[trdeg,trhol],ytr,np.c_[deg,hol]))
    print(f"  {path.split('/')[-1]:30s} seed={seed}  degree_only={a_deg:.4f}  +signed_holonomy={a_all:.4f}  lift={a_all-a_deg:+.4f}",flush=True)
for ds in ["signed/soc-sign-bitcoinalpha.csv","signed/soc-sign-Slashdot090221.txt","signed/soc-sign-epinions.txt"]:
    run("/tmp/hajdu/"+ds,0)
