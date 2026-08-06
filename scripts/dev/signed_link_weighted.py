import numpy as np
from scipy.sparse import csr_matrix
def load(p):   # keep real rating in [-1,1]
    e=[]
    for ln in open(p):
        if ln[:1]=='#':continue
        q=ln.replace(',',' ').split()
        if len(q)<3:continue
        e.append((int(q[0]),int(q[1]),float(q[2])))
    return e
def auroc(y,s):
    o=np.argsort(s,kind='mergesort');ss=s[o];r=np.empty(len(y));i=0
    while i<len(ss):
        j=i
        while j<len(ss) and ss[j]==ss[i]:j+=1
        r[o[i:j]]=(i+1+j)/2;i=j
    yb=y.astype(bool);P=int(yb.sum());Nn=len(y)-P
    return (r[yb].sum()-P*(P+1)/2)/(P*Nn)
def fit(X,y,Xt):
    m,sd=X.mean(0),X.std(0)+1e-6;X=np.c_[(X-m)/sd,np.ones(len(X))];Xt=np.c_[(Xt-m)/sd,np.ones(len(Xt))]
    w=np.zeros(X.shape[1])
    for _ in range(500):p=1/(1+np.exp(-np.clip(X@w,-30,30)));w-=0.4*(X.T@(p-y)/len(y)+1e-3*w)
    return Xt@w
def run(path,name,scale,seed=0):
    E=load(path);rng=np.random.default_rng(seed);idx=rng.permutation(len(E))
    r=np.array([e[2] for e in E]); uu=np.array([e[0] for e in E]);vv=np.array([e[1] for e in E])
    y_all=(r>0).astype(np.float32)  # sign target
    w_all=np.clip(r/scale,-1,1).astype(np.float32)  # real weight in [-1,1]
    c=int(0.8*len(E));trm,tem=idx[:c],idx[c:];N=int(max(uu.max(),vv.max()))+1
    def build(vals):
        A=csr_matrix((np.r_[vals[trm],vals[trm]],(np.r_[uu[trm],vv[trm]],np.r_[vv[trm],uu[trm]])),shape=(N,N))
        A2=(A@A).tocsr()
        def feat(m):
            a,b=uu[m],vv[m];h=np.array([A2[a[i],b[i]] for i in range(len(a))],np.float32)
            dp=np.asarray(A.multiply(A>0).sum(1)).ravel();dn=np.asarray((-A).multiply(A<0).sum(1)).ravel()
            return np.c_[np.log1p(np.abs(dp[a])),np.log1p(np.abs(dn[a])),np.log1p(np.abs(dp[b])),np.log1p(np.abs(dn[b])),
                        np.sign(h)*np.log1p(np.abs(h))]
        return feat(trm),feat(tem)
    Xb_tr,Xb_te=build(np.sign(w_all))   # BINARY +/-1
    Xw_tr,Xw_te=build(w_all)            # WEIGHTED real
    yb=y_all[trm];yt=y_all[tem]
    ab=auroc(yt,fit(Xb_tr,yb,Xb_te)); aw=auroc(yt,fit(Xw_tr,yb,Xw_te))
    print(f"  {name:12s} binary(+/-1)={ab:.4f}  weighted([-1,1])={aw:.4f}  gain={aw-ab:+.4f}",flush=True)
run("/tmp/hajdu/signed/soc-sign-bitcoinalpha.csv","BTC-Alpha",10.0)
run("/tmp/hajdu/signed/soc-sign-bitcoinotc.csv","BTC-OTC",10.0)
