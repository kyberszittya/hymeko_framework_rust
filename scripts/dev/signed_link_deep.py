import numpy as np
from scipy.sparse import csr_matrix
def load(p):
    e=[]
    for ln in open(p):
        if ln[:1]=='#':continue
        q=ln.replace(',',' ').split()
        if len(q)<3:continue
        e.append((int(q[0]),int(q[1]),1 if float(q[2])>0 else 0))
    return np.array(e,dtype=np.int64)
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
    for _ in range(500):
        p=1/(1+np.exp(-np.clip(X@w,-30,30)));w-=0.4*(X.T@(p-y)/len(y)+1e-3*w)
    return Xt@w
def run(path,name,seed=0,trsub=60000):
    E=load(path);rng=np.random.default_rng(seed);idx=rng.permutation(len(E))
    c=int(0.8*len(E));tr,te=E[idx[:c]],E[idx[c:]];N=int(E[:,:2].max())+1
    u,v,s=tr[:,0],tr[:,1],(tr[:,2]*2-1).astype(np.float32)
    A=csr_matrix((np.r_[s,s],(np.r_[u,v],np.r_[v,u])),shape=(N,N))
    B=csr_matrix((np.ones(2*len(s),np.float32),(np.r_[u,v],np.r_[v,u])),shape=(N,N))
    A2=(A@A).tocsr(); B2=(B@B).tocsr()
    po=np.asarray((A>0).sum(1)).ravel();ne=np.asarray((A<0).sum(1)).ravel()
    def feats(EE):
        a,b=EE[:,0],EE[:,1];n=len(a)
        h2=np.empty(n,np.float32);h3=np.empty(n,np.float32);h4=np.empty(n,np.float32);sp=np.empty(n,np.float32)
        for i in range(n):
            ra=A.getrow(a[i]);r2=A2.getrow(a[i])
            h2[i]=r2.multiply(csr_matrix(([1],([0],[b[i]])),shape=(1,ra.shape[1]))).sum() if False else A2[a[i],b[i]]
            h3[i]=r2.multiply(A.getrow(b[i])).sum()      # A^3[a,b] = A2row_a . Arow_b
            h4[i]=r2.multiply(A2.getrow(b[i])).sum()     # A^4[a,b] = A2row_a . A2row_b
            sp[i]=B2[a[i],b[i]]
        deg=np.stack([np.log1p(po[a]),np.log1p(ne[a]),np.log1p(po[b]),np.log1p(ne[b])],1)
        return deg,h2,h3,h4,np.log1p(np.abs(sp))*np.sign(sp)
    trs=tr[rng.permutation(len(tr))[:trsub]] if len(tr)>trsub else tr
    dtr,h2tr,h3tr,h4tr,sptr=feats(trs); ytr=trs[:,2].astype(np.float32)
    dte,h2te,h3te,h4te,spte=feats(te);  yte=te[:,2].astype(np.float32)
    def norm(x): return x/(np.abs(x).max()+1e-6)
    Xtri_tr=np.c_[dtr,norm(h2tr)]; Xtri_te=np.c_[dte,norm(h2te)]
    Xdeep_tr=np.c_[dtr,norm(h2tr),norm(h3tr),norm(h4tr),sptr]; Xdeep_te=np.c_[dte,norm(h2te),norm(h3te),norm(h4te),spte]
    a_tri=auroc(yte,fit(Xtri_tr,ytr,Xtri_te)); a_deep=auroc(yte,fit(Xdeep_tr,ytr,Xdeep_te))
    print(f"  {name:12s} triad(A2)={a_tri:.4f}  deep(A2+A3+A4)={a_deep:.4f}  lift={a_deep-a_tri:+.4f}",flush=True)
for p,n in [("signed/soc-sign-bitcoinalpha.csv","BTC-Alpha"),("signed/soc-sign-Slashdot090221.txt","Slashdot"),("signed/soc-sign-epinions.txt","Epinions")]:
    run("/tmp/hajdu/"+p,n)
