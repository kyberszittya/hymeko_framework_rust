import numpy as np, torch
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
def feats(path,seed):
    E=load(path);rng=np.random.default_rng(seed);idx=rng.permutation(len(E))
    c=int(0.8*len(E));tr,te=E[idx[:c]],E[idx[c:]];N=int(E[:,:2].max())+1
    u,v,s=tr[:,0],tr[:,1],(tr[:,2]*2-1).astype(float)
    A=csr_matrix((np.r_[s,s],(np.r_[u,v],np.r_[v,u])),shape=(N,N))
    B=csr_matrix((np.ones(2*len(s)),(np.r_[u,v],np.r_[v,u])),shape=(N,N))
    po=np.asarray((A>0).sum(1)).ravel();ne=np.asarray((A<0).sum(1)).ravel()
    def ef(EE):
        a,b=EE[:,0],EE[:,1];h=np.empty(len(a));sp=np.empty(len(a))
        for i in range(len(a)):
            h[i]=A.getrow(a[i]).multiply(A.getrow(b[i])).sum();sp[i]=B.getrow(a[i]).multiply(B.getrow(b[i])).sum()
        return np.stack([np.log1p(po[a]),np.log1p(ne[a]),np.log1p(po[b]),np.log1p(ne[b]),
                         h,np.log1p(sp),h/np.maximum(sp,1),np.sign(h)],1)
    return ef(tr),tr[:,2].astype(float),ef(te),te[:,2].astype(float)
def norm(X,Xt):
    m,sd=X.mean(0),X.std(0)+1e-6;return (X-m)/sd,(Xt-m)/sd
def lin(X,y,Xt):
    X=np.c_[X,np.ones(len(X))];Xt=np.c_[Xt,np.ones(len(Xt))];w=np.zeros(X.shape[1])
    for _ in range(400):p=1/(1+np.exp(-np.clip(X@w,-30,30)));w-=0.5*(X.T@(p-y)/len(y)+1e-3*w)
    return Xt@w
def mlp(X,y,Xt):
    torch.manual_seed(0);Xt_=torch.tensor(Xt,dtype=torch.float32);X_=torch.tensor(X,dtype=torch.float32);y_=torch.tensor(y,dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
    opt=torch.optim.Adam(net.parameters(),1e-2,weight_decay=1e-4);lossf=torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad();l=lossf(net(X_).squeeze(1),y_);l.backward();opt.step()
    with torch.no_grad():return net(Xt_).squeeze(1).numpy()
for ds in ["signed/soc-sign-bitcoinalpha.csv","signed/soc-sign-Slashdot090221.txt","signed/soc-sign-epinions.txt"]:
    X,y,Xt,yt=feats("/tmp/hajdu/"+ds,0);Xn,Xtn=norm(X,Xt)
    al=auroc(yt,lin(Xn,y,Xtn));am=auroc(yt,mlp(Xn,y,Xtn))
    print(f"  {ds.split('/')[-1]:30s} linear={al:.4f}  MLP(64-64)={am:.4f}  nonlin_gain={am-al:+.4f}",flush=True)
