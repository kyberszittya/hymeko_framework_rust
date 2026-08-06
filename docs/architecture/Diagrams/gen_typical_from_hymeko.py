import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
DIR="/sessions/sleepy-funny-maxwell/mnt/hymeko_framework_rust/data/typical_graphs"
PUR="#6D28D9"; BLU="#2563EB"; GRN="#0E9D6E"; TEAL="#14B8A6"; AMB="#E0A21B"; CORAL="#D8552F"; INK="#201A40"; MUT="#6B6690"
PAL=[PUR,BLU,GRN,TEAL,AMB,CORAL,"#9333EA"]

def parse(fn):
    s=open(fn).read()
    # nodes: identifier {}  (not @, not the header which is Name{} on first line / context)
    nodes=[]; 
    for m in re.finditer(r'^\s*([A-Za-z_]\w*)\s*\{\s*\}', s, re.M):
        nm=m.group(1)
        if nm in ('context',): continue
        nodes.append(nm)
    # drop the header description name (first token before first { } that is the doc name) -- header is 'Name{}' line 1
    # edges
    edges=[]
    for m in re.finditer(r'@\w+\s*\{\s*\(([^)]*)\)\s*;\s*\}', s):
        mem=[x.strip().lstrip('~+-').strip() for x in m.group(1).split(',')]
        edges.append([x for x in mem if x])
    # keep only nodes that actually appear in edges (filters header tokens)
    used=set(v for e in edges for v in e)
    nodes=[n for n in nodes if n in used] or sorted(used)
    return nodes, edges

def convex_hull(pts):
    pts=sorted(set(map(tuple,pts)))
    if len(pts)<=2: return np.array(pts,float)
    def cr(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and cr(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    up=[]
    for p in reversed(pts):
        while len(up)>=2 and cr(up[-2],up[-1],p)<=0: up.pop()
        up.append(p)
    return np.array(lo[:-1]+up[:-1],float)
def cmr(P,n=20):
    P=np.array(P); m=len(P); out=[]
    for i in range(m):
        p0,p1,p2,p3=P[(i-1)%m],P[i],P[(i+1)%m],P[(i+2)%m]
        for t in np.linspace(0,1,n,endpoint=False):
            t2,t3=t*t,t*t*t
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(out)
def blobpoly(pts,pad):
    pts=np.array(pts,float)
    if len(pts)==1:
        c=pts[0]; th=np.linspace(0,2*np.pi,40); return np.c_[c[0]+pad*np.cos(th),c[1]+pad*np.sin(th)]
    if len(pts)==2:
        a,b=pts; d=b-a; L=np.hypot(*d)+1e-9; u=d/L; pr=np.array([-u[1],u[0]]); r=[]
        for an in np.linspace(-np.pi/2,np.pi/2,16): r.append(b+pad*(np.cos(an)*u+np.sin(an)*pr))
        for an in np.linspace(np.pi/2,3*np.pi/2,16): r.append(a+pad*(np.cos(an)*u+np.sin(an)*pr))
        return np.array(r)
    h=convex_hull(pts); ctr=h.mean(0)
    exp=np.array([v+pad*(v-ctr)/(np.hypot(*(v-ctr))+1e-9) for v in h]); return cmr(exp)

def render_circle(nodes,edges,title,out,pad=0.32):
    n=len(nodes); R=2.2
    pos={nm:(R*np.cos(np.pi/2 - 2*np.pi*i/n), R*np.sin(np.pi/2 - 2*np.pi*i/n)) for i,nm in enumerate(nodes)}
    fig,ax=plt.subplots(figsize=(3.6,3.4),dpi=200)
    warn=[]
    for k,e in enumerate(edges):
        poly=blobpoly([pos[m] for m in e],pad); col=PAL[k%len(PAL)]
        ax.fill(poly[:,0],poly[:,1],facecolor=col,alpha=0.15,edgecolor=col,lw=2.0,zorder=1)
        path=MPath(poly)
        for nm,p in pos.items():
            if nm not in e and path.contains_point(p): warn.append((title,k,nm))
    P=np.array([pos[nm] for nm in nodes])
    ax.scatter(P[:,0],P[:,1],s=150,color="white",edgecolor=INK,linewidth=2.0,zorder=5)
    ax.scatter(P[:,0],P[:,1],s=34,color=INK,zorder=6)
    for nm,(x,y) in pos.items(): ax.text(x*1.2,y*1.2,nm,fontsize=8.5,ha="center",va="center",color=MUT,zorder=7)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-3.2,3.2); ax.set_ylim(-3.2,3.2)
    ax.set_title(title,fontsize=11,color=INK,pad=6)
    fig.tight_layout(); fig.savefig(out,transparent=True,bbox_inches="tight"); plt.close(fig)
    return warn

# Fano: iconic embedding (special), still generated from the fixture's |V|,|E|
def render_fano(nodes,edges,out):
    fig,ax=plt.subplots(figsize=(3.7,3.5),dpi=200)
    V1=np.array([0,2.0]);V2=np.array([-1.85,-1.07]);V3=np.array([1.85,-1.07])
    M12=(V1+V2)/2;M13=(V1+V3)/2;M23=(V2+V3)/2;C=(V1+V2+V3)/3
    P=[V1,V2,V3,M12,M13,M23,C]
    L=[(V1,M12,V2),(V1,M13,V3),(V2,M23,V3),(V1,C,M23),(V2,C,M13),(V3,C,M12)]
    for k,(a,b,c) in enumerate(L):
        p=np.array([a,b,c]); ax.plot(p[:,0],p[:,1],color=PAL[k%7],lw=2.5,zorder=2,solid_capstyle="round",alpha=0.92)
    icc=(M12+M13+M23)/3; rr=np.mean([np.hypot(*(m-icc)) for m in (M12,M13,M23)])
    th=np.linspace(0,2*np.pi,200); ax.plot(icc[0]+rr*np.cos(th),icc[1]+rr*np.sin(th),color=PAL[6],lw=2.5,zorder=2,alpha=0.92)
    P=np.array(P); ax.scatter(P[:,0],P[:,1],s=150,color="white",edgecolor=INK,linewidth=2.2,zorder=6); ax.scatter(P[:,0],P[:,1],s=44,color=INK,zorder=7)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("Fano plane — 7 points, 7 lines",fontsize=11,color=INK,pad=6)
    fig.tight_layout(); fig.savefig(out,transparent=True,bbox_inches="tight"); plt.close(fig)

allw=[]
specs=[("fano_graph.hymeko","hg_fano.png","Fano plane",True),
       ("generic_hypergraph.hymeko","hg_generic.png","Generic hypergraph",False),
       ("k4_3uniform.hymeko","hg_kuniform.png","Complete 3-uniform K4(3)",False),
       ("sunflower_delta_system.hymeko","hg_sunflower.png","Sunflower (Δ-system)",False)]
for f,out,ttl,fano in specs:
    nodes,edges=parse(f"{DIR}/{f}")
    print(f"{f}: |V|={len(nodes)} |E|={len(edges)}  edges={edges}")
    if fano: render_fano(nodes,edges,out)
    else: allw+=render_circle(nodes,edges,ttl,out)
print("ENCLOSURE WARNINGS:", allw if allw else "none")
