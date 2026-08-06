import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle

PUR="#6D28D9"; BLU="#2563EB"; GRN="#0E9D6E"; TEAL="#14B8A6"; AMB="#E0A21B"; CORAL="#D8552F"; INK="#201A40"; MUT="#6B6690"
PAL=[PUR,BLU,GRN,TEAL,AMB,CORAL,"#9333EA"]

def convex_hull(pts):
    pts=sorted(map(tuple,set(map(tuple,pts))))
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

def catmull_closed(P,n=20):
    P=np.array(P); m=len(P); out=[]
    for i in range(m):
        p0,p1,p2,p3=P[(i-1)%m],P[i],P[(i+1)%m],P[(i+2)%m]
        for t in np.linspace(0,1,n,endpoint=False):
            t2,t3=t*t,t*t*t
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(out)

def blob_polygon(pts,pad):
    pts=np.array(pts,float)
    if len(pts)==1:
        c=pts[0]; th=np.linspace(0,2*np.pi,40); return np.c_[c[0]+pad*np.cos(th),c[1]+pad*np.sin(th)]
    if len(pts)==2:
        a,b=pts; d=b-a; L=np.hypot(*d)+1e-9; u=d/L; perp=np.array([-u[1],u[0]]); ring=[]
        for ang in np.linspace(-np.pi/2,np.pi/2,18): ring.append(b+pad*(np.cos(ang)*u+np.sin(ang)*perp))
        for ang in np.linspace(np.pi/2,3*np.pi/2,18): ring.append(a+pad*(np.cos(ang)*u+np.sin(ang)*perp))
        return np.array(ring)
    h=convex_hull(pts); ctr=h.mean(0)
    exp=np.array([v+pad*(v-ctr)/(np.hypot(*(v-ctr))+1e-9) for v in h])
    s=catmull_closed(exp); return s

def draw(ax,Pdict,edges,pad=0.4,alpha=0.16,lw=2.2,check=True):
    warns=[]
    for k,(mem,col) in enumerate(edges):
        poly=blob_polygon([Pdict[m] for m in mem],pad)
        ax.fill(poly[:,0],poly[:,1],facecolor=col,alpha=alpha,edgecolor=col,lw=lw,zorder=1)
        if check:
            path=Path(poly)
            for name,p in Pdict.items():
                if name not in mem and path.contains_point(p):
                    warns.append(f"edge{k}{tuple(mem)} encloses non-member {name}")
    P=np.array(list(Pdict.values()))
    ax.scatter(P[:,0],P[:,1],s=130,color="white",edgecolor=INK,linewidth=2.0,zorder=5)
    ax.scatter(P[:,0],P[:,1],s=34,color=INK,zorder=6)
    return warns

def finish(ax,title):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title,fontsize=11,color=INK,pad=8)

allwarn=[]

# 1) FANO (iconic geometry — a valid Fano plane, isomorphic to the fixture)
fig,ax=plt.subplots(figsize=(4.3,4.2),dpi=200)
V1=np.array([0,2.0]); V2=np.array([-1.85,-1.07]); V3=np.array([1.85,-1.07])
M12=(V1+V2)/2; M13=(V1+V3)/2; M23=(V2+V3)/2; C=(V1+V2+V3)/3
P=[V1,V2,V3,M12,M13,M23,C]
lines=[(V1,M12,V2),(V1,M13,V3),(V2,M23,V3),(V1,C,M23),(V2,C,M13),(V3,C,M12)]
for k,(a,b,c) in enumerate(lines):
    p=np.array([a,b,c]); ax.plot(p[:,0],p[:,1],color=PAL[k%7],lw=2.6,zorder=2,solid_capstyle="round",alpha=0.92)
icc=(M12+M13+M23)/3; rr=np.mean([np.hypot(*(m-icc)) for m in (M12,M13,M23)])
th=np.linspace(0,2*np.pi,220); ax.plot(icc[0]+rr*np.cos(th),icc[1]+rr*np.sin(th),color=PAL[6],lw=2.6,zorder=2,alpha=0.92)
P=np.array(P); ax.scatter(P[:,0],P[:,1],s=150,color="white",edgecolor=INK,linewidth=2.2,zorder=6); ax.scatter(P[:,0],P[:,1],s=44,color=INK,zorder=7)
finish(ax,"Fano plane — 7 points, 7 lines (3-uniform)"); fig.tight_layout(); fig.savefig("hg_fano.png",transparent=True,bbox_inches="tight"); plt.close(fig)

# 2) GENERIC (convex-safe layout; edges from generic_hypergraph.hymeko)
fig,ax=plt.subplots(figsize=(4.7,3.5),dpi=200)
G={"a":(-2.0,1.4),"b":(-0.3,1.95),"c":(0.0,0.0),"d":(1.9,0.45),"e":(3.3,1.05),"f":(0.35,-1.65),"g":(-2.05,-1.35)}
GE=[(["a","b","c"],PUR),(["c","d","f"],BLU),(["d","e"],GRN),(["c","g","f"],AMB)]
allwarn+=["generic: "+w for w in draw(ax,G,GE,pad=0.38)]
for name,(x,y) in G.items(): ax.text(x,y-0.34,name,fontsize=9,ha="center",va="top",color=MUT,zorder=7)
finish(ax,"A hypergraph — 4 hyperedges over 7 vertices"); ax.set_xlim(-2.9,4.1); ax.set_ylim(-2.4,2.7)
fig.tight_layout(); fig.savefig("hg_generic.png",transparent=True,bbox_inches="tight"); plt.close(fig)

# 3) K4^(3): 4 vertices in convex position -> each triangle excludes the 4th
fig,ax=plt.subplots(figsize=(4.1,3.8),dpi=200)
ang=[90,200,340,20]
K={f"v{i+1}":(1.7*np.cos(np.deg2rad(a)),1.7*np.sin(np.deg2rad(a))) for i,a in enumerate(ang)}
KE=[(["v1","v2","v3"],BLU),(["v1","v2","v4"],PUR),(["v1","v3","v4"],GRN),(["v2","v3","v4"],AMB)]
allwarn+=["k4: "+w for w in draw(ax,K,KE,pad=0.3,alpha=0.13)]
for name,(x,y) in K.items(): ax.text(x*1.28,y*1.28,name,fontsize=9,ha="center",va="center",color=MUT,zorder=7)
finish(ax,"Complete 3-uniform hypergraph  K₄⁽³⁾"); ax.set_xlim(-2.6,2.6); ax.set_ylim(-2.4,2.6)
fig.tight_layout(); fig.savefig("hg_kuniform.png",transparent=True,bbox_inches="tight"); plt.close(fig)

# 4) SUNFLOWER (radiating petals; core shared)
fig,ax=plt.subplots(figsize=(4.4,3.9),dpi=200)
S={"core_a":(-0.42,0.0),"core_b":(0.42,0.0)}
SE=[]
for k,a in enumerate([90,210,330]):
    r=np.deg2rad(a); p1=(2.05*np.cos(r),2.05*np.sin(r)); p2=(3.0*np.cos(r+0.16),3.0*np.sin(r+0.16))
    S[f"p{k+1}a"]=p1; S[f"p{k+1}b"]=p2; SE.append((["core_a","core_b",f"p{k+1}a",f"p{k+1}b"],PAL[k]))
allwarn+=["sunflower: "+w for w in draw(ax,S,SE,pad=0.34,alpha=0.13)]
cn=np.array([S["core_a"],S["core_b"]]); ax.scatter(cn[:,0],cn[:,1],s=150,color=CORAL,edgecolor="white",linewidth=1.6,zorder=8)
ax.text(0,-0.5,"core",fontsize=9,ha="center",color=CORAL,style="italic",zorder=8)
finish(ax,"Sunflower (Δ-system) — hyperedges share a core"); ax.set_xlim(-3.5,3.5); ax.set_ylim(-3.5,3.5)
fig.tight_layout(); fig.savefig("hg_sunflower.png",transparent=True,bbox_inches="tight"); plt.close(fig)

print("WARNINGS:", allwarn if allwarn else "none (no hyperedge encloses a non-member)")
