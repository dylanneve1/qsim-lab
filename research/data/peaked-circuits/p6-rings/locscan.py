"""Localisation scan: for windows of 2k unit-layers centred at c, Heisenberg image of Z_q / X_q (q all wires);
score = mean over (q,P) of max_w (single-site weight on w). Permutation-like windows -> 1; scrambling windows -> small.
usage: locscan.py UNITS.npy K [eps]"""
import sys, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-p5p6')
from uheis import ptm
U=np.load(sys.argv[1],allow_pickle=True); K=int(sys.argv[2]); eps=float(sys.argv[3]) if len(sys.argv)>3 else 1e-3
U=[u for u in U if u[0]=='2'] if isinstance(U[0][0],str) else U
units=[(int(u[1][0]),int(u[1][1]),np.asarray(u[2])) if isinstance(u[0],str) else (int(u[0]),int(u[1]),np.asarray(u[2])) for u in U]
n=62; lay=[0]*n; L=[]
for a,b,M in units:
    l=max(lay[a],lay[b]); L.append(l); lay[a]=lay[b]=l+1
D=max(lay); L=np.array(L)
import os
if os.environ.get('NULL'):
    from scipy.stats import unitary_group
    rng=np.random.default_rng(int(os.environ['NULL']))
    units=[(a,b,unitary_group.rvs(4,random_state=rng)) for a,b,M in units]
T=[ptm(M) for a,b,M in units]
def image(ks,q,p,maxterms=20000):
    cur={((q,p),):1.0}
    for k in reversed(ks):
        a,b=units[k][0],units[k][1]; Tk=T[k]; new=collections.defaultdict(float)
        for key,c in cur.items():
            d=dict(key); pa=d.pop(a,0); pb=d.pop(b,0)
            if pa==0 and pb==0: new[key]+=c; continue
            row=Tk[pa*4+pb]
            for j in np.nonzero(np.abs(row)>1e-12)[0]:
                ja,jb=divmod(j,4); dd=dict(d)
                if ja: dd[a]=ja
                if jb: dd[b]=jb
                new[tuple(sorted(dd.items()))]+=c*row[j]
        cur={k_:v for k_,v in new.items() if abs(v)>eps}
        if len(cur)>maxterms: return None
    return cur
print('units',len(units),'depth',D,flush=True)
for c in range(K, D-K+1):
    ks=[i for i in range(len(units)) if c-K<=L[i]<c+K]
    sc=[]
    for q in range(n):
        for p in (1,3):
            im=image(ks,q,p)
            if im is None: sc.append(0.0); continue
            w=collections.defaultdict(float)
            for key,v in im.items():
                if len(key)==1: w[key[0][0]]+=v*v
            sc.append(max(w.values()) if w else 0.0)
    sc=np.array(sc)
    print(c, len(ks), round(sc.mean(),3), round(np.median(sc),3), int((sc>0.9).sum()), flush=True)
