"""For a range of ASAP unit layers [l0,l1), Heisenberg images of X_q, Z_q through all units in that range.
Report per q the best single wire w and single-site weights; check whether the region is (permutation x local)."""
import sys, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-p5p6')
from uheis import ptm
from parse import ringof, pos, rings
U=np.load('/tmp/p6resyn/P6_titan_pinnacle.qasm.units.npy',allow_pickle=True)
units=[(int(u[0]),int(u[1]),np.asarray(u[2])) for u in U]
n=62; lay=[0]*n; L=[]
for a,b,M in units:
    l=max(lay[a],lay[b]); L.append(l); lay[a]=lay[b]=l+1
L=np.array(L)
def image(ks,q,p,eps=1e-4,maxterms=50000):
    cur={((q,p),):1.0}
    for k in reversed(ks):
        a,b,M=units[k]; Tk=ptm(M); new=collections.defaultdict(float)
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
def analyse(l0,l1,verbose=True):
    ks=[i for i in range(len(units)) if l0<=L[i]<l1]
    perm={}; worst=1
    for q in range(n):
        ws={}
        for p in (1,3):
            im=image(ks,q,p)
            w=collections.defaultdict(float)
            if im:
                for key,v in im.items():
                    if len(key)==1: w[key[0][0]]+=v*v
            ws[p]=w
        # wire with best combined
        cand=set(ws[1])|set(ws[3])
        best=max(cand,key=lambda x: ws[1].get(x,0)+ws[3].get(x,0)) if cand else None
        sc=(ws[1].get(best,0),ws[3].get(best,0)) if best is not None else (0,0)
        perm[q]=(best,sc); worst=min(worst,min(sc))
    return ks,perm,worst
if __name__=='__main__':
    l0,l1=int(sys.argv[1]),int(sys.argv[2])
    ks,perm,worst=analyse(l0,l1)
    print('layers',l0,l1,'units',len(ks),'worst',round(worst,4))
    nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
    moved=[(nm(q),nm(perm[q][0]) if perm[q][0] is not None else None,np.round(perm[q][1],3)) for q in range(n) if perm[q][0]!=q]
    print('non-fixed',len(moved)); print(moved)
    tgt=[perm[q][0] for q in range(n)]; print('bijective',len(set(tgt))==n)
    low=[(nm(q),np.round(perm[q][1],3)) for q in range(n) if min(perm[q][1])<0.98]; print('low',low)
