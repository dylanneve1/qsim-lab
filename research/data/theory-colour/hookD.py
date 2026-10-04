# D(H) = min number of single data errors that, together with hook H, form a nontrivial X-logical.
import sys, itertools, json, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from minlog import setup
def coset_min(H,row0,S):
    m,n=H.shape
    s=H[:,S].sum(1)%2; par=(len(set(S)&set(row0))+1)%2
    nv=n+m+1; c=np.zeros(nv); c[:n]=1
    A=np.zeros((m+1,nv)); A[:m,:n]=H; A[:m,n:n+m]=-2*np.eye(m); A[m,row0]=1; A[m,n+m]=-2
    b=np.r_[s,par]
    r=milp(c,constraints=LinearConstraint(A,b,b),integrality=np.ones(nv),bounds=Bounds(np.zeros(nv),np.r_[np.ones(n),np.full(m,3),n]))
    return int(round(r.fun))
def run(d):
    data,P,H,row0=setup(d); deg=H.sum(0)
    out=[]
    for i,p in enumerate(P):
        qs=[q for q in p['q'] if q is not None]; w=len(qs)
        bnd=any(deg[q]<3 for q in qs)
        Dm={}
        for k in range(2,w//2+1):
            for S in itertools.combinations(qs,k):
                if k*2==w and qs[0] not in S: continue
                Dm[S]=coset_min(H,row0,list(S))
        out.append(dict(i=i,c=int(p['c']),x=p['x'],y=p['y'],w=w,bnd=bool(bnd),D={",".join(map(str,k)):v for k,v in Dm.items()}))
    return out
if __name__=="__main__":
    for d in map(int,sys.argv[1:]):
        res=run(d); json.dump(res,open(f"hookD_d{d}.json","w"))
        for r in res:
            bad=[k for k,v in r['D'].items() if v<=d-2]
            print(d,r['i'],"RGB"[r['c']],(r['x'],r['y']),"w",r['w'],"bnd" if r['bnd'] else "int","malign",len(bad),"/",len(r['D']),"minD",min(r['D'].values()))
