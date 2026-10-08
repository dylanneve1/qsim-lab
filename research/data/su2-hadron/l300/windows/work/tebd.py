import os
import numpy as np, scipy.linalg as sla, sys, json, time, argparse
from ladder import build
ap=argparse.ArgumentParser(); ap.add_argument('--circ',default='SCV'); ap.add_argument('--chi',type=int,default=64)
ap.add_argument('--cut',type=float,default=1e-12); ap.add_argument('--dtype',default='complex128'); ap.add_argument('--U0',action='store_true')
a=ap.parse_args()
D=os.environ.get('SU2_CIRCUITS','circuits')+'/'
state,gl=build(D+f'x_100_{a.circ}.qasm')
dt=np.dtype(a.dtype)
N=60
A=[]
for s in range(N):
    t=np.zeros((1,4,1),dtype=dt); t[0,2*state[s][0]+state[s][1],0]=1; A.append(t)
c=0  # orthogonality center
def move(to):
    global c
    while c<to:
        Dl,d,Dr=A[c].shape; Q,R=np.linalg.qr(A[c].reshape(Dl*d,Dr)); A[c]=Q.reshape(Dl,d,-1)
        A[c+1]=np.tensordot(R,A[c+1],1); c+=1
    while c>to:
        Dl,d,Dr=A[c].shape; Q,R=np.linalg.qr(A[c].reshape(Dl,d*Dr).T); A[c]=Q.T.reshape(-1,d,Dr)
        A[c-1]=np.tensordot(A[c-1],R.T,1); c-=1
Zi=np.array([1,1,-1,-1.]); Zo=np.array([1,-1,1,-1.])
def measure():
    # returns <Z_i(r)>, <Z_o(r)> using left/right environments (state is normalised, center c)
    move(0); out=[]
    for s in range(N):
        move(s); p=np.einsum('ajb,ajb->j',A[s],A[s].conj()).real; nrm=p.sum(); p=p/nrm
        out.append((float(p@Zi),float(p@Zo)))
    return out,float(nrm)
logF=0.0; maxchi=1; disc=0.0; rec=[]; t0=time.time()
for item in gl:
    if item[0]=='1':
        _,s,U=item; A[s]=np.einsum('jk,akb->ajb',U.astype(dt),A[s])
    elif item[0]=='2':
        _,r,U=item
        if c<r: move(r)
        elif c>r+1: move(r+1)
        th=np.einsum('ajb,bkc->ajkc',A[r],A[r+1]); Dl,_,_,Dr=th.shape
        th=np.einsum('xy,ayc->axc',U.astype(dt),th.reshape(Dl,16,Dr)).reshape(Dl*4,4*Dr)
        try: u,sv,vh=np.linalg.svd(th,full_matrices=False)
        except np.linalg.LinAlgError: u,sv,vh=sla.svd(th,full_matrices=False,lapack_driver='gesvd')
        w=sv**2; tot=w.sum()
        keep=min(a.chi, max(1,int(np.sum(w/tot>a.cut))))
        dw=w[keep:].sum()/tot; disc+=dw; logF+=np.log1p(-dw)
        sv=sv[:keep]/np.sqrt(w[:keep].sum())
        A[r]=u[:,:keep].reshape(Dl,4,keep); A[r+1]=(sv[:,None]*vh[:keep]).reshape(keep,4,Dr); c=r+1
        maxchi=max(maxchi,keep)
    else:
        k=item[1]; z,_=measure()
        Q=sum(1-(zi+zo)/2 for zi,zo in z); stag=sum((-1)**r*(2-(z[r][0]+z[r][1]))/2 for r in range(N))
        rec.append(dict(step=k,Q=Q,stag=stag,z=z,fid=float(np.exp(logF)),disc=disc,maxchi=maxchi,t=time.time()-t0))
        print(f"step {k:2d} Q={Q:.8f} stag={stag:.8f} fid={np.exp(logF):.6f} chi={maxchi} t={time.time()-t0:.0f}s",flush=True)
json.dump(rec,open(f'tebd_{a.circ}_chi{a.chi}.json','w'))
