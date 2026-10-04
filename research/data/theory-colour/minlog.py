import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from geom import code
def setup(d):
    data,P=code(d); n=len(data)
    H=np.zeros((len(P),n),int)
    for i,p in enumerate(P):
        for q in p['q']:
            if q is not None: H[i,q]=1
    row0=[i for i,(x,y) in enumerate(data) if y==0]
    return data,P,H,row0
def min_logical(H,row0,force=(),forbid=()):
    m,n=H.shape
    # vars: v (n binary), z (m int), w int for logical parity: sum row0 v - 2w = 1
    nv=n+m+1
    c=np.zeros(nv); c[:n]=1
    A=np.zeros((m+1,nv)); A[:m,:n]=H; A[:m,n:n+m]=-2*np.eye(m)
    A[m,row0]=1; A[m,n+m]=-2
    lb=np.r_[np.zeros(m),1]; ub=lb.copy()
    lo=np.zeros(nv); hi=np.r_[np.ones(n),np.full(m,3),np.full(1,n)]
    for q in force: lo[q]=1
    for q in forbid: hi[q]=0
    r=milp(c,constraints=LinearConstraint(A,lb,ub),integrality=np.ones(nv),bounds=Bounds(lo,hi))
    if r.status!=0: return None,None
    v=np.round(r.x[:n]).astype(int)
    return int(v.sum()),[i for i in range(n) if v[i]]
