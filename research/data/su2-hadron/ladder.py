"""Convert the QASM to a logical ladder gate list (swap-free), fused into
1-site (d=4) and nearest-neighbour 2-site (16x16) unitaries, with step markers."""
import numpy as np
from parse import load
from graph_util import logical_map
I2=np.eye(2); Hm=np.array([[1,1],[1,-1]])/np.sqrt(2); Xm=np.array([[0,1],[1,0]])
def rz(t): return np.diag([np.exp(-1j*t/2),np.exp(1j*t/2)])
CX=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)  # control first
def one(name,p): return {'h':Hm,'x':Xm}[name] if name!='rz' else rz(p)
def embed(op, legs, nlegs):
    """op acting on given leg indices (ordered) of an nlegs-qubit register (big-endian)."""
    k=len(legs); op=op.reshape([2]*(2*k))
    T=np.eye(2**nlegs,dtype=complex).reshape([2]*(2*nlegs))
    # apply op to output legs
    idx_out=list(range(nlegs)); 
    T=np.tensordot(op, T, axes=(list(range(k,2*k)), legs))
    # tensordot puts op's out legs first then remaining T axes; rebuild order
    rest=[a for a in range(2*nlegs) if a not in legs]
    order=[None]*(2*nlegs)
    for j,l in enumerate(legs): order[l]=j
    for j,a in enumerate(rest): order[a]=k+j
    T=np.transpose(T,order)
    return T.reshape(2**nlegs,2**nlegs)
def build(path):
    ops=load(path); k=0
    init=[]
    while ops[k][0]=='x': init.append(ops[k][2][0]); k+=1
    wire2site=logical_map(path)   # initial wire -> (site, leg)
    occ=[0]*120
    for q in init: occ[q]^=1
    state=[[0,0] for _ in range(60)]
    for w in range(120):
        s,l=wire2site[w]; state[s][l]=occ[w]
    lab=list(range(120))
    out=[]  # items: ('1',site,4x4) ('2',r,16x16) ('step',k)
    cur=None  # (r, U16) pending bond block on (r,r+1)
    nm=0
    def flush():
        nonlocal cur
        if cur is not None: out.append(('2',cur[0],cur[1])); cur=None
    for name,p,qs in ops[k:]:
        if name=='swap':
            a,b=qs; lab[a],lab[b]=lab[b],lab[a]; continue
        sl=[wire2site[lab[q]] for q in qs]
        sites=sorted(set(s for s,_ in sl))
        if len(sites)==1:
            s=sites[0]
            U=embed(CX if name=='cx' else one(name,p), [l for _,l in sl], 2)
            if cur is not None and s in (cur[0],cur[0]+1):
                legs=[2*(s-cur[0])+l for _,l in sl]
                U16=embed(CX if name=='cx' else one(name,p), legs, 4)
                cur=(cur[0],U16@cur[1])
            else:
                out.append(('1',s,U))
        else:
            assert name=='cx' and sites[1]==sites[0]+1, (name,sl)
            r=sites[0]
            legs=[2*(s-r)+l for s,l in sl]
            U16=embed(CX,legs,4)
            if cur is not None and cur[0]!=r:
                # does the new bond overlap pending one? if overlap -> flush; if disjoint we could keep both, but keep simple
                flush()
            if cur is None: cur=(r,np.eye(16,dtype=complex))
            cur=(r,U16@cur[1])
        if name=='rz' and abs(abs(p)-0.03)<1e-12:
            nm+=1
            if nm%120==0: flush(); out.append(('step',nm//120))
    flush()
    return state,out
