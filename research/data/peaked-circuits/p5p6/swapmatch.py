"""Makhlin-invariant matching of rank-4 units, allowing a SWAP absorbed on either side (G vs G.SWAP)."""
import numpy as np, sys, json, collections
U=np.load(sys.argv[1],allow_pickle=True)
Bm=np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
SW=np.eye(4)[[0,2,1,3]]
def mak(M):
    M=M/np.linalg.det(M)**0.25; Up=Bm.conj().T@M@Bm; m=Up.T@Up; tr=np.trace(m)
    return np.array([ (tr**2/16).real, (tr**2/16).imag, ((tr**2-np.trace(m@m))/4).real ])
def rank(M):
    T=M.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4); s=np.linalg.svd(T,compute_uv=False); return int((s>1e-6*s[0]).sum())
idx=[k for k in range(len(U)) if rank(np.asarray(U[k][2]))==4]
F=np.array([mak(np.asarray(U[k][2])) for k in idx]); FS=np.array([mak(np.asarray(U[k][2])@SW) for k in idx])
print('rank-4 units',len(idx))
def conjf(f): return np.array([f[0],-f[1],f[2]])
res=collections.Counter(); out=[]
for x in range(len(idx)):
    for y in range(x+1,len(idx)):
        for nm,a,b in (('plain',F[x],F[y]),('plain*',F[x],conjf(F[y])),('swap',FS[x],F[y]),('swap*',FS[x],conjf(F[y])),('swapB',F[x],FS[y]),('swapB*',F[x],conjf(FS[y]))):
            if np.linalg.norm(a-b)<1e-6:
                res[nm]+=1; out.append((idx[x],idx[y],nm)); break
print(res)
json.dump([(int(a),int(b),c) for a,b,c in out],open('p6_swapmatch.json','w'))
gaps=[b-a for a,b,c in out]; print('gap quantiles',np.quantile(gaps,[0,.25,.5,.75,1]) if gaps else None)
for a,b,c in out[:40]:
    print(a,(int(U[a][0]),int(U[a][1])),b,(int(U[b][0]),int(U[b][1])),c)
