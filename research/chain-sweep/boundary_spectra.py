#!/usr/bin/env python3
"""Schmidt spectra of a chain-sweep boundary tensor across every time cut.

Input: raw complex128 files from `chain_sweep dumpcut` (bit j of the index =
bond j of the cut edge, bonds in time order). See BOUNDARY_MPS.md."""
import numpy as np, sys
def spec(f):
    v=np.fromfile(f,dtype=np.float64).view(np.complex128)
    m=int(np.log2(len(v))); v=v/np.linalg.norm(v)
    # index bit j = bond j (time order); reshape C-order: last axis = bit 0
    out=[]
    T=v.reshape([2]*m)  # axis k <-> bit m-1-k
    T=np.transpose(T, list(range(m))[::-1])  # axis j <-> bit j
    for t in range(1,m):
        M=T.reshape(2**t, -1, order='F')  # bits 0..t-1 as rows
        s=np.linalg.svd(M,compute_uv=False); p=s**2; p=p[p>1e-14]
        S=-(p*np.log2(p)).sum(); r=len(p)
        # weight kept with chi = 2^k
        keep=[np.sum(np.sort(s**2)[::-1][:2**k]) for k in range(0,min(t,m-t)+1)]
        out.append((t,r,S,keep))
    return m,out
for f in sys.argv[1:]:
    m,out=spec(f); print(f,"m=",m)
    for t,r,S,keep in out:
        print(f"  t={t:2d} rank=2^{np.log2(r):5.2f} S={S:6.3f} kept(chi=2^k)=",' '.join(f"{x:.3f}" for x in keep))
