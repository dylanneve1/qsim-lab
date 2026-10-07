"""Operator Schmidt ranks of each epoch's unitary across ring cuts, in fold order (A folded | B twin-folded | C folded)."""
import sys, numpy as np, math, json
from parse import *
from segment import epochs
from mpo import MPO
from relabel import order
G=load(); E=epochs()
o=order(sys.argv[1] if len(sys.argv)>1 else 'twinfold'); site={q:i for i,q in enumerate(o)}
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
ne=max(E.values())+1
for e in range(ne):
    W=MPO(62,cutoff=1e-8)
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    for g in ops:
        if g[0]=='u3': W.g1(site[g[1][0]],U3(*g[2]))
        else: W.cz(site[g[1][0]],site[g[1][1]])
    b=W.bonds()
    print(e,'B' if e%2==0 else 'gap','ops',len(ops),'A|B',b[19],'B|C',b[39],'max',max(b),'bonds',b,'trunc %.1e'%W.trunc,flush=True)
