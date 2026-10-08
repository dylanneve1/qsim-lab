"""Temporal-entanglement diagnostic: left IM sweep from the real left edge (site 0) for depth T, record
per cut the temporal entropy, chi_t at tolerance and discarded weight, for several chi caps."""
import numpy as np, sys, time, json
from tim import Circuit, sweep_left
T=int(sys.argv[1]); chi=int(sys.argv[2]); eps=float(sys.argv[3]); ncut=int(sys.argv[4]); lam=float(sys.argv[5]) if len(sys.argv)>5 else 1.0
cir=Circuit('SCV',lam,0,60,T); t0=time.time()
Ls,st=sweep_left(cir,chi,eps,ncut)
for c in sorted(st):
    S,k,d=st[c]; print(f'T={T} lam={lam} chi={chi} cut {c-1}|{c}: S_t={S:.3f} chi_t={k} disc={d:.1e}',flush=True)
print('time',time.time()-t0,flush=True)
