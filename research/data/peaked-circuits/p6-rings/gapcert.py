"""Certify each gap: F = |Tr(W_eff^dag W_raw)| / 2^62 by exact TN contraction of the closed network (raw gap followed by the
inverse of the reconstructed cluster unitaries). usage: gapcert.py EFF.pkl SEG"""
import sys, os, pickle, numpy as np, quimb.tensor as qtn, cotengra as ctg, math
os.environ['SEG']=sys.argv[2]
from segment import epochs, epochs3, epochs4
from parse import load
G=load(); E={'3':epochs3,'4':epochs4}.get(sys.argv[2],epochs)()
ops=pickle.load(open(sys.argv[1],'rb')); ep=pickle.load(open(sys.argv[1]+'.ep','rb'))
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
ne=max(E.values())+1
for e in range(1,ne,2):
    circ=qtn.Circuit(62)
    for i,g in enumerate(G):
        if E[i]!=e: continue
        if g[0]=='u3': circ.apply_gate_raw(U3(*g[2]),g[1],contract=False)
        else: circ.apply_gate('CZ',*g[1],contract=False)
    for op,x in zip(ops,ep):
        if x!=e: continue
        W=op[2] if op[0]=='U' else None
        circ.apply_gate_raw(W.conj().T,op[1],contract=False)
    U=circ.get_uni()
    tn=U.copy()
    tn.reindex({f'k{q}':f'b{q}' for q in range(62)},inplace=True)
    val=tn.contract(all,optimize=ctg.HyperOptimizer(max_time=20,progbar=False,parallel=False))
    print(e,'F=|Tr|/2^62 = %.6f'%(abs(val)/2.0**62),flush=True)
