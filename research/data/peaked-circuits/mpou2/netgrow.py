import sys, numpy as np, collections, time
import gparse as G
from mpou2 import MPO2
from match2 import match_unswap, score_wires
CZ=np.diag([1,1,1,-1]).astype(complex)
n,units,tail=G.parse(sys.argv[1]); K=len(units)
def key(M):
    M=M/np.sqrt(np.linalg.det(M));
    if M[0,0].real<0 or (abs(M[0,0].real)<1e-9 and M[0,1].real<0): M=-M
    return tuple(np.round(np.concatenate([M.real.ravel(),M.imag.ravel()]),6))
cnt=collections.Counter()
for a,b,Pa,Pb in units: cnt[key(Pa)]+=1; cnt[key(Pb)]+=1
rep={k for k,c in cnt.items() if c>=3}
mode=sys.argv[2] if len(sys.argv)>2 else 'both'
N=[k for k,(a,b,Pa,Pb) in enumerate(units) if (key(Pa) in rep and key(Pb) in rep) or (mode=='any' and (key(Pa) in rep or key(Pb) in rep))]
Nset=set(N)
# convexity: descendants of N that are ancestors of N
succ=[[] for _ in range(K)]; last={}
for k,(a,b,_,_) in enumerate(units):
    for w in (a,b):
        if w in last: succ[last[w]].append(k)
        last[w]=k
desc=set(); stack=list(N)
while stack:
    k=stack.pop()
    for j in succ[k]:
        if j not in desc: desc.add(j); stack.append(j)
pred=[[] for _ in range(K)]
for k in range(K):
    for j in succ[k]: pred[j].append(k)
anc=set(); stack=list(N)
while stack:
    k=stack.pop()
    for j in pred[k]:
        if j not in anc: anc.add(j); stack.append(j)
bad=(desc&anc)-Nset
print('N units',len(N),'non-N units between N units (convexity violators):',len(bad))
Nc=sorted(Nset|bad)
print('convex hull size',len(Nc))
W=MPO2(n,eps=1e-10,mode='rel',max_bond=4096)
t0=time.time(); mx=0
for i,k in enumerate(Nc):
    a,b,Pa,Pb=units[k]
    W.absorb(a,b,CZ@np.kron(Pa,Pb),'up')
    for _ in range(3):
        if W.unswap_sweep(thr=2)<1: break
    mx=max(mx,W.stats()['max_bond'])
    if i%10==0: print(i,W.stats(),f'{time.time()-t0:.1f}s',flush=True)
print('final',W.stats(),'max during',mx)
S=score_wires(W,'smax'); print('smax diag of current pairing', np.round(sorted([S[w,W.su[W.pd[w]]] for w in range(n)])[:10],3))
S2=score_wires(W,'sum'); print('sum score current pairing min/median', np.min([S2[w,W.su[W.pd[w]]] for w in range(n)]), np.median([S2[w,W.su[W.pd[w]]] for w in range(n)]))
