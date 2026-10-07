import sys, numpy as np, time, collections
import gparse as G
import tno as TN, tno_fast
tno_fast.install()
CZ=np.diag([1,1,1,-1]).astype(complex)
n,units,tail=G.parse('P6_titan_pinnacle.qasm')
depth=[0]*n; lay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); lay.append(l); depth[a]=depth[b]=l+1
lo,hi=int(sys.argv[1]),int(sys.argv[2]); cut=float(sys.argv[3])
W=TN.TNO(n,cutoff=1e-10)
t0=time.time()
sel=[k for k in range(len(units)) if lo<=lay[k]<hi]
for l in range(lo,hi):
    for k in sel:
        if lay[k]==l:
            a,b,Pa,Pb=units[k]; W.gate(CZ@np.kron(Pa,Pb),a,b,'after')
    TN.canonical_compress(W, cutoff=cut); W.drop_trivial_bonds()
    while TN.unswap_pass(W): TN.canonical_compress(W, cutoff=cut); W.drop_trivial_bonds()
    E=W.bond_graph()
    print(l, 'elems',W.size(),'maxbond',max(E.values()) if E else 1,'nbonds',len(E),'moved',sum(1 for o,i in W.perm().items() if o!=i), f'{time.time()-t0:.1f}s', flush=True)
# clusters
adj=collections.defaultdict(set)
for (x,y),d in W.bond_graph().items(): adj[x].add(y); adj[y].add(x)
seen=set(); comps=[]
for t in W.T:
    if t in seen: continue
    st=[t]; c=[]
    while st:
        u=st.pop()
        if u in seen: continue
        seen.add(u); c.append(u); st+=list(adj[u]-seen)
    comps.append(c)
print('cluster sizes', sorted([len(c) for c in comps], reverse=True)[:15])
