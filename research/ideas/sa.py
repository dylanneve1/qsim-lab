import numpy as np, sys
D=int(sys.argv[1]); M=np.load(f'M_D{D}.npy').astype(np.int64); Sm=np.load(f'S_D{D}.npy')
nT=M.shape[0]; rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 0)
def cost(y, lam):
    v=(M@y)%2; inval=int(((1-v)*y).sum()); a=int(v.sum())
    nontriv = ((Sm@y)%2).any()
    return a + lam*inval + (1000 if not nontriv else 0), a, inval
best=(10**9,None)
for rest in range(40):
    y=np.zeros(nT,np.int64); y[rng.integers(nT)]=1
    lam=3.0; c,a,iv=cost(y,lam); T=4.0
    for it in range(20000):
        j=rng.integers(nT); y[j]^=1
        if y.sum()==0: y[j]^=1; continue
        c2,a2,iv2=cost(y,lam)
        if c2<=c or rng.random()<np.exp((c-c2)/T): c,a,iv=c2,a2,iv2
        else: y[j]^=1
        T=max(0.05,T*0.9997)
        if iv==0 and c<1000 and a<best[0]: best=(a,y.copy())
print(f"D={D} nT={nT} best valid a={best[0]} |J|={int(best[1].sum()) if best[1] is not None else None} |S|={int(((Sm@best[1])%2).sum()) if best[1] is not None else None}")
