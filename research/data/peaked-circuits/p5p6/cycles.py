import numpy as np,sys
B=np.load(sys.argv[1]); L=B[:,2].max()+1; n=B[:,:2].max()+1
S=[set(map(tuple,np.sort(B[B[:,2]==l][:,:2],axis=1))) for l in range(L)]
def comps(edges):
    par=list(range(n))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for a,b in edges: par[f(a)]=f(b)
    from collections import Counter
    return sorted(Counter(f(i) for i in range(n)).values())
for l in range(L-1):
    print(l,l+1,len(S[l]),len(S[l+1]),'comp',comps(S[l]|S[l+1]))
