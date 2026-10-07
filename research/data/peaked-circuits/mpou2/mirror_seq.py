import sys, numpy as np
import gparse as G
n,units,tail=G.parse(sys.argv[1]); K=len(units)
P=[(a,b) for a,b,_,_ in units]
def run(c, maxk=400, tol=3):
    fw={}; bw={}; good=0; bad=0; k=0
    while k<maxk and c-1-k>=0 and c+k<K:
        (a,b)=P[c-1-k]; (x,y)=P[c+k]
        ok=False
        for (u,v) in ((x,y),(y,x)):
            if fw.get(a,u)==u and fw.get(b,v)==v and bw.get(u,a)==a and bw.get(v,b)==b:
                fw[a]=u; fw[b]=v; bw[u]=a; bw[v]=b; ok=True; break
        if ok: good+=1
        else:
            bad+=1
            if bad>tol: break
        k+=1
    return good, k
res=[]
for c in range(700,1250):
    g,k=run(c); res.append((g,c,k))
res.sort(reverse=True); print(res[:15])
