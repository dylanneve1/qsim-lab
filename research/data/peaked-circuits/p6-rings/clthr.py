import pickle, sys, collections
from parse import *
nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
def clusters(imgs,thr):
    par=list(range(62))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for (q,p),im in imgs.items():
        # weight of each other wire in the image
        w=collections.defaultdict(float)
        for k,v in im.items():
            for x,_ in k: w[x]+=v*v
        for x,s in w.items():
            if x!=q and s>thr: par[f(x)]=f(q)
    cl=collections.defaultdict(list)
    for q in range(62): cl[f(q)].append(q)
    return [sorted(v) for v in cl.values()]
if __name__=='__main__':
    for e in [int(x) for x in sys.argv[1].split(',')]:
        imgs=pickle.load(open(f'private/imgs_e{e}.pkl','rb'))
        for thr in (1e-3,1e-2,5e-2,0.2):
            cl=clusters(imgs,thr); s=sorted((len(c) for c in cl),reverse=True)
            cross=[[nm(q) for q in c] for c in cl if len(set(ringof[q] for q in c))>1]
            print(e,thr,'sizes',s[:6],'cross',cross)
