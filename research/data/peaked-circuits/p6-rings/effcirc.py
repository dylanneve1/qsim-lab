"""Effective circuit: burst epochs as raw gates; gap epochs replaced by cluster unitaries reconstructed from cached Heisenberg
images (private/imgs_e{e}.pkl). Clusters = wires coupled with image weight > THR; couplings below THR are dropped (lossy).
Output: list of ops ('u3',(q,),params) | ('cz',(a,b),()) | ('U',tuple(wires),W) with W on kron(wires) (first = most significant).
usage: effcirc.py THR out.pkl"""
import sys, pickle, numpy as np, collections, math
from parse import *
from segment import epochs, epochs3, epochs4
import os
from clthr import clusters
PM=[np.eye(2),np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1]).astype(complex)]
def string_matrix(K,key):
    d=dict(key); M=np.ones((1,1),complex)
    for w in K: M=np.kron(M,PM[d.get(w,0)])
    return M
def reconstruct(K,imgs):
    k=len(K); D=2**k; Kset=set(K)
    def img(q,p):
        M=np.zeros((D,D),complex); kept=0; tot=0
        for key,v in imgs[(q,p)].items():
            tot+=v*v
            if all(w in Kset for w,_ in key): M+=v*string_matrix(K,key); kept+=v*v
        return M,kept/tot
    Zs=[img(q,3) for q in K]; Xs=[img(q,1) for q in K]
    keep=min([z[1] for z in Zs]+[x[1] for x in Xs])
    S=sum(z[0] for z in Zs); w,V=np.linalg.eigh((S+S.conj().T)/2); psi0=V[:,-1]
    cols=[]
    for x in range(D):
        v=psi0.copy()
        for j in range(k):
            if (x>>(k-1-j))&1: v=Xs[j][0]@v
        cols.append(v)
    Wd=np.array(cols).T             # columns psi_x = W^dag |x>
    u,s,vh=np.linalg.svd(Wd); Wd=u@vh
    W=Wd.conj().T
    # fidelity of reconstruction
    fid=[]
    for j,q in enumerate(K):
        for P,(Mimg,_) in ((PM[3],Zs[j]),(PM[1],Xs[j])):
            Pq=np.kron(np.kron(np.eye(2**j),P),np.eye(2**(k-1-j)))
            fid.append(abs(np.trace(W.conj().T@Pq@W@Mimg))/D)
    return W,keep,min(fid)
if __name__=='__main__':
    THR=float(sys.argv[1]); out=sys.argv[2]
    G=load(); E={'3':epochs3,'4':epochs4}.get(os.environ.get('SEG'),epochs)(); ne=max(E.values())+1
    if len(sys.argv)>3: ne=int(sys.argv[3])
    ops=[]; report=[]; opep=[]
    for e in range(ne):
        if e%2==0:
            new=[G[i] for i in range(len(G)) if E[i]==e]; ops+=new; opep+=[e]*len(new); continue
        imgs=pickle.load(open(f'private/imgs{os.environ.get("SEG","")}_e{e}.pkl','rb'))
        cl=clusters(imgs,THR)
        for K in cl:
            if len(K)>10: print('BIG cluster',e,len(K)); 
            W,keep,fid=reconstruct(K,imgs)
            ops.append(('U',tuple(K),W)); opep.append(e); report.append((e,len(K),keep,fid))
        ks=[r for r in report if r[0]==e]
        print(e,'clusters',len(ks),'max size',max(r[1] for r in ks),'min keep %.4f'%min(r[2] for r in ks),'min fid %.4f'%min(r[3] for r in ks),
              'cross',[[ 'ABC'[ringof[q]]+str(pos[q]) for q in K] for K in cl if len({ringof[q] for q in K})>1],flush=True)
    pickle.dump(ops,open(out,'wb')); pickle.dump(opep,open(out+'.ep','wb'))
    print('ops',len(ops))
