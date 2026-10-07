import sys, pickle, time, numpy as np
from ttn import StarTTN
from parse import rings
ops=pickle.load(open(sys.argv[1],'rb')); ep=pickle.load(open(sys.argv[1]+'.ep','rb'))
tol=float(sys.argv[2]); maxa=int(sys.argv[3]); stop=int(sys.argv[4]) if len(sys.argv)>4 else 99; tag=sys.argv[5] if len(sys.argv)>5 else 'x'
S=StarTTN([rings[0],rings[1],rings[2]],tol=tol,maxa=maxa,dtype=np.complex64)
t0=time.time(); last=-1
for op,e in zip(ops,ep):
    if e>stop: break
    S.apply(op)
    if e!=last:
        print(f'epoch {e} K {S.bonds()} trunc {S.trunc:.3e} t={time.time()-t0:.0f}s',flush=True); last=e
print(f'end K {S.bonds()} trunc {S.trunc:.3e}')
pickle.dump(dict(L=S.L,K=S.K,trunc=S.trunc,lognorm=S.lognorm),open(f'private/ttn_{tag}.pkl','wb'))
