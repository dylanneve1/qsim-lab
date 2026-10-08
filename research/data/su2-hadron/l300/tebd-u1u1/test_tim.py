import numpy as np, sys, time
from tim import *
L=int(sys.argv[1]); chi=int(sys.argv[2]); T=int(sys.argv[3]); circ=sys.argv[4] if len(sys.argv)>4 else 'SCV'
lo=30-L//2; hi=lo+L; eps=1e-12
ref=np.load(f'win_L{L}_{circ}.npy')
cir=Circuit(circ,1.0,lo,hi,T); t0=time.time()
Ls,sl=sweep_left(cir,chi,eps,hi-1,verbose=True); Rs,sr=sweep_right(cir,chi,eps,lo+1,verbose=True)
for c in range(lo,hi):
    d,den=density(cir,Ls,Rs,c); print(c,f'tim {d:.10f} exact {ref[T-1,c-lo].sum():.10f} diff {abs(d-ref[T-1,c-lo].sum()):.1e} norm {abs(den):.6f}')
print('time',time.time()-t0)
