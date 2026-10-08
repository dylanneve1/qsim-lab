import numpy as np, sys
from compile import compile_ops
from heis import evolve, rung_states
L=int(sys.argv[1]); D=int(sys.argv[2]); lam=float(sys.argv[3]); nT=int(sys.argv[4]); eps=float(sys.argv[5]) if len(sys.argv)>5 else 1e-12
rs=[int(x) for x in sys.argv[6].split(',')] if len(sys.argv)>6 else None
lo=30-L//2; hi=lo+L
n0s,steps=compile_ops('SCV',lam); n0m,_=compile_ops('meson',lam)
ref={c:np.load(f'win_L{L}_{c}.npy') for c in ('SCV','meson')} if (lam==1 and L<=14) else None
for r in (rs or range(lo,hi)):
    v,err,Ds=evolve(r,lo,hi,steps,nT,D,eps,states=[rung_states(n0s,lo,hi),rung_states(n0m,lo,hi)],verbose=(rs is not None))
    if ref is not None:
        e=np.array([[ref[c][k,r-lo].sum() for c in ('SCV','meson')] for k in range(nT)])
        print(r,'max|heis-exact| per step',' '.join(f'{x:.0e}' for x in np.abs(v-e).max(1)),'D',Ds[-1],flush=True)
