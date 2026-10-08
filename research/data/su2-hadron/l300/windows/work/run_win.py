import os,sys,functools,numpy as np
os.environ['SU2_CIRCUITS']='/tmp/su2-254-win/circuits'
import gauss
gauss.blocks=functools.lru_cache(maxsize=4)(gauss.blocks)
import sv_lam
L=int(sys.argv[1]); circ=sys.argv[2]; lams=[float(x) for x in sys.argv[3:]]
lo=30-L//2; hi=lo+L
for lam in lams:
    f=f'res/L{L}_{circ}_lam{lam:+.2f}.npy'
    if os.path.exists(f): continue
    r=np.array(sv_lam.run(circ,lo,hi,lam,20))   # (20,L,2)
    np.save(f,r); print('done',f,flush=True)
