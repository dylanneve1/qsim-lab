import os,numpy as np,gauss
os.environ['SU2_CIRCUITS']='/tmp/su2-254-win/circuits'
for c in ('SCV','meson'):
    for mode in ('free',):
        rec,mo,k=gauss.run(c,mode); print(c,'maxoff/det',mo,k)
