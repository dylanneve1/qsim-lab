import re,sys,numpy as np
exec(open('profile.py').read().split('# ---- layering')[0].replace('F = sys.argv[1]','F = "nq70_depth70_checks27_doped.qasm"'))
src=open('profile.py').read()
exec(src[src.index('def tableau'):src.index('def ent(')])
for lab,rz in (("T removed",None),("T->S",'s')):
    X,Z=tableau(ops,rz)
    r=gf2_rank(X.copy())
    print(lab,"X-part rank of stabilizers =",r,"-> Z-basis support dim",r,"; uniform-on-support XEB vs Clifford state =",2**(70-r)-1)
