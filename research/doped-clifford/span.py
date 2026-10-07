import re,numpy as np
exec(open('xrank.py').read().split('tidx = [i')[0].replace('sys.argv[1]','"nq70_depth70_checks27_doped.qasm"'))
tidx=[i for i,(nm,_) in enumerate(ops) if nm=='rz']
E=[];S=[]
for i in tidx:
    q=ops[i][1][0]
    for direction,store in ((1,E),(-1,S)):
        x=np.zeros(n,np.uint8); z=np.zeros(n,np.uint8); z[q]=1
        seq = ops[i+1:] if direction==1 else reversed(ops[:i])
        for nm2,qs2 in seq:
            if nm2!='rz': step(x,z,nm2,qs2)
        sup=np.nonzero(x|z)[0]; store.append((sup.min(),sup.max()))
E=np.array(E); S=np.array(S)
# graph-state entanglement per cut (from profile run): min(e,70-e) capped ~32
ent=[min(k,70-k,32) for k in range(1,70)]
best=[]
for e in range(1,70):
    ce=int(((E[:,0]<e)&(E[:,1]>=e)).sum()); cs=int(((S[:,0]<e)&(S[:,1]>=e)).sum())
    best.append((e,ent[e-1],ce,cs))
for e,g,ce,cs in best[::5]+[best[33]]:
    print(f"cut {e:2d}: stab-ent {g:2d}  end-pushed rotations spanning {ce:3d}  start-pushed spanning {cs:3d}")
print("max over cuts of (ent + end-spanning):", max(g+ce for e,g,ce,cs in best))
