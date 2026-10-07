import numpy as np
rng=np.random.default_rng(3); n=10
L=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{n}];']
for _ in range(22):
    a,b=rng.choice(n,2,replace=False)
    for q in (a,b): L.append(f'u3({rng.uniform(0,3)},{rng.uniform(-3,3)},{rng.uniform(-3,3)}) q[{q}];')
    L.append(f'cz q[{a}],q[{b}];')
    if rng.random()<0.3: L.append(f'cz q[{a}],q[{b}];')
for q in range(n): L.append(f'u3({rng.uniform(0,3)},{rng.uniform(-3,3)},{rng.uniform(-3,3)}) q[{q}];')
open('t10.qasm','w').write('\n'.join(L)+'\n')
