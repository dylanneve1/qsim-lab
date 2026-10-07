import numpy as np
rng=np.random.default_rng(5)
rings=[[0,3,6,9],[1,4,7,10],[2,5,8]]   # 11 qubits
n=11; L=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{n}];']
def u(q): L.append(f'u3({rng.uniform(0,3)},{rng.uniform(-3,3)},{rng.uniform(-3,3)}) q[{q}];')
for rep in range(6):
    for r in rings:
        for i in range(len(r)):
            a,b=r[i],r[(i+1)%len(r)]; u(a); u(b); L.append(f'cz q[{a}],q[{b}];')
    for _ in range(3):
        a,b=rng.choice(n,2,replace=False); u(a); u(b); L.append(f'cz q[{a}],q[{b}];')
for q in range(n): u(q)
open('ring11.qasm','w').write('\n'.join(L)+'\n')
