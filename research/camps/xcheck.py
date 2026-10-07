import numpy as np, subprocess, sys
from camps import *
n, d = 20, 40
ops = load_circuit(n=n, d=d)
ex = dense_state(ops, n)
rng = np.random.default_rng(1)
idx = list(rng.integers(0, 2**n, 5)) + list(np.argsort(-abs(ex))[:3])
with open('xs_chk.txt', 'w') as f:
    for i in idx:
        f.write(''.join(str((i >> q) & 1) for q in range(n)) + '\n')
out = subprocess.run(['/tmp/qsim-camps/target/release/examples/camps_ref', '--n', str(n), '--d', str(d), '--xs', 'xs_chk.txt', '--prec', '64'], capture_output=True, text=True).stdout
for i, line in zip(idx, out.split('\n')):
    _, re, im, _ = line.split()
    print(i, ex[i], complex(float(re), float(im)))
