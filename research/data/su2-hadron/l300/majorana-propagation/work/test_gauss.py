import gauss
import numpy as np

D = '/tmp/su2-254/circuits/'
occ, m, out = gauss.blocks(D + f'x_100_SCV.qasm')
print("Testing gauss.py directly for g=0")
rec, _, _ = gauss.run('meson', mode='free')
for r in rec[:3]:
    print(f"Step {r['step']}: stag = {r['stag']}")
