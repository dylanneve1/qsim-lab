import sys, numpy as np
from camps import *
n, d, lo, chi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ops = load_circuit(n=n, d=d, lo=lo)
st = run_camps(ops, n, chi)
v = st.dense()
ro = Readout(st)
rng = np.random.default_rng(0)
for _ in range(4):
    i = int(rng.integers(0, 2**n))
    bits = [(i >> q) & 1 for q in range(n)]
    a, _ = ro.amp(bits)
    print(i, v[i], a, abs(v[i]) - abs(a))
rs = []
for _ in range(20):
    i = int(rng.integers(0, 2**n))
    a, _ = ro.amp([(i >> q) & 1 for q in range(n)])
    rs.append(v[i] / a)
rs = np.array(rs); print("ratio spread", np.abs(rs - rs[0]).max(), rs[0], np.vdot(v, v))
for _ in range(3):
    i = int(rng.integers(0, 2**n))
    b = [(i >> q) & 1 for q in range(n)]
    print("fast", ro.amp(b)[0], readout_amp_fast(ro, b))
