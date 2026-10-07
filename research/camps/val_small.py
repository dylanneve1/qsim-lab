import sys, numpy as np, time
from camps import *
n, d = int(sys.argv[1]), int(sys.argv[2])
chis = [int(c) for c in sys.argv[3].split(',')]
lo = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ops = load_circuit(n=n, d=d, lo=lo)
nt = sum(1 for o in ops if o[0].startswith('rz'))
ex = dense_state(ops, n)
print(f"n={n} d={d} ops={len(ops)} T={nt} norm={np.linalg.norm(ex):.6f}")
for chi in chis:
    for mode in ['camps', 'camps-noDis', 'mps']:
        if mode == 'mps':
            m = run_mps(ops, n, chi); v = m.dense(); fe = m.fid; mb = m.max_bond(); tt = m.time; extra = ''
        else:
            st = run_camps(ops, n, chi, disentangle=(mode == 'camps'))
            v = st.dense(); fe = st.mps.fid; mb = st.mps.max_bond(); tt = st.time
            extra = f"ofd={st.stats['ofd']} mpo={st.stats['mpo']} triv={st.stats['trivial']} dis={st.stats['dis_applied']}"
        f = abs(np.vdot(ex, v))**2 / np.vdot(v, v).real
        print(f"chi={chi:5d} {mode:12s} F_true={f:.6f} F_est={fe:.6f} maxbond={mb} t={tt:.1f}s {extra}", flush=True)
