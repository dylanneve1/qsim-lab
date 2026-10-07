import numpy as np
from camps import *
for d in [20, 30, 40, 44, 48, 56, 63, 70]:
    ops = load_circuit(n=70, d=d)
    st = run_camps(ops, 70, 4, disentangle=False)
    e = readout_entropies(st.cinv); f = frame_out_entropies(st.cinv)
    # plain Clifford part only (T removed): entanglement of the undoped state
    cl = CAMPS(70, 1, disentangle=False)
    for name, qs, layer in ops:
        if not name.startswith('rz'): cl.clifford(name, qs)
    g = frame_out_entropies(cl.cinv)
    print(f"d={d} readout E(C^dag|0>) max={max(e)} mid={e[34]} | E(C|0>) max={max(f)} | undoped graph state max={max(g)}", flush=True)
