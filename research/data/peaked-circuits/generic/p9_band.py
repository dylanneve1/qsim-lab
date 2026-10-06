"""Heisenberg images of all generators through a band of layers [l0, l1) (exact Pauli algebra, eps tiny)."""
import sys, numpy as np, gparse as G, struct_probe as SP, mirror as MI
f='/tmp/peaked-gen/peaked_circuit_P9_Hqap_56x1917.qasm'
n, units, tail = G.parse(f); L = SP.layers(n, units)
for l0, l1 in [(46, 56), (45, 57), (46, 55)]:
    W = MI.Window(n, units, l0, eps=float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3, layer=L)
    ks = [k for k in range(len(units)) if l0 <= L[k] < l1]
    for k in ks:                      # serial order is a valid order; all are 'after' gates of the window
        W.commit(W.cand_after(k), 'after', k)
    wts = [W.wsum(S) for S in W.img]
    terms = [len(S) for S in W.img]
    pm = W.perm(); single = sum(1 for q,(w,p) in pm.items() if p > 0.999)
    print(f"band [{l0},{l1}): {len(ks)} gates; mean image weight {np.mean(wts):.3f}, max {max(wts):.2f}; "
          f"max terms {max(terms)}; Z images single-wire {single}/{n}; moved {sum(1 for q,(w,p) in pm.items() if w!=q)}")
