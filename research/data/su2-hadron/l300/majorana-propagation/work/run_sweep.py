import ctypes
import os
import gauss
import numpy as np
import time

lib = ctypes.CDLL('/tmp/su2-254-mp/work/heis_c.so')

lib.init.argtypes = []
lib.apply_phase.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
lib.apply_hop.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                          ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                          ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.apply_interact.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.do_truncation.argtypes = [ctypes.c_double, ctypes.c_int]
lib.get_eval.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
lib.get_eval.restype = ctypes.c_double
lib.add_term.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double, ctypes.c_double]

num_terms = ctypes.c_int.in_dll(lib, "num_terms")

def run_sim(eps, k_max):
    D = '/tmp/su2-254/circuits/'
    occ_scv, m, out = gauss.blocks(D + 'x_100_SCV.qasm')
    occ_meson, _, _ = gauss.blocks(D + 'x_100_meson.qasm')
    
    occ0_scv = 0; occ1_scv = 0
    occ0_mes = 0; occ1_mes = 0
    for w in range(120):
        s, l = m[w]
        if occ_scv[w]:
            if l == 0: occ0_scv |= (1 << s)
            else: occ1_scv |= (1 << s)
        if occ_meson[w]:
            if l == 0: occ0_mes |= (1 << s)
            else: occ1_mes |= (1 << s)
            
    steps = []
    curr = []
    for b in out:
        if 'step' in b:
            if curr: steps.append(curr)
            curr = []
        else:
            curr.append(b)
            
    res_scv = []
    res_mes = []
    times = []
    term_counts = []
    
    for t in range(1, len(steps) + 1):
        t0 = time.time()
        lib.init()
        for r in range(60):
            c = 1.0 if r % 2 == 0 else -1.0
            lib.add_term(1<<r, 1<<r, 0, 0, c, 0.0)
            lib.add_term(0, 0, 1<<r, 1<<r, c, 0.0)
        lib.do_truncation(0.0, 999)
        
        blocks_seq = []
        for i in range(t-1, -1, -1):
            blocks_seq.extend(reversed(steps[i]))
            
        exploded = False
        for b in blocks_seq:
            w = b['w']; U = b['U']; sl = [m[x] for x in w]
            if len(w) == 1:
                s, l = sl[0]; ph = U[1,1]/U[0,0]
                lib.apply_phase(s, l, ph.real, ph.imag)
            else:
                (s1, l1), (s2, l2) = sl
                if l1 == l2:
                    V = U[1:3, 1:3]/U[0,0]
                    Vsp = np.array([[V[1,1], V[1,0]], [V[0,1], V[0,0]]])
                    lib.apply_hop(s1, s2, l1, 
                                  Vsp[0,0].real, Vsp[0,0].imag, Vsp[0,1].real, Vsp[0,1].imag,
                                  Vsp[1,0].real, Vsp[1,0].imag, Vsp[1,1].real, Vsp[1,1].imag)
                else:
                    ph = np.angle(np.diag(U))
                    a0 = ph[2] - ph[0]; a1 = ph[1] - ph[0]
                    g = ph[3] - ph[2] - ph[1] + ph[0]
                    g = (g + np.pi) % (2 * np.pi) - np.pi
                    lib.apply_interact(s1, s2, a0, a1, g)
            lib.do_truncation(eps, k_max)
            if num_terms.value > 4500000:
                print(f"EXPLODED at step {t} during propagation (terms > 4.5M)")
                exploded = True
                break
                
        if exploded:
            break
            
        val_scv = lib.get_eval(occ0_scv, occ1_scv)
        val_mes = lib.get_eval(occ0_mes, occ1_mes)
        t1 = time.time()
        
        res_scv.append(val_scv)
        res_mes.append(val_mes)
        times.append(t1 - t0)
        term_counts.append(num_terms.value)
        
        print(f"Step {t:2d}: SCV={val_scv:10.5f} Mes={val_mes:10.5f} nf={val_mes-val_scv:10.5f} | terms: {num_terms.value:7d} | {t1-t0:.2f}s")
        
    return res_scv, res_mes, times, term_counts

if __name__ == '__main__':
    for eps, k in [(1e-3, 4), (1e-4, 6), (1e-5, 6)]:
        print(f"\n--- Running eps={eps}, k_max={k} ---")
        run_sim(eps, k)
