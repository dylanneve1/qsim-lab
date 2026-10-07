import sys, time, json, numpy as np
from camps import *
d, chi, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dis = (sys.argv[4] == 'dis') if len(sys.argv) > 4 else True
ops = load_circuit(n=70, d=d)
st = run_camps(ops, 70, chi, disentangle=dis)
ro = Readout(st)
ref = [complex(float(l.split()[1]), float(l.split()[2])) for l in open(f'ref_d{d}_u400.txt')][:N]
xs = [list(map(int, l.strip())) for l in open('xs_u400.txt')][:len(ref)]
t0 = time.time(); b = []; mk = 0
for x in xs:
    a, k = readout_amp_fast(ro, x); b.append(a); mk = max(mk, k)
ta = (time.time() - t0) / len(xs)
a = np.array(ref); b = np.array(b); M = len(a)
D2 = 2.0**70
terms = D2 * np.conj(a) * b
O = terms.mean()
F1 = abs(O)**2
F2 = abs(np.vdot(a, b))**2 / (np.vdot(a, a).real * np.vdot(b, b).real)
rng = np.random.default_rng(0)
bs1, bs2 = [], []
for _ in range(500):
    i = rng.integers(0, M, M)
    bs1.append(abs(terms[i].mean())**2)
    bs2.append(abs(np.vdot(a[i], b[i]))**2 / (np.vdot(a[i], a[i]).real * np.vdot(b[i], b[i]).real))
print(json.dumps(dict(d=d, chi=chi, dis=dis, M=M, F_est=st.mps.fid, F_true_unbiased=F1, se1=float(np.std(bs1)),
      F_true_ratio=float(F2), se2=float(np.std(bs2)), mean_2n_a2=float(D2 * np.mean(abs(a)**2)), mean_2n_b2=float(D2 * np.mean(abs(b)**2)),
      t_amp=ta, max_keys=mk, max_front=ro.max_front, readout_E=max(readout_entropies(st.cinv)))), flush=True)
