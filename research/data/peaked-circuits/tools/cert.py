# Certify the peak: any s' != s with p(s') >= p(s) must (a) agree with s on every wire whose minority
# marginal < p, and (b) have its ambiguous-wire configuration x with marginal P_amb(x) >= p(s').
# So compute the exact marginal over the ambiguous wires; if s's configuration is the only one with
# P_amb >= p(s), the peak is certified (exact, up to float error).
import sys, numpy as np, quimb.tensor as qtn, cotengra as ctg, time
f = sys.argv[1]; s = sys.argv[2]; p = float(sys.argv[3])
circ = qtn.Circuit.from_openqasm2_file(f); n = circ.N
zs = np.load(f.split('/')[-1] + '.zs.npy')
minority = (1 - np.abs(zs)) / 2
amb = [q for q in range(n) if minority[q] >= p - 1e-12]
print('ambiguous', len(amb), amb, 'max outside minority', max(minority[q] for q in range(n) if q not in amb), flush=True)
t = time.time()
P = circ.compute_marginal(where=amb, optimize=ctg.ReusableHyperOptimizer(max_repeats=16, methods=['greedy', 'kahypar'], parallel=False, progbar=False))
P = np.real(np.asarray(P)).reshape(-1)
xs = ''.join(s[q] for q in amb); ix = int(xs, 2)
order = np.argsort(P)[::-1]
print('marginal sum', P.sum(), 'time', round(time.time() - t), flush=True)
print('P_amb(s config)', P[ix], ' top configs', [(format(i, f'0{len(amb)}b'), round(float(P[i]), 5)) for i in order[:5]])
over = [i for i in range(len(P)) if P[i] >= p - 1e-9]
print('configs with P_amb >= p:', len(over), 'CERTIFIED' if over == [ix] else 'NOT CERTIFIED')
