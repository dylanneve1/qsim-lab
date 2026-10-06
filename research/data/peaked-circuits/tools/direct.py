# Direct exact solve for small/shallow circuits: exact single-qubit marginals by TN light cones,
# peak = argmax per qubit, then exact amplitude of that string.  Prints cost first.
import sys, time, numpy as np, quimb.tensor as qtn, cotengra as ctg
f = sys.argv[1]; t0 = time.time()
circ = qtn.Circuit.from_openqasm2_file(f)
n = circ.N; print(f, 'n', n, 'gates', len(circ.gates), flush=True)
opt = ctg.ReusableHyperOptimizer(max_repeats=16, methods=['greedy', 'kahypar'], minimize='flops', parallel=False, progbar=False)
Z = np.diag([1., -1.])
# cost probe on the middle qubit
tn = circ.local_expectation_tn(Z, (n // 2,)) if hasattr(circ, 'local_expectation_tn') else None
zs = np.empty(n)
for q in range(n):
    zs[q] = float(np.real(circ.local_expectation(Z, (q,), optimize=opt, simplify_sequence='ADCRS')))
    if q % 8 == 0: print(' q', q, round(zs[q], 4), round(time.time() - t0), 's', flush=True)
bits = ''.join('0' if z > 0 else '1' for z in zs)
amp = circ.amplitude(bits, optimize=opt)
p = abs(amp) ** 2
np.save(f.split('/')[-1]+'.zs.npy', zs)
minority = (1 - np.abs(zs)) / 2
amb = [q for q in range(n) if minority[q] >= p - 1e-12]
print('ambiguous wires', len(amb), amb, flush=True)
import itertools
best = (p, bits); comp = []
if len(amb) <= 14:
    for flips in itertools.product((0, 1), repeat=len(amb)):
        if not any(flips): continue
        b = list(bits)
        for q, fl in zip(amb, flips):
            if fl: b[q] = '1' if b[q] == '0' else '0'
        s2 = ''.join(b); p2 = abs(circ.amplitude(s2, optimize=opt)) ** 2; comp.append(p2)
        if p2 > best[0]: best = (p2, s2)
    outside = max([minority[q] for q in range(n) if q not in amb] + [0])
    print('CERT: best', best[1], best[0], 'max competitor enumerated', max(comp) if comp else None, 'outside bound', outside,
          'CERTIFIED' if best[0] > max(max(comp + [0]) if best[1] == bits else p, outside) else 'check', flush=True)
print('PEAK (q0 first)', bits, 'p', p, 'min|Z|', np.abs(zs).min().round(4), 'time', round(time.time() - t0), flush=True)
