"""Slice-dropping fidelity test. Open-output state TN for a small truncated instance; slice indices chosen
(a) by cotengra (general slicing to a target width) and (b) uniformly at random among inner indices; contract every slice
to a full state vector; measure Gram matrix (orthogonality, norms) and fidelity of keeping a random fraction f of slices:
F = |<psi|psi_S>|^2 / <psi_S|psi_S>, plus linear XEB of p_S against p (normalised by the ideal XEB of p)."""
import sys, json, itertools, numpy as np, cotengra as ctg, quimb.tensor as qtn
from circ import *
n, D, nsl, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(seed)
gates = load(D, n); nT = sum(g[0] == 't' for g in gates)
control = len(sys.argv) > 5 and sys.argv[5] == 'haar'
if control:  # replace every 1q gate by a Haar-random unitary (same CZ skeleton): generic, non-Clifford control
    from scipy.stats import unitary_group
    for k, g in enumerate(gates):
        if g[0] != 'cz': MATS[f'u{k}'] = unitary_group.rvs(2, random_state=rng); gates[k] = (f'u{k}', g[1])
c = qtn.Circuit(n)
for g in gates:
    if g[0] == 'cz': c.apply_gate('CZ', g[1], g[2])
    else: c.apply_gate_raw(MATS[g[0]], (g[1],))
tn = c.psi.copy(); out = tuple(f'k{i}' for i in range(n))
tn.full_simplify_(output_inds=out, atol=1e-12)
ref = statevector(gates, n).reshape(-1); ref /= np.linalg.norm(ref)
inner = [i for i in tn.ind_map if i not in out]
inputs = [t.inds for t in tn]; sd = {i: 2 for i in tn.ind_map}
def run(sliced, label):
    M = 2**len(sliced); vecs = []
    for vals in itertools.product((0, 1), repeat=len(sliced)):
        t2 = tn.isel(dict(zip(sliced, vals)))
        v = t2.contract(all, output_inds=out, optimize='auto-hq')
        v = v.transpose(*out).data.reshape(-1) if hasattr(v, 'data') else np.asarray(v).reshape(-1)
        vecs.append(v)
    V = np.array(vecs); psi = V.sum(0)
    assert np.allclose(psi/np.linalg.norm(psi)*np.exp(-1j*np.angle(np.vdot(ref, psi))), ref, atol=1e-8), 'sum of slices != state'
    nrm = np.linalg.norm(psi); V /= nrm; psi /= nrm
    G = V.conj() @ V.T; norms = np.real(np.diag(G))
    off = G - np.diag(np.diag(G)); p = np.abs(psi)**2; xeb_ideal = 2**n*np.sum(p**2) - 1
    res = dict(label=label + (' [haar control]' if control else ''), n=n, D=D, T=nT, nsliced=len(sliced), M=M, max_offdiag_over_mean_norm=float(np.abs(off).max()/norms.mean()),
               norm_cv=float(norms.std()/norms.mean()), fid={})
    for f in (1/2, 1/4, 1/8):
        k = max(1, int(round(f*M))); Fs, Xs = [], []
        for _ in range(40):
            S = rng.choice(M, k, replace=False); ps = V[S].sum(0)
            Fs.append(abs(np.vdot(psi, ps))**2/np.vdot(ps, ps).real)
            pS = np.abs(ps)**2/np.sum(np.abs(ps)**2); Xs.append((2**n*np.sum(pS*p) - 1)/xeb_ideal)
        res['fid'][f'{k}/{M}'] = dict(F_mean=float(np.mean(Fs)), F_std=float(np.std(Fs)), xeb_rel_mean=float(np.mean(Xs)))
    print(json.dumps(res), flush=True); open('/tmp/doped-zx/slicefid.jsonl', 'a').write(json.dumps(res) + '\n')
# (a) cotengra general slicing
opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy'], max_repeats=32, max_time=60, minimize='flops', progbar=False,
                         )
tree = opt.search(inputs, out, sd).slice(target_slices=2**nsl)
run(list(tree.sliced_inds)[:nsl], 'cotengra-sliced')
# (b) random inner indices
for r in range(2):
    run(list(rng.choice(inner, nsl, replace=False)), f'random-inner-{r}')
