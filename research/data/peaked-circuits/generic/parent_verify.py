# Parent's verifier: rebuild the compressed state network, then (1) exact amplitudes of all single-bit flips of the
# candidate (local optimality), (2) collision-probability certificate if affordable:
#   sum_x p(x)^2 - p*^2 < p*^2  =>  every other x has p(x) < p*  (unique global peak).
import sys, json, numpy as np, quimb.tensor as qtn, cotengra as ctg, time
sys.argv_saved = list(sys.argv)
import solve_generic as SG
path, centre, cutoff, cand = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
n, units, tail = SG.G.parse(path); L = SG.SP.layers(n, units)
W, lo, hi, hist = SG.tnoq.grow(n, units, L, centre, cutoff=cutoff, max_elems=50 * 2e4, log=False)
psi = SG.state_network(n, units, L, W, lo, hi, tail)
nrm = float(np.real((psi | psi.conj()).contract(all, optimize='greedy')))
def prob(bits):
    t = psi.copy()
    for q in range(n):
        t |= qtn.Tensor(np.array([1, 0] if bits[q] == '0' else [0, 1], dtype=complex), inds=(f"s{q}",))
    return float(abs(t.contract(all, optimize='greedy')) ** 2 / nrm)
t0 = time.time(); p0 = prob(cand)
flips = []
for q in range(n):
    b = list(cand); b[q] = '1' if b[q] == '0' else '0'; flips.append(prob(''.join(b)))
flips = np.array(flips)
print(json.dumps({'p_candidate': p0, 'norm': nrm, 'max_single_flip': float(flips.max()), 'ratio': p0 / flips.max(),
                  'n_flips_higher': int((flips > p0).sum()), 'secs': round(time.time() - t0)}), flush=True)
# collision probability: 4 copies, per-qubit delta joining s-legs of psi, psi*, psi, psi*
A = psi.reindex({f"s{q}": f"a{q}" for q in range(n)}); Ac = psi.conj().reindex({f"s{q}": f"a{q}" for q in range(n)})
B = psi.reindex({f"s{q}": f"a{q}" for q in range(n)}); Bc = psi.conj().reindex({f"s{q}": f"a{q}" for q in range(n)})
# make inner indices unique per copy
def uniq(t, tag):
    inner = [i for i in t.ind_map if not i.startswith('a')]
    return t.reindex({i: f"{tag}_{i}" for i in inner})
net = uniq(A, 'A') | uniq(Ac, 'Ac') | uniq(B, 'B') | uniq(Bc, 'Bc')   # hyperindex a{q} shared by 4 tensors = delta
opt = ctg.HyperOptimizer(methods=['greedy', 'kahypar'], max_repeats=48, parallel=False, progbar=False, minimize='combo', slicing_reconf_opts={'target_size': 2**24})
tree = net.contraction_tree(optimize=opt, output_inds=())
w = tree.contraction_width(); print('collision net width', round(w, 1), 'log2 flops', round(tree.contraction_cost(log=2), 1), flush=True)
print('sliced: nslices', tree.nslices, 'max size log2', round(tree.max_size(log=2),1), flush=True)
if tree.max_size(log=2) <= 25:
    arrays=[t.data for t in net]; S = float(np.real(tree.contract(arrays))) / nrm ** 2
    print(json.dumps({'sum_p2': S, 'p2': p0 ** 2, 'others_max_bound': float(np.sqrt(max(S - p0 ** 2, 0))),
                      'CERTIFIED': bool(S - p0 ** 2 < p0 ** 2)}), flush=True)
