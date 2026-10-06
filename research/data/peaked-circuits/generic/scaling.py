"""Contraction cost of the final reduced cores (cotengra hyper-optimiser, kahypar+greedy, flops objective)."""
import sys, json, time, numpy as np, quimb.tensor as qtn, cotengra as ctg
sys.path.insert(0, '/tmp/peaked-generic')
import solve_anchor as SA

def cost(tn, reps=64):
    opt = ctg.HyperOptimizer(methods=['greedy', 'kahypar'], max_repeats=reps, minimize='flops', parallel=False, progbar=False)
    tree = tn.contraction_tree(optimize=opt, output_inds=())
    return round(tree.contraction_width(), 1), round(tree.contraction_cost(log=2), 1)

for f, res in [('/tmp/pk/research/data/peaked-circuits/peaked_circuit_P11_Hqap_98x1999.qasm', 'results/p11_anchor.json'),
               ('/tmp/pk/research/data/peaked-circuits/peaked_circuit_P12_Hqap_98x2457.qasm', 'results/p12_anchor.json')]:
    r = json.load(open('/tmp/peaked-generic/' + res))
    core = SA.Core(f, log=lambda s: None)
    ops = core.ops(frozenset(r['extra_r']), frozenset(r['extra_p']))
    circ = qtn.Circuit(core.n)
    for M, w, _ in ops: circ.apply_gate_raw(M, w)
    bits = ''.join(r['peak'][w] for w in np.argsort(core.L))  # core-wire order
    amp = circ.amplitude_tn(bits)
    w_a, c_a = cost(amp)
    # the most expensive single-qubit marginal (largest backward light cone)
    worst = None
    for q in range(core.n):
        cone, gates = {q}, 0
        for M, wires, key in reversed(ops):
            if any(s in cone for s in wires): cone.update(wires); gates += 1
        if worst is None or gates > worst[1]: worst = (q, gates)
    tnq = circ.local_expectation_tn(np.diag([1., -1.]), (worst[0],))
    w_m, c_m = cost(tnq) if hasattr(tnq, 'contraction_tree') else (None, None)
    print(json.dumps(dict(file=f.split('/')[-1], core_units=len([o for o in ops if o[2][0] in 'RP']),
                          amp_width=w_a, amp_log2flops=c_a, worst_marginal_qubit=int(worst[0]), worst_cone_gates=worst[1],
                          marg_width=w_m, marg_log2flops=c_m)), flush=True)
