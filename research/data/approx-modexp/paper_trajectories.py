"""Runs the paper's own verification (scatter_script, a few random classical
trajectories, verify_clean_finish) on one of our instances, for context.

python3 paper_trajectories.py N g m f mask w1 w3a w3b w4 [branches]
"""
import sys
import time

import gidney_env  # noqa: F401
from facto.algorithm import approx_modexp
from facto.algorithm.prep import ExecutionConfig, ProblemConfig
from scatter_script import QPU, rvalue_multi_int

N, g, m, f, mask, w1, w3a, w3b, w4 = map(int, sys.argv[1:10])
nb = int(sys.argv[10]) if len(sys.argv) > 10 else 32
t0 = time.monotonic()
conf = ExecutionConfig.from_problem_config(ProblemConfig.from_ini_content(f"""
modulus = {N}
generator = {g}
num_input_qubits = {m}
num_shots = 1
window1 = {w1}
window3a = {w3a}
window3b = {w3b}
window4 = {w4}
min_wraparound_gap = {f}
len_accumulator = {f}
mask_bits = {mask}
"""))
t1 = time.monotonic()
qpu = QPU(num_branches=nb)
Q_e = qpu.alloc_quint(scatter=True, length=m)
init = Q_e.UNPHYSICAL_copy()
res = approx_modexp(qpu=qpu, conf=conf, Q_exponent=Q_e)
assert Q_e == init
gg = rvalue_multi_int.from_value(g, expected_count=nb)
exact = pow(gg, Q_e, N)
devs = []
for e, a in zip(exact.UNPHYSICAL_branch_vals, res.UNPHYSICAL_branch_vals):
    err = (e - (a << conf.dropped_bits)) % N
    devs.append(min(err, N - err) / N)
Q_e.UNPHYSICAL_force_del(dealloc=True)
res.UNPHYSICAL_force_del(dealloc=True)
qpu.verify_clean_finish()
t2 = time.monotonic()
print(f"paper's verifier: N={N} m={m} |P|={len(conf.periods)} ell={conf.rns_primes_bit_length} "
      f"{nb} trajectories clean (verify_clean_finish passed); max |acc·2^t − f(e)|/N over them {max(devs):.4f} (includes the mask offset s); "
      f"precompute {t1 - t0:.1f}s, simulate {t2 - t1:.1f}s")
