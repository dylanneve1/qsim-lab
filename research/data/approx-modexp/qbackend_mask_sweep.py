"""Mask sweep on the genuinely quantum backend (the paper's own approx_modexp and prime
search, full superposition): TV between the approximate circuit and exact arithmetic with the
same mask, N=899 m=10 f=8. (m=10 is far below 2n, so the success values are not meaningful.)"""
import sys, time, math, fractions
sys.path.insert(0, '.')
import numpy as np
import gidney_env as ge
from facto.algorithm.prep import ExecutionConfig, ProblemConfig
import qbackend as qb
from facto.algorithm._detailed_example_code import approx_modexp
N, g, m, f = 899, 2, 10, 8
def succ_mask(m):
    out = np.zeros(1 << m, dtype=bool)
    for j in range(1 << m):
        d = fractions.Fraction(j, 1 << m).limit_denominator(N).denominator
        ff = math.gcd(pow(g, d // 2, N) + 1, N)
        out[j] = 1 < ff < N
    return out
S = succ_mask(m)
base = None
for mask in (0, 1, 2, 3, 4):
    conf = ExecutionConfig.from_problem_config(ProblemConfig.from_ini_content(f"""
modulus = {N}
generator = {g}
num_input_qubits = {m}
num_shots = 1
window1 = 2
window3a = 2
window3b = 2
window4 = 2
min_wraparound_gap = {f}
len_accumulator = {f}
mask_bits = {mask}
""")) if base is None else base.with_edits(base.conf.with_edits(mask_bits=mask))
    base = conf
    t0 = time.monotonic()
    qpu = qb.QuantumQPU(seed=5)
    Q_e = qpu.alloc_quint(length=m, scatter=True)
    Q_res = approx_modexp(Q_exponent=Q_e, conf=conf, qpu=qpu)
    qpu.fold_phases()
    e = np.array(Q_e.UNPHYSICAL_branch_vals); acc = np.array(Q_res.UNPHYSICAL_branch_vals)
    T = conf.truncated_modulus; W = 1 << mask; t = conf.dropped_bits
    psi = np.zeros((T, 1 << m), dtype=complex); psi[acc, e] += qpu.amp
    P = (np.abs(np.fft.fft(psi, axis=1, norm='ortho'))**2).sum(0)
    # ideal masked (exact arithmetic) for the same mask
    fe = np.array([pow(g, int(x), N) for x in range(1 << m)]) >> t
    fe %= T
    psi2 = np.zeros((T, 1 << m))
    for s in range(W):
        psi2[(fe + s) % T, np.arange(1 << m)] += 1
    psi2 /= np.linalg.norm(psi2)
    P2 = (np.abs(np.fft.fft(psi2, axis=1, norm='ortho'))**2).sum(0)
    print(f"N={N} m={m} f={f} mask={mask} |P|={len(conf.periods)} merges={qpu.merges} succ_actual={P[S].sum():.4f} succ_ideal={P2[S].sum():.4f} TV={0.5*np.abs(P-P2).sum():.4f} secs={time.monotonic()-t0:.0f}", flush=True)
