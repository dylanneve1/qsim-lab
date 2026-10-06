"""qsimlab.shor against independent references.

* number theory: brute-force orders / Carmichael values in pure Python;
* oracle circuits: a pure-Python bit-level evaluator (X/CNOT/CCX) and the
  dense numpy reference (Beauregard), every input x < N, both control values;
* distributions: the textbook QPE distribution computed with numpy, against
  the exact measurement-tree distribution and against sampling the exported
  circuit with qsimlab.simulate;
* factoring: factors / orders checked classically; same seed + base gives the
  same measured integer on every engine and oracle (research/shor/shor.md);
* counts and runs: the numbers recorded in research/shor/shor.md,
  research/data/mbu-shor/counts.txt and research/shor/ge-shor.md;
* the support law (theory-shor T1) against the measured support traces;
* noisy trajectories: against qsimlab.simulate on the same circuit with
  explicit X_ERROR ops (an independent noise implementation).
"""

import doctest
import math
import os
from fractions import Fraction

import numpy as np
import pytest

import qsimlab as qs
import qsimlab.shor as shor
from qsimlab.errors import ResourceLimitError, UnsupportedOperationError

import reference as ref

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "shor.md")


# --------------------------------------------------------------------------- references


def order_brute(a, n):
    x, r = a % n, 1
    while x != 1:
        x = x * a % n
        r += 1
    return r


def qpe_distribution(N, a):
    """Textbook distribution of the 2n-bit counting register (numpy, independent)."""
    n = (N - 1).bit_length()
    t = 2 * n
    T = 1 << t
    r = order_brute(a, N)
    y = np.arange(T)
    p = np.zeros(T)
    for j in range(r):
        k = np.arange(j, T, r)
        amp = np.exp(2j * np.pi * np.outer(y, k) / T).sum(axis=1) / T
        p += np.abs(amp) ** 2
    return p


def eval_reversible(c, bits):
    """Pure-Python evaluation of an X/CNOT/CCX/SWAP circuit on a basis state (int)."""
    for ins in c.instructions():
        q = ins.qubits
        if ins.name == "x":
            bits ^= 1 << q[0]
        elif ins.name == "cx":
            if bits >> q[0] & 1:
                bits ^= 1 << q[1]
        elif ins.name == "ccx":
            if bits >> q[0] & 1 and bits >> q[1] & 1:
                bits ^= 1 << q[2]
        elif ins.name == "swap":
            b0, b1 = bits >> q[0] & 1, bits >> q[1] & 1
            if b0 != b1:
                bits ^= (1 << q[0]) | (1 << q[1])
        else:
            raise AssertionError(f"non-permutation gate {ins.name}")
    return bits


def good_y(y, r, t):
    s = round(y * r / 2**t)
    return abs(Fraction(y, 2**t) - Fraction(s, r)) < Fraction(1, 2 * r * r)


# --------------------------------------------------------------------------- number theory


def test_orders_and_carmichael_match_brute_force(rng):
    for _ in range(40):
        N = int(rng.integers(15, 5000)) | 1
        lam = 1
        for a in range(2, min(N, 60)):
            if math.gcd(a, N) == 1:
                r = order_brute(a, N)
                assert shor.multiplicative_order(a, N) == r
                lam = lam * r // math.gcd(lam, r)
        if N < 60:
            assert shor.carmichael(N) == lam
        else:
            assert shor.carmichael(N) % lam == 0
    assert shor.carmichael(1_537_596_787) == 256_252_500  # research/shor/shor.md, 31-bit record


def test_support_bounds_closed_form():
    for r in [1, 2, 3, 12, 60, 41_832, 2_538_720, 110_806_800]:
        t = 2 * r.bit_length() + 2
        b = shor.support_bounds(r, t)
        nu = (r & -r).bit_length() - 1
        r_odd = r >> nu
        assert b.max() == (r if nu == 0 else max(r_odd, r // 2))
        for i, bi in enumerate(b):
            assert bi == min(2**i, r // math.gcd(r, 2 ** (t - i)))


# --------------------------------------------------------------------------- circuits


@pytest.mark.parametrize("oracle,window", [("ripple", None), ("windowed", 2), ("windowed-opt", 3)])
@pytest.mark.parametrize("N,a", [(15, 7), (21, 2), (35, 3)])
def test_oracle_circuit_is_the_modular_multiplication(oracle, window, N, a):
    o = shor.oracle_circuit(N, a, oracle, window=window)
    assert o.control == 0 and o.circuit.num_qubits == o.num_qubits
    assert o.circuit.stats()["is_unitary"]
    work = o.work
    for c in (0, 1):
        for x in range(N):
            inp = c | sum(((x >> k) & 1) << q for k, q in enumerate(work))
            out = eval_reversible(o.circuit, inp)
            y = pow(a, c, N) * x % N
            want = c | sum(((y >> k) & 1) << q for k, q in enumerate(work))
            assert out == want, (c, x)


def test_beauregard_oracle_against_dense_reference():
    N, a = 15, 7
    o = shor.oracle_circuit(N, a, "beauregard")
    n = o.num_qubits
    assert n == 2 * 4 + 3
    for c in (0, 1):
        for x in range(N):
            prep = qs.Circuit(n)
            if c:
                prep.x(0)
            for k, q in enumerate(o.work):
                if x >> k & 1:
                    prep.x(q)
            psi = ref.statevector(prep + o.circuit)
            y = pow(a, c, N) * x % N
            idx = c | sum(((y >> k) & 1) << q for k, q in enumerate(o.work))
            assert abs(psi[idx] - 1) < 1e-9


def test_beauregard_oracle_in_qiskit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qsimlab.interop import to_qiskit

    N, a = 15, 2
    o = shor.oracle_circuit(N, a, "beauregard")
    qc = to_qiskit(o.circuit)
    for x in (1, 7, 14):
        idx = 1 | sum(((x >> k) & 1) << q for k, q in enumerate(o.work))
        sv = Statevector.from_int(idx, 2**o.num_qubits).evolve(qc)
        y = a * x % N
        want = 1 | sum(((y >> k) & 1) << q for k, q in enumerate(o.work))
        assert abs(sv.data[want] - 1) < 1e-9
    assert qiskit is not None


def test_mbu_oracles_are_not_circuits():
    with pytest.raises(UnsupportedOperationError, match="feed-forward"):
        shor.oracle_circuit(15, 7, "windowed-mbu")
    with pytest.raises(UnsupportedOperationError):
        shor.oracle_circuit(15, 7, "permutation")


@pytest.mark.parametrize(
    "oracle", ["permutation", "beauregard", "ripple", "windowed", "windowed-opt",
               "windowed-mbu-lookup", "windowed-mbu"]
)
@pytest.mark.parametrize("N,a", [(15, 7), (21, 2), (33, 5)])
def test_exact_distribution_is_the_textbook_qpe(oracle, N, a):
    if oracle == "beauregard" and N > 21:
        pytest.skip("dense 2n+3 qubits")
    p = shor.exact_distribution(N, a, oracle, window=2 if "windowed" in oracle else None)
    np.testing.assert_allclose(p, qpe_distribution(N, a), atol=1e-12)


def test_sampling_the_exported_circuit_matches_qpe():
    # Beauregard (11 qubits, dense shots through qsimlab.simulate): full distribution
    N, a = 15, 7
    c = shor.shor_circuit(N, a, "beauregard")
    r = qs.simulate(c, qs.samples(1000), seed=11)
    ys = (r.bits.astype(np.int64) << np.arange(r.bits.shape[1])).sum(axis=1)
    emp = np.bincount(ys, minlength=256) / len(ys)
    assert 0.5 * np.abs(emp - qpe_distribution(N, a)).sum() < 0.06
    # ripple (16 qubits, ~10k gates): a few shots, all in the support of the QPE distribution
    c = shor.shor_circuit(N, a, "ripple")
    assert c.num_measurements == 8 and c.stats()["gates_3q"] > 0
    r = qs.simulate(c, qs.samples(12), seed=12)
    ys = (r.bits.astype(np.int64) << np.arange(8)).sum(axis=1)
    assert set(ys.tolist()) <= {0, 64, 128, 192}


# --------------------------------------------------------------------------- factoring


@pytest.mark.parametrize("oracle", ["windowed-opt", "windowed", "ripple", "windowed-mbu-lookup",
                                    "windowed-mbu", "permutation", "beauregard"])
def test_factor_small_semiprimes(oracle):
    for N in [15, 21, 33, 35, 143] if oracle != "beauregard" else [15, 21, 35]:
        r = shor.factor(N, oracle=oracle, seed=N, tries=20)
        assert r.factors is not None, (oracle, N)
        p, q = r.factors
        assert 1 < p <= q and p * q == N
        assert r.order == shor.multiplicative_order(r.base, N) == order_brute(r.base, N)
        assert pow(r.base, r.order, N) == 1
        assert r.runs[-1].succeeded and all(not x.succeeded for x in r.runs[:-1])
        assert r.components == [(x.qubits, x.total_gates, x.engine) for x in r.runs]


def test_same_seed_same_measured_integer_on_every_engine():
    N, a = 1003, 2  # research/shor/shor.md: same seed -> same measured integer on every path
    ys = {}
    for oracle, engine in [
        ("permutation", "dense"),
        ("permutation", "sparse"),
        ("ripple", "sliced"),
        ("ripple", "sparse"),
        ("windowed", "auto"),
        ("windowed-opt", "auto"),
        ("windowed-mbu-lookup", "auto"),
        ("windowed-mbu", "auto"),
    ]:
        r = shor.factor(N, oracle=oracle, base=a, seed=5, tries=1, engine=engine)
        ys[(oracle, engine)] = r.runs[0].measured
    assert len(set(ys.values())) == 1, ys
    f32 = shor.factor(N, base=a, seed=5, tries=1, precision="f32")
    assert f32.precision == "f32" and f32.measured == ys[("windowed-opt", "auto")]


def test_reproduces_the_research_runs():
    # research/shor/shor.md round 4: N = 1 005 973, a = 980 062, seed 1, measured
    # 475 634 978 396 on every path; (f) ripple sliced 1 148 438 gates, (g) windowed 501 948
    r = shor.factor(1_005_973, oracle="ripple", base=980_062, seed=1, tries=1)
    assert r.measured == 475_634_978_396 and r.total_gates == 1_148_438
    assert r.qubits == 64 and r.order == 41_832 and r.factors == (997, 1009)
    r = shor.factor(1_005_973, oracle="windowed", base=980_062, seed=1, tries=1)
    assert r.measured == 475_634_978_396 and r.total_gates == 501_948 and r.qubits == 88
    # 24-bit row of the scaling table: random base (seed 1), 104 qubits, run 1
    r = shor.factor(10_161_323, oracle="windowed", seed=1, tries=1)
    assert (r.base, r.order, r.qubits) == (9_899_614, 2_538_720, 104)
    assert r.factors == (2_833, 3_587) or r.factors[0] * r.factors[1] == 10_161_323


def test_support_trace_follows_the_law():
    rounds = deficient = 0
    for N, seed in [(1_005_973, 1), (143, 2), (4_087, 3), (10_403, 4), (65_533 * 3, 5)]:
        try:
            r = shor.factor(N, seed=seed, tries=3)
        except ValueError:
            continue
        for run in r.runs:
            tr, b = run.support_trace, run.predicted_support
            assert tr is not None and len(tr) == 2 * (N - 1).bit_length()
            assert (tr <= b).all(), (N, run.base)
            rounds += len(tr)
            deficient += int((tr < b).sum())
            assert run.peak_support >= tr.max()
            assert run.predicted_peak_support == b.max()
            # p1 is a rayon parallel sum: reduction order varies with thread count, so allow ulp-level overshoot
            assert run.p1_trace is not None and ((-1e-12 <= run.p1_trace) & (run.p1_trace <= 1 + 1e-12)).all()
    # theory-shor T1(c): deficient rounds have probability <= 4/r_odd each
    assert rounds > 50 and deficient <= 0.05 * rounds


def test_ekera_hastad_and_ge():
    for N in [143, 323, 899]:
        r = shor.factor(N, oracle="eh", seed=7, tries=10)
        assert r.factors is not None and r.factors[0] * r.factors[1] == N
        assert isinstance(r.measured, tuple) and len(r.measured) == 2
        assert r.order is None
        r = shor.factor(N, oracle="ge", seed=8, tries=10)
        assert r.factors is not None and r.factors[0] * r.factors[1] == N
        assert pow(r.base, r.order, N) == 1
    with pytest.raises(ValueError, match="balanced"):
        shor.factor(1_003, oracle="eh")  # 17 × 59: 59 > 2^5


# --------------------------------------------------------------------------- counts


def test_resource_counts_match_research_tables():
    # research/data/mbu-shor/counts.txt (w = 4, whole-run sums over the 2n blocks)
    N, a = 1_005_973, 980_062
    c = shor.resource_counts(N, "windowed-opt", base=a)
    assert (c.qubits, c.oracle_gates, c.toffoli, c.measurements) == (88, 271_380, 70_720, 0)
    c = shor.resource_counts(N, "windowed-mbu-lookup", base=a)
    assert (c.qubits, c.oracle_gates, c.toffoli, c.measurements, c.fixups) == (
        88, 224_622, 55_169, 14_089, 5_469)
    assert (c.cnot, c.x) == (143_815, 6_080)
    c = shor.resource_counts(N, "windowed-mbu", base=a)
    assert (c.qubits, c.oracle_gates, c.toffoli, c.measurements, c.fixups) == (
        107, 294_892, 31_399, 36_839, 16_999)
    c = shor.resource_counts(10_161_323, "windowed-opt", base=9_899_614)
    assert (c.qubits, c.oracle_gates, c.toffoli) == (104, 463_922, 120_672)
    # the factoring run's Toffoli count is the oracle total
    r = shor.factor(N, base=a, seed=1, tries=1)
    assert r.toffoli_gates == 70_720


def test_resource_counts_ge_and_permutation():
    c = shor.resource_counts(10_161_323, "ge", base=9_899_614)
    assert c.rounds == 48 and c.qubits == 106 and c.slice_steps is not None
    assert c.toffoli < shor.resource_counts(10_161_323, "windowed-mbu-lookup", base=9_899_614).toffoli
    e = shor.resource_counts(10_161_323, "eh", base=9_899_614)
    assert e.rounds == 36 and e.toffoli < c.toffoli
    p = shor.resource_counts(143, "permutation")
    assert not p.gate_level and p.oracle_gates == 0 and p.qubits == 9
    pr = shor.resource_counts(143, "ripple", per_round=True)
    assert pr.per_round_gates.sum() == pr.oracle_gates and len(pr.per_round_gates) == 16
    o = shor.oracle_circuit(143, pr.base, "ripple")
    assert pr.per_round_gates[-1] == o.gates  # last round multiplies by a itself


# --------------------------------------------------------------------------- guards, errors


def test_memory_guard_refuses_before_running():
    p = shor.predict_support(1_005_973, 980_062)
    assert p.order == 41_832 and p.peak == 20_916 and p.engine == "sliced"
    assert p.predicted_bytes == 20_916 * 51
    with pytest.raises(ResourceLimitError, match="order r = 41832") as e:
        shor.factor(1_005_973, base=980_062, budget=100_000)
    assert e.value.needed == p.predicted_bytes and e.value.limit == 100_000
    assert shor.predict_support(1_005_973, 980_062, precision="f32").predicted_bytes == 20_916 * 33
    # the dense paths are predicted from the register size
    assert shor.predict_support(143, 2, "beauregard").predicted_bytes == 16 << 19
    with pytest.raises(ResourceLimitError):
        shor.factor(1003, oracle="beauregard", base=2, budget="1MiB")


def test_bad_inputs():
    with pytest.raises(ValueError, match="even"):
        shor.factor(1002)
    with pytest.raises(ValueError, match="prime power"):
        shor.factor(3**7)
    with pytest.raises(ValueError, match="prime"):
        shor.factor(1009)
    with pytest.raises(ValueError, match="shares the factor"):
        shor.factor(15, base=5)
    with pytest.raises(ValueError, match="unknown oracle"):
        shor.factor(15, oracle="magic")
    with pytest.raises(ValueError, match="cannot run"):
        shor.factor(15, oracle="windowed", engine="dense")
    with pytest.raises(TypeError):
        shor.factor(15.0)
    with pytest.raises(ValueError):
        shor.factor(15, precision="f16")


# --------------------------------------------------------------------------- noise


def test_noiseless_trajectories_match_the_exact_distribution():
    N, a = 143, 5
    r = shor.noisy_success(N, a, 0.0, trajectories=400, seed=3)
    assert r.mean_faults == 0 and r.capped == 0 and r.locations > 0
    t = 2 * (N - 1).bit_length()
    ro = order_brute(a, N)
    p = shor.exact_distribution(N, a, "windowed", window=4)
    p_good = sum(p[y] for y in range(2**t) if p[y] > 0 and good_y(y, ro, t))
    sd = math.sqrt(p_good * (1 - p_good) / 400)
    assert abs(r.peak_rate - p_good) < 4 * sd + 1e-9


def test_noisy_trajectories_against_explicit_noise_circuit():
    """Bit-flip noise: shor.noisy (bit-sliced trajectories) vs qsimlab.simulate on the
    same circuit with explicit X_ERROR ops (the reference_circuit of shor/noisy.rs,
    rebuilt here from the public oracle_circuit blocks)."""
    N, a, p = 15, 7, 6e-5
    n = 4
    t = 2 * n
    mults = [pow(a, 2**k, N) for k in range(t)]
    blocks = [shor.oracle_circuit(N, m, "ripple").circuit for m in mults]
    nq = blocks[0].num_qubits
    c = qs.Circuit(nq)
    c.x(1)
    for i in range(t):
        if i > 0:
            c.x(0, c_if=i - 1)
        c.x_error(0, p)  # Prep
        c.h(0)
        c.x_error(0, p)  # H1
        for ins in blocks[t - 1 - i].instructions():
            c.append(ins.name, ins.qubits)
            for q in ins.qubits:
                c.x_error(q, p)
        for l in range(i):
            c.p(0, -math.pi / 2 ** (i - l), c_if=l)
        c.x_error(0, p)  # Phase
        c.h(0)
        c.x_error(0, p)  # H2
        c.x_error(0, p)  # Meas
        c.measure(0)
    # dense noisy shots of a 16-qubit circuit with 10k gates and 25k noise ops cost
    # ~0.5-1.5 s each; 150 shots passed on the Mac (241 s, loaded), default 40
    shots = int(os.environ.get("QSIMLAB_TEST_SHOTS", "40"))
    ref_r = qs.simulate(c, qs.samples(shots), seed=21)
    ys = (ref_r.bits.astype(np.int64) << np.arange(t)).sum(axis=1)
    ro = order_brute(a, N)
    ref_good = np.mean([good_y(int(y), ro, t) for y in ys])
    eng = shor.noisy_success(N, a, p, noise="bitflip", oracle="ripple", trajectories=4000, seed=22)
    assert eng.mean_faults == pytest.approx(p * eng.locations, rel=0.1)
    assert 0.2 < eng.peak_rate < 0.9  # the noise matters at this p
    sd = math.sqrt(eng.peak_rate * (1 - eng.peak_rate) / shots)
    assert abs(eng.peak_rate - ref_good) < 4 * sd, (eng.peak_rate, ref_good)
    # the useful outcomes (y = 64, 192 give r = 4 directly) agree too
    ref_u = np.isin(ys, [64, 192]).mean()
    eng_u = np.mean([y in (64, 192) for y in eng.measured])
    assert abs(ref_u - eng_u) < 4 * math.sqrt(max(eng_u * (1 - eng_u), 0.01) / shots)


def test_noise_degrades_and_is_reproducible():
    a = shor.noisy_success(143, 5, 2e-3, trajectories=60, seed=4, threads=1)
    b = shor.noisy_success(143, 5, 2e-3, trajectories=60, seed=4, threads=2)
    assert np.array_equal(a.factor_ok, b.factor_ok) and a.measured == b.measured
    k1 = shor.noisy_success(143, 5, 0.0, faults=1, trajectories=200, seed=5)
    assert (k1.faults == 1).all()
    assert k1.peak_rate < 0.9  # a random single fault is fatal ~70 % of the time
    lo, hi = k1.ci
    assert lo <= k1.success <= hi


# --------------------------------------------------------------------------- docs


def test_docs_tutorial_runs():
    if not os.path.exists(DOCS):
        pytest.skip("docs not present")
    res = doctest.testfile(
        os.path.abspath(DOCS),
        module_relative=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert res.failed == 0 and res.attempted > 0
