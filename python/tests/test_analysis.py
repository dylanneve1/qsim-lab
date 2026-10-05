"""qsimlab.analysis against independent references.

* stabilizer nullity / Rényi entropy: brute force over all Pauli strings (numpy);
* magic profile: the active dimension d recomputed in Python (Heisenberg
  conjugation of the rotation axes + GF(2) rank), and its bounds checked on
  dense states (nullity ≤ d, support, Schmidt rank ≤ E + d, exact E for
  Clifford circuits);
* branching rank: its dense state against the numpy reference simulator;
* monitored: Born probabilities and the post-measurement state against a numpy
  simulation forced to the same outcomes; Rényi-2 cut entropies; the
  outcome-independence of d(t);
* simulability: features against circuit statistics, explanation against plan().
"""

import doctest
import itertools
import math
import os

import numpy as np
import pytest

import qsimlab as qs
import qsimlab.analysis as an
from qsimlab.errors import ResourceLimitError, UnsupportedOperationError

import reference as ref

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "analysis.md")
CLIFFORD_T = ["h", "s", "sdg", "x", "z", "cx", "cz", "swap", "t", "tdg"]


def random_ct(rng, n, depth, gates=CLIFFORD_T):
    c = qs.Circuit(n)
    for _ in range(depth):
        g = gates[rng.integers(len(gates))]
        k = qs.GATES[g][0]
        c.append(g, rng.choice(n, size=k, replace=False).tolist())
    return c


# --------------------------------------------------------------------------- references

P1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
      "Z": np.diag([1, -1])}


def brute_magic(psi):
    n = int(np.log2(len(psi)))
    vals = []
    for s in itertools.product("IXYZ", repeat=n):
        m = np.array([[1.0]])
        for ch in s:  # leftmost = highest qubit, as in a little-endian kron
            m = np.kron(m, P1[ch])
        vals.append(np.real(np.vdot(psi, m @ psi)))
    vals = np.array(vals)
    stab = int((np.abs(vals) > 1 - 1e-9).sum())
    return n - math.log2(stab), -math.log2((vals**4).sum() / 2**n)


def conj_pauli_back(x, z, gates):
    """Heisenberg-conjugate the Pauli (x, z) (bit vectors) backwards through Clifford gates:
    returns the x, z of C† P C for C = gates applied in order."""
    for name, qs_ in reversed(gates):
        if name == "h":
            q = qs_[0]
            x[q], z[q] = z[q], x[q]
        elif name in ("s", "sdg"):
            q = qs_[0]
            z[q] ^= x[q]
        elif name in ("x", "z"):
            pass
        elif name == "cx":
            a, b = qs_
            x[b] ^= x[a]
            z[a] ^= z[b]
        elif name == "cz":
            a, b = qs_
            z[a] ^= x[b]
            z[b] ^= x[a]
        elif name == "swap":
            a, b = qs_
            x[a], x[b] = x[b], x[a]
            z[a], z[b] = z[b], z[a]
        else:
            raise AssertionError(name)
    return x, z


def gf2_rank(rows):
    rows = [int("".join(map(str, r[::-1])), 2) for r in rows if any(r)]
    rank = 0
    while rows:
        piv = max(rows)
        if piv == 0:
            break
        rows.remove(piv)
        top = piv.bit_length() - 1
        rows = [r ^ piv if r >> top & 1 else r for r in rows]
        rows = [r for r in rows if r]
        rank += 1
    return rank


def active_dimension(c):
    """d of the rotation frame for a Clifford+T circuit, recomputed from scratch."""
    n = c.num_qubits
    cliffs, axes = [], []
    for ins in c.instructions():
        if ins.name in ("t", "tdg"):
            x, z = [0] * n, [0] * n
            z[ins.qubits[0]] = 1
            x, _ = conj_pauli_back(x, z, list(cliffs))
            axes.append(x)
        else:
            cliffs.append((ins.name, ins.qubits))
    return gf2_rank(axes)


def renyi2(psi, n, region):
    t = psi.reshape([2] * n)  # axis j <-> qubit n-1-j
    a_axes = [n - 1 - q for q in region]
    rest = [ax for ax in range(n) if ax not in a_axes]
    m = t.transpose(a_axes + rest).reshape(2 ** len(a_axes), -1)
    rho = m @ m.conj().T
    return -math.log2(np.real(np.trace(rho @ rho)))


def schmidt_bits(psi, n, cut):
    t = psi.reshape([2] * n)
    a_axes = [n - 1 - q for q in range(cut)]
    rest = [ax for ax in range(n) if ax not in a_axes]
    m = t.transpose(a_axes + rest).reshape(2**cut, -1)
    s = np.linalg.svd(m, compute_uv=False)
    return math.log2(int((s > 1e-9).sum()))


def same_ray(a, b, tol=1e-8):
    return abs(abs(np.vdot(a, b)) - 1) < tol


# --------------------------------------------------------------------------- state magic


def test_state_magic_against_brute_force(rng):
    for trial in range(12):
        n = int(rng.integers(1, 5))
        c = random_ct(rng, n, 25) if trial % 2 else ref.random_circuit(rng, n, 20)
        psi = ref.statevector(c)
        nu, m2 = brute_magic(psi)
        m = an.state_magic(c)
        assert m.nullity == pytest.approx(nu, abs=1e-9)
        assert m.m2 == pytest.approx(m2, abs=1e-9)
        assert an.state_magic(psi).m2 == pytest.approx(m2, abs=1e-9)
        assert an.stabilizer_nullity(psi) == pytest.approx(nu, abs=1e-9)
        assert an.stabilizer_renyi_entropy(c) == pytest.approx(m2, abs=1e-9)
    assert an.state_magic(qs.Circuit(3).h(0).cx(0, 1).s(2)).m2 == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match="normalised"):
        an.state_magic(np.ones(4))
    with pytest.raises(ValueError, match="2\\^n"):
        an.state_magic(np.ones(3) / math.sqrt(3))


# --------------------------------------------------------------------------- magic profile


def test_magic_profile_active_dimension_matches_python(rng):
    for _ in range(25):
        n = int(rng.integers(2, 9))
        c = random_ct(rng, n, int(rng.integers(5, 60)))
        p = an.magic_profile(c)
        names = [i.name for i in c.instructions()]
        assert p.t_count == p.rotations == names.count("t") + names.count("tdg")
        assert p.d == active_dimension(c)
        assert len(p.d_profile) == p.rotations
        assert (np.diff(p.d_profile.astype(int)) >= 0).all() and p.d <= n
        assert (p.f_profile <= p.d_profile).all() and p.f <= p.d
        assert len(p.rotation_gate) == p.rotations
        assert p.num_qubits == n and p.gates == len(c)


def test_magic_profile_bounds_hold_on_dense_states(rng):
    for _ in range(20):
        n = int(rng.integers(2, 8))
        c = random_ct(rng, n, int(rng.integers(5, 40)))
        p = an.magic_profile(c)
        psi = ref.statevector(c)
        assert an.state_magic(psi).nullity <= p.d + 1e-9  # nu <= d
        supp = int((np.abs(psi) > 1e-12).sum())
        assert math.log2(supp) <= p.support_log2 + 1e-9
        assert schmidt_bits(psi, n, p.cut) <= p.e_bound_max + 1e-9
        if p.rotations == 0:  # Clifford: the state is its skeleton, E is exact
            assert p.d == 0 and p.log2_work == -1
            assert schmidt_bits(psi, n, p.cut) == p.checkpoints[-1][5]


def test_magic_profile_families():
    # research/magic-atlas.md: QFT of a basis state has d = n - 1 and f = 1
    n = 10
    qft = qs.Circuit(n)
    for q in (0, 3, 4, 8):
        qft.x(q)
    for j in reversed(range(n)):
        qft.h(j)
        for k in reversed(range(j)):
            qft.cp(k, j, math.pi / 2 ** (j - k))
    p = an.magic_profile(qft)
    assert (p.d, p.f) == (n - 1, 1)
    assert an.magic_profile(qs.Circuit(5).h(0).cx(0, 1).measure_all()).d == 0
    with pytest.raises(UnsupportedOperationError):
        an.magic_profile(qs.Circuit(2).h(0).measure(0).h(1).cx(0, 1))


# --------------------------------------------------------------------------- branching rank


def test_branching_rank_state_matches_reference(rng):
    gates = CLIFFORD_T + ["ccx", "rz", "p", "cp", "rx", "ry"]
    for _ in range(20):
        n = int(rng.integers(3, 7))
        c = ref.random_circuit(rng, n, int(rng.integers(5, 25)), gates=gates)
        b = an.branching_rank(c, state=True)
        assert not b.overflow and len(b.trace) == len(c)
        assert b.rank == b.trace[-1] and b.max_rank == b.trace.max() >= 1
        psi = ref.statevector(c)
        np.testing.assert_allclose(b.state, psi, atol=1e-9)


def test_branching_rank_counts():
    c = qs.Circuit(4)
    for q in range(4):
        c.h(q)
    for q in range(4):
        c.t(q)
    assert an.branching_rank(c).trace.tolist()[-4:] == [2, 4, 8, 16]
    clif = qs.Circuit(6).h(0).cx(0, 1).s(1).cz(1, 2).swap(2, 3)
    assert an.branching_rank(clif).max_rank == 1
    big = qs.Circuit(12)
    for q in range(12):
        big.h(q).t(q)
    r = an.branching_rank(big, max_terms=100)
    assert r.overflow and r.max_rank <= 100


# --------------------------------------------------------------------------- simulability


def test_simulability_features_and_explanation():
    c = qs.Circuit(30)
    for q in range(29):
        c.h(q).cx(q, q + 1)
    s = an.simulability(c)
    f = s.features
    assert f["n"] == 30 and f["gates"] == len(c) and f["t_count"] == 0 and f["d"] == 0
    assert s.explanation.engine == qs.plan(c, qs.samples(1024)).engine == "tableau"
    costs = [x for _, x in s.log2_costs]
    assert costs == sorted(costs) and {e for e, _ in s.log2_costs} >= {"statevector", "mps"}
    assert f["sv_l"] >= 30
    t = qs.Circuit(6)
    for q in range(6):
        t.h(q).t(q)
    for q in range(5):
        t.cx(q, q + 1).t(q + 1)
    s2 = an.simulability(t, qs.statevector(), hsf=False)
    assert s2.features["t_count"] == 11 and s2.explanation.engine != "tableau"
    assert "log2 work" in str(s2)


# --------------------------------------------------------------------------- monitored


def apply_named(psi, n, name, qubits, params=()):
    m = ref.mat1(name, params) if len(qubits) == 1 else ref.mat_multi(name, params)
    return ref.apply(psi, n, np.asarray(m, dtype=complex), list(qubits))


def project(psi, n, q, outcome):
    idx = np.arange(len(psi))
    keep = ((idx >> q) & 1) == outcome
    phi = np.where(keep, psi, 0)
    p = float(np.vdot(phi, phi).real)
    return phi / math.sqrt(p) if p > 0 else phi, p


def random_monitored(rng, n, depth):
    c = qs.Circuit(n)
    for _ in range(depth):
        u = rng.random()
        if u < 0.12:
            c.measure(int(rng.integers(n)))
        elif u < 0.17 and c.num_measurements:
            c.x(int(rng.integers(n)), c_if=int(rng.integers(c.num_measurements)))
        else:
            g = CLIFFORD_T[rng.integers(len(CLIFFORD_T))]
            k = qs.GATES[g][0]
            c.append(g, rng.choice(n, size=k, replace=False).tolist())
    return c


def test_monitored_matches_forced_outcome_reference(rng):
    for trial in range(25):
        n = int(rng.integers(2, 6))
        c = random_monitored(rng, n, int(rng.integers(10, 50)))
        r = an.monitored(c, seed=trial, state=True, cuts=[[0], list(range(n // 2 + 1))])
        assert len(r.d) == len(c) and len(r.outcomes) == c.num_measurements
        psi = np.zeros(2**n, complex)
        psi[0] = 1
        k = 0
        for ins in c.instructions():
            if ins.name == "measure":
                q = ins.qubits[0]
                psi, p = project(psi, n, q, int(r.outcomes[k]))
                assert r.qubits[k] == q
                assert r.probabilities[k] == pytest.approx(p, abs=1e-9)
                assert p > 1e-12
                if r.kinds[k] == 2:
                    assert p == pytest.approx(1, abs=1e-9)
                k += 1
                continue
            if ins.c_if is not None:
                m, v = ins.c_if
                if bool(r.outcomes[m]) != v:
                    continue
            psi = apply_named(psi, n, ins.name, ins.qubits, ins.params)
        assert same_ray(r.state, psi)
        _, ents = r.entropies[-1]
        for region, (lo, hi, s2) in zip([[0], list(range(n // 2 + 1))], ents):
            want = renyi2(psi, n, region)
            assert lo - 1e-9 <= want <= hi + 1e-9
            assert s2 is not None and s2 == pytest.approx(want, abs=1e-8)


def test_monitored_dimension_is_outcome_independent(rng):
    for trial in range(8):
        n = int(rng.integers(3, 8))
        c = random_monitored(rng, n, 60)
        d0 = an.monitored(c, seed=1).d
        for s in (2, 3):
            assert np.array_equal(an.monitored(c, seed=s).d, d0)
        assert np.array_equal(an.monitored(c, seed=9, exact=False).d, d0)
        unitary = qs.Circuit(n)
        for ins in c.instructions():
            if ins.name != "measure" and ins.c_if is None:
                unitary.append(ins.name, ins.qubits, ins.params)
        assert an.monitored(unitary, seed=0).final_d == an.magic_profile(unitary).d


def test_monitored_clifford_reset_noise_and_limits():
    c = qs.Circuit(3).h(0).cx(0, 1).measure(0).cx(1, 2).measure(2)
    r = an.monitored(c, seed=4)
    assert (r.d == 0).all() and r.outcomes[0] == r.outcomes[1]
    assert r.probabilities.tolist() == [0.5, 1.0] and r.kinds.tolist() == [0, 2]
    rs = an.monitored(qs.Circuit(2).h(0).t(0).cx(0, 1).reset(0), seed=3, state=True)
    assert len(rs.outcomes) == 0
    p0 = (np.abs(rs.state.reshape(2, 2)[:, 1]) ** 2).sum()  # qubit 0 = 1
    assert p0 == pytest.approx(0, abs=1e-12)
    flip = an.monitored(qs.Circuit(1).x_error(0, 1.0).measure(0), seed=1)
    assert flip.outcomes.tolist() == [1]
    big = qs.Circuit(30)
    for q in range(30):
        big.h(q).t(q)
    with pytest.raises(ResourceLimitError, match="max_d"):
        an.monitored(big, max_d=10)
    dim = an.monitored(big, exact=False)
    assert dim.final_d == 30 and dim.entropies[-1][1][0][2] is None
    lo, hi, _ = dim.entropies[-1][1][0]
    assert lo == 0 and hi == 15


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
