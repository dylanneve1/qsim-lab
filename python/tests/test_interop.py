import math

import numpy as np
import pytest

import qsimlab as qs
from qsimlab import Circuit, simulate, statevector, samples
from qsimlab.errors import UnsupportedOperationError

import reference as ref
from conftest import tv_distance


# --------------------------------------------------------------------------- Qiskit



def _qiskit():
    return pytest.importorskip("qiskit")


def random_qiskit_circuit(rng, n, depth):
    q = _qiskit()
    from qiskit.circuit import library as lib

    qc = q.QuantumCircuit(n)
    one = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"]
    rot = ["rx", "ry", "rz", "p"]
    two = ["cx", "cz", "swap", "cy", "ch", "iswap", "ecr", "dcx"]
    two_rot = ["crx", "cry", "crz", "cp", "rxx", "ryy", "rzz", "rzx"]
    for _ in range(depth):
        kind = rng.integers(6)
        a, b, c = (int(x) for x in rng.choice(n, size=3, replace=False))
        t = float(rng.uniform(-math.pi, math.pi))
        if kind == 0:
            getattr(qc, one[rng.integers(len(one))])(a)
        elif kind == 1:
            getattr(qc, rot[rng.integers(len(rot))])(t, a)
        elif kind == 2:
            getattr(qc, two[rng.integers(len(two))])(a, b)
        elif kind == 3:
            getattr(qc, two_rot[rng.integers(len(two_rot))])(t, a, b)
        elif kind == 4:
            qc.u(t, 0.3 * t, -0.7, a)
            qc.cu(t, 0.1, 0.2, 0.3, a, b)
        else:
            [qc.ccx, qc.cswap][rng.integers(2)](a, b, c)
    qc.global_phase = 0.321
    return qc


def test_from_qiskit_matches_qiskit_statevector(rng):
    _qiskit()
    from qiskit.quantum_info import Statevector

    for trial in range(8):
        n = int(rng.integers(3, 6))
        qc = random_qiskit_circuit(rng, n, 40)
        c = qs.interop.from_qiskit(qc)
        ours = simulate(c, statevector()).state
        theirs = Statevector(qc).data
        np.testing.assert_allclose(ours, theirs, atol=1e-9)  # global phase included


def test_qiskit_round_trip_and_aer_sampling(rng):
    _qiskit()
    from qiskit.quantum_info import Statevector

    c = ref.random_circuit(rng, 4, 50)
    c.global_phase = -1.2
    qc = qs.interop.to_qiskit(c)
    np.testing.assert_allclose(Statevector(qc).data, simulate(c, statevector()).state, atol=1e-9)
    back = qs.interop.from_qiskit(qc)
    np.testing.assert_allclose(simulate(back, statevector()).state, simulate(c, statevector()).state, atol=1e-9)

    aer = pytest.importorskip("qiskit_aer")
    import qiskit

    qc_m = qc.copy()
    qc_m.measure_all()
    shots = 20000
    res = aer.AerSimulator(seed_simulator=3).run(qiskit.transpile(qc_m, optimization_level=0), shots=shots).result()
    aer_counts = {k.replace(" ", ""): v for k, v in res.get_counts().items()}
    ours = simulate(c.copy().measure_all(), samples(shots), seed=3).counts()
    # both use Qiskit's key order (qubit/measurement 0 rightmost)
    probs = {k: v / shots for k, v in aer_counts.items()}
    assert tv_distance(ours, probs, shots) < 0.04


def test_from_qiskit_measurements_and_errors():
    q = _qiskit()
    qc = q.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    c = qs.interop.from_qiskit(qc)
    assert c.num_measurements == 2
    r = simulate(c, samples(1000), seed=1)
    assert set(r.counts()) <= {"00", "11"}
    from qiskit.circuit import Parameter

    p = q.QuantumCircuit(1)
    p.rx(Parameter("a"), 0)
    with pytest.raises(UnsupportedOperationError):
        qs.interop.from_qiskit(p)


# --------------------------------------------------------------------------- Cirq


def _cirq():
    return pytest.importorskip("cirq")


def little_endian(state, n):
    """Cirq's big-endian state -> qsimlab's little-endian order."""
    return state.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)


def random_cirq_circuit(rng, n, depth):
    cirq = _cirq()
    q = cirq.LineQubit.range(n)
    ops = []
    for _ in range(depth):
        a, b, c = (int(x) for x in rng.choice(n, size=3, replace=False))
        t = float(rng.uniform(-1, 1))
        choice = rng.integers(10)
        if choice == 0:
            ops.append([cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.S, cirq.T][rng.integers(6)](q[a]))
        elif choice == 1:
            ops.append(cirq.rx(t * 3)(q[a]))
        elif choice == 2:
            ops.append((cirq.X**t)(q[a]))  # XPowGate: has a global phase vs rx
        elif choice == 3:
            ops.append(cirq.PhasedXPowGate(phase_exponent=t, exponent=0.3)(q[a]))
        elif choice == 4:
            ops.append([cirq.CNOT, cirq.CZ, cirq.SWAP, cirq.ISWAP][rng.integers(4)](q[a], q[b]))
        elif choice == 5:
            ops.append((cirq.CZ**t)(q[a], q[b]))
        elif choice == 6:
            ops.append((cirq.XX**t)(q[a], q[b]))
        elif choice == 7:
            ops.append(cirq.FSimGate(theta=t, phi=0.4)(q[a], q[b]))
        elif choice == 8:
            ops.append([cirq.TOFFOLI, cirq.CCZ, cirq.CSWAP][rng.integers(3)](q[a], q[b], q[c]))
        else:
            ops.append((cirq.ISWAP**t)(q[a], q[b]))
    return cirq.Circuit(ops), q


def test_from_cirq_exact_including_global_phase(rng):
    cirq = _cirq()
    for trial in range(8):
        n = int(rng.integers(3, 6))
        cc, q = random_cirq_circuit(rng, n, 30)
        c = qs.interop.from_cirq(cc, qubit_order=q)
        ours = simulate(c, statevector()).state
        theirs = little_endian(cirq.final_state_vector(cc, qubit_order=q, dtype=np.complex128), n)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_cirq_round_trip(rng):
    cirq = _cirq()
    c = ref.random_circuit(rng, 4, 40)
    c.global_phase = 0.9
    cc = qs.interop.to_cirq(c)
    q = cirq.LineQubit.range(4)
    theirs = little_endian(cirq.final_state_vector(cc, qubit_order=q, dtype=np.complex128), 4)
    np.testing.assert_allclose(theirs, simulate(c, statevector()).state, atol=1e-8)
    back = qs.interop.from_cirq(cc, qubit_order=q)
    np.testing.assert_allclose(simulate(back, statevector()).state, simulate(c, statevector()).state, atol=1e-8)


def test_cirq_measurement_noise_and_control():
    cirq = _cirq()
    a, b = cirq.LineQubit.range(2)
    cc = cirq.Circuit(
        cirq.H(a),
        cirq.measure(a, key="m"),
        cirq.X(b).with_classical_controls("m"),
        cirq.bit_flip(0.0)(b),
        cirq.measure(b, key="out"),
    )
    c = qs.interop.from_cirq(cc)
    r = simulate(c, samples(2000), seed=3)
    assert (r.bits[:, 0] == r.bits[:, 1]).all()
    back = qs.interop.to_cirq(c)
    assert len(list(back.all_operations())) == 5


# --------------------------------------------------------------------------- Stim


def test_stim_generated_surface_code_matches_stim_statistics():
    stim = pytest.importorskip("stim")
    sc = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    c = qs.interop.from_stim(sc)
    assert c.num_measurements == sc.num_measurements
    assert len(c.detectors) == sc.num_detectors
    shots = 20000
    r = simulate(c, samples(shots), seed=11)
    assert r.engine == "symphase"
    ours = np.array([r.parity(d).mean() for d in c.detectors])
    det, obs = sc.compile_detector_sampler(seed=11).sample(shots, separate_observables=True)
    theirs = det.mean(axis=0)
    # detector firing rates agree within sampling error (rates ~ 2-6 %)
    assert np.max(np.abs(ours - theirs)) < 0.012
    # raw (undecoded) logical observable flip rate agrees too
    assert abs(r.parity(c.observables[0]).mean() - obs[:, 0].mean()) < 0.015
    back = qs.interop.to_stim(c)
    assert back.num_detectors == sc.num_detectors


def test_noiseless_stim_reference_sample():
    stim = pytest.importorskip("stim")
    sc = stim.Circuit("H 0\nCX 0 1\nCX 1 2\nM 0 1 2")
    r = simulate(qs.interop.from_stim(sc), samples(200), seed=0)
    assert all(row.min() == row.max() for row in r.bits)
