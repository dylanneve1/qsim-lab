import math
import pickle

import numpy as np
import pytest

import qsimlab as qs
from qsimlab import Circuit
from qsimlab.errors import (
    CircuitError,
    ParseError,
    QsimError,
    QubitIndexError,
    UnsupportedOperationError,
)

import reference as ref


def test_every_gate_has_a_builder_and_round_trips(rng):
    c = Circuit(3)
    for name, (k, np_) in qs.GATES.items():
        qubits = list(range(k))
        params = [0.1 * (j + 1) for j in range(np_)]
        getattr(c, name)(*qubits, *params)
    names = [i.name for i in c.instructions()]
    assert names == list(qs.GATES)
    again = Circuit.from_instructions(3, c.instructions())
    assert again == c


def test_aliases():
    a = Circuit(3).cnot(0, 1).phase(0, 0.3).cphase(0, 1, 0.2).toffoli(0, 1, 2).id(2)
    b = Circuit(3).cx(0, 1).p(0, 0.3).cp(0, 1, 0.2).ccx(0, 1, 2).i(2)
    assert a == b
    assert Circuit(1).append("U3", 0, (1, 2, 3)) == Circuit(1).u(0, 1, 2, 3)


def test_validation_errors():
    c = Circuit(2)
    with pytest.raises(QubitIndexError):
        c.h(2)
    with pytest.raises(IndexError):
        c.cx(0, 5)
    with pytest.raises(CircuitError):
        c.cx(1, 1)
    with pytest.raises(CircuitError):
        c.append("nope", 0)
    with pytest.raises(CircuitError):
        c.append("rx", 0)  # missing parameter
    with pytest.raises(CircuitError):
        c.rx(0, float("nan"))
    with pytest.raises(CircuitError):
        c.x_error(0, 1.5)
    with pytest.raises(QubitIndexError):
        c.x(1, c_if=0)  # no measurement yet
    with pytest.raises(QubitIndexError):
        c.detector([0])
    assert len(c) == 0, "failed appends must not leave partial ops"
    assert issubclass(QubitIndexError, QsimError)


def test_conditionals_and_metadata():
    c = Circuit(3).h(0).measure(0).x(1, c_if=0).z(2, c_if=(0, False)).measure(1, 2)
    ins = c.instructions()
    assert ins[2].c_if == (0, True) and ins[3].c_if == (0, False)
    d = c.detector([1, 2])
    c.observable_include(0, [2])
    assert d == 0 and c.detectors == [[1, 2]] and c.observables == [[2]]
    s = c.stats()
    assert s["conditionals"] == 2 and s["measurements"] == 3 and not s["is_unitary"]


def test_compose_shifts_measurement_indices():
    body = Circuit(2).measure(0).x(1, c_if=0)
    body.detector([0])
    c = Circuit(2).measure(1)
    c.compose(body)
    assert c.instructions()[2].c_if == (1, True)
    assert c.detectors == [[1]]
    assert Circuit(2).repeat(Circuit(2).h(0), 2) == Circuit(2).h(0).h(0)
    r = Circuit(2).repeat(body, 3)
    assert r.num_measurements == 3
    assert [i.c_if for i in r.instructions() if i.c_if] == [(0, True), (1, True), (2, True)]
    assert r.detectors == [[0], [1], [2]]
    mapped = Circuit(3).compose(Circuit(2).cx(0, 1), qubits=[2, 0])
    assert mapped.instructions()[0].qubits == (2, 0)


def test_add_copy_eq_pickle():
    a = Circuit(2).h(0)
    b = Circuit(2).cx(0, 1).measure_all()
    c = a + b
    assert len(c) == 4 and len(a) == 1
    d = c.copy()
    d.x(0)
    assert len(c) == 4
    c.global_phase = 0.25
    c.detector([0, 1])
    e = pickle.loads(pickle.dumps(c))
    assert e == c and e.global_phase == 0.25 and e.detectors == [[0, 1]]


def test_inverse_is_exact(rng):
    c = ref.random_circuit(rng, 3, 30)
    c.global_phase = 0.4
    full = c + c.inverse()
    psi = qs.simulate(full, qs.statevector()).state
    assert abs(psi[0] - 1) < 1e-10
    with pytest.raises(UnsupportedOperationError):
        Circuit(1).measure(0).inverse()


def test_remove_final_measurements():
    c = Circuit(2).h(0).measure(0).cx(0, 1).measure(0, 1)
    r = c.remove_final_measurements()
    assert [i.name for i in r.instructions()] == ["h", "measure", "cx"]


def test_stats_and_draw():
    c = Circuit(3).h(0).t(1).ccx(0, 1, 2).depolarize1(2, 0.1).measure_all()
    s = c.stats()
    assert (s["gates_1q"], s["gates_3q"], s["t_gates"], s["noise_channels"]) == (2, 1, 1, 1)
    assert s["depth"] >= 3 and not s["is_clifford"]
    text = c.draw()
    assert text.count("\n") >= 2 and "H" in text


def test_qasm_round_trip_exact(rng):
    c = ref.random_circuit(rng, 4, 60)
    c.measure_all()
    back = Circuit.from_qasm(c.to_qasm())
    # iswap/iswapdg are exported as decompositions; compare states instead of ops
    np.testing.assert_allclose(
        qs.simulate(back, qs.statevector()).state, qs.simulate(c, qs.statevector()).state, atol=1e-10
    )
    plain = Circuit(2).h(0).rz(1, 0.123456789).cp(0, 1, -0.5).u(0, 1, 2, 3).measure(0, 1)
    assert Circuit.from_qasm(plain.to_qasm()) == plain


def test_qasm_export_refuses_what_it_cannot_express():
    with pytest.raises(UnsupportedOperationError):
        Circuit(2).measure(0).x(1, c_if=0).to_qasm()
    with pytest.raises(UnsupportedOperationError):
        Circuit(1).x_error(0, 0.1).to_qasm()


QASM_FEATURES = """
OPENQASM 2.0;
include "qelib1.inc";
/* block comment */
qreg a[2];
qreg b[1];
creg c[2];
creg flag[1];
gate mygate(theta) x, y { h x; cx x, y; rz(theta/2) y; }
gate wrapper x, y { mygate(pi) y, x; barrier x, y; }
h a;                       // broadcast over a register
mygate(0.5) a[0], b[0];
wrapper a[1], b[0];
cx a, b;                   // broadcast: cx a[0],b[0]; cx a[1],b[0]
barrier a, b;
measure a -> c;
measure b[0] -> flag[0];
if (flag == 1) x a[0];
reset b[0];
"""


def test_qasm_parser_features():
    c = Circuit.from_qasm(QASM_FEATURES)
    assert c.num_qubits == 3
    names = [i.name for i in c.instructions()]
    assert names[:2] == ["h", "h"]
    assert names.count("measure") == 3
    cond = [i for i in c.instructions() if i.c_if is not None]
    assert len(cond) == 1 and cond[0].c_if == (2, True) and cond[0].qubits == (0,)
    assert names[-1] == "reset"


@pytest.mark.parametrize(
    "bad",
    [
        "qreg q[1]; foo q[0];",
        "qreg q[1]; rz(1/0) q[0];",
        "qreg q[2]; cx q[0];",
        "qreg q[1]; h r[0];",
        "qreg q[1]; h q[3];",
        'include "other.inc";',
        "qreg q[2]; creg c[2]; if (c == 1) x q[0];",
        "OPENQASM 3.0;",
        "qreg q[1]; opaque g a; g q[0];",
    ],
)
def test_qasm_parse_errors(bad):
    with pytest.raises(ParseError) as e:
        Circuit.from_qasm(bad)
    assert "line" in str(e.value)


def unitary_of(c):
    n = c.num_qubits
    cols = []
    for j in range(2**n):
        prep = Circuit(n)
        for q in range(n):
            if (j >> q) & 1:
                prep.x(q)
        cols.append(qs.simulate(prep + c, qs.statevector()).state)
    return np.array(cols).T


def test_qasm_qelib_extras_match_their_definitions():
    # gates without an engine equivalent are expanded exactly (global phase included)
    rz = lambda t: ref.mat1("rz", (t,))
    rx = lambda t: ref.mat1("rx", (t,))
    ry = lambda t: ref.mat1("ry", (t,))
    I2 = np.eye(2)
    X, Y, Z = ref.mat1("x"), ref.mat1("y"), ref.mat1("z")

    def ctrl(u):  # first argument is the control (most significant in the matrix)
        return np.block([[I2, 0 * I2], [0 * I2, u]])

    def expm_pauli(p, t):
        return math.cos(t / 2) * np.eye(4) - 1j * math.sin(t / 2) * p

    th = 0.731
    cases = {
        f"crz({th}) q[0],q[1];": ctrl(rz(th)),
        f"cry({th}) q[0],q[1];": ctrl(ry(th)),
        f"crx({th}) q[0],q[1];": ctrl(rx(th)),
        "cy q[0],q[1];": ctrl(Y),
        "ch q[0],q[1];": ctrl(ref.mat1("h")),
        f"rzz({th}) q[0],q[1];": expm_pauli(np.kron(Z, Z), th),
        f"rxx({th}) q[0],q[1];": expm_pauli(np.kron(X, X), th),
        f"ryy({th}) q[0],q[1];": expm_pauli(np.kron(Y, Y), th),
        f"rzx({th}) q[0],q[1];": expm_pauli(np.kron(Z, X), th),
        f"cu3({th},0.2,0.3) q[0],q[1];": ctrl(ref.mat1("u", (th, 0.2, 0.3))),
        f"cu({th},0.2,0.3,0.4) q[0],q[1];": ctrl(np.exp(0.4j) * ref.mat1("u", (th, 0.2, 0.3))),
        "csx q[0],q[1];": ctrl(ref.mat1("sx")),
        f"u2(0.2,0.3) q[0];": ref.mat1("u", (math.pi / 2, 0.2, 0.3)),
    }
    for stmt, want in cases.items():
        k = 1 if stmt.startswith("u2") else 2
        c = Circuit.from_qasm(f"OPENQASM 2.0; include \"qelib1.inc\"; qreg q[{k}]; {stmt}")
        u = unitary_of(c)
        if k == 2:
            # reference matrices index the first argument (qubit 0) as most significant
            perm = [0, 2, 1, 3]
            u = u[np.ix_(perm, perm)]
        np.testing.assert_allclose(u, want, atol=1e-10, err_msg=stmt)


def test_stim_round_trip():
    src = """
    R 0 1 2
    H 0
    CX 0 1
    X_ERROR(0.01) 2
    DEPOLARIZE2(0.02) 0 1
    REPEAT 3 {
        CX 1 2
        M(0.05) 2
        DETECTOR rec[-1]
    }
    M(0.05) 0 1
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
    """
    c = Circuit.from_stim(src)
    assert c.num_measurements == 5
    assert c.readout_error == pytest.approx(0.05)
    assert c.detectors == [[0], [1], [2]]
    assert [sorted(o) for o in c.observables] == [[3, 4]]
    assert c._core.has_repeats
    again = Circuit.from_stim(c.to_stim())
    assert again.instructions() == c.instructions()
    assert again.detectors == c.detectors and again.observables == c.observables


def test_stim_parse_error():
    with pytest.raises(ParseError):
        Circuit.from_stim("FOO 0")
    with pytest.raises(UnsupportedOperationError):
        Circuit(1).t(0).to_stim()
