#!/usr/bin/env python3
"""Sycamore-pattern random circuits (53 qubits, plus a 12-qubit validation sub-grid) as plain text.

Usage (shared venv, cirq-core 1.5):

    /dev/shm/qsim/venv/bin/python sycamore.py --out circuits

writes, into the --out directory,

    syc53_m{m}_s{seed}.txt        m in {10, 12, 14, 16, 18, 20}, seed in {0, 1}
    sycsub12_m8_s{seed}.txt       12-qubit 3x4 block (rows 3-5, cols 3-6), m = 8, seed in {0, 1}
    sycsub12_m8_s{seed}.amps.json exact amplitudes of the 12-qubit circuits from cirq (complex128)

Qubits. cirq_google's 54-qubit Sycamore grid (rows top to bottom, '-' = no qubit):

    -----AB---
    ----ABCD--
    ---ABCDEF-
    --ABCDEFGH
    -ABCDEFGHI
    ABCDEFGHI-
    -CDEFGHI--
    --EFGHI---
    ---GHI----
    ----I-----

minus GridQubit(2, 3), the qubit that was inoperable in the 2019 supremacy experiment (Arute et al.,
Nature 574, 505). That identification is inferred, not quoted: Google's published m=10 circuit
(circuit_n53_m10_s0_e0_pABCDCDAB.qsim, Dryad dataset, mirrored in cotengra/examples) numbers its 53
qubits 0..52, and of the 54 possible single-qubit removals only (2, 3), with the remaining qubits
numbered in sorted (row, col) order, reproduces its set of 86 fSim couplers exactly. Our qubits are
numbered 0..52 in sorted (row, col) order as well.

Circuit. cirq.experiments.random_rotations_between_grid_interaction_layers_circuit with depth m
(m cycles of [random single-qubit layer, fSim layer] plus a final single-qubit layer, so 2m+1
moments), two-qubit gate fSim(theta=pi/2, phi=pi/6) on every active coupler, pattern
GRID_STAGGERED_PATTERN (ABCDCDAB), single-qubit gates drawn from {X**0.5, Y**0.5,
PhasedXPowGate(phase_exponent=0.25, exponent=0.5)} with no qubit repeating its previous gate, seeded
by `seed` (numpy RandomState). These are the idealised supremacy circuits: no per-coupler
calibrated fSim angles and no Rz phase corrections (Google's "e0" files carry both).

File format: one op per line, whitespace separated; '#' starts a comment (anywhere on a line).

    # syc53 m=14 seed=0 pattern=ABCDCDAB fsim(pi/2, pi/6) removed=(2,3)
    qubits 53
    # map <index> <row> <col>          (one comment line per qubit)
    # moment <k>                       (comment before each moment's ops)
    sx <q>
    sy <q>
    sw <q>
    fsim <a> <b> <theta> <phi>         (floats printed with repr(), full precision)

Ops are listed moment by moment (all ops of moment k, then moment k+1). The circuit acts on
|0...0>. Gate matrices (for fsim the two-qubit basis index is 2*bit(a) + bit(b): the first listed
qubit is the more significant one); every gate emitted is checked against cirq.unitary to 1e-12
(global phase included) before a file is written:

    sx = X**0.5                 = [[1+i, 1-i], [1-i, 1+i]] / 2
    sy = Y**0.5                 = [[1+i, -1-i], [1+i, 1+i]] / 2
    sw = PhasedXPowGate(0.25, 0.5) = [[(1+i)/2, -i/sqrt2], [1/sqrt2, (1+i)/2]]
    fsim(t, p)                  = [[1, 0, 0, 0], [0, cos t, -i sin t, 0],
                                   [0, -i sin t, cos t, 0], [0, 0, 0, exp(-i p)]]

Amplitude JSON (12-qubit circuits only): basis index convention is "qubit q = bit q of the index"
(qubit 0 least significant), so index = sum_q b_q 2^q; the "bits_q0_first" strings list b_0 first.
Amplitudes are <x|C|0...0> from cirq.final_state_vector (complex128).
"""

import argparse
import json
import os
import sys

import cirq
import numpy as np

GRID = """\
-----AB---
----ABCD--
---ABCDEF-
--ABCDEFGH
-ABCDEFGHI
ABCDEFGHI-
-CDEFGHI--
--EFGHI---
---GHI----
----I-----
"""

REMOVED = (2, 3)
THETA = np.pi / 2
PHI = np.pi / 6
DEPTHS = (10, 12, 14, 16, 18, 20)
SEEDS = (0, 1)
SUB_BLOCK = [(r, c) for r in range(3, 6) for c in range(3, 7)]  # 3x4 block, 12 qubits
SUB_DEPTH = 8
SUB_SEEDS = (0, 1)

S2 = 1 / np.sqrt(2)
MATRICES_1Q = {
    "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    "sy": np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2,
    "sw": np.array([[(1 + 1j) / 2, -1j * S2], [S2, (1 + 1j) / 2]]),
}
GATES_1Q = {
    "sx": cirq.X**0.5,
    "sy": cirq.Y**0.5,
    "sw": cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
}


def fsim_matrix(theta, phi):
    """fSim(theta, phi) in the basis index 2*bit(a) + bit(b)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]],
        dtype=complex,
    )


def gate_matrix(name, params=()):
    """Matrix of a file op by name (used by the readers too)."""
    if name == "fsim":
        return fsim_matrix(*params)
    return MATRICES_1Q[name]


def grid_qubits():
    """The 54 Sycamore grid sites as sorted (row, col) tuples."""
    sites = [(r, c) for r, line in enumerate(GRID.splitlines()) for c, ch in enumerate(line) if ch != "-"]
    assert len(sites) == 54
    return sorted(sites)


def syc53_qubits():
    """The 53 qubits (54-site grid minus REMOVED), sorted by (row, col); index = position."""
    qs = [q for q in grid_qubits() if q != REMOVED]
    assert len(qs) == 53
    return qs


def make_circuit(sites, depth, seed):
    """The cirq circuit on the given (row, col) sites (order fixes the RNG stream)."""
    qubits = [cirq.GridQubit(r, c) for (r, c) in sites]
    return cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits,
        depth=depth,
        two_qubit_op_factory=lambda a, b, _: cirq.FSimGate(theta=THETA, phi=PHI)(a, b),
        pattern=cirq.experiments.GRID_STAGGERED_PATTERN,
        single_qubit_gates=(
            cirq.X**0.5,
            cirq.Y**0.5,
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
        ),
        add_final_single_qubit_layer=True,
        seed=seed,
    ), qubits


def circuit_ops(circuit, qubits):
    """Translate to [(moment, name, qubit indices, params)], checking every matrix against cirq."""
    index = {q: i for i, q in enumerate(qubits)}
    out = []
    for k, moment in enumerate(circuit.moments):
        for op in moment.operations:
            g = op.gate
            if isinstance(g, cirq.FSimGate):
                name, params = "fsim", (float(g.theta), float(g.phi))
            else:
                matches = [n for n, ref in GATES_1Q.items() if g == ref]
                if len(matches) != 1:
                    raise ValueError(f"unexpected gate {g!r}")
                name, params = matches[0], ()
            ours = gate_matrix(name, params)
            theirs = cirq.unitary(op.gate)
            if ours.shape != theirs.shape or not np.allclose(ours, theirs, rtol=0, atol=1e-12):
                raise AssertionError(f"matrix mismatch for {name}{params}: {ours} vs {theirs}")
            out.append((k, name, tuple(index[q] for q in op.qubits), params))
    return out


def write_circuit(path, header, sites, ops):
    lines = [f"# {header}", f"qubits {len(sites)}"]
    lines += [f"# map {i} {r} {c}" for i, (r, c) in enumerate(sites)]
    last = None
    for k, name, qs, params in ops:
        if k != last:
            lines.append(f"# moment {k}")
            last = k
        fields = [name] + [str(q) for q in qs] + [repr(float(p)) for p in params]
        lines.append(" ".join(fields))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_circuit(path):
    """Parse a circuit file -> (n_qubits, [(name, (qubits...), (params...))]). Shared by the readers."""
    n = None
    ops = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            t = line.split()
            if t[0] == "qubits":
                n = int(t[1])
            elif t[0] in ("sx", "sy", "sw"):
                ops.append((t[0], (int(t[1]),), ()))
            elif t[0] == "fsim":
                ops.append(("fsim", (int(t[1]), int(t[2])), (float(t[3]), float(t[4]))))
            else:
                raise ValueError(f"{path}: unknown op {t[0]!r}")
    assert n is not None, f"{path}: missing 'qubits' line"
    return n, ops


def check_roundtrip(path, ops):
    """Re-read a written file and check it reproduces the op list bit-exactly."""
    n, parsed = read_circuit(path)
    assert [(name, qs, params) for _, name, qs, params in ops] == parsed, path


def amplitudes_json(circuit, qubits, sites, header):
    """Exact amplitudes <x|C|0..0> (cirq, complex128) in the 'qubit q = bit q' index convention."""
    psi = cirq.final_state_vector(circuit, qubit_order=qubits, dtype=np.complex128)
    n = len(qubits)
    # cirq: qubit_order[0] is the MOST significant bit. Convert to qubit 0 least significant.
    psi_lsb = psi.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)

    def entry(idx):
        bits = "".join(str((idx >> q) & 1) for q in range(n))
        a = complex(psi_lsb[idx])
        return {"index": idx, "bits_q0_first": bits, "re": a.real, "im": a.imag}

    rng = np.random.RandomState(12345)
    extra = [1, 1 << (n - 1), 0b101] + [int(i) for i in rng.randint(0, 2**n, size=5)]
    return {
        "circuit": header,
        "qubits": n,
        "convention": (
            "basis index = sum_q b_q * 2^q, i.e. qubit q is bit q of the index (qubit 0 least "
            "significant); bits_q0_first lists b_0 b_1 ... b_{n-1}; amplitude = <x|C|0...0>"
        ),
        "source": f"cirq {cirq.__version__} final_state_vector, dtype complex128",
        "zeros": entry(0),
        "ones": entry(2**n - 1),
        "extra": [entry(i) for i in extra],
        "norm2": float(np.vdot(psi, psi).real),
        "sum_abs2_check": float(np.sum(np.abs(psi_lsb) ** 2)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="circuits", help="output directory")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    sites = syc53_qubits()
    for m in DEPTHS:
        for seed in SEEDS:
            circuit, qubits = make_circuit(sites, m, seed)
            ops = circuit_ops(circuit, qubits)
            header = (
                f"syc53 m={m} seed={seed} pattern=ABCDCDAB fsim(pi/2, pi/6) "
                f"removed=({REMOVED[0]},{REMOVED[1]}) moments={len(circuit.moments)} cirq={cirq.__version__}"
            )
            path = os.path.join(args.out, f"syc53_m{m}_s{seed}.txt")
            write_circuit(path, header, sites, ops)
            check_roundtrip(path, ops)
            n2 = sum(1 for o in ops if o[1] == "fsim")
            print(f"{path}: {len(ops)} ops ({n2} fsim), {len(circuit.moments)} moments")

    grid = set(grid_qubits())
    assert all(s in grid and s != REMOVED for s in SUB_BLOCK)
    sub = sorted(SUB_BLOCK)
    for seed in SUB_SEEDS:
        circuit, qubits = make_circuit(sub, SUB_DEPTH, seed)
        ops = circuit_ops(circuit, qubits)
        header = (
            f"sycsub12 m={SUB_DEPTH} seed={seed} pattern=ABCDCDAB fsim(pi/2, pi/6) "
            f"block=rows3-5,cols3-6 moments={len(circuit.moments)} cirq={cirq.__version__}"
        )
        stem = os.path.join(args.out, f"sycsub12_m{SUB_DEPTH}_s{seed}")
        write_circuit(stem + ".txt", header, sub, ops)
        check_roundtrip(stem + ".txt", ops)
        amps = amplitudes_json(circuit, qubits, sub, header)
        with open(stem + ".amps.json", "w") as f:
            json.dump(amps, f, indent=1)
            f.write("\n")
        z, o = amps["zeros"], amps["ones"]
        print(
            f"{stem}.txt: {len(ops)} ops; <0|C|0> = {z['re']!r} + {z['im']!r}j, "
            f"<1..1|C|0> = {o['re']!r} + {o['im']!r}j"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
