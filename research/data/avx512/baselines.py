#!/usr/bin/env python3
"""Run one simulator on one gate-list circuit file and time the simulation only.

usage:
  baselines.py run <fw> <file> <c64|c128> <reps> [knob=val ...] [dump=<path.npy>]
      fw in: qsim, qulacs, qulacs_src, aer, lightning, qsimlab
      prints one JSON line per repetition: {"fw", "file", "n", "prec", "threads",
      "config", "rep", "seconds", "timer", ...}
  baselines.py info            # modules / kernels each framework dispatches to

Threads: env THREADS (default 8). The driver also sets OMP_NUM_THREADS /
RAYON_NUM_THREADS to the same value before this process starts (OpenMP reads them
at load time).

Bit layout: gate-list qubit k = bit k of the state index for every framework
(cirq: LineQubit(n-1-k) with qubit_order LineQubit.range(n), because qsim numbers
qubits in reverse; PennyLane/lightning: wire n-1-k, because wire 0 is the most
significant bit). `dump=` writes the final state (complex128, that layout).

Timers (what "seconds" means; `timer` field):
  qsim       "qsim_fuse+simu": qsim's own C++ timers (verbosity=2: fusion + gate
             application; excludes state allocation/initialisation and Python);
             also reported: "call" = wall time of qsim_simulate_fullstate after
             translation (incl. allocation + zeroing of the state), "init".
  qulacs     "update": wall time of circuit.update_quantum_state(state) on a
             preallocated |0> state (optimizer passes, if any, run before timing).
             `qulacs` = the pip wheel (no AVX2/AVX-512 code in qulacs_core: 0 ymm/zmm
             instructions); `qulacs_src` = the same version built from source with
             -march=native (AVX2 SIMD on), as the qulacs README recommends.
  aer        "time_taken": result.results[0].time_taken (experiment time inside
             Aer's C++, after transpile; includes state allocation/initialisation);
             also "run_wall" (wall of backend.run().result()). enable_truncation=False
             (otherwise Aer simulates only the qubits it deems relevant to the saved
             data); "sim_qubits" records the simulated width as a check.
  lightning  "apply": wall time of the loop of C++ gate calls on a preallocated
             |0> StateVectorC64/C128 (Python call overhead ~1 us/gate included).
  qsimlab    "apply": wall time of apply_gates_blocked (examples/sv_file_bench.rs:
             lowering, fusion, planning and execution on a preallocated, touched state).
"""
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_circuits import read, u3_mat  # noqa: E402

THREADS = int(os.environ.get("THREADS", "8"))
QSIMLAB_BIN = os.environ.get("QSIMLAB_BIN", "/dev/shm/qsim/avx512/bin/sv_file_bench-main")


def emit(d):
    print(json.dumps(d), flush=True)


def parse_knobs(args):
    kn, dump = {}, None
    for a in args:
        k, v = a.split("=", 1)
        if k == "dump":
            dump = v
        else:
            kn[k] = v
    return kn, dump


# ----------------------------------------------------------------------------- qsim

def run_qsim(path, prec, reps, kn, dump):
    assert prec == "c64", "qsim is single precision only"
    import cirq
    import qsimcirq
    import qsimcirq.qsim_circuit as qsimc

    n, gates = read(path)
    qs = [cirq.LineQubit(n - 1 - k) for k in range(n)]
    ops = []
    for name, q, p in gates:
        if name == "h":
            ops.append(cirq.H(qs[q[0]]))
        elif name == "x":
            ops.append(cirq.X(qs[q[0]]))
        elif name == "rx":
            ops.append(cirq.rx(p[0])(qs[q[0]]))
        elif name == "ry":
            ops.append(cirq.ry(p[0])(qs[q[0]]))
        elif name == "rz":
            ops.append(cirq.rz(p[0])(qs[q[0]]))
        elif name == "u3":
            ops.append(cirq.MatrixGate(u3_mat(*p)).on(qs[q[0]]))
        elif name == "cx":
            ops.append(cirq.CNOT(qs[q[0]], qs[q[1]]))
        elif name == "cz":
            ops.append(cirq.CZ(qs[q[0]], qs[q[1]]))
        elif name == "cp":
            ops.append(cirq.CZPowGate(exponent=p[0] / math.pi)(qs[q[0]], qs[q[1]]))
        elif name == "swap":
            ops.append(cirq.SWAP(qs[q[0]], qs[q[1]]))
        elif name == "rzz":
            ops.append(cirq.ZZPowGate(exponent=p[0] / math.pi, global_shift=-0.5)(qs[q[0]], qs[q[1]]))
        elif name == "u4":
            # cirq is big-endian over the operand list: (b, a) gives index 2*bit_b + bit_a
            ops.append(cirq.MatrixGate(p[0]).on(qs[q[1]], qs[q[0]]))
        else:
            raise ValueError(name)
    circuit = qsimc.QSimCircuit(cirq.Circuit(ops))
    f = int(kn.get("f", 3))
    sim = qsimcirq.QSimSimulator(qsim_options=qsimcirq.QSimOptions(
        cpu_threads=THREADS, max_fused_gate_size=f, verbosity=2))
    order = cirq.LineQubit.range(n)
    options = dict(sim.qsim_options)
    options["c"], _ = sim._translate_circuit(circuit, "translate_cirq_to_qsim", order)
    options["s"] = sim.get_seed()
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    sys.stdout.flush()
    # capture C++ stdout (timers) per call via a pipe on fd 1
    for rep in range(reps):
        r, w = os.pipe()
        saved = os.dup(1)
        os.dup2(w, 1)
        t0 = time.perf_counter()
        st = sim._sim_module.qsim_simulate_fullstate(options, 0)
        call = time.perf_counter() - t0
        libc.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(w)
        out = os.read(r, 1 << 20).decode()
        os.close(r)
        tm = {k: float(v) for k, v in re.findall(r"(\w+) time is ([0-9.eE+-]+) seconds", out)}
        sec = tm.get("fuse", 0.0) + tm.get("simu", float("nan"))
        emit(dict(fw="qsim", file=os.path.basename(path), n=n, prec=prec, threads=THREADS,
                  config=f"f={f}", rep=rep, seconds=sec, timer="qsim_fuse+simu",
                  call=call, init=tm.get("init"), fuse=tm.get("fuse"), simu=tm.get("simu"),
                  module=sim._sim_module.__name__))
        if dump and rep == reps - 1:
            np.save(dump, st.view(np.complex64).astype(np.complex128))
        del st


# ----------------------------------------------------------------------------- qulacs

QULACS_SRC = os.environ.get("QULACS_SRC", "/dev/shm/qsim/ext/qulacs-site")


def run_qulacs_src(path, prec, reps, kn, dump):
    """qulacs 0.6.14 built from source with AVX2 (-march=native), as its README recommends
    for best performance; installed privately (pip --target) so the shared venv is untouched."""
    sys.path.insert(0, QULACS_SRC)
    run_qulacs(path, prec, reps, kn, dump, fw="qulacs_src")


def run_qulacs(path, prec, reps, kn, dump, fw="qulacs"):
    assert prec == "c128", "qulacs is double precision only"
    import qulacs
    from qulacs import QuantumCircuit, QuantumState
    from qulacs import gate as G

    n, gates = read(path)
    c = QuantumCircuit(n)
    for name, q, p in gates:
        if name == "h":
            c.add_gate(G.H(q[0]))
        elif name == "x":
            c.add_gate(G.X(q[0]))
        elif name == "rx":
            c.add_gate(G.RotX(q[0], p[0]))
        elif name == "ry":
            c.add_gate(G.RotY(q[0], p[0]))
        elif name == "rz":
            c.add_gate(G.RotZ(q[0], p[0]))
        elif name == "u3":
            c.add_gate(G.DenseMatrix(q[0], u3_mat(*p)))
        elif name == "cx":
            c.add_gate(G.CNOT(q[0], q[1]))
        elif name == "cz":
            c.add_gate(G.CZ(q[0], q[1]))
        elif name == "cp":
            g = G.DenseMatrix(q[1], np.diag([1, np.exp(1j * p[0])]))
            g.add_control_qubit(q[0], 1)
            c.add_gate(g)
        elif name == "swap":
            c.add_gate(G.SWAP(q[0], q[1]))
        elif name == "rzz":
            # qulacs PauliRotation(angle) = exp(+i angle/2 P)
            c.add_gate(G.PauliRotation([q[0], q[1]], [3, 3], -p[0]))
        elif name == "u4":
            # qulacs: first target = least significant local bit
            c.add_gate(G.DenseMatrix([q[0], q[1]], p[0]))
        else:
            raise ValueError(name)
    opt = kn.get("opt", "none")
    t0 = time.perf_counter()
    if opt == "light":
        qulacs.circuit.QuantumCircuitOptimizer().optimize_light(c)
    elif opt.startswith("block"):
        qulacs.circuit.QuantumCircuitOptimizer().optimize(c, int(opt[5:]))
    topt = time.perf_counter() - t0
    st = QuantumState(n)
    for rep in range(reps):
        st.set_zero_state()
        t0 = time.perf_counter()
        c.update_quantum_state(st)
        sec = time.perf_counter() - t0
        emit(dict(fw=fw, file=os.path.basename(path), n=n, prec=prec, threads=THREADS,
                  config=f"opt={opt}", rep=rep, seconds=sec, timer="update",
                  optimizer_seconds=topt, gates=c.get_gate_count(),
                  module=os.path.dirname(qulacs.__file__)))
    if dump:
        np.save(dump, np.asarray(st.get_vector(), dtype=np.complex128))


# ----------------------------------------------------------------------------- aer

def run_aer(path, prec, reps, kn, dump):
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator

    n, gates = read(path)
    qc = QuantumCircuit(n)
    for name, q, p in gates:
        if name == "h":
            qc.h(q[0])
        elif name == "x":
            qc.x(q[0])
        elif name == "rx":
            qc.rx(p[0], q[0])
        elif name == "ry":
            qc.ry(p[0], q[0])
        elif name == "rz":
            qc.rz(p[0], q[0])
        elif name == "u3":
            qc.u(p[0], p[1], p[2], q[0])
        elif name == "cx":
            qc.cx(q[0], q[1])
        elif name == "cz":
            qc.cz(q[0], q[1])
        elif name == "cp":
            qc.cp(p[0], q[0], q[1])
        elif name == "swap":
            qc.swap(q[0], q[1])
        elif name == "rzz":
            qc.rzz(p[0], q[0], q[1])
        elif name == "u4":
            qc.unitary(p[0], [q[0], q[1]])  # qiskit: first qubit = least significant
        else:
            raise ValueError(name)
    if dump:
        qc.save_statevector()
    else:
        qc.save_expectation_value(Pauli("Z"), [0])
    fusion = kn.get("fusion", "1") == "1"
    # enable_truncation=False: Aer otherwise drops qubits it deems irrelevant to the saved
    # data (with only <Z0> saved it simulated 2 of 16 qubits in a test), which would time a
    # smaller circuit
    opts = dict(method="statevector", precision="single" if prec == "c64" else "double",
                max_parallel_threads=THREADS, fusion_enable=fusion, enable_truncation=False)
    if "fmax" in kn:
        opts["fusion_max_qubit"] = int(kn["fmax"])
    if "fthr" in kn:
        opts["fusion_threshold"] = int(kn["fthr"])
    if "block" in kn:
        opts["blocking_enable"] = True
        opts["blocking_qubits"] = int(kn["block"])
    sim = AerSimulator(**opts)
    tqc = transpile(qc, sim, optimization_level=0)
    cfg = ",".join(f"{k}={v}" for k, v in sorted(kn.items())) or "fusion=1"
    for rep in range(reps):
        t0 = time.perf_counter()
        res = sim.run(tqc, shots=1).result()
        wall = time.perf_counter() - t0
        r0 = res.results[0]
        md = r0.metadata or {}
        emit(dict(fw="aer", file=os.path.basename(path), n=n, prec=prec, threads=THREADS,
                  config=cfg, rep=rep, seconds=float(r0.time_taken), timer="time_taken",
                  run_wall=wall, success=bool(r0.success),
                  fusion=str(md.get("fusion", {}).get("applied", ""))
                  if isinstance(md.get("fusion"), dict) else "",
                  parallel_state_update=md.get("parallel_state_update"),
                  sim_qubits=md.get("num_qubits"),
                  status=str(r0.status)[:80]))
    if dump:
        np.save(dump, np.asarray(res.get_statevector(), dtype=np.complex128))


# ----------------------------------------------------------------------------- lightning

def run_lightning(path, prec, reps, kn, dump):
    import pennylane_lightning.lightning_qubit_ops as L

    n, gates = read(path)
    cdt = np.complex64 if prec == "c64" else np.complex128
    sv = (L.StateVectorC64 if prec == "c64" else L.StateVectorC128)(n)
    W = lambda k: n - 1 - k
    prog = []  # (callable, args)
    for name, q, p in gates:
        if name == "h":
            prog.append((sv.Hadamard, ([W(q[0])], False, [])))
        elif name == "x":
            prog.append((sv.PauliX, ([W(q[0])], False, [])))
        elif name == "rx":
            prog.append((sv.RX, ([W(q[0])], False, [p[0]])))
        elif name == "ry":
            prog.append((sv.RY, ([W(q[0])], False, [p[0]])))
        elif name == "rz":
            prog.append((sv.RZ, ([W(q[0])], False, [p[0]])))
        elif name == "u3":
            prog.append((sv.applyMatrix, (np.ascontiguousarray(u3_mat(*p), dtype=cdt).ravel(),
                                          [W(q[0])], False)))
        elif name == "cx":
            prog.append((sv.CNOT, ([W(q[0]), W(q[1])], False, [])))
        elif name == "cz":
            prog.append((sv.CZ, ([W(q[0]), W(q[1])], False, [])))
        elif name == "cp":
            prog.append((sv.ControlledPhaseShift, ([W(q[0]), W(q[1])], False, [p[0]])))
        elif name == "swap":
            prog.append((sv.SWAP, ([W(q[0]), W(q[1])], False, [])))
        elif name == "rzz":
            prog.append((sv.IsingZZ, ([W(q[0]), W(q[1])], False, [p[0]])))
        elif name == "u4":
            # big-endian over the wire list: [wire(b), wire(a)] gives 2*bit_b + bit_a
            prog.append((sv.applyMatrix, (np.ascontiguousarray(p[0], dtype=cdt).ravel(),
                                          [W(q[1]), W(q[0])], False)))
        else:
            raise ValueError(name)
    km = sv.kernel_map()
    used = sorted({km.get(k, "?") for k in ("CNOT", "CZ", "SingleQubitOp", "TwoQubitOp",
                                            "IsingZZ", "RX", "Hadamard", "SWAP",
                                            "ControlledPhaseShift")})
    for rep in range(reps):
        sv.resetStateVector()
        t0 = time.perf_counter()
        for f, a in prog:
            f(*a)
        sec = time.perf_counter() - t0
        emit(dict(fw="lightning", file=os.path.basename(path), n=n, prec=prec, threads=THREADS,
                  config="default", rep=rep, seconds=sec, timer="apply",
                  kernels="/".join(used)))
    if dump:
        out = np.zeros(1 << n, dtype=cdt)
        sv.getState(out)
        np.save(dump, out.astype(np.complex128))


# ----------------------------------------------------------------------------- qsim-lab

def run_qsimlab(path, prec, reps, kn, dump):
    prec_rs = "f32" if prec == "c64" else "f64"
    args = [QSIMLAB_BIN, path, prec_rs, str(reps)] + [f"{k}={v}" for k, v in kn.items()]
    raw = None
    if dump:
        raw = dump + ".raw"
        args.append(f"dump={raw}")
    env = dict(os.environ, RAYON_NUM_THREADS=str(THREADS))
    out = subprocess.run(args, env=env, capture_output=True, text=True, check=True).stdout
    # | name | n | prec | threads | mode | min | median | all... | norm |
    line = [l for l in out.splitlines() if l.startswith("|")][-1]
    cells = [c.strip() for c in line.strip("|").split("|")]
    n = int(cells[1])
    times = [float(x) for x in cells[7].split()]
    cfg = ",".join(f"{k}={v}" for k, v in sorted(kn.items())) or "default"
    for rep, t in enumerate(times):
        emit(dict(fw="qsimlab", file=os.path.basename(path), n=n, prec=prec, threads=THREADS,
                  config=cfg, rep=rep, seconds=t, timer="apply",
                  binary=os.path.basename(QSIMLAB_BIN), norm=cells[8]))
    if dump:
        v = np.fromfile(raw, dtype=np.float64)
        np.save(dump, v[0::2] + 1j * v[1::2])
        os.remove(raw)


RUNNERS = dict(qsim=run_qsim, qulacs=run_qulacs, qulacs_src=run_qulacs_src, aer=run_aer,
               lightning=run_lightning, qsimlab=run_qsimlab)


def info():
    import qsimcirq
    from qsimcirq import qsim_decide
    print("qsim module:", qsimcirq.qsim.__name__, "detect_instructions:",
          qsim_decide.detect_instructions(), "(0 = AVX512)")
    import pennylane_lightning.lightning_qubit_ops as L
    print("lightning compile:", L.compile_info(), "runtime:", L.runtime_info())
    for n in (12, 24, 28):
        sv = L.StateVectorC64(n)
        km = sv.kernel_map()
        print(f"lightning kernel map n={n}:", {k: km[k] for k in sorted(km) if km[k] != "LM"}, "(others LM)")
        del sv
    import qiskit_aer
    import qulacs
    print("qiskit-aer", qiskit_aer.__version__, "qulacs", qulacs.__version__ if hasattr(qulacs, "__version__") else "?")


if __name__ == "__main__":
    if sys.argv[1] == "info":
        info()
    elif sys.argv[1] == "run":
        fw, path, prec, reps = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        kn, dump = parse_knobs(sys.argv[6:])
        RUNNERS[fw](path, prec, reps, kn, dump)
    else:
        raise SystemExit(__doc__)
