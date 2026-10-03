import time, math, random, sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def qft(n, init):
    qc = QuantumCircuit(n)
    for q in range(n):
        if (init >> q) & 1: qc.x(q)
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(math.pi / (1 << (j - k)), k, j)
    for j in range(n // 2):
        qc.swap(j, n - 1 - j)
    return qc

def brick(n, depth=20, seed=7):
    rnd = random.Random(seed)
    qc = QuantumCircuit(n)
    for layer in range(depth):
        for q in range(n):
            qc.ry(rnd.random() * math.pi, q); qc.rz(rnd.random() * math.pi, q)
        for q in range(layer % 2, n - 1, 2):
            qc.cx(q, q + 1)
    return qc

for name in sys.argv[1].split(','):
    for n in [int(x) for x in sys.argv[2].split(',')]:
        for fusion in (True, False):
            qc = qft(n, 0x5A5A5A5A & ((1 << n) - 1)) if name == 'qft' else brick(n)
            sim = AerSimulator(method='statevector', precision='single', max_parallel_threads=int(__import__("os").environ.get("T","8")), fusion_enable=fusion)
            qc2 = qc.copy(); qc2.save_expectation_value  # noqa
            from qiskit.quantum_info import Pauli
            qc2.save_expectation_value(Pauli('Z' + 'I' * (n - 1)), list(range(n)))
            tqc = transpile(qc2, sim, optimization_level=0)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter(); r = sim.run(tqc, shots=1).result(); dt = time.perf_counter() - t
                best = min(best, dt)
            print(f"| aer {name} | {n} | f32 | {qc.size()} gates | fusion={fusion} | {best:.4f} s | sim-only {r.results[0].time_taken:.4f} s |", flush=True)
