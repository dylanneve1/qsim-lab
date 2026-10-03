import time, math, random, sys
import cirq, qsimcirq

def qft(n, init):
    q = cirq.LineQubit.range(n); ops = []
    for i in range(n):
        if (init >> i) & 1: ops.append(cirq.X(q[i]))
    for j in reversed(range(n)):
        ops.append(cirq.H(q[j]))
        for k in reversed(range(j)):
            ops.append(cirq.CZPowGate(exponent=1.0 / (1 << (j - k))).on(q[k], q[j]))
    for j in range(n // 2):
        ops.append(cirq.SWAP(q[j], q[n - 1 - j]))
    return cirq.Circuit(ops), q

def brick(n, depth=20, seed=7):
    rnd = random.Random(seed); q = cirq.LineQubit.range(n); ops = []
    for layer in range(depth):
        for i in range(n):
            ops.append(cirq.ry(rnd.random() * math.pi)(q[i])); ops.append(cirq.rz(rnd.random() * math.pi)(q[i]))
        for i in range(layer % 2, n - 1, 2):
            ops.append(cirq.CNOT(q[i], q[i + 1]))
    return cirq.Circuit(ops), q

for name in sys.argv[1].split(','):
    for n in [int(x) for x in sys.argv[2].split(',')]:
        c, q = qft(n, 0x5A5A5A5A & ((1 << n) - 1)) if name == 'qft' else brick(n)
        for f in (2, 3, 4):
            opts = qsimcirq.QSimOptions(cpu_threads=int(__import__("os").environ.get("T","8")), max_fused_gate_size=f, verbosity=0)
            sim = qsimcirq.QSimSimulator(qsim_options=opts)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter(); sim.simulate(c, qubit_order=q); best = min(best, time.perf_counter() - t)
            print(f"| qsim {name} | {n} | f32 | {len(list(c.all_operations()))} ops | fuse={f} | {best:.4f} s |", flush=True)
