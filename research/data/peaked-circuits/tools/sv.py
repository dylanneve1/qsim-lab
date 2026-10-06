# Exact state-vector solve for <=30-qubit circuits (rz, sx, x, h, u3/u, cz, cx). complex64.
# Peak = argmax |amp|^2 over all 2^n basis states (exact, no heuristics). Prints top 5.
import sys, re, math, time, numpy as np
f = sys.argv[1]; src = open(f).read()
n = int(re.search(r'qreg\s+\w+\[(\d+)\]', src).group(1))
def U3(t, p, l):
    return np.array([[math.cos(t/2), -np.exp(1j*l)*math.sin(t/2)], [np.exp(1j*p)*math.sin(t/2), np.exp(1j*(p+l))*math.cos(t/2)]], dtype=np.complex128)
def num(x): return float(eval(x, {'pi': math.pi, '__builtins__': {}}))
SX = 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]]); X = np.array([[0, 1], [1, 0]], complex); H = np.array([[1, 1], [1, -1]], complex)/math.sqrt(2)
ops = []
for line in src.splitlines():
    line = line.strip()
    m = re.match(r'(\w+)(?:\(([^)]*)\))?\s+(.*);', line)
    if not m or m.group(1) in ('OPENQASM', 'include', 'qreg', 'creg', 'measure', 'barrier'): continue
    g, args, qs = m.group(1), m.group(2), [int(x) for x in re.findall(r'\[(\d+)\]', m.group(3))]
    a = [num(x) for x in args.split(',')] if args else []
    if g == 'rz': ops.append(('1', qs[0], np.diag([np.exp(-0.5j*a[0]), np.exp(0.5j*a[0])])))
    elif g == 'sx': ops.append(('1', qs[0], SX))
    elif g == 'x': ops.append(('1', qs[0], X))
    elif g == 'h': ops.append(('1', qs[0], H))
    elif g in ('u3', 'u'): ops.append(('1', qs[0], U3(*a)))
    elif g == 'ry': ops.append(('1', qs[0], U3(a[0], 0, 0)))
    elif g == 'cz': ops.append(('cz', qs, None))
    elif g == 'cx': ops.append(('cx', qs, None))
    else: raise SystemExit('unsupported gate ' + g)
# fuse consecutive 1q gates per wire
pend = {}; fused = []
for k, q, M in ops:
    if k == '1': pend[q] = M @ pend.get(q, np.eye(2))
    else:
        for w in q:
            if w in pend: fused.append(('1', w, pend.pop(w)))
        fused.append((k, q, None))
for w, M in pend.items(): fused.append(('1', w, M))
print('n', n, 'ops', len(ops), 'fused', len(fused), flush=True)
t = time.time()
psi = np.zeros(2**n, np.complex64); psi[0] = 1
T = psi.reshape((2,)*n)          # axis j <-> qubit n-1-j (qubit 0 = least significant)
ax = lambda q: n-1-q
for k, q, M in fused:
    if k == '1':
        a = ax(q); M = M.astype(np.complex64)
        V = psi.reshape(2**a, 2, 2**(n-a-1)); R = V.shape[2]
        cr = min(R, 1 << 22); cl = max(1, (1 << 22) // cr)
        for i in range(0, V.shape[0], cl):
            for j in range(0, R, cr):
                blk = V[i:i+cl, :, j:j+cr]; x0 = blk[:, 0].copy(); x1 = blk[:, 1]
                blk[:, 0] = M[0, 0]*x0 + M[0, 1]*x1
                blk[:, 1] = M[1, 0]*x0 + M[1, 1]*x1
    elif k == 'cz':
        a, b = ax(q[0]), ax(q[1]); idx = [slice(None)]*n; idx[a] = 1; idx[b] = 1; T[tuple(idx)] *= -1
    else:
        a, b = ax(q[0]), ax(q[1]); idx = [slice(None)]*n; idx[a] = 1
        sub = T[tuple(idx)]; bb = b if b < a else b-1
        sub = np.moveaxis(sub, bb, 0); tmp = sub[0].copy(); sub[0] = sub[1]; sub[1] = tmp
print('simulated in', round(time.time()-t), 's; norm', float(np.vdot(psi, psi).real), flush=True)
pr = (psi.real**2 + psi.imag**2)
top = np.argpartition(pr, -5)[-5:]; top = top[np.argsort(pr[top])[::-1]]
for i in top:
    s = format(int(i), f'0{n}b')[::-1]   # qubit 0 first
    print('  ', s, float(pr[i]))
print('PEAK (q0 first)', format(int(top[0]), f'0{n}b')[::-1], 'p', float(pr[top[0]]), 'runner-up', float(pr[top[1]]))
