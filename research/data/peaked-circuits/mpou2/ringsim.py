"""P6 independent check: exact ring-block state simulation with low-rank cross-ring coupling (our method).

The wires split into three groups (rings) found from the circuit itself: the most frequent wire pairs (one per wire)
form disjoint cycles.  The state is kept exactly inside each ring and with a small Schmidt rank between rings, as a
3-block chain  psi = sum_{k1,k2} T0[k1, x0] T1[k1, k2, x1] T2[k2, x2]  (x_r = full 2^{n_r} basis of ring r).
Ring-internal gates act directly on the ring block (exact).  Cross-ring gates are applied through their operator
Schmidt decomposition G = sum_s L_s (x) R_s (rank <= 4), then the two inter-ring bonds are recompressed exactly via
Gram matrices (singular values below tol * s_max dropped).  Peak = signs of the exact single-qubit marginals; the
peak and all single flips get exact amplitudes.
usage: ringsim.py QASM [tol] [out.json]"""
import sys, time, json, collections, hashlib, os, resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gparse as G

qasm = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-10
out = sys.argv[3] if len(sys.argv) > 3 else None
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.eye(4)[[0, 2, 1, 3]].astype(complex)
t0 = time.time()
n, units, tail = G.parse(qasm)
# blocks (consecutive same-pair units merged), tails folded into the last block of the wire
last = {}; blocks = []
for k, (a, b, Pa, Pb) in enumerate(units):
    M = CZ @ np.kron(Pa, Pb)
    ba, bb = last.get(a), last.get(b)
    if ba is not None and ba == bb:
        if (blocks[ba][0], blocks[ba][1]) == (b, a):
            M = SWAP @ M @ SWAP
        blocks[ba][2] = M @ blocks[ba][2]
    else:
        blocks.append([a, b, M]); last[a] = last[b] = len(blocks) - 1
lone = {}
for w, T in enumerate(tail):
    if w in last:
        bi = last[w]; a, b = blocks[bi][0], blocks[bi][1]
        blocks[bi][2] = (np.kron(T, np.eye(2)) if w == a else np.kron(np.eye(2), T)) @ blocks[bi][2]
    else:
        lone[w] = T
# rings: most frequent pair per wire -> cycles
cnt = collections.Counter(frozenset((a, b)) for a, b, _ in blocks)
deg = collections.Counter(); heavy = []
for pr, c in cnt.most_common():
    a, b = tuple(pr)
    if deg[a] < 2 and deg[b] < 2:
        heavy.append(pr); deg[a] += 1; deg[b] += 1
par = list(range(n))
def f(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for pr in heavy:
    a, b = tuple(pr); par[f(a)] = f(b)
comps = collections.defaultdict(list)
for q in range(n):
    comps[f(q)].append(q)
groups = sorted(comps.values(), key=len)
print('groups', [len(g) for g in groups], flush=True)
assert len(groups) == 3, 'expected three rings'
gid = {q: i for i, g in enumerate(groups) for q in g}
xc = collections.Counter()
for a, b, _ in blocks:
    if gid[a] != gid[b]:
        xc[frozenset((gid[a], gid[b]))] += 1
print('cross-group blocks', {tuple(sorted(k)): v for k, v in xc.items()}, flush=True)
# chain order: the pair with the fewest cross blocks is the non-adjacent (end) pair
if os.environ.get('CHAIN') == 'size':      # smallest group in the middle (memory)
    mid = min(range(3), key=lambda i: (len(groups[i]), i)); e0, e1 = sorted({0, 1, 2} - {mid})
else:
    ends = min(xc, key=lambda k: xc[k]) if len(xc) == 3 else frozenset((0, 2))
    mid = ({0, 1, 2} - set(ends)).pop(); e0, e1 = sorted(ends)
order = [e0, mid, e1]
G3 = [groups[i] for i in order]
pos = {}
for r, g in enumerate(G3):
    for j, q in enumerate(g):
        pos[q] = (r, j)
nr = [len(g) for g in G3]
print('chain', nr, flush=True)

DT = np.complex64 if os.environ.get('C64') else complex
T = [np.zeros((1, 2 ** nr[0]), DT), np.zeros((1, 1, 2 ** nr[1]), DT), np.zeros((1, 2 ** nr[2]), DT)]
T[0][0, 0] = 1; T[1][0, 0, 0] = 1; T[2][0, 0] = 1
lognorm = 0.0


def apply1(X, r, j, M):
    """M (2x2) on qubit j of ring r; X has leading bond axes then 2^{nr} (qubit 0 most significant)."""
    lead = X.shape[:-1]; m = nr[r]
    Y = X.reshape(lead + (2 ** j, 2, 2 ** (m - j - 1)))
    Y = np.einsum('ab,...xbz->...xaz', M.astype(X.dtype), Y)
    return Y.reshape(lead + (2 ** m,))


def apply2(X, r, j1, j2, M):
    lead = X.shape[:-1]; m = nr[r]
    if j1 > j2:
        j1, j2 = j2, j1; M = SWAP @ M @ SWAP
    Y = X.reshape(lead + (2 ** j1, 2, 2 ** (j2 - j1 - 1), 2, 2 ** (m - j2 - 1)))
    M4 = M.reshape(2, 2, 2, 2).astype(X.dtype)
    Y = np.einsum('abcd,...xcydz->...xaybz', M4, Y, optimize=True)
    return Y.reshape(lead + (2 ** m,))


def osr(M):
    """M on kron(a, b) = sum_s L_s (x) R_s."""
    R4 = M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)     # (a_out a_in) x (b_out b_in)
    U, s, Vh = np.linalg.svd(R4)
    r = int(np.sum(s > 1e-12 * s[0]))
    return [(U[:, i].reshape(2, 2) * s[i], Vh[i].reshape(2, 2)) for i in range(r)]


def apply_on(X, r, j, M, axis_mode):
    return apply1(X, r, j, M)


def cross(a, b, M):
    (ra, ja), (rb, jb) = pos[a], pos[b]
    terms = osr(M)
    if ra > rb:
        terms = [(R, L) for L, R in terms]; (ra, ja), (rb, jb) = (rb, jb), (ra, ja)
    S = len(terms)
    if (ra, rb) == (0, 1):
        A = np.stack([apply1(T[0], 0, ja, L) for L, R in terms], axis=1)          # (k1, s, x0)
        T[0] = A.reshape(-1, A.shape[-1])
        B = np.stack([apply1(T[1], 1, jb, R) for L, R in terms], axis=1)          # (k1, s, k2, x1)
        T[1] = B.reshape(-1, B.shape[2], B.shape[3])
    elif (ra, rb) == (1, 2):
        B = np.stack([apply1(T[1], 1, ja, L) for L, R in terms], axis=2)          # (k1, k2, s, x1)
        T[1] = B.reshape(B.shape[0], -1, B.shape[3])
        C = np.stack([apply1(T[2], 2, jb, R) for L, R in terms], axis=1)          # (k2, s, x2)
        T[2] = C.reshape(-1, C.shape[-1])
    else:   # (0, 2): carry s through the middle block
        A = np.stack([apply1(T[0], 0, ja, L) for L, R in terms], axis=1)
        T[0] = A.reshape(-1, A.shape[-1])
        k1, k2, x1 = T[1].shape
        B = np.einsum('ijx,st->isjtx', T[1], np.eye(S)).reshape(k1 * S, k2 * S, x1)
        T[1] = B
        C = np.stack([apply1(T[2], 2, jb, R) for L, R in terms], axis=1)
        T[2] = C.reshape(-1, C.shape[-1])


def _sqrt_gram(G):
    w, V = np.linalg.eigh((G + G.conj().T) / 2)
    keep = w > max(w.max(), 0) * (1e-11 if os.environ.get('C64') else 1e-28)
    return w[keep], V[:, keep]


def compress():
    """exact Schmidt recompression of both bonds via Gram matrices (drop s < tol * s_max)."""
    global lognorm
    # bond 1 (between block 0 and blocks 1,2)
    GC = T[2].conj() @ T[2].T                                   # (k2, k2'): <C_k2|C_k2'>
    GA = T[0].conj() @ T[0].T                                   # (k1, k1')
    GR = np.einsum('akx,blx,kl->ab', T[1].conj(), T[1], GC, optimize=True)
    for which in (1, 2):
        if which == 1:
            wa, Va = _sqrt_gram(GA); wr, Vr = _sqrt_gram(GR)
        else:
            GA = T[0].conj() @ T[0].T
            GL = np.einsum('akx,blx,ab->kl', T[1].conj(), T[1], GA, optimize=True)
            GC = T[2].conj() @ T[2].T
            wa, Va = _sqrt_gram(GL); wr, Vr = _sqrt_gram(GC)
        Mx = (np.sqrt(wa)[:, None] * Va.conj().T) @ (np.sqrt(wr)[:, None] * Vr.conj().T).T
        U, s, Wh = np.linalg.svd(Mx)
        r = max(1, int(np.sum(s > tol * s[0])))
        PA = (Va / np.sqrt(wa)[None, :]) @ U[:, :r] * s[:r][None, :]     # (k, r)
        PR = (Vr / np.sqrt(wr)[None, :]) @ Wh[:r].T                      # (k, r)
        nrm = np.linalg.norm(s[:r])
        PA = (PA / nrm).astype(T[0].dtype); PR = PR.astype(T[0].dtype); lognorm += np.log(nrm)
        if which == 1:
            T[0] = PA.T @ T[0]
            T[1] = np.einsum('kr,klx->rlx', PR, T[1], optimize=True)
        else:
            T[1] = np.einsum('kr,akx->arx', PA, T[1], optimize=True)
            T[2] = PR.T @ T[2]




def applyk(X, r, js, U):
    """dense k-qubit operator U (2^k x 2^k, bit order = js) on ring r of block X."""
    lead = X.shape[:-1]; m = nr[r]; k = len(js)
    Y = X.reshape(lead + (2,) * m)
    nl = len(lead)
    Uk = U.reshape((2,) * (2 * k)).astype(X.dtype)
    Y = np.tensordot(Uk, Y, axes=(list(range(k, 2 * k)), [nl + j for j in js]))
    Y = np.moveaxis(Y, list(range(k)), [nl + j for j in js])
    return Y.reshape(lead + (2 ** m,))


def apply_cluster(ws, U):
    """U (2^k x 2^k) on wires ws (bit order = ws); multi-ring clusters via an operator-Schmidt chain over the rings."""
    rs = sorted({pos[w][0] for w in ws})
    if len(rs) == 1:
        r = rs[0]
        T[r] = applyk(T[r], r, [pos[w][1] for w in ws], U)
        return False
    k = len(ws); Ut = U.reshape((2,) * (2 * k))
    parts = [[i for i, w in enumerate(ws) if pos[w][0] == r] for r in rs]
    perm = []
    for p in parts:
        perm += p + [k + i for i in p]
    V = np.transpose(Ut, perm)
    ops = []; left = 1; rest = V
    for idx, p in enumerate(parts[:-1]):
        q = len(p)
        Mx = rest.reshape(left * 4 ** q, -1)
        Uu, sv, Vh = np.linalg.svd(Mx, full_matrices=False)
        rk = max(1, int(np.sum(sv > float(os.environ.get('OSRCUT', '1e-12')) * sv[0])))
        ops.append(Uu[:, :rk].reshape(left, 2 ** q, 2 ** q, rk))
        rest = sv[:rk, None] * Vh[:rk]; left = rk
    q = len(parts[-1]); ops.append(rest.reshape(left, 2 ** q, 2 ** q, 1))
    js = [[pos[ws[i]][1] for i in p] for p in parts]
    def on(r, X, js_r, op):          # op (2^q, 2^q)
        return applyk(X, r, js_r, op)
    if rs == [0, 1]:
        o0, o1 = ops; S = o0.shape[3]
        A = np.stack([on(0, T[0], js[0], o0[0, :, :, s]) for s in range(S)], axis=1); T[0] = A.reshape(-1, A.shape[-1])
        B = np.stack([on(1, T[1], js[1], o1[s, :, :, 0]) for s in range(S)], axis=1); T[1] = B.reshape(-1, B.shape[2], B.shape[3])
    elif rs == [1, 2]:
        o1, o2 = ops; S = o1.shape[3]
        B = np.stack([on(1, T[1], js[0], o1[0, :, :, s]) for s in range(S)], axis=2); T[1] = B.reshape(B.shape[0], -1, B.shape[3])
        C = np.stack([on(2, T[2], js[1], o2[s, :, :, 0]) for s in range(S)], axis=1); T[2] = C.reshape(-1, C.shape[-1])
    elif rs == [0, 2]:
        o0, o2 = ops; S = o0.shape[3]
        A = np.stack([on(0, T[0], js[0], o0[0, :, :, s]) for s in range(S)], axis=1); T[0] = A.reshape(-1, A.shape[-1])
        k1, k2, x1 = T[1].shape
        T[1] = np.einsum('ijx,st->isjtx', T[1], np.eye(S, dtype=T[1].dtype)).reshape(k1 * S, k2 * S, x1)
        C = np.stack([on(2, T[2], js[1], o2[s, :, :, 0]) for s in range(S)], axis=1); T[2] = C.reshape(-1, C.shape[-1])
    else:
        o0, o1, o2 = ops; S = o0.shape[3]; Tt = o1.shape[3]
        A = np.stack([on(0, T[0], js[0], o0[0, :, :, s]) for s in range(S)], axis=1); T[0] = A.reshape(-1, A.shape[-1])
        k1, k2, x1 = T[1].shape
        B = np.stack([np.stack([on(1, T[1], js[1], o1[s, :, :, t]) for t in range(Tt)], axis=2) for s in range(S)], axis=1)
        T[1] = B.reshape(k1 * S, k2 * Tt, x1)                      # (k1, s, k2, t, x1)
        C = np.stack([on(2, T[2], js[2], o2[t, :, :, 0]) for t in range(Tt)], axis=1); T[2] = C.reshape(-1, C.shape[-1])
    return True


def apply_tn_multi(c, Wt):
    """apply a TNO cluster spanning several rings: each ring's tensors are contracted into its block; the cluster's
    cross-ring bonds become extra Schmidt indices (A-C bonds are carried through the middle block as identities).
    A tensor whose output and input legs belong to different rings (a wire moved across rings) is attached to the ring of
    its output leg; its input leg becomes a dimension-2 cross bond to an identity tensor in the input wire's ring."""
    import quimb.tensor as qtn
    parts = {0: [], 1: [], 2: []}
    for t in c:
        A_, lg = Wt.T[t]
        names = []; ro = None
        for l in lg:
            if l[0] == 'o':
                ro = pos[l[1]][0]
        for l in lg:
            if l[0] == 'o':
                names.append(f'y{l[1]}')
            elif l[0] == 'i':
                if pos[l[1]][0] == ro:
                    names.append(f'x{l[1]}')
                else:
                    names.append(f'mv{l[1]}')
                    parts[pos[l[1]][0]].append((np.eye(2, dtype=complex), [f'x{l[1]}', f'mv{l[1]}']))
            else:
                names.append(f'b{l[1]}')
        parts[ro].append((A_, names))
    # bonds and the rings that hold them
    where = collections.defaultdict(set); dim = {}
    for r, lst in parts.items():
        for A_, names in lst:
            for d_, nm in zip(A_.shape, names):
                if nm.startswith('b') or nm.startswith('mv'):
                    where[nm].add(r); dim[nm] = d_
    cb = {(0, 1): [], (1, 2): [], (0, 2): []}
    for nm, rs_ in where.items():
        if len(rs_) == 2:
            cb[tuple(sorted(rs_))].append(nm)
    for key in cb:
        cb[key].sort()
    def ring_contract(r, extra_out):
        X = T[r]; lead = X.shape[:-1]; m = nr[r]
        ids = [f'L{i}' for i in range(len(lead))]
        ts = [qtn.Tensor(X.reshape(lead + (2,) * m), inds=ids + [f'x{w}' for w in G3[r]])]
        outw = set()
        for A_, names in parts[r]:
            ts.append(qtn.Tensor(A_, inds=names))
            for nm in names:
                if nm.startswith('y'):
                    outw.add(int(nm[1:]))
        # wires of ring r: output index y_w if produced by the cluster, else x_w if not consumed
        consumed = {int(nm[1:]) for A_, names in parts[r] for nm in names if nm.startswith('x')}
        outi = ids + list(extra_out)
        for w in G3[r]:
            if w in outw:
                outi.append(f'y{w}')
            elif w not in consumed:
                outi.append(f'x{w}')
            else:
                raise RuntimeError('ring wire consumed but not produced')
        tn_ = qtn.TensorNetwork(ts)
        tree = tn_.contraction_tree(optimize=OPT, output_inds=outi)
        print(f'      ring {r} part: {len(ts)-1} tensors, contraction width {tree.contraction_width():.1f}', flush=True)
        res = tn_.contract(all, output_inds=outi, optimize=tree)
        return np.asarray(res.data), lead
    d = lambda ls: int(np.prod([dim[l] for l in ls])) if ls else 1
    D01, D12, D02 = d(cb[(0, 1)]), d(cb[(1, 2)]), d(cb[(0, 2)])
    R0, lead0 = ring_contract(0, cb[(0, 1)] + cb[(0, 2)])
    T0n = R0.reshape(lead0[0] * D01 * D02, 2 ** nr[0])
    R2, lead2 = ring_contract(2, cb[(1, 2)] + cb[(0, 2)])
    T2n = R2.reshape(lead2[0] * D12 * D02, 2 ** nr[2])
    R1, lead1 = ring_contract(1, cb[(0, 1)] + cb[(1, 2)])
    k1, k2 = lead1
    R1 = R1.reshape(k1, k2, D01, D12, 2 ** nr[1])
    T1n = np.einsum('abcdx,ef->acebdfx', R1, np.eye(D02, dtype=R1.dtype)).reshape(k1 * D01 * D02, k2 * D12 * D02, 2 ** nr[1])
    T[0] = T0n.astype(T[0].dtype); T[1] = T1n.astype(T[1].dtype); T[2] = T2n.astype(T[2].dtype)


MODE = os.environ.get('MODE', 'gates')
OPT = os.environ.get('OPT', 'auto-hq')
ncross = 0; maxr = (1, 1); tl = time.time()
if MODE == 'gates':
    for bi, (a, b, M) in enumerate(blocks):
        (ra, ja), (rb, jb) = pos[a], pos[b]
        if ra == rb:
            T[ra] = apply2(T[ra], ra, ja, jb, M)
        else:
            cross(a, b, M); ncross += 1
            if T[1].size * T[1].itemsize > float(os.environ.get('MAXBYTES', '3e9')):
                print('ABORT: middle block too large', T[1].shape, flush=True); sys.exit(3)
            compress()
            maxr = (max(maxr[0], T[0].shape[0]), max(maxr[1], T[2].shape[0]))
    for w, M in lone.items():
        r, j = pos[w]; T[r] = apply1(T[r], r, j, M)
else:
    # slabs of ASAP unit layers: 'burst' layers (<= XB cross units) applied gate by gate; maximal runs of other layers
    # ('gaps') compressed first as an operator with the swap-aware TNO; its bond-graph components = clusters, each
    # contracted to a dense unitary and applied (multi-ring clusters through an operator-Schmidt chain).
    import tno as TN, tno_fast
    tno_fast.install()
    XB = int(os.environ.get('XB', '1')); TCUT = float(os.environ.get('TCUT', '1e-8')); KMAX = int(os.environ.get('KMAX', '12'))
    depth = [0] * n; ulay = []
    for a, b, _, _ in units:
        l = max(depth[a], depth[b]); ulay.append(l); depth[a] = depth[b] = l + 1
    D = max(ulay) + 1
    bylay = [[] for _ in range(D)]
    for k in range(len(units)):
        bylay[ulay[k]].append(k)
    isb = [sum(1 for k in bylay[l] if pos[units[k][0]][0] != pos[units[k][1]][0]) <= XB for l in range(D)]
    slabs = []; l = 0
    while l < D:
        l2 = l
        while l2 < D and isb[l2] == isb[l]:
            l2 += 1
        slabs.append((isb[l], l, l2)); l = l2
    print('slabs', [(('B' if b else 'G'), l0, l1) for b, l0, l1 in slabs], flush=True)
    # cross units inside a burst that are the first (last) unit on both wires within the burst move to the end of the
    # preceding gap (start of the following gap); causal order is preserved.
    movedB = set(); extra_end = collections.defaultdict(list); extra_start = collections.defaultdict(list)
    for si, (sb, l0, l1) in enumerate(slabs):
        if not sb:
            continue
        ks = sorted(k for l in range(l0, l1) for k in bylay[l])
        first = {}; lastu = {}
        for k in ks:
            for w in units[k][:2]:
                first.setdefault(w, k); lastu[w] = k
        for k in ks:
            a, b = units[k][:2]
            if pos[a][0] == pos[b][0]:
                continue
            if first[a] == k and first[b] == k and si > 0:
                extra_end[si - 1].append(k); movedB.add(k)
            elif lastu[a] == k and lastu[b] == k and si + 1 < len(slabs):
                extra_start[si + 1].append(k); movedB.add(k)
    print('cross units moved from bursts into gaps:', len(movedB), flush=True)
    def ugate(k):
        a, b, Pa, Pb = units[k]
        return a, b, CZ @ np.kron(Pa, Pb)
    maxk = 0; nclus = 0
    def slab_groups(si):
        sb, l0, l1 = slabs[si]
        if sb:
            return [[k for k in bylay[l] if k not in movedB] for l in range(l0, l1)]
        return [sorted(extra_start[si])] + [bylay[l] for l in range(l0, l1)] + [sorted(extra_end[si])]

    def clusters_of(Wt):
        adj = collections.defaultdict(set)
        for (x, y), d in Wt.bond_graph().items():
            adj[x].add(y); adj[y].add(x)
        seen = set(); comps = []
        for t in Wt.T:
            if t in seen:
                continue
            st = [t]; c = []
            while st:
                u = st.pop()
                if u in seen:
                    continue
                seen.add(u); c.append(u); st += list(adj[u] - seen)
            comps.append(c)
        return comps

    def bad_cluster(Wt, c):
        """estimated element count of the largest ring block after applying a multi-ring cluster."""
        rr = {}
        for t in c:
            for l in Wt.T[t][1]:
                if l[0] == 'o':
                    rr[t] = pos[l[1]][0]
        if len(set(rr.values())) < 2:
            return 0
        owner = collections.defaultdict(list); dim = {}
        for t in c:
            A_, lg = Wt.T[t]
            for d_, l in zip(A_.shape, lg):
                if l[0] == 'b':
                    owner[l].append(t); dim[l] = d_
        pr = collections.Counter()
        for l, ts_ in owner.items():
            if len(ts_) == 2 and rr[ts_[0]] != rr[ts_[1]]:
                pr[rr[ts_[0]]] += np.log2(dim[l]); pr[rr[ts_[1]]] += np.log2(dim[l])
        est = 0
        for r in range(3):
            lead = T[r].size // T[r].shape[-1]
            est = max(est, np.log2(lead) + pr[r] + nr[r])
        return est

    LIMB = float(os.environ.get('LIMB', '26')); MAXEXT = int(os.environ.get('MAXEXT', '3'))
    if os.environ.get('SEGW'):
        # wire-aware segmentation: L(u) = latest burst behind u (DAG), F(u) = earliest burst ahead of u.
        # u with F(u) == L(u) sits inside burst L(u) (applied gate by gate there); otherwise u belongs to gap L(u).
        bidx = {}; kb = 0
        for sb_, l0_, l1_ in slabs:
            if sb_:
                for l in range(l0_, l1_):
                    for k in bylay[l]:
                        if pos[units[k][0]][0] == pos[units[k][1]][0]:
                            bidx[k] = kb
                kb += 1
        NBUR = kb
        Lv = [None] * len(units); lastw = {}
        for k, (a, b, _, _) in enumerate(units):
            L_ = -1
            for w in (a, b):
                if w in lastw:
                    p_ = lastw[w]; L_ = max(L_, bidx[p_] if p_ in bidx else Lv[p_])
            Lv[k] = L_
            lastw[a] = lastw[b] = k
        Fv = [None] * len(units); nextw = {}
        for k in range(len(units) - 1, -1, -1):
            a, b = units[k][:2]; F_ = 10 ** 9
            for w in (a, b):
                if w in nextw:
                    q_ = nextw[w]; F_ = min(F_, bidx[q_] if q_ in bidx else Fv[q_])
            Fv[k] = F_
            nextw[a] = nextw[b] = k
        Bu = collections.defaultdict(list); Gu = collections.defaultdict(list)
        for k in range(len(units)):
            if k in bidx:
                Bu[bidx[k]].append(k)
            elif Fv[k] == Lv[k]:
                Bu[Lv[k]].append(k)
            else:
                Gu[Lv[k]].append(k)
        newslabs = []
        for g in range(-1, NBUR):
            if g >= 0:
                newslabs.append((True, min(ulay[k] for k in Bu[g]), max(ulay[k] for k in Bu[g]) + 1, sorted(Bu[g])))
            if Gu[g]:
                newslabs.append((False, min(ulay[k] for k in Gu[g]), max(ulay[k] for k in Gu[g]) + 1, sorted(Gu[g])))
        print('wire-aware segmentation:', [('B' if x[0] else 'G', x[1], x[2], len(x[3])) for x in newslabs], flush=True)
        print('in-burst non-ring units:', sum(1 for k in range(len(units)) if k not in bidx and Fv[k] == Lv[k]), flush=True)
        slabs = [x[:3] for x in newslabs]
        _units_of = [x[3] for x in newslabs]
        def slab_groups(si):
            us = _units_of[si]
            if slabs[si][0]:
                return [us]
            return [us[i:i + 25] for i in range(0, len(us), 25)]
    si = 0
    while si < len(slabs):
        sb, l0, l1 = slabs[si]
        if sb:
            for grp in slab_groups(si):
                for k in grp:
                    a, b, M = ugate(k)
                    (ra, ja), (rb, jb) = pos[a], pos[b]
                    if ra == rb:
                        T[ra] = apply2(T[ra], ra, ja, jb, M)
                    else:
                        cross(a, b, M); ncross += 1; compress()
            si += 1
            continue
        Wt = TN.TNO(n, cutoff=1e-12)
        sl = [si]; ext = 0
        def absorb_groups(gs):
            for grp in gs:
                for k in grp:
                    a, b, M = ugate(k)
                    Wt.gate(M, a, b, 'after')
                TN.canonical_compress(Wt, cutoff=TCUT); Wt.drop_trivial_bonds()
                while TN.unswap_pass(Wt):
                    TN.canonical_compress(Wt, cutoff=TCUT); Wt.drop_trivial_bonds()
        absorb_groups(slab_groups(si))
        while True:
            comps = clusters_of(Wt)
            worst = max([bad_cluster(Wt, c) for c in comps] + [0])
            if worst <= LIMB or ext >= MAXEXT or sl[-1] + 2 >= len(slabs):
                break
            print(f'    gap window {[slabs[x][1:] for x in sl]}: multi-ring cluster needs 2^{worst:.0f} elements -> extend by the next burst+gap', flush=True)
            absorb_groups(slab_groups(sl[-1] + 1)); absorb_groups(slab_groups(sl[-1] + 2))
            sl += [sl[-1] + 1, sl[-1] + 2]; ext += 1
        l0, l1 = slabs[sl[0]][1], slabs[sl[-1]][2]
        si = sl[-1] + 1
        sizes = []
        for c in comps:
            wsc = sorted({l[1] for t in c for l in Wt.T[t][1] if l[0] == 'o'})
            rsc = {pos[w][0] for w in wsc}
            if len(wsc) > int(os.environ.get('QTNK', '10')) and len(rsc) == 1:          # large single-ring cluster: contract its tensors into the ring block
                import quimb.tensor as qtn
                print(f'    single-ring cluster of {len(wsc)} wires via contraction', flush=True)
                r = rsc.pop(); X = T[r]; lead = X.shape[:-1]; m = nr[r]
                ids = [f'L{i}' for i in range(len(lead))]
                ts = [qtn.Tensor(X.reshape(lead + (2,) * m), inds=ids + [f'x{w}' for w in G3[r]])]
                for t in c:
                    A_, lg = Wt.T[t]
                    ts.append(qtn.Tensor(A_, inds=[(f'y{l[1]}' if l[0] == 'o' else (f'x{l[1]}' if l[0] == 'i' else f'b{l[1]}')) for l in lg]))
                outi = ids + [(f'y{w}' if w in wsc else f'x{w}') for w in G3[r]]
                tn_ = qtn.TensorNetwork(ts)
                tree = tn_.contraction_tree(optimize=OPT, output_inds=outi)
                print(f'      width {tree.contraction_width():.1f}', flush=True)
                res = tn_.contract(all, output_inds=outi, optimize=tree)
                T[r] = np.asarray(res.data).reshape(lead + (2 ** m,)).astype(X.dtype)
                sizes.append(len(wsc)); maxk = max(maxk, len(wsc))
                continue
            moved_x = any(pos[l[1]][0] != pos[[m_[1] for m_ in Wt.T[t][1] if m_[0] == 'o'][0]][0]
                          for t in c for l in Wt.T[t][1] if l[0] == 'i')
            if len(rsc) > 1 and len(wsc) > KMAX:          # large multi-ring cluster: split along its own cross-ring bonds
                print(f'    multi-ring cluster of {len(wsc)} wires (rings {sorted(rsc)}) via contraction', flush=True)
                apply_tn_multi(c, Wt); nclus += 1; ncross += 1
                if max(t.size * t.itemsize for t in T) > float(os.environ.get('MAXBYTES', '3e9')):
                    print('ABORT: block too large', [t.shape for t in T], flush=True); sys.exit(3)
                compress(); sizes.append(len(wsc)); maxk = max(maxk, len(wsc))
                continue
            Aq, lq = Wt.T[c[0]]
            for t in c[1:]:
                Aq, lq = TN.TNO._contract(Aq, lq, Wt.T[t][0], Wt.T[t][1])
            outs = sorted(l[1] for l in lq if l[0] == 'o'); ins = sorted(l[1] for l in lq if l[0] == 'i')
            if outs != ins:
                raise RuntimeError(f'cluster with a wire permutation {outs} {ins}')
            ws = outs; kq = len(ws); sizes.append(kq)
            if kq > KMAX:
                raise RuntimeError(f'cluster too large {kq}')
            Aq = np.transpose(Aq, [lq.index(('o', w)) for w in ws] + [lq.index(('i', w)) for w in ws]).reshape(2 ** kq, 2 ** kq)
            if apply_cluster(ws, Aq):
                nclus += 1; ncross += 1
                if max(t.size * t.itemsize for t in T) > float(os.environ.get('MAXBYTES', '3e9')):
                    print('ABORT: block too large', [t.shape for t in T], 'cluster', ws, flush=True); sys.exit(3)
                compress()
            maxk = max(maxk, kq)
        compress()
        maxr = (max(maxr[0], T[0].shape[0]), max(maxr[1], T[2].shape[0]))
        print(f'  gap [{l0},{l1}): TNO {Wt.size()} elems, clusters {sorted(sizes, reverse=True)[:6]}, cross clusters so far {nclus}, '
              f'ranks {T[0].shape[0]},{T[2].shape[0]} t={time.time()-t0:.0f}s', flush=True)
    for w in range(n):
        r, j = pos[w]; T[r] = apply1(T[r], r, j, tail[w])
compress()
print('final ranks', T[0].shape[0], T[2].shape[0], 'max', maxr, flush=True)
# marginals
GA = T[0].conj() @ T[0].T; GC = T[2].conj() @ T[2].T
nrm = np.einsum('akx,blx,ab,kl->', T[1].conj(), T[1], GA, GC, optimize=True).real
GR = np.einsum('akx,blx,kl->ab', T[1].conj(), T[1], GC, optimize=True)
GL = np.einsum('akx,blx,ab->kl', T[1].conj(), T[1], GA, optimize=True)
Z = np.diag([1., -1.]).astype(complex)
zs = np.zeros(n)
for q in range(n):
    r, j = pos[q]
    if r == 0:
        v = np.einsum('ax,bx,ab->', T[0].conj(), apply1(T[0], 0, j, Z), GR)
    elif r == 2:
        v = np.einsum('ax,bx,ab->', T[2].conj(), apply1(T[2], 2, j, Z), GL)
    else:
        v = np.einsum('akx,blx,ab,kl->', T[1].conj(), apply1(T[1], 1, j, Z), GA, GC, optimize=True)
    zs[q] = (v / nrm).real
peak = ''.join('0' if z >= 0 else '1' for z in zs)


def amp(bits):
    idx = []
    for r in range(3):
        x = 0
        for q in G3[r]:
            x = 2 * x + int(bits[q])
        idx.append(x)
    return np.einsum('a,ab,b->', T[0][:, idx[0]], T[1][:, :, idx[1]], T[2][:, idx[2]])


p = abs(amp(peak)) ** 2 / nrm
flips = [abs(amp(peak[:q] + ('1' if peak[q] == '0' else '0') + peak[q + 1:])) ** 2 / nrm for q in range(n)]
res = dict(qasm=qasm, tol=tol, peak=peak, p=float(p), max_flip=float(max(flips)), n_flips_higher=int(sum(fl > p for fl in flips)),
           min_abs_z=float(np.min(np.abs(zs))), zs=[round(float(z), 4) for z in zs], groups=G3, max_ranks=maxr,
           seconds=round(time.time() - t0))
if out:
    json.dump(res, open(out, 'w'), indent=1)
print(f"PEAK sha {hashlib.sha256(peak.encode()).hexdigest()[:12]} p={p:.5f} maxflip={max(flips):.3g} higher={res['n_flips_higher']} "
      f"min|Z|={res['min_abs_z']:.3f} {res['seconds']}s", flush=True)
