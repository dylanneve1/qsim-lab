"""Approximate Clifford-augmented MPS (CAMPS) and plain MPS for the IBM
doped-Clifford circuit (tracker issue 228).

psi = C |phi>, C a Clifford kept as the stim tableau of C^{-1} (`cinv`),
phi an MPS with max bond chi.
  * Clifford gate g:   C <- g C       (cinv.prepend(g^-1))
  * rz(pi/4) on q:     phi <- exp(-i pi/8 P) phi,  P = C^dag Z_q C = cinv(Z_q)
      - OFD (optimization-free disentangling, Liu/Clark arXiv:2412.17209): if P
        has X/Y on a "free" site j (phi_j still exactly |0>, never touched by
        magic) we pick a Clifford D with D P D^dag = X_j and D phi = local(phi)
        (S/CZ/CNOT controlled on |0>), so the rotation is single-site.
      - otherwise: bond-2 MPO over the support of P, SVD truncation to chi,
        then a greedy sweep of 2-qubit Clifford disentanglers (720 unsigned
        2-qubit Cliffords, scored by Renyi-2 purity) absorbed into C.
  * fidelity estimate = prod(1 - discarded weight).
"""
import math
import sys
import time

import numpy as np
import stim

QASM_DEFAULT = "/tmp/qsim-camps/research/chain-sweep/nq70_depth70_checks27_doped.qasm"

# ---------------------------------------------------------------- circuit
GATE_STIM = {"h": "H", "s": "S", "sx": "SQRT_X", "sxdg": "SQRT_X_DAG", "cz": "CZ"}
GATE_INV = {"h": "H", "s": "S_DAG", "sx": "SQRT_X_DAG", "sxdg": "SQRT_X", "cz": "CZ"}


def load_circuit(path=QASM_DEFAULT, n=70, d=70, lo=0):
    """Gate list [(name, qubits, layer)] truncated exactly like
    qsim_lab::engines::chain_sweep::truncate (CZ-depth d, first n qubits).
    `layer` = CZ-layer index of the gate (for 1q gates: depth of its qubit)."""
    ops = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "//")):
            continue
        name, rest = line.split(None, 1) if " " in line else (line, "")
        qs = [int(t.split("[")[1].split("]")[0]) for t in rest.rstrip(";").split(",")]
        ops.append((name, qs))
    depth = [0] * 70
    out = []
    for name, qs in ops:
        if name == "cz":
            a, b = qs
            layer = max(depth[a], depth[b]) + 1
            depth[a] = depth[b] = layer
            if layer <= d and lo <= a < lo + n and lo <= b < lo + n:
                out.append(("cz", [a - lo, b - lo], layer))
        else:
            if all(lo <= q < lo + n and depth[q] <= d for q in qs):
                out.append((name, [q - lo for q in qs], depth[qs[0]]))
    return out


# ---------------------------------------------------------------- dense reference
def _u1(name):
    s2 = 1 / math.sqrt(2)
    if name == "h":
        return np.array([[1, 1], [1, -1]]) * s2
    if name == "s":
        return np.diag([1, 1j])
    if name == "sdg":
        return np.diag([1, -1j])
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    if name == "sxdg":
        return 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]])
    if name == "rz":
        return np.diag([np.exp(-1j * math.pi / 8), np.exp(1j * math.pi / 8)])
    raise KeyError(name)


def dense_state(ops, n):
    """State vector, index bit q = qubit q (little endian)."""
    psi = np.zeros(2**n, complex)
    psi[0] = 1
    t = psi.reshape([2] * n)  # axis k <-> qubit n-1-k
    for name, qs, _ in ops:
        if name == "cz":
            a, b = qs
            idx = [slice(None)] * n
            idx[n - 1 - a] = 1
            idx[n - 1 - b] = 1
            t[tuple(idx)] *= -1
        else:
            q = qs[0]
            ax = n - 1 - q
            t = np.moveaxis(np.tensordot(_u1(name.split("(")[0]), t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


# ---------------------------------------------------------------- MPS
class MPS:
    """Open-boundary MPS, tensors (l, 2, r); orthogonality centre `c`."""

    def __init__(self, n, chi, cutoff=1e-12):
        self.n, self.chi, self.cutoff = n, chi, cutoff
        self.A = []
        for _ in range(n):
            a = np.zeros((1, 2, 1), complex)
            a[0, 0, 0] = 1
            self.A.append(a)
        self.c = 0
        self.fid = 1.0  # product of (1 - discarded weight)
        self.ntrunc = 0
        self.disc = []  # (tag, discarded weight)
        self.tag = None

    def bond_dims(self):
        return [a.shape[2] for a in self.A[:-1]]

    def max_bond(self):
        return max(a.shape[2] for a in self.A)

    def move(self, k):
        A = self.A
        while self.c < k:
            i = self.c
            l, d, r = A[i].shape
            q, rr = np.linalg.qr(A[i].reshape(l * d, r))
            A[i] = q.reshape(l, d, -1)
            A[i + 1] = np.tensordot(rr, A[i + 1], axes=(1, 0))
            self.c += 1
        while self.c > k:
            i = self.c
            l, d, r = A[i].shape
            q, rr = np.linalg.qr(A[i].reshape(l, d * r).T)
            A[i] = q.T.reshape(-1, d, r)
            A[i - 1] = np.tensordot(A[i - 1], rr.T, axes=(2, 0))
            self.c -= 1

    def apply1(self, u, i):
        self.A[i] = np.einsum("st,ltr->lsr", u, self.A[i])

    def _split(self, theta, i, right=True):
        """theta (l,2,2,r) at sites i,i+1 with centre inside; SVD-truncate."""
        l, _, _, r = theta.shape
        m = theta.reshape(l * 2, 2 * r)
        L, R = m.shape
        if min(L, R) <= 64:
            try:
                u, s, vh = np.linalg.svd(m, full_matrices=False)
            except np.linalg.LinAlgError:
                u, s, vh = _svd_fallback(m)
            s2 = s**2
        else:  # eigh of the smaller Gram matrix (faster than SVD here)
            if L <= R:
                w, u = np.linalg.eigh(m @ m.conj().T)
                w, u = w[::-1], u[:, ::-1]
            else:
                w, v = np.linalg.eigh(m.conj().T @ m)
                w, v = w[::-1], v[:, ::-1]
            s2 = np.clip(w, 0, None)
        tot = float(np.sum(s2))
        keep = int(np.sum(s2 > self.cutoff * tot))
        keep = max(1, min(keep, self.chi))
        disc = float(np.sum(s2[keep:])) / tot if tot > 0 else 0.0
        if disc > 0:
            self.fid *= 1 - disc
            self.ntrunc += 1
            self.disc.append((self.tag, disc))
        if min(L, R) > 64:
            s = np.sqrt(s2[:keep])
            if L <= R:
                u = u[:, :keep]
                vh = (u.conj().T @ m) / s[:, None]
            else:
                vh = v[:, :keep].conj().T
                u = (m @ v[:, :keep]) / s[None, :]
        u, s, vh = u[:, :keep], s[:keep] / math.sqrt(tot - disc * tot), vh[:keep]
        if right:
            self.A[i] = u.reshape(l, 2, keep)
            self.A[i + 1] = (s[:, None] * vh).reshape(keep, 2, r)
            self.c = i + 1
        else:
            self.A[i] = (u * s).reshape(l, 2, keep)
            self.A[i + 1] = vh.reshape(keep, 2, r)
            self.c = i
        return disc

    def theta(self, i):
        self.move(i)
        return np.tensordot(self.A[i], self.A[i + 1], axes=(2, 0))

    def apply2(self, u4, i, right=True):
        """u4 (4x4) on sites i,i+1, index 2*s_i + s_{i+1}."""
        th = self.theta(i)
        th = np.einsum("xy,lyr->lxr", u4, th.reshape(th.shape[0], 4, th.shape[3]))
        th = th.reshape(th.shape[0], 2, 2, th.shape[2])
        return self._split(th, i, right)

    def apply_pauli_rot(self, ops, theta):
        """exp(-i theta P), P = prod of 2x2 ops {site: matrix} (sign folded in)."""
        sites = sorted(ops)
        a, b = sites[0], sites[-1]
        c, s = math.cos(theta), math.sin(theta)
        if a == b:
            self.apply1(c * np.eye(2) - 1j * s * ops[a], a)
            return
        self.move(a)
        A = self.A
        I2 = np.eye(2)
        for k in range(a, b + 1):
            p = ops.get(k, I2)
            t = A[k]
            l, _, r = t.shape
            pt = np.einsum("st,ltr->lsr", p, t)
            if k == a:
                new = np.zeros((l, 2, 2 * r), complex)
                new[:, :, :r] = c * t
                new[:, :, r:] = -1j * s * pt
            elif k == b:
                new = np.zeros((2 * l, 2, r), complex)
                new[:l] = t
                new[l:] = pt
            else:
                new = np.zeros((2 * l, 2, 2 * r), complex)
                new[:l, :, :r] = t
                new[l:, :, r:] = pt
            A[k] = new
        # canonicalise left->right (centre ends at b), then truncate right->left
        self.c = a
        self.move(b)
        for k in range(b - 1, a - 1, -1):
            th = np.tensordot(A[k], A[k + 1], axes=(2, 0))
            self._split(th, k, right=False)

    def norm2(self):
        return float(np.linalg.norm(self.A[self.c]) ** 2)

    def dense(self):
        v = self.A[0]
        for k in range(1, self.n):
            v = np.tensordot(v, self.A[k], axes=(v.ndim - 1, 0))
        v = v.reshape([2] * self.n)  # axis k <-> site k
        # to little-endian index (bit q = site q): reverse axes
        return np.transpose(v, list(range(self.n))[::-1]).reshape(-1)

    def amp(self, bits):
        """<bits|phi>, bits[i] for site i."""
        v = self.A[0][:, bits[0], :]
        for k in range(1, self.n):
            v = v @ self.A[k][:, bits[k], :]
        return v[0, 0]

    def entropies(self):
        """von Neumann entropy (bits) at every bond."""
        out = []
        self.move(0)
        for k in range(self.n - 1):
            th = np.tensordot(self.A[k], self.A[k + 1], axes=(2, 0))
            s = np.linalg.svd(th.reshape(th.shape[0] * 2, -1), compute_uv=False)
            p = s**2 / np.sum(s**2)
            p = p[p > 1e-16]
            out.append(float(-np.sum(p * np.log2(p))))
            self.move(k + 1)
        return out


def _svd_fallback(m):
    import scipy.linalg

    return scipy.linalg.svd(m, full_matrices=False, lapack_driver="gesvd")


# ---------------------------------------------------------------- Clifford tools
PAULI_M = {
    1: np.array([[0, 1], [1, 0]], complex),
    2: np.array([[0, -1j], [1j, 0]], complex),
    3: np.array([[1, 0], [0, -1]], complex),
}


def _two_qubit_cliffords():
    """720 unsigned 2-qubit Cliffords: (tableau, 4x4 unitary in index 2*q0+q1)."""
    tabs = list(stim.Tableau.iter_all(2, unsigned=True))
    us = np.array([t.to_unitary_matrix(endian="big") for t in tabs])
    return tabs, us


class CAMPS:
    def __init__(self, n, chi, disentangle=True, ofd=True, cutoff=1e-12, dis_sweeps=1):
        self.n = n
        self.mps = MPS(n, chi, cutoff)
        self.cinv = stim.Tableau(n)
        self.free = [True] * n
        self.disentangle = disentangle
        self.ofd = ofd
        self.dis_sweeps = dis_sweeps
        self.tabs, self.us = _two_qubit_cliffords()
        self.stats = {"ofd": 0, "mpo": 0, "trivial": 0, "dis_applied": 0, "support": []}
        self._tab_cache = {}
        G = self.us.reshape(-1, 2, 2, 4)  # (g, s, t, x)
        # rho_L block (s,s') = sum_xy c^{ss'}_{xy} P_xy, c = sum_t G[s,t,x] G*[s',t,y]
        cL = np.einsum("gstx,guty->gsuxy", G, G.conj())
        # rho_R block (t,t') (conjugated) = sum_xy G*[s,t,x] G[s,t',y] P_xy
        cR = np.einsum("gstx,gsuy->gtuxy", G.conj(), G)
        g = len(self.us)
        self.cL = (cL.reshape(g, 4, 16), cL.transpose(0, 2, 1, 3, 4).reshape(g, 4, 16))
        self.cR = (cR.reshape(g, 4, 16), cR.transpose(0, 2, 1, 3, 4).reshape(g, 4, 16))
        self.ident = [k for k, u in enumerate(self.us) if np.allclose(u, np.eye(4))][0]

    def is_free(self, k):
        return float(np.linalg.norm(self.mps.A[k][:, 1, :])) < 1e-12

    def _tab(self, name):
        if name not in self._tab_cache:
            self._tab_cache[name] = stim.Tableau.from_named_gate(name)
        return self._tab_cache[name]

    def clifford(self, name, qs):
        self.cinv.prepend(self._tab(GATE_INV[name]), qs)

    def _frame_gate(self, name, qs):
        """Absorb D (gate applied to phi) into the frame: C <- C D^-1, cinv <- D cinv."""
        self.cinv.append(self._tab(name), qs)

    def tgate(self, q, theta=math.pi / 8):
        n = self.n
        P = self.cinv(stim.PauliString("_" * q + "Z" + "_" * (n - q - 1)))
        sign = P.sign  # +1/-1 (real for a Hermitian Pauli)
        xs, zs = P.to_numpy()
        kind = {}  # site -> 1 X,2 Y,3 Z
        for k in range(n):
            x, z = bool(xs[k]), bool(zs[k])
            if x or z:
                kind[k] = 1 if (x and not z) else (2 if (x and z) else 3)
        self.free = [self.is_free(k) for k in range(n)]
        # free sites: Z acts as +1 on |0>
        for k in list(kind):
            if self.free[k] and kind[k] == 3:
                del kind[k]
        if not kind:
            self.stats["trivial"] += 1
            return
        sgn = complex(sign).real
        frees = [k for k in kind if self.free[k]]  # these have X or Y
        if self.ofd and frees:
            # pick the free site closest to the centre of mass of the support
            cm = np.mean(list(kind))
            j = min(frees, key=lambda k: abs(k - cm))
            m = self.mps
            if kind[j] == 2:  # S^dag Y S = X ; S^dag|0>=|0>
                self._frame_gate("S_DAG", [j])  # phi_j unchanged
            for k, p in sorted(kind.items()):
                if k == j:
                    continue
                if self.free[k]:
                    if p == 2:
                        self._frame_gate("S_DAG", [k])
                    # X_j X_k --CNOT(j->k)--> X_j ; control |0> -> identity on phi
                    self._frame_gate("CX", [j, k])
                else:
                    # local Clifford L on k mapping p -> Z, applied to phi_k
                    L = {1: "H", 2: "H_YZ", 3: None}[p]
                    if L is not None:
                        self._frame_gate(L, [k])
                        m.apply1(stim.Tableau.from_named_gate(L).to_unitary_matrix(endian="big"), k)
                    self._frame_gate("CZ", [j, k])  # X_j Z_k -> X_j ; control |0>
            # verify (cheap) and apply the single-site rotation
            P2 = self.cinv(stim.PauliString("_" * q + "Z" + "_" * (n - q - 1)))
            x2, z2 = P2.to_numpy()
            assert x2[j] and not z2[j], "OFD failed"
            for k in range(n):
                if k != j and (x2[k] or z2[k]):
                    assert self.free[k] and z2[k] and not x2[k], "OFD residual"
            s2 = complex(P2.sign).real
            m.apply1(math.cos(theta) * np.eye(2) - 1j * math.sin(theta) * s2 * PAULI_M[1], j)
            self.free[j] = False
            self.stats["ofd"] += 1
            self.stats["support"].append(1)
            return
        ops = {k: PAULI_M[p] for k, p in kind.items()}
        first = min(ops)
        ops[first] = sgn * ops[first]
        a, b = min(kind), max(kind)
        self.mps.apply_pauli_rot(ops, theta)
        self.stats["mpo"] += 1
        self.stats["support"].append(b - a + 1)
        if self.disentangle:
            lo, hi = max(0, a - 1), min(n - 2, b)
            for _ in range(self.dis_sweeps):
                for i in range(lo, hi + 1):
                    self.dis_bond(i, right=True)
                for i in range(hi, lo - 1, -1):
                    self.dis_bond(i, right=False)

    def dis_bond(self, i, right=True):
        """Choose the 2-qubit Clifford on (i,i+1) maximising Renyi-2 purity."""
        m = self.mps
        th = m.theta(i)
        l, _, _, r = th.shape
        if l == 1 and r == 1:
            return
        M = np.transpose(th, (1, 2, 0, 3)).reshape(4, l, r)  # M[x] l x r, x=2s+t
        if l <= r:
            Mf = M.reshape(4 * l, r)
            P = (Mf @ Mf.conj().T).reshape(4, l, 4, l).transpose(0, 2, 1, 3)  # M_x M_y^dag
            d = l
        else:
            Mf = M.transpose(0, 2, 1).reshape(4 * r, l)
            P = (Mf.conj() @ Mf.T).reshape(4, r, 4, r).transpose(0, 2, 1, 3)  # conj(M_x^T M_y^*)
            d = r
        Pf = P.reshape(16, d * d)
        PT = P.transpose(0, 1, 3, 2).reshape(16, d * d)
        N = (Pf @ PT.T).reshape(4, 4, 4, 4)  # tr(P_xy P_zw)
        c, cs = self.cL if l <= r else self.cR
        pur = np.einsum("gaz,gaz->g", c @ N.reshape(16, 16), cs).real
        best = int(np.argmax(pur))
        ident = self.ident
        if pur[best] <= pur[ident] * (1 + 1e-9):
            best = ident
        if best != ident:
            m._split(np.einsum("xy,lyr->lxr", self.us[best], th.reshape(l, 4, r)).reshape(l, 2, 2, r), i, right)
            self.cinv.append(self.tabs[best], [i, i + 1])
            self.stats["dis_applied"] += 1
        else:
            m._split(th, i, right)

    # ------------------------------------------------------------ readout (small n)
    def dense(self):
        """Full state vector C|phi> (little endian), for validation."""
        v = self.mps.dense()
        circ = self.cinv.inverse().to_circuit("elimination")
        sim_ops = []
        for inst in circ:
            nm = inst.name
            ts = [t.value for t in inst.targets_copy()]
            ar = 2 if stim.GateData(nm).is_two_qubit_gate else 1
            for k in range(0, len(ts), ar):
                sim_ops.append((nm, ts[k : k + ar]))
        n = self.n
        t = v.reshape([2] * n)
        for nm, qs in sim_ops:
            if len(qs) == 1:
                u = stim.Tableau.from_named_gate(nm).to_unitary_matrix(endian="big")
                ax = n - 1 - qs[0]
                t = np.moveaxis(np.tensordot(u, t, axes=([1], [ax])), 0, ax)
            else:
                u = stim.Tableau.from_named_gate(nm).to_unitary_matrix(endian="big").reshape(2, 2, 2, 2)
                a1, a2 = n - 1 - qs[0], n - 1 - qs[1]
                t = np.moveaxis(np.tensordot(u, t, axes=([2, 3], [a1, a2])), [0, 1], [a1, a2])
        return t.reshape(-1)


def run_camps(ops, n, chi, log=None, **kw):
    st = CAMPS(n, chi, **kw)
    t0 = time.time()
    lastlayer = 0
    for name, qs, layer in ops:
        if name.startswith("rz"):
            st.mps.tag = layer
            st.tgate(qs[0])
        else:
            st.clifford(name, qs)
        if log and name == "cz" and layer != lastlayer:
            lastlayer = layer
    st.time = time.time() - t0
    return st


def run_mps(ops, n, chi):
    m = MPS(n, chi)
    t0 = time.time()
    rz = _u1("rz")
    for name, qs, layer in ops:
        m.tag = layer
        if name == "cz":
            a, b = sorted(qs)
            assert b == a + 1
            m.apply2(np.diag([1, 1, 1, -1]).astype(complex), a, right=(m.c <= a))
        elif name.startswith("rz"):
            m.apply1(rz, qs[0])
        else:
            m.apply1(_u1(name).astype(complex), qs[0])
    m.time = time.time() - t0
    return m


# ---------------------------------------------------------------- frame analysis
def gf2_rank(M):
    M = M.copy().astype(np.uint8)
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        nz = np.nonzero(M[:, c])[0]
        nz = nz[nz != r]
        M[nz] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def stab_entropies(gens):
    """gens: list of stim.PauliString stabilisers of an n-qubit stabiliser state.
    Returns entanglement (bits) across every chain cut k|n-k."""
    xs = np.array([g.to_numpy()[0] for g in gens], dtype=np.uint8)
    zs = np.array([g.to_numpy()[1] for g in gens], dtype=np.uint8)
    n = xs.shape[1]
    out = []
    for k in range(1, n):
        A = np.concatenate([xs[:, :k], zs[:, :k]], axis=1)
        out.append(gf2_rank(A) - k)
    return out


def readout_entropies(cinv):
    """Entanglement of the stabiliser state C^dag|0> (needed to contract
    <x|C|phi> = <C^dag x|phi>): stabilisers C^dag Z_i C = cinv(Z_i)."""
    n = len(cinv)
    gens = [cinv.z_output(i) for i in range(n)]
    return stab_entropies(gens)


def frame_out_entropies(cinv):
    """Entanglement of C|0> (stabilisers C Z_i C^dag)."""
    c = cinv.inverse()
    return stab_entropies([c.z_output(i) for i in range(len(c))])


# ---------------------------------------------------------------- CAMPS amplitudes
class Readout:
    """<x|C|phi> = <s|Q_x|phi>, s = C^dag|0> = Lloc Prod_e CZ_e |+>^n (stim
    'graph_state' form), Q_x = C^dag X^x C.  Contraction sweeps the chain
    keeping (MPS bond) x (pending CZ phase pattern); cost ~ n chi^2 2^E,
    E = cut rank of the graph (= entanglement of s)."""

    def __init__(self, st):
        self.st = st
        n = self.n = st.n
        circ = st.cinv.to_circuit("graph_state")  # prepares cinv|0> = C^dag|0>
        self.edges = []
        self.loc = [np.eye(2, dtype=complex) for _ in range(n)]
        seen_cz = False
        for inst in circ:
            if inst.name == "TICK":
                continue
            ts = [t.value for t in inst.targets_copy()]
            if inst.name == "RX":
                continue
            if inst.name == "CZ":
                for k in range(0, len(ts), 2):
                    self.edges.append(tuple(sorted(ts[k : k + 2])))
                continue
            u = stim.Tableau.from_named_gate(inst.name).to_unitary_matrix(endian="big")
            for q in ts:
                self.loc[q] = u @ self.loc[q]
        self.adj = [set() for _ in range(n)]
        for a, b in self.edges:
            self.adj[a].add(b)
            self.adj[b].add(a)
        # frontier bit positions: future vertex j gets bit j in the key
        self.cut_rank = None

    def amp(self, bits):
        st, n = self.st, self.n
        xs = "".join("X" if b else "_" for b in bits)
        Q = st.cinv(stim.PauliString(xs))
        qx, qz = Q.to_numpy()
        sign = complex(Q.sign)
        A = st.mps.A
        # tensors of Lloc^dag Q phi
        keys = np.zeros(1, dtype=object)
        keys = np.array([0], dtype=np.int64) if n <= 62 else None
        keys = [0]
        vec = np.ones((1, 1), complex)  # rows = keys, cols = bond
        key_arr = np.zeros(1, dtype=object)
        key_arr[0] = 0
        for k in range(n):
            p = np.eye(2, dtype=complex)
            if qx[k] and qz[k]:
                p = PAULI_M[2]
            elif qx[k]:
                p = PAULI_M[1]
            elif qz[k]:
                p = PAULI_M[3]
            t = np.einsum("st,ltr->lsr", self.loc[k].conj().T @ p, A[k])
            # <+| CZ-phase: for y_k, phase (-1)^{y_k * l_k}, l_k = bit k of key
            fut = 0
            for j in self.adj[k]:
                if j > k:
                    fut |= 1 << j
            newkeys = []
            newvecs = []
            lk = np.array([(int(kk) >> k) & 1 for kk in key_arr])
            base = np.array([int(kk) & ~(1 << k) for kk in key_arr], dtype=object)
            for y in (0, 1):
                v = vec @ t[:, y, :]
                if y:
                    v = v * (1 - 2 * lk)[:, None]
                    nk = np.array([int(b) ^ fut for b in base], dtype=object)
                else:
                    nk = base
                newkeys.append(nk)
                newvecs.append(v)
            allk = np.concatenate(newkeys)
            allv = np.concatenate(newvecs)
            uk, inv = np.unique(allk, return_inverse=True)
            vec = np.zeros((len(uk), allv.shape[1]), complex)
            np.add.at(vec, inv, allv)
            key_arr = uk
        assert len(key_arr) == 1
        return sign * vec[0, 0] / 2 ** (n / 2), len(key_arr)


def readout_amp_fast(ro, bits):
    """Vectorised Readout.amp: frontier vertex j <-> bit j of a 128-bit key
    stored as two int64 words (lo = vertices 0..62, hi = 63..)."""
    st, n = ro.st, ro.n
    if not hasattr(ro, "fm"):
        ro.fm = []
        ro.max_front = 0
        front = set()
        for k in range(n):
            lo = hi = 0
            for j in ro.adj[k]:
                if j > k:
                    if j < 63:
                        lo |= 1 << j
                    else:
                        hi |= 1 << (j - 63)
                    front.add(j)
            front.discard(k)
            ro.fm.append((np.int64(lo), np.int64(hi)))
            ro.max_front = max(ro.max_front, len(front))
    xs = "".join("X" if b else "_" for b in bits)
    Q = st.cinv(stim.PauliString(xs))
    qx, qz = Q.to_numpy()
    sign = complex(Q.sign)
    A = st.mps.A
    keys = np.zeros((1, 2), dtype=np.int64)
    vec = np.ones((1, 1), complex)
    maxk = 1
    for k in range(n):
        p = np.eye(2, dtype=complex)
        if qx[k] and qz[k]:
            p = PAULI_M[2]
        elif qx[k]:
            p = PAULI_M[1]
        elif qz[k]:
            p = PAULI_M[3]
        t = np.einsum("st,ltr->lsr", ro.loc[k].conj().T @ p, A[k])
        w, b = (0, k) if k < 63 else (1, k - 63)
        lk = (keys[:, w] >> b) & 1
        base = keys.copy()
        base[:, w] &= ~np.int64(1 << b)
        v0 = vec @ t[:, 0, :]
        v1 = (vec * (1 - 2 * lk)[:, None]) @ t[:, 1, :]
        k1 = base.copy()
        k1[:, 0] ^= ro.fm[k][0]
        k1[:, 1] ^= ro.fm[k][1]
        allk = np.concatenate([base, k1])
        allv = np.concatenate([v0, v1])
        uk, inv = np.unique(allk, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        vec = np.zeros((len(uk), allv.shape[1]), complex)
        np.add.at(vec, inv, allv)
        keys = uk
        maxk = max(maxk, len(uk))
    return sign * vec[0, 0] / 2 ** (n / 2), maxk
