"""Independent exact reference: LITERAL qubit circuit of the paper (pair-interleaved JW ordering, RXX/RYY, RZZ, RZ, fSWAP
layers, virtual relabelling P) on a dense 2^(2L) statevector, plus an alternative direct Pauli-string implementation
(XZZX/YZZY long hops, no fSWAP) and continuous-time exact evolution.  Used only to validate the d=4 TEBD.
Spec (arXiv:2605.04025 supp. Sec. 'Trotterization'):
  U_step = A S F U S A  (A=exp(-i dt/2 H1Q), S=exp(-i dt H_S), U=exp(-i dt H_U2Q), F=fSWAP layer on wires (2J,2J+1)),
  applied right-to-left (A,S,U,F,S,A); n_step steps; fermion labels of wires toggle (up<->down) at every F.
H_S=-1/2 sum_{J=0}^{L-2}(X_{2J+1}X_{2J+2}+Y..Y), H_L=-1/2 sum (X_{2J}Z_{2J+1}Z_{2J+2}X_{2J+3}+YZZY),
H1Q=1/2(mu-U/2) sum Z, H_U2Q=U/4 sum_J Z_{2J}Z_{2J+1}; Neel = |down up down up ...> = X on wires 0,2,4,...
"""
import numpy as np, sys, json

class Q:
    def __init__(s, L):
        s.L = L; s.n = 2 * L; s.N = 1 << s.n; s.idx = np.arange(s.N, dtype=np.int64)
        s.bits = [((s.idx >> q) & 1) for q in range(s.n)]

    def pauli(s, v, ops):
        """apply Pauli string ops={qubit:'X'|'Y'|'Z'} to vector v (basis state bit q = occupation of wire q)."""
        flip = 0; ph = np.ones(s.N, dtype=complex)
        for q, o in ops.items():
            b = s.bits[q]
            if o == 'Z': ph = ph * (1 - 2 * b)
            elif o == 'X': flip |= (1 << q)
            elif o == 'Y': flip |= (1 << q); ph = ph * 1j * (1 - 2 * b)   # Y|0>=i|1>, Y|1>=-i|0>
        out = np.zeros_like(v); out[s.idx ^ flip] = ph * v
        return out

    def expP(s, v, ops, phi):  # exp(i phi P) v
        return np.cos(phi) * v + 1j * np.sin(phi) * s.pauli(v, ops)

    def rz(s, v, q, phi):  # exp(i phi Z_q)
        return v * np.exp(1j * phi * (1 - 2 * s.bits[q]))

    def rzz(s, v, a, b, phi):
        return v * np.exp(1j * phi * (1 - 2 * s.bits[a]) * (1 - 2 * s.bits[b]))

    def fswap(s, v, a, b):
        # SWAP . CZ : swap amplitudes of |01>,|10>, sign -1 on |11>
        ba, bb = s.bits[a], s.bits[b]
        out = np.empty_like(v)
        sw = s.idx ^ ((1 << a) | (1 << b)) * (ba != bb)   # partner index (same if ba==bb)
        out = v[sw] * np.where((ba == 1) & (bb == 1), -1.0, 1.0)
        return out

    # --- terms
    def S(s, v, dt):   # exp(-i dt H_S) = prod exp(+i dt/2 (XX+YY))
        for J in range(s.L - 1):
            a, b = 2 * J + 1, 2 * J + 2
            v = s.expP(v, {a: 'X', b: 'X'}, dt / 2); v = s.expP(v, {a: 'Y', b: 'Y'}, dt / 2)
        return v

    def Lg(s, v, dt):  # exp(-i dt H_L), direct four-qubit rotations
        for J in range(s.L - 1):
            a, b, c, d = 2 * J, 2 * J + 1, 2 * J + 2, 2 * J + 3
            v = s.expP(v, {a: 'X', b: 'Z', c: 'Z', d: 'X'}, dt / 2); v = s.expP(v, {a: 'Y', b: 'Z', c: 'Z', d: 'Y'}, dt / 2)
        return v

    def U2(s, v, dt, U):  # exp(-i dt U/4 sum Z Z)
        for J in range(s.L): v = s.rzz(v, 2 * J, 2 * J + 1, -dt * U / 4)
        return v

    def H1(s, v, dt, U, mu=0.0):  # exp(-i dt 1/2 (mu-U/2) sum Z)
        for q in range(s.n): v = s.rz(v, q, -dt * 0.5 * (mu - U / 2))
        return v

    def F(s, v):
        for J in range(s.L): v = s.fswap(v, 2 * J, 2 * J + 1)
        return v

    def neel(s):
        v = np.zeros(s.N, dtype=complex); st = sum(1 << (2 * i) for i in range(s.L)); v[st] = 1; return v

    def obs(s, v, swapped):
        """returns nup,ndn,ndd per site. wire pair of site i is (2i,2i+1). Unswapped labeling: even i: wire2i=down, odd i: wire2i=up."""
        p = np.abs(v) ** 2
        nu = np.zeros(s.L); nd = np.zeros(s.L); dd = np.zeros(s.L)
        for i in range(s.L):
            w0 = float(p @ s.bits[2 * i]); w1 = float(p @ s.bits[2 * i + 1])
            dd[i] = float(p @ (s.bits[2 * i] * s.bits[2 * i + 1]))
            first_is_up = (i % 2 == 1) ^ swapped
            nu[i], nd[i] = (w0, w1) if first_is_up else (w1, w0)
        return nu, nd, dd


def literal_circuit(L, nsteps, dt=0.2, U=-2.0, mu=0.0):
    q = Q(L); v = q.neel(); sw = False; out = []
    for k in range(1, nsteps + 1):
        v = q.H1(v, dt / 2, U, mu); v = q.S(v, dt); v = q.U2(v, dt, U); v = q.F(v); sw = not sw
        v = q.S(v, dt); v = q.H1(v, dt / 2, U, mu)
        out.append(q.obs(v, sw))
    return out


def direct_pauli(L, nsteps, dt=0.2, U=-2.0, mu=0.0):
    """no fSWAP: odd steps  A,S,U,L,A ; even steps A,L,U,S,A  (Eq. first_order_trotter and mirrored)"""
    q = Q(L); v = q.neel(); out = []
    for k in range(1, nsteps + 1):
        v = q.H1(v, dt / 2, U, mu)
        if k % 2 == 1: v = q.S(v, dt); v = q.U2(v, dt, U); v = q.Lg(v, dt)
        else: v = q.Lg(v, dt); v = q.U2(v, dt, U); v = q.S(v, dt)
        v = q.H1(v, dt / 2, U, mu)
        out.append(q.obs(v, False))
    return out


def continuous(L, times, U=-2.0, mu=0.0):
    """exact exp(-iHt) via Krylov (expm_multiply) with H = H_S+H_L+H_U2Q+H1Q built from Pauli strings."""
    from scipy.sparse.linalg import expm_multiply, LinearOperator
    q = Q(L)
    def Hv(v):
        w = np.zeros_like(v)
        for J in range(L - 1):
            a, b, c, d = 2 * J, 2 * J + 1, 2 * J + 2, 2 * J + 3
            w += -0.5 * (q.pauli(v, {b: 'X', c: 'X'}) + q.pauli(v, {b: 'Y', c: 'Y'}))
            w += -0.5 * (q.pauli(v, {a: 'X', b: 'Z', c: 'Z', d: 'X'}) + q.pauli(v, {a: 'Y', b: 'Z', c: 'Z', d: 'Y'}))
        diag = np.zeros(q.N)
        for J in range(L): diag += U / 4 * (1 - 2 * q.bits[2 * J]) * (1 - 2 * q.bits[2 * J + 1])
        for qq in range(q.n): diag += 0.5 * (mu - U / 2) * (1 - 2 * q.bits[qq])
        return w + diag * v
    v = q.neel(); out = []; tprev = 0.0
    for t in times:
        nsub = max(1, int(np.ceil((t - tprev) / 0.02))); h = (t - tprev) / nsub
        for _ in range(nsub):
            term = v; acc = v.copy()
            for m in range(1, 11):
                term = (-1j * h / m) * Hv(term); acc = acc + term
            v = acc
        tprev = t
        out.append(q.obs(v, False))
    return out

if __name__ == '__main__':
    L = int(sys.argv[1]); n = int(sys.argv[2])
    a = literal_circuit(L, n); b = direct_pauli(L, n)
    print('literal vs direct-pauli max diff', max(np.abs(np.array(x) - np.array(y)).max() for x, y in zip(a, b)))
