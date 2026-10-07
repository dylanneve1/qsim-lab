"""Endgame for mpou2: once the middle operator W (MPO2, both leg kinds) is small, the outermost input-side blocks Rg
and output-side blocks Pg (shallow, all-to-all) are not absorbed into W in 1D; instead the exact single-qubit
marginals of  psi = Pg . W . Rg |0>  are computed by tensor-network contraction (quimb + cotengra path search),
with the output-side gates pruned to the backward light cone of the measured qubit (unitary cancellation).
Rg, Pg: lists of (a, b, M) in forward time order, M 4x4 on kron(a, b)."""
import numpy as np
import quimb.tensor as qtn

Z = np.diag([1., -1.]).astype(complex)


class Builder:
    def __init__(self, n, tag):
        self.n = n; self.tag = tag; self.cnt = 0
        self.cur = {}
        self.ts = []

    def new(self, w):
        self.cnt += 1
        ix = f'{self.tag}{w}_{self.cnt}'
        self.cur[w] = ix
        return ix


def psi_tn(W, Rg, Pg, n, tag='k', bra=False):
    """tensor network of psi (open output indices out{w}); bond indices get `tag` so two copies do not clash."""
    B = Builder(n, tag)
    ts = []
    z = np.array([1, 0], dtype=complex)
    for w in range(n):
        ts.append(qtn.Tensor(z, inds=[B.new(w)]))
    for a, b, M in Rg:
        ia, ib = B.cur[a], B.cur[b]
        oa, ob = B.new(a), B.new(b)
        ts.append(qtn.Tensor(M.reshape(2, 2, 2, 2), inds=[oa, ob, ia, ib]))
    # W: site s, up leg = wire su[s], dn leg = wire sd[s]
    din = {W.sd[s]: B.cur[W.sd[s]] for s in range(n)}
    outs = {}
    for s in range(n):
        A = W.A[s]
        outs[W.su[s]] = f'{tag}W{W.su[s]}_o'
    for s in range(n):
        A = W.A[s] * np.exp(W.lognorm / n)
        inds = [f'{tag}b{s - 1}', outs[W.su[s]], din[W.sd[s]], f'{tag}b{s}']
        if s == n - 1:
            A = A[..., 0]; inds = inds[:3]
        if s == 0:
            A = A[0]; inds = inds[1:]
        ts.append(qtn.Tensor(A, inds=inds))
    for w in range(n):
        B.cur[w] = outs[w]
    for a, b, M in Pg:
        ia, ib = B.cur[a], B.cur[b]
        oa, ob = B.new(a), B.new(b)
        ts.append(qtn.Tensor(M.reshape(2, 2, 2, 2), inds=[oa, ob, ia, ib]))
    tn = qtn.TensorNetwork(ts)
    final = dict(B.cur)
    return tn, final


def cone(Pg, q):
    S = {q}; keep = []
    for g in reversed(Pg):
        a, b, _ = g
        if a in S or b in S:
            S |= {a, b}; keep.append(g)
    return keep[::-1]


def contract(tn, opt):
    tn = tn.full_simplify(seq='ADCR', output_inds=(), equalize_norms=True)
    v = tn.contract(all, optimize=opt, output_inds=())
    return v


def marginals(W, Rg, Pg, n, opt='auto-hq', log=print):
    """returns (<Z_q> list, norm^2)."""
    # norm: Pg cancels
    k, fk = psi_tn(W, Rg, [], n, 'k')
    bb, fb = psi_tn(W, Rg, [], n, 'c')
    bb = bb.conj()
    bb = bb.reindex({fb[w]: fk[w] for w in range(n)})
    nrm = complex(contract(k | bb, opt))
    zs = []
    for q in range(n):
        Pq = cone(Pg, q)
        k, fk = psi_tn(W, Rg, Pq, n, 'k')
        bb, fb = psi_tn(W, Rg, Pq, n, 'c')
        bb = bb.conj()
        mp = {fb[w]: fk[w] for w in range(n) if w != q}
        bb = bb.reindex(mp)
        tz = qtn.Tensor(Z, inds=[fk[q], fb[q]])
        v = complex(contract(k | bb | tz, opt))
        zs.append((v / nrm).real)
        if log and q % 8 == 0:
            log(f'    endgame marginal {q}: <Z>={zs[-1]:+.4f} cone {len(Pq)} gates')
    return np.array(zs), nrm.real


def amplitude(W, Rg, Pg, n, bits, opt='auto-hq'):
    k, fk = psi_tn(W, Rg, Pg, n, 'k')
    ts = []
    for w in range(n):
        v = np.zeros(2, complex); v[int(bits[w])] = 1
        ts.append(qtn.Tensor(v, inds=[fk[w]]))
    return complex(contract(k | qtn.TensorNetwork(ts), opt))
