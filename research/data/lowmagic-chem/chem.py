#!/usr/bin/env python3
"""Low-magic chemistry driver (research/simulability/lowmagic-chem.md).

Molecules (PySCF) -> FCIDUMP + classical references (HF, MP2, CCSD, CCSD(T), FCI where
tractable) and Pauli-rotation programs (OpenFermion JW/BK) for the Rust driver
`examples/lowmagic_chem.rs`.

  python chem.py prep NAME OUT_DIR           # one molecule: fcidump, refs.json, programs
  python chem.py list                        # molecule names
  python chem.py rank NAME OUT_DIR           # GF(2) rank of excitation x-vectors (any size)

Program format: see src/chem.rs (`n`, `param`, `x q`, `h q`, `rot A P`, `prot K MULT P`).
Spin orbitals are interleaved (qubit 2p + sigma), Jordan-Wigner as in OpenFermion.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------- molecules


def hchain(nh, r):
    return [("H", (0.0, 0.0, i * r)) for i in range(nh)]


def hsheet(a, b, r):
    return [("H", (i * r, j * r, 0.0)) for i in range(a) for j in range(b)]


MOLS = {
    # name: (atoms, basis, frozen core orbitals, fci?)
    "h2": (hchain(2, 0.7414), "sto-3g", 0),
    "h2_s": (hchain(2, 2.5), "sto-3g", 0),
    "lih": ([("Li", (0, 0, 0)), ("H", (0, 0, 1.5949))], "sto-3g", 0),
    "h4": (hchain(4, 1.0), "sto-3g", 0),
    "h4_s": (hchain(4, 2.0), "sto-3g", 0),
    "h6": (hchain(6, 1.0), "sto-3g", 0),
    "h6_s": (hchain(6, 2.0), "sto-3g", 0),
    "beh2": ([("Be", (0, 0, 0)), ("H", (0, 0, 1.3264)), ("H", (0, 0, -1.3264))], "sto-3g", 0),
    "h2o": ([("O", (0, 0, 0)), ("H", (0.7572, 0.5865, 0)), ("H", (-0.7572, 0.5865, 0))], "sto-3g", 0),
    "n2": ([("N", (0, 0, 0)), ("N", (0, 0, 1.0977))], "sto-3g", 2),
    "n2_s": ([("N", (0, 0, 0)), ("N", (0, 0, 2.0))], "sto-3g", 2),
    "h8": (hchain(8, 1.0), "sto-3g", 0),
    "h10": (hchain(10, 1.0), "sto-3g", 0),
    "h10_s": (hchain(10, 1.8), "sto-3g", 0),
    "h3x4": (hsheet(3, 4, 1.0), "sto-3g", 0),
    "h3x4_s": (hsheet(3, 4, 1.8), "sto-3g", 0),
    # beyond state-vector size
    "h4x4": (hsheet(4, 4, 1.0), "sto-3g", 0),
    "h4x4_s": (hsheet(4, 4, 1.8), "sto-3g", 0),
    "h4x5_s": (hsheet(4, 5, 1.8), "sto-3g", 0),
    "h20": (hchain(20, 1.0), "sto-3g", 0),
    "h30": (hchain(30, 1.0), "sto-3g", 0),
    "h50": (hchain(50, 1.0), "sto-3g", 0),
    "h50_s": (hchain(50, 1.8), "sto-3g", 0),
    "h2o_dz": ([("O", (0, 0, 0)), ("H", (0.7572, 0.5865, 0)), ("H", (-0.7572, 0.5865, 0))], "cc-pvdz", 1),
    "n2_dz": ([("N", (0, 0, 0)), ("N", (0, 0, 1.0977))], "cc-pvdz", 2),
    "n2_dz_s": ([("N", (0, 0, 0)), ("N", (0, 0, 2.0))], "cc-pvdz", 2),
}

FCI_MAX_DIM = 3e7


def build(name):
    from pyscf import gto, scf, mcscf, ao2mo, cc, mp, fci

    atoms, basis, ncore = MOLS[name]
    mol = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        mf = mf.newton().run()
    nmo = mf.mo_coeff.shape[1]
    norb = nmo - ncore
    nel = mol.nelectron - 2 * ncore
    mc = mcscf.CASCI(mf, norb, nel)
    h1, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), norb)
    refs = {"name": name, "basis": basis, "norb": norb, "nelec": nel, "ncore": ncore, "qubits": 2 * norb}
    refs["e_hf"] = float(mf.e_tot)
    t0 = time.time()
    pt = mp.MP2(mf, frozen=ncore or None).run()
    refs["e_mp2"] = float(pt.e_tot)
    mycc = cc.CCSD(mf, frozen=ncore or None)
    mycc.conv_tol = 1e-9
    mycc.max_cycle = 300
    mycc.kernel()
    refs["e_ccsd"] = float(mycc.e_tot)
    refs["ccsd_converged"] = bool(mycc.converged)
    try:
        refs["e_ccsd_t"] = float(mycc.e_tot + mycc.ccsd_t())
    except Exception as e:  # pragma: no cover
        refs["e_ccsd_t"] = None
        refs["ccsd_t_error"] = str(e)
    refs["cc_secs"] = time.time() - t0
    from math import comb

    na = nel // 2
    dim = comb(norb, na) ** 2
    refs["fci_dim"] = dim
    if dim <= FCI_MAX_DIM:
        t0 = time.time()
        cis = fci.direct_spin1.FCI()
        cis.conv_tol = 1e-11
        e, _ = cis.kernel(h1, eri, norb, nel, ecore=ecore)
        refs["e_fci"] = float(e)
        refs["fci_secs"] = time.time() - t0
    else:
        refs["e_fci"] = None
    return mf, mycc, h1, eri, ecore, refs


# ----------------------------------------------------------------------------- excitations


def ccsd_spin_amplitudes(mycc, norb, nel):
    """Spin-orbital singles/doubles (interleaved qubits) with CCSD amplitudes.

    Returns lists of (occ tuple, vir tuple, amplitude), occ/vir as qubit indices.
    """
    from pyscf.cc import addons

    nocc = nel // 2
    t1 = addons.spatial2spin(mycc.t1)  # (2nocc, 2nvir), interleaved within blocks
    t2 = addons.spatial2spin(mycc.t2)  # (2nocc, 2nocc, 2nvir, 2nvir)
    no, nv = t1.shape
    singles, doubles = [], []
    for i in range(no):
        for a in range(nv):
            if abs(t1[i, a]) > 1e-10:
                singles.append(((i,), (2 * nocc + a,), float(t1[i, a])))
    for i, j in itertools.combinations(range(no), 2):
        for a, b in itertools.combinations(range(nv), 2):
            v = t2[i, j, a, b]
            if abs(v) > 1e-10:
                doubles.append(((i, j), (2 * nocc + a, 2 * nocc + b), float(v)))
    return singles, doubles


def excitation_op(occ, vir):
    import openfermion as of

    term = tuple((v, 1) for v in vir) + tuple((o, 0) for o in reversed(occ))
    op = of.FermionOperator(term, 1.0)
    return op - of.hermitian_conjugated(op)


def generator_paulis(occ, vir, n, enc="jw"):
    """exp(theta (T - T^dag)) = prod_j exp(-i (mult_j theta)/2 P_j); returns [(mult_j, P_j)]."""
    import openfermion as of

    op = excitation_op(occ, vir)
    q = of.jordan_wigner(op) if enc == "jw" else of.bravyi_kitaev(op, n_qubits=n)
    out = []
    for term, v in q.terms.items():
        if abs(v) < 1e-14:
            continue
        assert abs(v.real) < 1e-12, (term, v)
        out.append((-2.0 * v.imag, " ".join(f"{s}{i}" for i, s in term)))
    return out


def write_program(path, n, occ_qubits, gens, params, meta, enc="jw"):
    """gens: list of (occ, vir); params: initial angles (one per generator)."""
    with open(path, "w") as f:
        f.write(f"n {n}\n")
        for k, v in meta.items():
            f.write(f"# {k} {v}\n")
        for k, p in enumerate(params):
            f.write(f"param {k} {float(p)!r}\n")
        for q in occ_qubits:
            f.write(f"x {q}\n")
        for k, (o, v) in enumerate(gens):
            for mult, ps in generator_paulis(o, v, n, enc):
                f.write(f"prot {k} {float(mult)!r} {ps}\n")


def hf_qubits(nel, enc, n):
    occ = list(range(nel))
    if enc == "jw":
        return occ
    # Bravyi-Kitaev basis state of the HF occupation vector
    vec = np.zeros(n, dtype=int)
    vec[occ] = 1
    # (only used for profiles: d does not depend on the reference bit string)
    enc_mat = bk_matrix(n)
    b = enc_mat.dot(vec) % 2
    return [int(i) for i in np.nonzero(b)[0]]


def bk_matrix(n):
    """Bravyi-Kitaev encoder matrix (Seeley-Richard-Love 2012), qubit j = sum_k beta_jk n_k mod 2."""
    size = 1
    while size < n:
        size *= 2
    beta = np.array([[1]], dtype=int)
    while beta.shape[0] < size:
        m = beta.shape[0]
        nb = np.zeros((2 * m, 2 * m), dtype=int)
        nb[:m, :m] = beta
        nb[m:, m:] = beta
        nb[2 * m - 1, :m] = 1
        beta = nb
    return beta[:n, :n]


def pair_excitations(nocc, norb):
    gens = []
    for i in range(nocc):
        for a in range(nocc, norb):
            gens.append(((2 * i, 2 * i + 1), (2 * a, 2 * a + 1)))
    return gens


def upccgsd(norb, k):
    gens = []
    for _ in range(k):
        for p, q in itertools.combinations(range(norb), 2):
            gens.append(((2 * p, 2 * p + 1), (2 * q, 2 * q + 1)))
        for p, q in itertools.combinations(range(norb), 2):
            for s in (0, 1):
                gens.append(((2 * p + s,), (2 * q + s,)))
    return gens


# ----------------------------------------------------------------------------- Hamiltonians


def qubit_hamiltonian(h1, eri, ecore, enc="jw"):
    import openfermion as of

    norb = h1.shape[0]
    op = of.FermionOperator((), ecore)
    for p in range(norb):
        for q in range(norb):
            if abs(h1[p, q]) > 1e-12:
                for s in (0, 1):
                    op += of.FermionOperator(((2 * p + s, 1), (2 * q + s, 0)), h1[p, q])
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        v = eri[p, q, r, s]
        if abs(v) < 1e-12:
            continue
        for sg in (0, 1):
            for tu in (0, 1):
                i, j, k, l = 2 * p + sg, 2 * q + sg, 2 * r + tu, 2 * s + tu
                if i == k or j == l:
                    continue
                op += of.FermionOperator(((i, 1), (k, 1), (l, 0), (j, 0)), 0.5 * v)
    n = 2 * norb
    return of.jordan_wigner(op) if enc == "jw" else of.bravyi_kitaev(op, n_qubits=n)


def write_trotter(path, n, occ_qubits, qham, dt, steps, meta):
    terms = [(t, float(np.real(c))) for t, c in sorted(qham.terms.items()) if t and abs(c) > 1e-12]
    with open(path, "w") as f:
        f.write(f"n {n}\n")
        for k, v in meta.items():
            f.write(f"# {k} {v}\n")
        f.write(f"# rotations_per_step {len(terms)}\n")
        for q in occ_qubits:
            f.write(f"x {q}\n")
        for _ in range(steps):
            for t, c in terms:
                ps = " ".join(f"{s}{i}" for i, s in t)
                f.write(f"rot {2.0 * c * dt!r} {ps}\n")
    return len(terms)


def write_qpe(path, n, occ_qubits, qham, tau, t, meta):
    """Textbook QPE: t counting qubits (n..n+t-1) in |+>, controlled U^(2^k) with U one
    first-order Trotter step exp(-i tau H), inverse QFT; all as Pauli rotations + H."""
    terms = [(tt, float(np.real(c))) for tt, c in sorted(qham.terms.items()) if tt and abs(c) > 1e-12]
    nt = n + t
    lines = [f"n {nt}"] + [f"# {k} {v}" for k, v in meta.items()]
    lines += [f"x {q}" for q in occ_qubits] + [f"h {n + k}" for k in range(t)]
    for k in range(t):
        c = n + k
        for _ in range(2**k):
            for tt, co in terms:
                th = 2.0 * co * tau
                ps = " ".join(f"{s_}{i}" for i, s_ in tt)
                # controlled exp(-i th/2 P) = exp(-i th/4 P) exp(+i th/4 Z_c P)
                lines.append(f"rot {th / 2!r} {ps}")
                lines.append(f"rot {-th / 2!r} {ps} Z{c}")
    # inverse QFT on the counting register (qubit n+t-1 most significant)
    for j in reversed(range(t)):
        for m in reversed(range(j + 1, t)):
            phi = -np.pi / 2 ** (m - j)
            a, b = n + j, n + m
            lines += [f"rot {phi / 2!r} Z{a}", f"rot {phi / 2!r} Z{b}", f"rot {-phi / 2!r} Z{a} Z{b}"]
        lines.append(f"h {n + j}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def lattice_programs(out):
    """Trotter programs for 1D Hubbard (JW, 2L qubits) and J1-J2 Heisenberg chains (L qubits)."""
    import openfermion as of

    rows = []
    for L, u, tt in ((6, 4.0, 1.0), (6, 0.0, 1.0), (6, 4.0, 0.0), (10, 4.0, 1.0)):
        h = of.fermi_hubbard(1, L, tunneling=tt, coulomb=u, periodic=False)
        q = of.jordan_wigner(h)
        occ = [2 * i + (i % 2) for i in range(L)]  # Neel-like half filling
        name = f"hubbard_L{L}_U{u}_t{tt}"
        write_trotter(os.path.join(out, f"{name}.trot0.1.jw.prog"), 2 * L, occ, q, 0.1, 2,
                      {"model": name, "ansatz": "trotter dt=0.1 x2", "enc": "jw"})
        rows.append(name)
    for L, j2 in ((12, 0.0), (12, 0.5)):
        h = of.QubitOperator()
        for i in range(L - 1):
            for s_ in "XYZ":
                h += of.QubitOperator(f"{s_}{i} {s_}{i + 1}", 1.0)
        for i in range(L - 2):
            for s_ in "XYZ":
                h += of.QubitOperator(f"{s_}{i} {s_}{i + 2}", j2)
        name = f"heis_L{L}_J2{j2}"
        # dimer (singlet-product) reference: X on odd sites then (H, CNOT) is Clifford; we write
        # the singlet product with h/cx/x/z lines: |01>-|10> = CNOT (H x I)|0 1> then Z
        p = os.path.join(out, f"{name}.trot0.1.dimer.prog")
        write_trotter(p, L, [], h, 0.1, 2, {"model": name, "ansatz": "trotter dt=0.1 x2 from dimer state", "enc": "spin"})
        body = open(p).read().splitlines()
        hdr = [ln for ln in body if ln.startswith(("n ", "#"))]
        rest = [ln for ln in body if not ln.startswith(("n ", "#"))]
        prep = []
        for i in range(0, L, 2):
            prep += [f"x {i + 1}", f"h {i}", f"cx {i} {i + 1}", f"z {i}"]
        open(p, "w").write("\n".join(hdr + prep + rest) + "\n")
        rows.append(name)
    return rows


# ----------------------------------------------------------------------------- GF(2) rank


def gf2_rank(vectors):
    rows = []
    for v in vectors:
        x = int(v)
        for r in rows:
            x = min(x, x ^ r)
        if x:
            rows.append(x)
    return len(rows)


def span_select(allx, dmax):
    """Span-closed selection: excitations by |t|, kept while the GF(2) span of their
    x-vectors has dimension <= dmax (members of the span are always kept)."""
    rows, sel = [], []
    for o, v, t in allx:
        r = xvec(o, v)
        for b in rows:
            r = min(r, r ^ b)
        if r == 0:
            sel.append((o, v, t))
        elif len(rows) < dmax:
            rows.append(r)
            sel.append((o, v, t))
    return sel, len(rows)


def xvec(occ, vir):
    v = 0
    for q in tuple(occ) + tuple(vir):
        v ^= 1 << q
    return v


# ----------------------------------------------------------------------------- prep


def prep(name, out, small_only=False):
    from pyscf.tools import fcidump

    os.makedirs(out, exist_ok=True)
    mf, mycc, h1, eri, ecore, refs = build(name)
    norb, nel = refs["norb"], refs["nelec"]
    n = 2 * norb
    nocc = nel // 2
    fcidump.from_integrals(os.path.join(out, f"{name}.fcidump"), h1, eri, norb, nel, nuc=ecore, ms=0, tol=1e-12)
    singles, doubles = ccsd_spin_amplitudes(mycc, norb, nel)
    refs["n_singles"] = len(singles)
    refs["n_doubles"] = len(doubles)
    refs["l1_t1"] = float(sum(abs(t) for *_, t in singles))
    refs["l1_t2"] = float(sum(abs(t) for *_, t in doubles))
    # d of the standard ansaetze from the GF(2) rank (any size)
    refs["d_uccsd"] = gf2_rank([xvec(o, v) for o, v, _ in singles + doubles])
    refs["d_uccd"] = gf2_rank([xvec(o, v) for o, v, _ in doubles])
    refs["d_pair"] = gf2_rank([xvec(o, v) for o, v in pair_excitations(nocc, norb)])
    refs["d_upccgsd"] = gf2_rank([xvec(o, v) for o, v in upccgsd(norb, 1)])
    occ_q = list(range(nel))
    # selected UCC: top-K CCSD amplitudes (doubles and singles together), largest first
    allx = sorted(singles + doubles, key=lambda e: -abs(e[2]))
    refs["top_amplitudes"] = [[list(o), list(v), t] for o, v, t in allx[:40]]
    for k in (1, 2, 4, 8, 12, 16, 20, 24, 28):
        if k > len(allx):
            break
        sel = allx[:k]
        refs[f"d_sel{k}"] = gf2_rank([xvec(o, v) for o, v, _ in sel])
        write_program(
            os.path.join(out, f"{name}.sel{k}.jw.prog"),
            n,
            occ_q,
            [(o, v) for o, v, _ in sel],
            [t for *_, t in sel],
            {"mol": name, "ansatz": f"sel{k}", "enc": "jw"},
        )
    # span-budget selection: walk the CCSD excitations by |t|; keep one if the GF(2)
    # span stays within D dimensions (excitations already in the span are free)
    for dmax in (8, 12, 16, 20, 24):
        sel, dim = span_select(allx, dmax)
        refs[f"span{dmax}_k"] = len(sel)
        write_program(
            os.path.join(out, f"{name}.span{dmax}.jw.prog"),
            n,
            occ_q,
            [(o, v) for o, v, _ in sel],
            [t for *_, t in sel],
            {"mol": name, "ansatz": f"span{dmax}", "enc": "jw", "k": len(sel)},
        )
        if dim < dmax:
            break
    small = n <= 24
    if small:
        # full ansaetze (CCSD amplitudes as angles), JW and BK
        for enc in ("jw", "bk"):
            occb = hf_qubits(nel, enc, n)
            write_program(
                os.path.join(out, f"{name}.uccsd.{enc}.prog"),
                n,
                occb,
                [(o, v) for o, v, _ in singles + doubles],
                [t for *_, t in singles + doubles],
                {"mol": name, "ansatz": "uccsd", "enc": enc},
                enc,
            )
        write_program(
            os.path.join(out, f"{name}.uccd.jw.prog"),
            n,
            occ_q,
            [(o, v) for o, v, _ in doubles],
            [t for *_, t in doubles],
            {"mol": name, "ansatz": "uccd", "enc": "jw"},
        )
        pg = pair_excitations(nocc, norb)
        write_program(
            os.path.join(out, f"{name}.pucc.jw.prog"),
            n,
            occ_q,
            pg,
            [0.01] * len(pg),
            {"mol": name, "ansatz": "pUCCD", "enc": "jw"},
        )
        for k in (1, 2):
            gg = upccgsd(norb, k)
            rng = np.random.default_rng(k)
            write_program(
                os.path.join(out, f"{name}.upccgsd{k}.jw.prog"),
                n,
                occ_q,
                gg,
                list(rng.normal(0, 0.05, len(gg))),
                {"mol": name, "ansatz": f"{k}-UpCCGSD", "enc": "jw"},
            )
    if n <= 12:
        qh = qubit_hamiltonian(h1, eri, ecore, "jw")
        for t in (3, 5):
            write_qpe(os.path.join(out, f"{name}.qpe{t}.jw.prog"), n, occ_q, qh, 0.1, t,
                      {"mol": name, "ansatz": f"QPE t={t} tau=0.1, HF reference", "enc": "jw"})
    if n <= 20:
        for enc in ("jw", "bk"):
            qh = qubit_hamiltonian(h1, eri, ecore, enc)
            occb = hf_qubits(nel, enc, n)
            for dt in (0.01, 0.1, 0.5):
                nt = write_trotter(
                    os.path.join(out, f"{name}.trot{dt}.{enc}.prog"),
                    n,
                    occb,
                    qh,
                    dt,
                    2,
                    {"mol": name, "ansatz": f"trotter dt={dt} x2", "enc": enc},
                )
            refs[f"ham_terms_{enc}"] = nt
    with open(os.path.join(out, f"{name}.refs.json"), "w") as f:
        json.dump(refs, f, indent=1)
    print(json.dumps({k: v for k, v in refs.items() if k != "top_amplitudes"}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        print(" ".join(MOLS))
    elif cmd == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif cmd == "lattice":
        print(lattice_programs(sys.argv[2]))
    else:
        print(__doc__)
