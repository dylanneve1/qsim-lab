#!/usr/bin/env python3
"""Independent check of the large exact energies: a plain sparse-determinant UCC simulator
(dict of determinants, exact Givens rotations exp(theta (tau - tau^dag)), JW sign rule) and a
Slater-Condon energy, sharing nothing with the Rust path except the PySCF integrals.

  python crosscheck.py NAME DMAX DATA_DIR      -> prints one JSON line (energy, nnz, seconds)

Compare with `lowmagic_chem energy DATA/NAME.fcidump DATA/NAME.spanDMAX.jw.prog 0` (e_init).
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chem  # noqa: E402


def apply_ladder(det, ops):
    """Apply a product of ladder ops (rightmost first) to a determinant bitmask.
    ops: list of (q, dagger) in operator order (leftmost first). Returns (sign, det) or None."""
    sign = 1
    for q, dag in reversed(ops):
        occ = (det >> q) & 1
        if dag == occ:
            return None
        if bin(det & ((1 << q) - 1)).count("1") % 2:
            sign = -sign
        det ^= 1 << q
    return sign, det


def ucc(gens, thetas, hf):
    psi = {hf: 1.0}
    for (occ, vir), th in zip(gens, thetas):
        tau = [(v, True) for v in vir] + [(o, False) for o in reversed(occ)]
        taud = [(o, True) for o in occ] + [(v, False) for v in reversed(vir)]
        c, s = math.cos(th), math.sin(th)
        new = {}
        for det, a in psi.items():
            r = apply_ladder(det, tau)
            rd = apply_ladder(det, taud)
            if r is None and rd is None:
                new[det] = new.get(det, 0.0) + a
                continue
            # exp(th(T - T^dag)) on the pair {det, T det}: det -> c det + s T det
            new[det] = new.get(det, 0.0) + c * a
            if r is not None:
                sg, d2 = r
                new[d2] = new.get(d2, 0.0) + s * sg * a
            else:
                sg, d2 = rd
                new[d2] = new.get(d2, 0.0) - s * sg * a
        psi = {k: v for k, v in new.items() if abs(v) > 1e-15}
    return psi


def energy(psi, h1, eri, ecore):
    def h(p, q):
        return h1[p // 2, q // 2] if p % 2 == q % 2 else 0.0

    def g(p, q, r, s):  # chemist (pq|rs) over spin orbitals
        if p % 2 != q % 2 or r % 2 != s % 2:
            return 0.0
        return eri[p // 2, q // 2, r // 2, s // 2]

    def anti(a, b, i, j):  # <ab||ij>
        return g(a, i, b, j) - g(a, j, b, i)

    dets = list(psi.items())
    e = 0.0
    for x, (d1, c1) in enumerate(dets):
        occ1 = [q for q in range(d1.bit_length()) if d1 >> q & 1]
        # diagonal
        ed = ecore + sum(h(i, i) for i in occ1)
        for a_, i in enumerate(occ1):
            for j in occ1[a_ + 1:]:
                ed += g(i, i, j, j) - g(i, j, j, i)
        e += c1 * c1 * ed
        for d2, c2 in dets[x + 1:]:
            diff = d1 ^ d2
            nd = bin(diff).count("1")
            if nd == 2:
                i = (d1 & diff).bit_length() - 1
                a = (d2 & diff).bit_length() - 1
                r = apply_ladder(d1, [(a, True), (i, False)])
                sg = r[0]
                m = h(a, i) + sum(g(a, i, j, j) - g(a, j, j, i) for j in occ1 if j != i)
                e += 2 * c1 * c2 * sg * m
            elif nd == 4:
                rem = d1 & diff
                add = d2 & diff
                i, j = [q for q in range(rem.bit_length()) if rem >> q & 1]
                a, b = [q for q in range(add.bit_length()) if add >> q & 1]
                r = apply_ladder(d1, [(a, True), (b, True), (j, False), (i, False)])
                e += 2 * c1 * c2 * r[0] * anti(a, b, i, j)
    return e


def main():
    name, dmax, data = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    from pyscf.tools import fcidump
    from pyscf import ao2mo

    t0 = time.time()
    mf, mycc, h1, eri, ecore, refs = chem.build(name)
    norb, nel = refs["norb"], refs["nelec"]
    singles, doubles = chem.ccsd_spin_amplitudes(mycc, norb, nel)
    allx = sorted(singles + doubles, key=lambda e: -abs(e[2]))
    sel, dim = chem.span_select(allx, dmax)
    # integrals exactly as in the FCIDUMP the Rust run used
    fd = fcidump.read(os.path.join(data, f"{name}.fcidump"), verbose=False)
    h1f = np.asarray(fd["H1"])
    erif = ao2mo.restore(1, fd["H2"], fd["NORB"])
    t1 = time.time()
    psi = ucc([(o, v) for o, v, _ in sel], [t for *_, t in sel], (1 << nel) - 1)
    norm = sum(a * a for a in psi.values())
    e = energy(psi, h1f, erif, fd["ECORE"])
    print(json.dumps({"name": name, "dmax": dmax, "k": len(sel), "dim": dim, "nnz": len(psi), "norm": norm,
                      "energy": e, "prep_secs": t1 - t0, "sim_secs": time.time() - t1}))


if __name__ == "__main__":
    main()
