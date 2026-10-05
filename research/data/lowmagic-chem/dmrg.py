#!/usr/bin/env python3
"""DMRG references (block2, SU(2)) for the molecules too large for FCI.

  python dmrg.py NAME OUT_DIR [MAXM]   ->  OUT_DIR/NAME.dmrg.json

Same frozen core / active space as chem.py. Hydrogen chains and sheets use Loewdin
orthogonalised AOs in their natural (snake) order; other molecules use the canonical
active MOs with block2's Fiedler reordering. The energy is the variational DMRG energy
of the last sweep at bond dimension MAXM (an upper bound; no extrapolation).
"""
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chem  # noqa: E402


def main():
    name, out = sys.argv[1], sys.argv[2]
    maxm = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    from pyscf import gto, scf, mcscf, ao2mo, lo
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    atoms, basis, ncore = chem.MOLS[name]
    mol = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    nmo = mf.mo_coeff.shape[1]
    norb, nel = nmo - ncore, mol.nelectron - 2 * ncore
    hydrogen_only = all(a[0] == "H" for a in atoms)
    if hydrogen_only and ncore == 0:
        c = lo.orth_ao(mol, "lowdin")
        h1 = c.T @ mf.get_hcore() @ c
        eri = ao2mo.restore(1, ao2mo.kernel(mol, c), norb)
        ecore = mol.energy_nuc()
        order = None
    else:
        mc = mcscf.CASCI(mf, norb, nel)
        h1, ecore = mc.get_h1eff()
        eri = ao2mo.restore(1, mc.get_h2eff(), norb)
        order = "fiedler"
    scratch = os.path.expanduser(f"~/qsim-lmc-scratch/{name}")
    os.makedirs(scratch, exist_ok=True)
    t0 = time.time()
    drv = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SU2, n_threads=int(os.environ.get("DMRG_THREADS", "1")), stack_mem=int(1.0e9))
    drv.initialize_system(n_sites=norb, n_elec=nel, spin=0)
    if order == "fiedler":
        idx = drv.orbital_reordering(h1, eri)
        h1 = h1[idx][:, idx]
        eri = eri[idx][:, idx][:, :, idx][:, :, :, idx]
    mpo = drv.get_qc_mpo(h1e=h1, g2e=eri, ecore=ecore, iprint=0)
    ket = drv.get_random_mps(tag="KET", bond_dim=min(250, maxm), nroots=1)
    ms = [m for m in (200,) if m < maxm] + [maxm]
    bond_dims, noises, thrds = [], [], []
    for m in ms:
        bond_dims += [m] * 4
        noises += [1e-5] * 4
        thrds += [1e-9] * 4
    bond_dims += [maxm] * 3
    noises += [0.0] * 3
    thrds += [1e-10] * 3
    e = drv.dmrg(mpo, ket, n_sweeps=len(bond_dims), bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1)
    sweep_e = drv.get_dmrg_results()[1] if hasattr(drv, "get_dmrg_results") else None
    rec = {
        "name": name,
        "norb": norb,
        "nelec": nel,
        "maxm": maxm,
        "e_dmrg": float(e),
        "basis_for_dmrg": "lowdin-ao" if order is None else "canonical-mo-fiedler",
        "secs": time.time() - t0,
    }
    try:
        bd, dw, en = drv.get_dmrg_results()
        rec["sweep_energies"] = [float(np.ravel(x)[0]) for x in en]
        rec["discarded_weights"] = [float(x) for x in dw]
    except Exception as ex:  # pragma: no cover
        rec["note"] = str(ex)
    del sweep_e
    shutil.rmtree(scratch, ignore_errors=True)
    with open(os.path.join(out, f"{name}.dmrg.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("sweep_energies", "discarded_weights")}))


if __name__ == "__main__":
    main()
