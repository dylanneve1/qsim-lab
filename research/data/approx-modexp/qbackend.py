"""A genuinely quantum backend for the paper's `scatter_script` simulator, used
to run the paper's own `approx_modexp` (`facto/algorithm/_detailed_example_code.py`,
unchanged) on the FULL superposition of a small instance.

Differences from scatter_script's QPU (which follows a few random classical
trajectories):

* superposed registers are allocated as the whole superposition: every value
  of the exponent register times every value of the mask (Cartesian product of
  branches), each branch with a complex amplitude;
* `mx_rz` / `del_measure_x` is a real projective X-basis measurement: the
  outcome's probability is computed from the amplitudes (branches that agree
  on every other live register are summed, i.e. they interfere), the state is
  projected, renormalised and the register reset; merging branches is
  recorded as an information-leak finding (it cannot happen if the measured
  register is a function of the rest);
* at the end the exponent register is measured in the frequency basis by an
  exact QFT (numpy FFT) for every value of the measured output register.

Usage (from this directory, with the Rust driver built):

    python3 qbackend.py N=143 g=2 m=10 f=6 mask=2 [seed=1]

writes `qb_<tag>_*.txt` and compares with the Rust simulator replaying the
same outcomes (`approx_modexp replay`) and with its `dist` output.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
import time

import numpy as np

import gidney_env as ge

from facto.algorithm._detailed_example_code import approx_modexp  # the paper's code
from scatter_script import QPU, quint  # noqa: F401
from scatter_script import _quint as _qmod

TWO64 = float(2**64)


class QuantumQPU(QPU):
    """scatter_script QPU whose branches are the full superposition."""

    def __init__(self, seed: int = 1):
        super().__init__(num_branches=1)
        self.amp = np.ones(1, dtype=np.complex128)
        self.rng = random.Random(seed)
        self.log: list[tuple[int, int]] = []
        self.merges = 0
        self.max_p_err = 0.0
        self.dead: set[int] = set()

    # -- superposition bookkeeping --------------------------------------
    def fold_phases(self) -> None:
        ph = self.branch_phases_u64
        if np.any(ph):
            turns = ph.astype(np.float64) / TWO64
            fac = np.exp(2j * np.pi * turns)
            # exact signs for the half-turn phases this algorithm uses
            fac[ph == np.uint64(1 << 63)] = -1.0
            fac[ph == 0] = 1.0
            self.amp = self.amp * fac
            self.branch_phases_u64[:] = 0

    def live_buffers(self):
        return [q._buffer for q in self.allocated if id(q._buffer) not in self.dead]

    def _dealloc_register(self, register):
        super()._dealloc_register(register)
        if register._offset == 0 and register._length == len(
            next(q for q in self.allocated if q._buffer is register._buffer)
        ):
            self.dead.add(id(register._buffer))

    def alloc_quint(self, *, length, val=None, scatter=False, scatter_range=None):
        if not scatter:
            return super().alloc_quint(length=length, val=val)
        if scatter_range is None:
            scatter_range = range(1 << length)
        elif isinstance(scatter_range, int):
            scatter_range = range(scatter_range)
        vals = list(scatter_range)
        r = len(vals)
        nb = self.num_branches
        # repeat every existing branch r times (Cartesian product)
        for q in self.allocated:
            q._buffer[:] = [v for v in q._buffer for _ in range(r)]
        self.branch_phases_u64 = np.repeat(self.branch_phases_u64, r)
        self.amp = np.repeat(self.amp, r) / math.sqrt(r)
        self.num_branches = nb * r
        buf = [v for _ in range(nb) for v in vals]
        result = quint(buffer=buf, offset=0, length=length, parent=self, alloc_id=len(self.allocated))
        self.allocated.append(result)
        self._alloc_count += length
        return result

    def measure_x(self, view) -> int:
        """Projective X-basis measurement of `view`, then reset to |0>."""
        self.fold_phases()
        L = len(view)
        vals = view.UNPHYSICAL_branch_vals
        x = self.rng.randrange(1 << L)
        self.log.append((x, L))
        own = view._buffer
        m = ((1 << view._length) - 1) << view._offset
        bufs = self.live_buffers()
        nb = self.num_branches
        keys = [tuple((b[k] & ~m) if b is own else b[k] for b in bufs) for k in range(nb)]
        sgn = np.array([-1.0 if (x & v).bit_count() & 1 else 1.0 for v in vals])
        contrib = self.amp * sgn
        groups: dict = {}
        for k, key in enumerate(keys):
            groups.setdefault(key, []).append(k)
        # P(x) = 2^-L * sum_groups |sum_b amp_b (-1)^{x.R_b}|^2
        if len(groups) == nb:
            p_x = float(np.sum(np.abs(contrib) ** 2)) / (1 << L)
            self.amp = contrib / math.sqrt((1 << L) * p_x)
            view.UNPHYSICAL_force_del(dealloc=False)
        else:
            self.merges += 1
            new_amp = []
            keep = []
            for key, ks in groups.items():
                new_amp.append(sum(contrib[k] for k in ks))
                keep.append(ks[0])
            new_amp = np.array(new_amp)
            p_x = float(np.sum(np.abs(new_amp) ** 2)) / (1 << L)
            for q in self.allocated:
                q._buffer[:] = [q._buffer[k] for k in keep]
            self.branch_phases_u64 = self.branch_phases_u64[keep]
            self.num_branches = len(keep)
            self.amp = new_amp / math.sqrt((1 << L) * p_x)
            view.UNPHYSICAL_force_del(dealloc=False)
        self.max_p_err = max(self.max_p_err, abs(p_x * (1 << L) - 1.0))
        return x


def _mx_rz(self):
    return self._parent.measure_x(self)


def _del_measure_x(self):
    r = self._parent.measure_x(self)
    self._parent._dealloc_register(self)
    return r


_qmod.quint.mx_rz = _mx_rz
_qmod.quint.del_measure_x = _del_measure_x


def main(argv: list[str]) -> int:
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    tag = kv.get("tag", "small")
    seed = int(kv.get("seed", "1"))
    out = ge.HERE / "xcheck"
    out.mkdir(exist_ok=True)
    rust_args = [f"{k}={v}" for k, v in kv.items() if k not in ("tag", "seed")]
    dump_dir = out / f"qb_{tag}"
    ge.run_driver("dump", *rust_args, f"out={dump_dir}")
    rc = ge.read_rust_config(dump_dir / "rust_config.txt")
    eh = kv.get("mode") == "eh"
    mult = ge.eh_multipliers(rc, int(kv.get("s", "1"))) if eh else None
    conf = ge.paper_exec_config(rc, multipliers=mult)
    t0 = time.monotonic()
    qpu = QuantumQPU(seed=seed)
    m = conf.num_input_qubits
    Q_e = qpu.alloc_quint(length=m, scatter=True)
    Q_res = approx_modexp(Q_exponent=Q_e, conf=conf, qpu=qpu)
    qpu.fold_phases()
    secs = time.monotonic() - t0
    # every other register must be clean
    others = [q for q in qpu.allocated if q._buffer is not Q_e._buffer and q._buffer is not Q_res._buffer]
    dirty = sum(1 for q in others if any(q._buffer))
    nb = qpu.num_branches
    e_vals = np.array(Q_e.UNPHYSICAL_branch_vals, dtype=np.int64)
    acc_vals = np.array(Q_res.UNPHYSICAL_branch_vals, dtype=np.int64)
    W = 1 << rc["mask_bits"]
    # amplitudes: all equal (global phase aside) iff every phase kickback was corrected
    a0 = qpu.amp[0]
    phase_spread = float(np.max(np.abs(qpu.amp / a0 - 1.0)))
    expect_mag = 1.0 / math.sqrt((1 << m) * W)
    mag_err = float(np.max(np.abs(np.abs(qpu.amp) - expect_mag)))
    print(
        f"[python-quantum] N={rc['modulus']} m={m} mask={rc['mask_bits']} branches={nb} "
        f"draws={len(qpu.log)} merges={qpu.merges} max|2^L P(x)-1|={qpu.max_p_err:.2e} "
        f"dirty_registers={dirty} amp_phase_spread={phase_spread:.2e} mag_err={mag_err:.2e} "
        f"global_amp={a0 / expect_mag:.6f} secs={secs:.1f}"
    )
    # outcome log for the Rust replay
    log_path = out / f"qb_{tag}_outcomes.txt"
    log_path.write_text("".join(f"{v},{n}\n" for v, n in qpu.log))
    # Rust replays the same outcomes
    f_path = out / f"qb_{tag}_rust_F.txt"
    rep = ge.run_driver("replay", *rust_args, f"log={log_path}", f"out={f_path}")
    print("[rust replay]", rep.strip().replace("\n", " | "))
    F = np.zeros(1 << m, dtype=np.int64)
    for line in f_path.read_text().splitlines()[1:]:
        e, f = line.split(",")
        F[int(e)] = int(f)
    trunc = rc["modulus"] >> max(0, rc["modulus"].bit_length() - rc["len_accumulator"])
    # Python's (e, acc) for every branch vs Rust's (s + F(e)) mod T: the branch
    # order is e-major, s-minor (allocation order)
    s_vals = np.tile(np.arange(W), 1 << m)
    want = (s_vals + F[e_vals]) % trunc
    mism = int(np.sum(want != acc_vals))
    print(f"[compare] per-branch accumulator mismatches python vs rust: {mism} of {nb}")
    # exact distribution of (V, j) from the Python state: FFT over e for each V
    T = int(trunc)
    psi = np.zeros((T, 1 << m), dtype=np.complex128)
    psi[acc_vals, e_vals] += qpu.amp
    P = np.abs(np.fft.fft(psi, axis=1, norm="ortho")) ** 2
    Pj = P.sum(axis=0)
    PV = P.sum(axis=1)
    # Rust's distribution of the same instance (its own outcome stream)
    d_path = out / f"qb_{tag}_rust_dist.csv"
    ge.run_driver("dist", *rust_args, f"dist_out={d_path}", "unmasked=0")
    Pr = np.zeros(1 << m)
    for line in d_path.read_text().splitlines()[1:]:
        j, a, _ = line.split(",")
        Pr[int(j)] = float(a)
    diff = float(np.max(np.abs(Pr - Pj)))
    print(f"[compare] max |P_python(j) - P_rust(j)| = {diff:.3e}  (sum P_python = {Pj.sum():.12f})")
    np.savetxt(out / f"qb_{tag}_python_Pj.txt", Pj)
    if tag == "n8":
        ref = ge.REPO / "tests" / "data" / "approx-modexp"
        ref.mkdir(parents=True, exist_ok=True)
        with open(ref / "qbackend_n8_Pj.txt", "w") as fh:
            fh.write("# P(j) of the paper's approx_modexp (Gidney 2025 release, CC-BY-4.0) run on the full\n")
            fh.write("# superposition by research/data/approx-modexp/qbackend.py (real X-basis measurements,\n")
            fh.write(f"# numpy FFT); N=143 g=2 m=10 w=(2,2,2,2) f=6 mask=2; periods={list(map(int, conf.periods))}\n")
            for x in Pj:
                fh.write(f"{x:.17e}\n")
    np.savetxt(out / f"qb_{tag}_python_PV.txt", PV)
    ok = qpu.merges == 0 and dirty == 0 and mism == 0 and diff < 1e-10 and phase_spread < 1e-12
    print(f"XCHECK_OK={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
