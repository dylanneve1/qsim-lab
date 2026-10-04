"""AlphaQubit-lite data layer: Google Sycamore 2022 (Zenodo 6804040) and Willow 2024 (Zenodo 13273331)
surface-code memory experiments -> per-round, per-stabilizer tensors on a canonical (d+1)x(d+1) grid.

Detector layout (both datasets, Stim circuits shipped with the data): DETECTOR(r, c, t). For a
memory experiment with R measured rounds the detectors are
    t = 0        : the (d^2-1)/2 on-basis stabilizers (compared with the prepared state)
    t = 1..R-1   : all d^2-1 stabilizers (compared with the previous round)
    t = R        : the on-basis stabilizers recomputed from the final data-qubit measurements
Qubits sit on a 45-degree rotated lattice: with u = r + c, v = r - c the data qubits are at
u, v = umin + 2i, vmin + 2j (i, j < d) and stabilizers at odd offsets; stabilizer grid cell
((u - umin + 1) / 2, (v - vmin + 1) / 2) lies in 0..d.

Canonical frame. Every experiment is mapped by one of the 8 symmetries of the square onto a
canonical patch in which the on-basis stabilizers occupy the same cells and the logical observable
runs along the same line, so one model (and its pooling direction) serves every area and basis.
"""
import math, os, re, zipfile
import numpy as np

# ---------------------------------------------------------------- circuits and layout

def detector_coords(stim_text):
    """[(r, c, t)] in detector order from a Stim circuit (handles REPEAT/SHIFT_COORDS via stim)."""
    import stim
    c = stim.Circuit(stim_text)
    co = c.get_detector_coordinates()
    return np.array([co[k][:3] for k in range(c.num_detectors)], dtype=np.float64)


def qubit_coords(stim_text):
    import stim
    c = stim.Circuit(stim_text)
    return {q: tuple(v) for q, v in c.get_final_qubit_coordinates().items()}


def observable_data_qubits(stim_text):
    """data qubits whose final measurement enters OBSERVABLE_INCLUDE (via measurement record)."""
    import stim
    c = stim.Circuit(stim_text).flattened()
    meas = []  # qubit per measurement record entry
    obs = set()
    for inst in c:
        if inst.name in ("M", "MZ", "MX", "MR", "MRZ", "MRX", "MY"):
            meas += [t.value for t in inst.targets_copy()]
        elif inst.name == "OBSERVABLE_INCLUDE":
            for t in inst.targets_copy():
                obs.add(meas[len(meas) + t.value])
    return sorted(obs)


SYMS = [(sw, fu, fv) for sw in (0, 1) for fu in (0, 1) for fv in (0, 1)]  # swap axes, flip u, flip v


def _apply(sym, i, j, n):
    sw, fu, fv = sym
    if fu:
        i = n - i
    if fv:
        j = n - j
    return (j, i) if sw else (i, j)


class Layout:
    """Stabilizer grid of one experiment, mapped to the canonical frame.

    Attributes
      d, R          distance, measured rounds
      cell[s]       canonical (i, j) grid cell of stabilizer s (s = 0..d^2-2, canonical order)
      onbasis[s]    bool
      det_round, det_stab : per detector, its round t (0..R) and canonical stabilizer index
      obs_line      'row' or 'col': canonical data-qubit line the observable lies on
    """

    def __init__(self, stim_text, d, R):
        self.d, self.R = d, R
        dc = detector_coords(stim_text)
        qc = qubit_coords(stim_text)
        oq = observable_data_qubits(stim_text)
        uv = lambda rc: (int(round(rc[0] + rc[1])), int(round(rc[0] - rc[1])))
        # data qubits are exactly the observable qubits plus others; get them as qubits measured last.
        # Simpler: stabilizer positions are the distinct (r, c) of detectors with t = 1.
        stab_rc = sorted({(dc[k, 0], dc[k, 1]) for k in range(len(dc)) if round(dc[k, 2]) == 1})
        assert len(stab_rc) == d * d - 1, (len(stab_rc), d)
        su = np.array([uv(p) for p in stab_rc])
        umin, vmin = su[:, 0].min() + 1, su[:, 1].min() + 1  # data-qubit origin (stabs at -1 offset)
        raw = [((u - umin + 1) // 2, (v - vmin + 1) // 2) for u, v in su]
        on = {(dc[k, 0], dc[k, 1]) for k in range(len(dc)) if round(dc[k, 2]) == 0}
        onb = np.array([p in on for p in stab_rc])
        ouv = [uv(qc[q]) for q in oq]
        odata = [((u - umin) // 2, (v - vmin) // 2) for u, v in ouv]  # data cell in 0..d-1
        # canonical: the on-basis stabilizer pattern of the reference and observable along row i = 0
        best = None
        for sym in SYMS:
            cells = [_apply(sym, i, j, d) for i, j in raw]
            dat = [_apply(sym, i, j, d - 1) for i, j in odata]
            on_cells = sorted(c for c, o in zip(cells, onb) if o)
            line = "row" if len({a for a, _ in dat}) == 1 else ("col" if len({b for _, b in dat}) == 1 else None)
            key = (line != "row", tuple(on_cells), min(dat))
            if best is None or key < best[0]:
                best = (key, sym, cells, dat, line)
        _, self.sym, cells, self.obs_cells, self.obs_line = best
        order = sorted(range(len(cells)), key=lambda s: cells[s])
        self.cell = np.array([cells[s] for s in order], dtype=np.int64)
        self.onbasis = onb[order]
        inv = {stab_rc[s]: k for k, s in enumerate(order)}
        self.det_round = np.rint(dc[:, 2]).astype(np.int64)
        self.det_stab = np.array([inv[(dc[k, 0], dc[k, 1])] for k in range(len(dc))], dtype=np.int64)
        self.nd = len(dc)
        assert self.det_round.max() == R, (self.det_round.max(), R)

    def signature(self):
        return (self.d, tuple(map(tuple, self.cell)), tuple(self.onbasis), self.obs_line, tuple(self.obs_cells))


def to_grid(dets, lay):
    """(B, nd) 0/1 -> events (B, R+1, S) uint8; final round t = R only on-basis entries are valid."""
    B = dets.shape[0]
    S = lay.d * lay.d - 1
    ev = np.zeros((B, lay.R + 1, S), np.uint8)
    ev[:, lay.det_round, lay.det_stab] = dets
    return ev


# ---------------------------------------------------------------- Sycamore 2022 files

SYC_RE = re.compile(r"surface_code_b([XZ])_d(\d)_r(\d\d)_center_(\d)_(\d)")


def syc_experiments(root):
    out = []
    for n in sorted(os.listdir(root)):
        m = SYC_RE.fullmatch(n)
        if m:
            b, d, r, cr, cc = m.groups()
            out.append(dict(name=n, basis=b, d=int(d), R=int(r), area=f"{cr}_{cc}", path=os.path.join(root, n)))
    return out


def read_b8(path, bits):
    nb = (bits + 7) // 8
    a = np.fromfile(path, dtype=np.uint8).reshape(-1, nb)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :bits]


def read_01(path):
    return np.frombuffer(open(path, "rb").read().replace(b"\n", b""), dtype=np.uint8) - ord("0")


def syc_load(exp, nd=None):
    """dets (N, nd) uint8, obs (N,) uint8, circuit text"""
    txt = open(os.path.join(exp["path"], "circuit_ideal.stim")).read()
    if nd is None:
        import stim
        nd = stim.Circuit(txt).num_detectors
    dets = read_b8(os.path.join(exp["path"], "detection_events.b8"), nd)
    obs = read_01(os.path.join(exp["path"], "obs_flips_actual.01"))
    assert len(obs) == len(dets)
    return dets, obs, txt


# ---------------------------------------------------------------- metrics (paper conventions)

def eps_from_E(E, n):
    """per-round LER from the error rate after n rounds, E(n) = (1 - (1 - 2 eps)^n) / 2"""
    return 0.5 * (1 - max(1e-12, 1 - 2 * E) ** (1.0 / n))


def fit_ler(rounds, fails, shots):
    """paper eq. (5): least-squares fit of log F(n) = log F0 + n log(1 - 2 eps), F = 1 - 2E.
    returns eps, F0, R^2"""
    n = np.asarray(rounds, float)
    F = 1 - 2 * np.asarray(fails, float) / np.asarray(shots, float)
    y = np.log(np.maximum(F, 1e-9))
    A = np.stack([np.ones_like(n), n], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * n
    r2 = 1 - ((y - yhat) ** 2).sum() / max(1e-30, ((y - y.mean()) ** 2).sum())
    return 0.5 * (1 - math.exp(b)), math.exp(a), r2


def fit_ler_boot(rounds, fail_vecs, reps=499, seed=0):
    """fail_vecs: list of bool arrays (one per round count). Bootstrap (resample shots of every
    round count, 499 resamples as the paper) -> (eps, std, F0, R2)"""
    rng = np.random.default_rng(seed)
    shots = [len(f) for f in fail_vecs]
    fails = [int(f.sum()) for f in fail_vecs]
    e, F0, r2 = fit_ler(rounds, fails, shots)
    bs = []
    for _ in range(reps):
        fb = [rng.binomial(s, f / s) for f, s in zip(fails, shots)]
        bs.append(fit_ler(rounds, fb, shots)[0])
    return e, float(np.std(bs)), F0, r2


def wilson(f, n, z=1.96):
    p = f / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- DEM -> FastSampler circuit

def dem_to_circuit(dem_text, scale=1.0):
    """A Stim circuit whose detector/observable distribution equals the (flattened) DEM's:
    qubits 0..nd-1 carry detectors, nd..nd+no-1 observables, one ancilla `a` is reused:
        R a; X_ERROR(p) a; CX a q_1 a q_2 ...      (per error mechanism)
    then M on every detector/observable qubit with DETECTOR / OBSERVABLE_INCLUDE.
    Every X_ERROR is an independent noise variable, so qsim-lab's FastSampler samples it exactly.
    `scale` multiplies every probability (noise curriculum)."""
    import stim
    dem = stim.DetectorErrorModel(dem_text).flattened()
    nd, no = dem.num_detectors, dem.num_observables
    a = nd + no
    lines = []
    for inst in dem:
        if inst.type != "error":
            continue
        p = min(0.5, inst.args_copy()[0] * scale)
        tg = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                tg.append(t.val)
            elif t.is_logical_observable_id():
                tg.append(nd + t.val)
        # '^' separators (suggested decompositions) are dropped: the mechanism flips the XOR of all parts
        par = {}
        for q in tg:
            par[q] = par.get(q, 0) ^ 1
        tg = [q for q, v in par.items() if v]
        if not tg or p <= 0:
            continue
        lines.append(f"R {a}\nX_ERROR({p:.12g}) {a}\nCX " + " ".join(f"{a} {q}" for q in tg))
    lines.append("M " + " ".join(map(str, range(nd + no))))
    n = nd + no
    lines += [f"DETECTOR rec[{k - n}]" for k in range(nd)]
    lines += [f"OBSERVABLE_INCLUDE({j}) rec[{nd + j - n}]" for j in range(no)]
    return "\n".join(lines) + "\n"
