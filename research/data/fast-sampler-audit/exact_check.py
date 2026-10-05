#!/usr/bin/env python3
"""Independent exact check of the FastSampler (audit, research/qec/fast-sampler-audit.md).

For small Stim circuits:
  (A) EXACT ground truth: a branching TableauSimulator interpreter that follows Stim's channel
      definitions literally (each DEPOLARIZE1/2, X/Y/Z_ERROR, M(p)/MR(p)/MX(p) flip branches
      over its outcomes with exact Fraction weights; random measurements branch 1/2-1/2 via
      postselection). No linearity / Pauli-frame / DEM assumption. Outputs are reported the way
      Stim does: detector/observable parity XOR its value in the noiseless reference sample
      (random results forced to 0).
  (B) the FastSampler's hit model, evaluated analytically at 60 digits: every channel
      instance with p <= 0.25 is m independent pattern slots, each flipped an odd number of times
      with probability (1 - exp(-2 lambda / m)) / 2, lambda = -(m/(m+1)) ln(1 - (m+1) p / m)
      (the formula in fast_sampler.rs); p > 0.25 uses the channel itself (dense path). Pattern
      signatures come from single-fault runs of (A)'s interpreter. Only for circuits whose
      detectors/observables are deterministic.
  (C) the actual binary: `stim_compare sample-fast` (and Stim's own sampler as a control, and a
      perturbed circuit as a negative control) at N shots; Pearson chi^2 of the full joint
      histogram against (A).

usage: exact_check.py <stim_compare> <workdir> <shots> [seed]
"""
import sys, os, json, random, subprocess, itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
import stim
from scipy.stats import chi2

mp.mp.dps = 60
NOISE1 = {"X_ERROR": ["X"], "Y_ERROR": ["Y"], "Z_ERROR": ["Z"], "DEPOLARIZE1": ["X", "Y", "Z"]}
P2 = ["".join(t) for t in itertools.product("IXYZ", repeat=2)][1:]


def channels(circ):
    """Flattened instruction list."""
    return list(circ.flattened())


def merge(states):
    """Merge branches with the same stabilizer state (canonical stabilizers, with signs) and
    the same record: exact, keeps the branch count at most 2^n * #records."""
    acc = {}
    for sim, rec, pr in states:
        key = (tuple(str(p) for p in sim.canonical_stabilizers()), tuple(rec))
        if key in acc:
            acc[key][2] += pr
        else:
            acc[key] = [sim, rec, pr]
    return [tuple(v) for v in acc.values()]


def run_branches(circ, force=None):
    """Exact output distribution {bits: Fraction}. force: None = all channels random;
    else dict {(inst_index, target_index): outcome_index} with every other channel at identity
    (single-fault signature runs; random measurements still branch)."""
    insts = channels(circ)
    n = circ.num_qubits
    # state: (sim, record, prob)
    states = [(stim.TableauSimulator(), [], Fr(1))]
    states[0][0].set_num_qubits(n)
    dets, obs = [], {}
    for ii, inst in enumerate(insts):
        name = inst.name
        args = inst.gate_args_copy()
        tg = inst.targets_copy()
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name == "DETECTOR":
            dets.append(1)
            new = []
            for sim, rec, pr in states:
                meas = [x for x in rec if x[0] == "M"]
                par = 0
                for t in tg:
                    par ^= meas[t.value][1]
                new.append((sim, rec + [("D", par)], pr))
            states = new
            continue
        if name == "OBSERVABLE_INCLUDE":
            k = int(args[0])
            new = []
            for sim, rec, pr in states:
                meas = [x for x in rec if x[0] == "M"]
                par = 0
                for t in tg:
                    par ^= meas[t.value][1]
                new.append((sim, rec + [("O", k, par)], pr))
            states = new
            continue
        if name in NOISE1 or name == "DEPOLARIZE2":
            p = Fr(args[0])
            groups = [[t.value] for t in tg] if name != "DEPOLARIZE2" else [
                [tg[i].value, tg[i + 1].value] for i in range(0, len(tg), 2)]
            pats = NOISE1.get(name, P2)
            for gi, qs in enumerate(groups):
                outs = [(None, 1 - p)] + [(pt, p / len(pats)) for pt in pats]
                if force is not None:
                    k = force.get((ii, gi), 0)
                    outs = [(outs[k][0], Fr(1))]
                new = []
                for sim, rec, pr in states:
                    for pt, w in outs:
                        if w == 0:
                            continue
                        s2 = sim.copy()
                        if pt is not None:
                            for q, c in zip(qs, pt):
                                if c != "I":
                                    s2.do(stim.CircuitInstruction(c, [q]))
                        new.append((s2, rec, pr * w))
                states = merge(new)
            continue
        if name in ("M", "MZ", "MX", "MR", "MRZ", "MRX", "R", "RZ", "RX"):
            p = Fr(args[0]) if args else Fr(0)
            xb = name.endswith("X")
            for ti, t in enumerate(tg):
                q = t.value
                new = []
                for sim, rec, pr in states:
                    if xb:
                        sim.h(q)
                    v = sim.peek_z(q)
                    branches = [(1 if v == -1 else 0, Fr(1))] if v != 0 else [(0, Fr(1, 2)), (1, Fr(1, 2))]
                    for val, w in branches:
                        s2 = sim.copy() if len(branches) > 1 else sim
                        if v == 0:
                            s2.postselect_z(q, desired_value=bool(val))
                        if name.startswith("R"):
                            if val:
                                s2.x(q)
                            if xb:
                                s2.h(q)
                            new.append((s2, rec, pr * w))
                            continue
                        flips = [(0, 1 - p), (1, p)]
                        if force is not None:
                            flips = [(force.get((ii, ti), 0), Fr(1))]
                        for f, wf in flips:
                            if wf == 0:
                                continue
                            s3 = s2.copy() if len(flips) > 1 else s2
                            if name.startswith("MR") and val:
                                s3.x(q)
                            if xb:
                                s3.h(q)
                            new.append((s3, rec + [("M", val ^ f)], pr * w * wf))
                states = merge(new)
            continue
        # Clifford gate
        for sim, rec, pr in states:
            sim.do(inst)
    nd = len(dets)
    dist = {}
    for sim, rec, pr in states:
        d = [x[1] for x in rec if x[0] == "D"]
        o = {}
        for x in rec:
            if x[0] == "O":
                o[x[1]] = o.get(x[1], 0) ^ x[2]
        bits = tuple(d + [o[k] for k in sorted(o)])
        dist[bits] = dist.get(bits, Fr(0)) + pr
    return dist


def reference_bits(circ):
    """Stim's convention: XOR with the noiseless reference sample (random results -> 0)."""
    ref = circ.without_noise()
    return run_branches_ref(ref)


def run_branches_ref(circ):
    dist = run_branches_det_ref(circ)
    return dist


def run_branches_det_ref(circ):
    # noiseless run with random measurement results forced to 0
    s = stim.TableauSimulator()
    s.set_num_qubits(circ.num_qubits)
    rec, d, o = [], [], {}
    for inst in circ.flattened():
        name, tg, args = inst.name, inst.targets_copy(), inst.gate_args_copy()
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name == "DETECTOR":
            d.append(sum(rec[t.value] for t in tg) % 2)
        elif name == "OBSERVABLE_INCLUDE":
            k = int(args[0]); o[k] = (o.get(k, 0) + sum(rec[t.value] for t in tg)) % 2
        elif name in ("M", "MZ", "MX", "MR", "MRZ", "MRX", "R", "RZ", "RX"):
            xb = name.endswith("X")
            for t in tg:
                q = t.value
                if xb:
                    s.h(q)
                v = s.peek_z(q)
                if v == 0:
                    s.postselect_z(q, desired_value=False)
                    val = 0
                else:
                    val = 1 if v == -1 else 0
                if not name.startswith("R"):
                    rec.append(val)
                if (name.startswith("R") or name.startswith("MR")) and val:
                    s.x(q)
                if xb:
                    s.h(q)
        else:
            s.do(inst)
    return tuple(d + [o[k] for k in sorted(o)])


def exact_truth(circ):
    ref = run_branches_det_ref(circ.without_noise())
    raw = run_branches(circ)
    out = {}
    for b, w in raw.items():
        k = tuple(x ^ y for x, y in zip(b, ref))
        out[k] = out.get(k, Fr(0)) + w
    assert sum(out.values()) == 1
    return out


def hit_model(circ):
    """(B): FastSampler's analytic distribution; needs deterministic outputs."""
    clean = circ.without_noise()
    base = run_branches(circ, force={})
    if len(base) != 1:
        return None  # non-deterministic outputs: (B) not applicable
    b0 = next(iter(base))
    insts = channels(circ)
    nbits = len(b0)
    dist = {tuple([0] * nbits): mp.mpf(1)}

    def conv(dist, outs):
        new = {}
        for k, w in dist.items():
            for sig, q in outs:
                kk = tuple(a ^ b for a, b in zip(k, sig))
                new[kk] = new.get(kk, 0) + w * q
        return new

    def sig_of(force):
        r = run_branches(circ, force=force)
        assert len(r) == 1
        return tuple(a ^ b for a, b in zip(next(iter(r)), b0))

    for ii, inst in enumerate(insts):
        name, args, tg = inst.name, inst.gate_args_copy(), inst.targets_copy()
        if name in NOISE1 or name == "DEPOLARIZE2":
            p = mp.mpf(Fr(args[0]).numerator) / Fr(args[0]).denominator
            ng = len(tg) if name != "DEPOLARIZE2" else len(tg) // 2
            m = len(NOISE1.get(name, P2))
            for gi in range(ng):
                sigs = [sig_of({(ii, gi): k}) for k in range(1, m + 1)]
                if p == 0:
                    continue
                if p <= mp.mpf("0.25"):
                    lam = -(mp.mpf(m) / (m + 1)) * mp.log(1 - (m + 1) * p / m)
                    q = (1 - mp.exp(-2 * lam / m)) / 2
                    for s in sigs:  # independent pattern slots, parity of Poisson(lam/m)
                        dist = conv(dist, [(tuple([0] * nbits), 1 - q), (s, q)])
                else:  # dense path: the channel itself
                    dist = conv(dist, [(tuple([0] * nbits), 1 - p)] + [(s, p / m) for s in sigs])
        elif name in ("M", "MZ", "MX", "MR", "MRZ", "MRX") and args and args[0] > 0:
            p = mp.mpf(Fr(args[0]).numerator) / Fr(args[0]).denominator
            for ti in range(len(tg)):
                s = sig_of({(ii, ti): 1})
                if p <= mp.mpf("0.25"):
                    lam = -mp.log(1 - 2 * p) / 2
                    q = (1 - mp.exp(-2 * lam)) / 2
                else:
                    q = p
                dist = conv(dist, [(tuple([0] * nbits), 1 - q), (s, q)])
    return dist


def read_ptb64(path, nbits, shots):
    a = np.fromfile(path, dtype="<u8").reshape(-1, nbits)
    bits = np.unpackbits(a.view(np.uint8).reshape(a.shape[0], nbits, 8), axis=2, bitorder="little")
    bits = bits.transpose(0, 2, 1).reshape(-1, nbits)[:shots]
    keys = bits @ (1 << np.arange(nbits, dtype=np.int64))
    return np.bincount(keys, minlength=1 << nbits)


def chi2_test(counts, truth, nbits, shots):
    exp_ = np.zeros(1 << nbits)
    for k, w in truth.items():
        exp_[sum(b << i for i, b in enumerate(k))] = float(w) * shots
    bad = counts[exp_ == 0].sum()
    # pool cells with expected < 5 into one
    small = (exp_ < 5) & (exp_ > 0)
    e = np.append(exp_[(exp_ >= 5)], exp_[small].sum())
    o = np.append(counts[(exp_ >= 5)], counts[small].sum())
    keep = e > 0
    e, o = e[keep], o[keep]
    x2 = float(((o - e) ** 2 / e).sum())
    dof = len(e) - 1
    return dict(chi2=x2, dof=dof, pval=float(chi2.sf(x2, dof)), impossible_outcomes=int(bad))


def sample_ours(binary, path, shots, out, seed, rng="wy"):
    r = subprocess.run([binary, "sample-fast", path, str(shots), out, str(seed), rng],
                       capture_output=True, text=True)
    return r


def sample_stim(circ, shots, out, seed):
    circ.compile_detector_sampler(seed=seed).sample_write(shots, filepath=out, format="ptb64",
                                                          append_observables=True)


def perturb(circ, factor):
    s = str(circ)
    import re
    def f(m):
        return f"{m.group(1)}({float(m.group(2)) * factor!r})"
    return stim.Circuit(re.sub(r"(DEPOLARIZE2|DEPOLARIZE1|X_ERROR)\(([0-9.e-]+)\)", f, s, count=1))


# ---------------------------------------------------------------------------------------------
HAND = {
 "mixed_rare": """
R 0 1 2 3 4
X_ERROR(0.02) 0 1 2
DEPOLARIZE1(0.05) 0 1 2
CX 0 3 1 3
DEPOLARIZE2(0.1) 0 3 1 3
CX 1 4 2 4
DEPOLARIZE2(0.03) 1 4 2 4
Z_ERROR(0.07) 3
Y_ERROR(0.04) 4
MR(0.01) 3 4
DETECTOR rec[-2]
DETECTOR rec[-1]
X_ERROR(0.01) 3 4
DEPOLARIZE1(0.2) 0 1 2
CX 0 3 1 3 1 4 2 4
DEPOLARIZE2(0.25) 0 3
MR(0.01) 3 4
DETECTOR rec[-2] rec[-4]
DETECTOR rec[-1] rec[-3]
M(0.01) 0 1 2
DETECTOR rec[-3] rec[-2] rec[-5]
DETECTOR rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
""",
 "near_max_dense": """
R 0 1 2 3
RX 4
X_ERROR(0.5) 0
X_ERROR(0.95) 1
DEPOLARIZE1(0.74) 2
DEPOLARIZE1(0.25) 3
X_ERROR(0.2500001) 3
CX 0 1 2 3
DEPOLARIZE2(0.93) 0 1
DEPOLARIZE2(0.9375) 2 3
DEPOLARIZE1(0.75) 4
DEPOLARIZE2(0.3) 1 2
Z_ERROR(0.6) 4
M(0.3) 0 1 2 3
MX(0.3) 4
DETECTOR rec[-5]
DETECTOR rec[-4]
DETECTOR rec[-3] rec[-4]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
""",
 "over_depolarized": """
R 0 1
DEPOLARIZE1(0.9) 0
DEPOLARIZE2(0.99) 0 1
DEPOLARIZE1(1.0) 1
X_ERROR(1.0) 0
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
""",
 "many_groups_lowp": """
R 0 1 2 3
REPEAT 60 {
    DEPOLARIZE1(0.002) 0 1 2 3
    CX 0 1 2 3
    DEPOLARIZE2(0.002) 0 1 2 3
    SWAP 1 2
    X_ERROR(0.002) 0 1 2 3
    CZ 0 2
}
M(0.002) 0 1 2 3
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
""",
 "many_groups_midp": """
R 0 1 2
REPEAT 40 {
    DEPOLARIZE1(0.03) 0 1 2
    CX 0 1
    DEPOLARIZE2(0.03) 0 1
    X_ERROR(0.03) 2
    SWAP 0 2
}
M 0 1 2
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
""",
 "deterministic_one": """
R 0 1
X 0
DEPOLARIZE1(0.1) 0 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]
""",
}

GATES1 = ["H", "S", "S_DAG", "X", "Y", "Z"]
GATES2 = ["CX", "CZ", "SWAP"]
PS = [0.001, 0.01, 0.05, 0.2, 0.25, 0.3, 0.6]


def random_circuit(rng, nq=4, depth=12):
    pm = rng.choice([0.0, 0.01, 0.2])
    L = [f"R {' '.join(map(str, range(nq)))}"]
    nm = 0
    for _ in range(depth):
        r = rng.random()
        if r < 0.35:
            q = rng.randrange(nq); L.append(f"{rng.choice(GATES1)} {q}")
        elif r < 0.6:
            a, b = rng.sample(range(nq), 2); L.append(f"{rng.choice(GATES2)} {a} {b}")
        elif r < 0.8:
            ch = rng.choice(["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"])
            p = rng.choice(PS)
            if ch == "DEPOLARIZE2":
                a, b = rng.sample(range(nq), 2); L.append(f"{ch}({p}) {a} {b}")
            else:
                L.append(f"{ch}({p}) {rng.randrange(nq)}")
        else:
            op = rng.choice(["M", "MR", "MX", "R", "RX"])
            q = rng.randrange(nq)
            if op in ("R", "RX"):
                L.append(f"{op} {q}")
            else:
                L.append(f"{op}({pm}) {q}" if pm else f"{op} {q}"); nm += 1
    L.append(f"M({pm}) {' '.join(map(str, range(nq)))}" if pm else f"M {' '.join(map(str, range(nq)))}")
    nm += nq
    for _ in range(rng.randint(2, 5)):
        k = rng.randint(1, min(3, nm))
        L.append("DETECTOR " + " ".join(f"rec[-{i}]" for i in rng.sample(range(1, nm + 1), k)))
    L.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{i}]" for i in rng.sample(range(1, nm + 1), 2)))
    return stim.Circuit("\n".join(L))


def check(name, circ, binary, work, shots, seed):
    path = f"{work}/{name}.stim"
    open(path, "w").write(str(circ))
    nbits = circ.num_detectors + circ.num_observables
    res = dict(name=name, nbits=nbits)
    truth = exact_truth(circ)
    res["truth_outcomes"] = len(truth)
    try:
        nch = sum(1 for i in circ.flattened() if i.name in NOISE1 or i.name == "DEPOLARIZE2" for _ in i.targets_copy())
        hm = hit_model(circ) if nch <= 300 else None
    except Exception as e:  # noqa
        hm = None
        res["hit_model_error"] = repr(e)
    if hm is not None:
        keys = set(hm) | set(truth)
        res["max_abs_diff_hitmodel_vs_exact"] = float(max(abs(hm.get(k, 0) - (mp.mpf(truth[k].numerator) / truth[k].denominator if k in truth else 0)) for k in keys))
    out = f"{work}/{name}.o.ptb64"
    r = sample_ours(binary, path, shots, out, seed)
    if r.returncode != 0:
        res["ours"] = "ERROR: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    else:
        res["ours"] = chi2_test(read_ptb64(out, nbits, shots), truth, nbits, shots)
        os.remove(out)
    sample_stim(circ, shots, out, seed + 1)
    res["stim"] = chi2_test(read_ptb64(out, nbits, shots), truth, nbits, shots)
    os.remove(out)
    return res


if __name__ == "__main__":
    binary, work, shots = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    nrand = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    os.makedirs(work, exist_ok=True)
    for name, txt in HAND.items():
        print(json.dumps(check(name, stim.Circuit(txt), binary, work, shots, seed)), flush=True)
    # negative controls: our binary on a wrong circuit vs the truth of the right one
    import math
    def naive(m, p):  # what a "lambda = p" Poisson approximation would sample
        return m / (m + 1) * (1 - math.exp(-(m + 1) * p / m))
    for name in ("mixed_rare", "many_groups_midp", "near_max_dense"):
        c = stim.Circuit(HAND[name])
        variants = {"first_channel_x1.3": perturb(c, 1.3)}
        import re
        def nv(mt):
            ch, p = mt.group(1), float(mt.group(2))
            m = {"DEPOLARIZE1": 3, "DEPOLARIZE2": 15}.get(ch, 1)
            return f"{ch}({naive(m, p)!r})" if p <= 0.25 else mt.group(0)
        variants["naive_poisson_rate_lambda_eq_p"] = stim.Circuit(re.sub(
            r"(DEPOLARIZE2|DEPOLARIZE1|X_ERROR|Y_ERROR|Z_ERROR)\(([0-9.e-]+)\)", nv, str(c)))
        truth = exact_truth(c)
        nbits = c.num_detectors + c.num_observables
        for vn, pc in variants.items():
            path = f"{work}/{name}_{vn}.stim"; open(path, "w").write(str(pc))
            out = f"{work}/neg.ptb64"
            sample_ours(binary, path, shots, out, seed)
            print(json.dumps(dict(name=f"{name}_NEGATIVE_CONTROL_{vn}",
                                  ours=chi2_test(read_ptb64(out, nbits, shots), truth, nbits, shots))), flush=True)
            os.remove(out)
    rng = random.Random(seed)
    for i in range(nrand):
        c = random_circuit(rng)
        print(json.dumps(check(f"random{i}", c, binary, work, shots, seed + 10 * i)), flush=True)
