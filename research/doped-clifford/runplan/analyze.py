#!/usr/bin/env python3
"""Run-side bookkeeping and scoring for the D=70 doped-Clifford sampling run (see ../RUNPLAN.md).

Conventions
- Bitstrings are written q0-first ("bitstring_q0_first"), exactly like SUTD's Zenodo file.
  Qiskit's little-endian integer is int(s[::-1], 2).
- A tail batch of m open qubits q0..q(m-1) is a list of 2^m amplitudes, index j = int(s[:m], 2)
  (q0 is the MSB of j), i.e. SUTD's own index rule restricted to the first m qubits.
- Amplitudes may carry any common complex scale; everything below is scale/phase invariant
  except the 'zscale' diagnostic.

Subcommands
  commit   --seed-file F                 print sha256(seed) to publish BEFORE production
  prefixes --seed-file F --m M --n N [--start S]   production prefixes (q_m..q69, uniform)
  calrows  --seed-file F --m M --k K     calibration prefixes taken from SUTD rows
  calib    REC.jsonl [...]               F_hat, X_hat (+ jackknife) against SUTD exact amplitudes
  samples  REC.jsonl [...] --xhat X --xse S     XEB prediction + significance for the sample set
  score    PROBS.json                    linear XEB of our samples from third-party ideal probs
  compare  REF.jsonl TEST.jsonl          batch fidelity of TEST records against REF records (same keys)
  rowjobs  --rows R1,R2 [--m 8] [--sub S]  calibration jobs for chosen SUTD rows
  synth    --F F --m M --k K --out O     synthetic calib records (noise model) to test the pipeline
"""
import argparse, hashlib, json, sys
import numpy as np

SUTD = __file__.rsplit('/', 2)[0] + '/sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz'
N = 70


def sutd():
    d = np.load(SUTD)
    amps = d['raw_vectors'].astype(np.complex128) * float(d['recovery_factor'])
    return amps, d['assignments_q8_q69'], d['bitstrings_q0_first']


def stream(seed: bytes, tag: str, i: int, nbits: int) -> str:
    """Deterministic uniform bits from sha256(seed|tag|i|ctr)."""
    out, ctr = '', 0
    while len(out) < nbits:
        h = hashlib.sha256(seed + f'|{tag}|{i}|{ctr}'.encode()).digest()
        out += ''.join(f'{b:08b}' for b in h)
        ctr += 1
    return out[:nbits]


def unif(seed: bytes, tag: str, i: int) -> float:
    return int(stream(seed, tag, i, 53), 2) / 2 ** 53


def read(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


def amps_of(r):
    a = np.array(r['amps'], dtype=float)
    return a[:, 0] + 1j * a[:, 1]


def jack(fn, k):
    full = fn(np.ones(k, bool))
    if k < 3:
        return full, float('nan')
    vals = []
    for i in range(k):
        m = np.ones(k, bool); m[i] = False; vals.append(fn(m))
    vals = np.array(vals)
    return full, float(np.sqrt((k - 1) / k * ((vals - vals.mean()) ** 2).sum()))


def cmd_calib(a):
    S, assign, _ = sutd()
    recs = [r for r in read(a.recs) if r.get('kind') == 'calib']
    E, L = [], []
    for r in recs:
        m = r['tail_m']; B = 2 ** m; s = r['prefix_bits']
        assert len(s) == N and set(s[:m]) == {'.'}, 'prefix_bits must be 70 chars with the tail as dots'
        row = r['row']
        assert s[8:] == assign[row], f'record row {row}: q8..q69 do not match SUTD row'
        sub = int(s[m:8], 2) if m < 8 else 0
        # SUTD index j = int(q0..q7), q0 MSB: open q0..q(m-1) are the HIGH bits, fixed q_m..q7 the low bits
        E.append(S[row, sub::2 ** (8 - m)][:B]); L.append(amps_of(r))
    k = len(E)
    if not k:
        sys.exit('no calib records')

    def fid(mask):
        num = sum(np.vdot(E[i], L[i]) for i in range(k) if mask[i])
        de = sum(np.vdot(E[i], E[i]).real for i in range(k) if mask[i])
        dl = sum(np.vdot(L[i], L[i]).real for i in range(k) if mask[i])
        return abs(num) ** 2 / (de * dl)

    def xeb(mask, exact=False):
        v = []
        for i in range(k):
            if not mask[i]:
                continue
            z = abs(E[i]) ** 2 * 2.0 ** N
            q = abs(E[i] if exact else L[i]) ** 2
            v.append((q / q.sum() * z).sum() - 1)
        return float(np.mean(v))

    F, Fse = jack(fid, k)

    def fid_alt(tr):
        num = sum(np.vdot(E[i], tr(L[i])) for i in range(k))
        de = sum(np.vdot(E[i], E[i]).real for i in range(k))
        dl = sum(np.vdot(L[i], L[i]).real for i in range(k))
        return float(abs(num) ** 2 / (de * dl))

    def bitrev(v):
        mm = int(np.log2(len(v)))
        return v[[int(format(j, f'0{mm}b')[::-1], 2) for j in range(len(v))]]
    # bootstrap over individual amplitudes (pooled), usable with 1-2 records
    rng = np.random.default_rng(0)
    Ea, La = np.concatenate(E), np.concatenate(L)
    bs = []
    for _ in range(2000):
        ix = rng.integers(0, len(Ea), len(Ea))
        e, l = Ea[ix], La[ix]
        bs.append(abs(np.vdot(e, l)) ** 2 / (np.vdot(e, e).real * np.vdot(l, l).real))
    Fbse = float(np.std(bs))
    # convention diagnostics: if F_hat ~ 0 but one of these is large, that is the bug
    diag = {'F_if_conjugated': fid_alt(np.conj), 'F_if_tail_index_bit_reversed': fid_alt(bitrev),
            'F_if_both': fid_alt(lambda v: np.conj(bitrev(v)))}
    X, Xse = jack(xeb, k)
    X0 = xeb(np.ones(k, bool), exact=True)
    Xr, Xrse = jack(lambda m: xeb(m) / xeb(m, exact=True), k)
    Bs = sorted({len(e) for e in E})
    per = [abs(np.vdot(E[i], L[i])) ** 2 / (np.vdot(E[i], E[i]).real * np.vdot(L[i], L[i]).real) for i in range(k)]
    # common-phase consistency: per-record phase of <e|l> should agree (same global phase convention)
    ph = np.angle([np.vdot(E[i], L[i]) for i in range(k)])
    print(json.dumps({
        'records': k, 'amplitudes': int(sum(len(e) for e in E)),
        'F_hat': F, 'F_se_jackknife': Fse, 'F_se_bootstrap_amplitudes': Fbse,
        'X_hat_sampler_xeb': X, 'X_se_jackknife': Xse,
        'X_exact_sampler_same_prefixes': X0,
        'X_ratio_ours_over_exact': Xr, 'X_ratio_se_jackknife': Xrse,
        'predicted_production_xeb_uniform_prefix': [Xr * (B - 1) / (B + 1) for B in Bs],
        'F_per_record': [round(x, 4) for x in per],
        'phase_of_overlap_rad': [round(float(x), 3) for x in ph],
        'convention_diagnostics': diag,
        'formats': sorted({r.get('format', '?') for r in recs}),
        'R_passes': sorted({r.get('R', -1) for r in recs}),
    }, indent=1))


def cmd_samples(a):
    recs = [r for r in read(a.recs) if r.get('kind') == 'sample']
    idx = sorted(r['i'] for r in recs)
    gaps = sorted(set(range(idx[-1] + 1)) - set(idx)) if idx else []
    n = len(recs)
    # Var of z(sample) for a fidelity-X sampler under Porter-Thomas: 1 + 2X - X^2
    var = 1 + 2 * a.xhat - a.xhat ** 2
    se_samp = np.sqrt(var / n)
    se = np.sqrt(se_samp ** 2 + a.xse ** 2)
    out = {'samples': n, 'missing_indices_redo_these': gaps,
           'predicted_xeb': a.xhat, 'se_sampling': se_samp, 'se_total': se,
           'sigma_over_0.044': (a.xhat - 0.044) / se,
           'sigma_over_IBM_0.342': (a.xhat - 0.342) / np.sqrt(se ** 2 + 0.028 ** 2)}
    print(json.dumps(out, indent=1))


def cmd_score(a):
    d = json.load(open(a.probs))  # {"bitstring_q0_first": p_ideal, ...}
    z = np.array(list(d.values()), float) * 2.0 ** N
    n = len(z); x = z.mean() - 1; se = z.std(ddof=1) / np.sqrt(n)
    print(json.dumps({'n': n, 'linear_xeb': x, 'se': se, 'sigma_over_0.044': (x - 0.044) / se,
                      'log_xeb': float(np.euler_gamma + np.log(z).mean())}, indent=1))


def cmd_commit(a):
    seed = open(a.seed_file, 'rb').read().strip()
    print(hashlib.sha256(seed).hexdigest())


def cmd_prefixes(a):
    seed = open(a.seed_file, 'rb').read().strip()
    for i in range(a.start, a.start + a.n):
        print(json.dumps({'i': i, 'prefix_bits': '.' * a.m + stream(seed, 'prefix', i, N - a.m),
                          'u_tail': unif(seed, 'tail', i)}))


def cmd_calrows(a):
    seed = open(a.seed_file, 'rb').read().strip()
    _, assign, _ = sutd()
    rows = []
    i = 0
    while len(rows) < a.k:
        r = int(stream(seed, 'calrow', i, 32), 2) % len(assign); i += 1
        if r not in [x[0] for x in rows]:
            rows.append((r, stream(seed, 'calsub', r, max(0, 8 - a.m))))
    for r, sb in rows:
        print(json.dumps({'row': r, 'prefix_bits': '.' * a.m + sb + str(assign[r])}))


def cmd_compare(a):
    """Batch fidelity of records in B (e.g. packed) against records in A (e.g. exact) with the same key."""
    def keyed(p):
        out = {}
        for r in read([p]):
            if r.get('kind') in ('sample', 'calib'):
                out[(r['kind'], r.get('i', r.get('row')))] = r
        return out
    A, B = keyed(a.ref), keyed(a.test)
    num, de, dl, per = 0, 0.0, 0.0, {}
    for k in sorted(set(A) & set(B), key=str):
        assert A[k]['prefix_bits'] == B[k]['prefix_bits'], k
        e, l = amps_of(A[k]), amps_of(B[k])
        num += np.vdot(e, l); de += np.vdot(e, e).real; dl += np.vdot(l, l).real
        per[str(k)] = round(float(abs(np.vdot(e, l)) ** 2 / (np.vdot(e, e).real * np.vdot(l, l).real)), 6)
    print(json.dumps({'matched': len(per), 'F_pooled': float(abs(num) ** 2 / (de * dl)) if per else None,
                      'F_per_record': per, 'R': sorted({r.get('R') for r in B.values()}),
                      'formats': [sorted({r.get('format') for r in A.values()}), sorted({r.get('format') for r in B.values()})]},
                     indent=1))


def cmd_rowjobs(a):
    """Calibration jobs for given SUTD rows (q8..q69 = that IBM sample; q_m..q7 seeded or 0)."""
    _, assign, _ = sutd()
    for r in [int(x) for x in a.rows.split(',')]:
        sb = format(a.sub, f'0{8 - a.m}b') if a.m < 8 else ''
        print(json.dumps({'row': r, 'prefix_bits': '.' * a.m + sb + str(assign[r])}))


def cmd_synth(a):
    S, assign, _ = sutd()
    rng = np.random.default_rng(a.rseed)
    B = 2 ** a.m
    with open(a.out, 'w') as f:
        for i, r in enumerate(rng.choice(len(assign), a.k, replace=False)):
            sub = int(rng.integers(0, 256 // B))
            e = S[r, sub::256 // B][:B]
            sig = np.sqrt((abs(e) ** 2).mean())
            g = sig * (rng.standard_normal(B) + 1j * rng.standard_normal(B)) / np.sqrt(2)
            l = (np.sqrt(a.F) * e + np.sqrt(1 - a.F) * g) * 3.7 * np.exp(0.4j)  # arbitrary common scale/phase
            sb = format(sub, f'0{8 - a.m}b') if a.m < 8 else ''
            f.write(json.dumps({'kind': 'calib', 'row': int(r), 'tail_m': a.m, 'format': 'synthetic', 'R': 71,
                                'prefix_bits': '.' * a.m + sb + str(assign[r]),
                                'amps': [[float(x.real), float(x.imag)] for x in l]}) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('commit'); x.add_argument('--seed-file', required=True)
    x = s.add_parser('prefixes'); x.add_argument('--seed-file', required=True); x.add_argument('--m', type=int, required=True)
    x.add_argument('--n', type=int, required=True); x.add_argument('--start', type=int, default=0)
    x = s.add_parser('calrows'); x.add_argument('--seed-file', required=True); x.add_argument('--m', type=int, required=True)
    x.add_argument('--k', type=int, required=True)
    x = s.add_parser('calib'); x.add_argument('recs', nargs='+')
    x = s.add_parser('samples'); x.add_argument('recs', nargs='+'); x.add_argument('--xhat', type=float, required=True)
    x.add_argument('--xse', type=float, required=True)
    x = s.add_parser('score'); x.add_argument('probs')
    x = s.add_parser('compare'); x.add_argument('ref'); x.add_argument('test')
    x = s.add_parser('rowjobs'); x.add_argument('--rows', required=True); x.add_argument('--m', type=int, default=8)
    x.add_argument('--sub', type=int, default=0)
    x = s.add_parser('synth'); x.add_argument('--F', type=float, required=True); x.add_argument('--m', type=int, required=True)
    x.add_argument('--k', type=int, required=True); x.add_argument('--out', required=True); x.add_argument('--rseed', type=int, default=0)
    a = p.parse_args()
    globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
