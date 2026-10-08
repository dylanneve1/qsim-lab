#!/usr/bin/env python3
"""Independent bookkeeping check of the D=70 production records (verification task 1).

Re-implements the seeded streams from the spec (sha256(seed|tag|i|ctr), bits MSB-first, concatenated)
without importing analyze.py, then cross-checks against analyze.py's own functions.
Never prints the production seed.
"""
import hashlib, json, sys, os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUB_PROD = 'b9d2c1262c305229e5c2a12a1e0c4cad1eb2f933226b767878547950b37a2cf2'
PUB_S1 = 'bbf15613ae727b71e319c7b413d2487589e75e4176e1511d91f2fc6555bcf9e6'
PUB_S2 = '0635c3a949a78820271bda5c6a51e4e3a02c6c436fce78e8935e328e31a77622'
N = 70
ok = True
def check(cond, msg):
    global ok
    print(('PASS ' if cond else 'FAIL ') + msg)
    ok &= bool(cond)

def seed_of(p):
    return open(os.path.join(ROOT, p), 'rb').read().strip()

# ---- own implementation (written from the spec, different code path) ----
def bits(seed, tag, i, nbits):
    acc = bytearray(); ctr = 0
    while len(acc) * 8 < nbits:
        acc += hashlib.sha256(b'%s|%s|%d|%d' % (seed, tag.encode(), i, ctr)).digest(); ctr += 1
    v = int.from_bytes(bytes(acc), 'big')
    return bin(v)[2:].zfill(len(acc) * 8)[:nbits]

def u01(seed, tag, i):
    return int(bits(seed, tag, i, 53), 2) * 2.0 ** -53

# ---- analyze.py, for cross-check ----
spec = importlib.util.spec_from_file_location('an', os.path.join(ROOT, 'runplan/analyze.py'))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

prod = seed_of('production/prod-seed.txt'); s1 = seed_of('production/s1-calseed.txt'); s2 = seed_of('production/s2-calseed.txt')
check(hashlib.sha256(prod).hexdigest() == PUB_PROD, 'sha256(prod seed) == published b9d2c126...2cf2')
check(hashlib.sha256(s1).hexdigest() == PUB_S1, 'sha256(s1 cal seed) == bbf15613... (SESSION1.md)')
check(hashlib.sha256(s2).hexdigest() == PUB_S2, 'sha256(s2 cal seed) == 0635c3a9... (SESSION2.md)')

M = 6
own = [('.' * M + bits(prod, 'prefix', i, N - M), u01(prod, 'tail', i)) for i in range(200)]
ref = [('.' * M + an.stream(prod, 'prefix', i, N - M), an.unif(prod, 'tail', i)) for i in range(200)]
check(own == ref, 'own stream implementation == analyze.py stream/unif for i=0..199')
check(len({p for p, _ in own}) == 200, '200 production prefixes are distinct')

# job files
jobs1 = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session1/jobs.jsonl')) if l.strip()]
jobs2 = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session2/jobs_s2.jsonl')) if l.strip()]
pj1 = [j for j in jobs1 if 'i' in j]; pj2 = [j for j in jobs2 if 'i' in j]
check([j['i'] for j in pj1] == list(range(200)) and all(j['prefix_bits'] == own[j['i']][0] and j['u_tail'] == own[j['i']][1] for j in pj1),
      'session1 jobs.jsonl production jobs == seed-derived prefixes/u_tail, i=0..199 in order')
check(jobs2[2:] == jobs1, 'jobs_s2.jsonl = 2 new calib jobs + session1 jobs.jsonl unchanged')

# calibration rows from the calibration seeds (own re-implementation of calrows)
S = np.load(an.SUTD); assign = S['assignments_q8_q69']; bs = S['bitstrings_q0_first']
def calrows(seed, m, k):
    rows, i = [], 0
    while len(rows) < k:
        r = int(bits(seed, 'calrow', i, 32), 2) % len(assign); i += 1
        if r not in [x for x, _ in rows]:
            rows.append((r, bits(seed, 'calsub', r, max(0, 8 - m))))
    return [{'row': r, 'prefix_bits': '.' * m + sb + str(assign[r])} for r, sb in rows]
c1 = calrows(s1, 6, 2); c2 = calrows(s2, 6, 2)
print('  s1 cal rows', [c['row'] for c in c1], ' s2 cal rows', [c['row'] for c in c2])
check([j for j in jobs1 if 'row' in j] == c1, 'session1 calibration jobs == calrows(s1 seed, m=6, k=2)')
check([j for j in jobs2 if 'row' in j] == c2 + c1, 'session2 calibration jobs == calrows(s2 seed) + calrows(s1 seed)')
for c in c1 + c2:
    r = c['row']
    check(c['prefix_bits'][8:] == assign[r] == bs[r][8:], f'cal row {r}: q8..q69 == SUTD assignment == IBM bitstring tail')

# records
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session2/out_final74.jsonl')) if l.strip()]
res = [r for r in recs if r['kind'] in ('calib', 'sample')]
cal = [r for r in res if r['kind'] == 'calib']; smp = [r for r in res if r['kind'] == 'sample']
check(len(cal) == 4 and len(smp) == 74, f'out_final74: {len(cal)} calib + {len(smp)} sample records')
check([r['row'] for r in cal] == [555, 1784, 1180, 495], 'calib rows in order 555, 1784, 1180, 495')
check(all(any(r['prefix_bits'] == c['prefix_bits'] and r['row'] == c['row'] for c in c1 + c2) for r in cal),
      'every calib record prefix_bits == seed-derived calibration job')
check([r['i'] for r in smp] == list(range(74)), 'sample indices exactly 0..73, in file order, no gaps or duplicates')
check(all(r['prefix_bits'] == own[r['i']][0] for r in smp), 'every sample prefix_bits == its committed prefix')
check(all(r['u_tail'] == own[r['i']][1] for r in smp), 'every sample u_tail == committed u (bit-exact float)')
check(all(r['tail_m'] == 6 and r['R'] == 65 and r['format'] == 'int5:b16:h' and r['n'] == 70 and r['d'] == 70 for r in res),
      'all records: m=6, R=65, int5:b16:h, n=70, d=70')
check(all(r['underflow'] == 0 and r['overflow'] == 0 for r in res), 'no underflow/overflow flagged in any record')
check(all(len(r['amps']) == 64 for r in res), 'every record has 64 amplitudes')
# sampler draw: smallest j with cumsum |l|^2 > u * sum
bad = []
for r in smp:
    a = np.array(r['amps']); p = a[:, 0] ** 2 + a[:, 1] ** 2; c = np.cumsum(p)
    j = int(np.argmax(c > r['u_tail'] * c[-1]))
    bstr = format(j, '06b') + r['prefix_bits'][6:]
    if j != r['j_tail'] or bstr != r['bitstring_q0_first']:
        bad.append(r['i'])
check(not bad, f'j_tail and bitstring_q0_first re-derived from amps + u for all samples (mismatches: {bad})')
check(len({r['bitstring_q0_first'] for r in smp}) == 74, '74 distinct sample bitstrings')
# timing/order
t = [r['t_end_unix'] for r in res]
check(all(x < y for x, y in zip(t, t[1:])), 'result records strictly increasing in t_end_unix (calib555,1784,s0..s41,c1180,c495,s42..)')
order = [(r['kind'], r.get('i', r.get('row'))) for r in res]
print('  result order head:', order[:4], '... around 46:', order[43:48])
# start lines: each job has >=1 start; extra starts = redone jobs
from collections import Counter
st = Counter((r.get('i'), r.get('row')) for r in recs if r['kind'] == 'start')
extra = {k: v for k, v in st.items() if v > 1}
print('  jobs with more than one start line (redone after kill/outage):', extra)
nostart = [r for r in recs if r['kind'] == 'start' and r.get('i') is not None and r['i'] >= 74]
print('  start lines without a result (in progress at stop):', [(r.get('i')) for r in nostart])
# session-1 file is a prefix-consistent subset of the final file
s1recs = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session1/out.jsonl')) if l.strip()]
s1res = [r for r in s1recs if r['kind'] in ('calib', 'sample')]
fin = {(r['kind'], r.get('i', r.get('row'))): r for r in res}
check(all(fin[(r['kind'], r.get('i', r.get('row')))] == r for r in s1res), f'all {len(s1res)} session-1 result records identical in out_final74')
check(recs[:len(s1recs)] == s1recs, 'out_final74 begins with session1 out.jsonl verbatim (append-only)')
# sample bitstrings never coincide with IBM's samples
ibm = set(bs)
check(not any(r['bitstring_q0_first'] in ibm for r in smp), 'no production sample equals an IBM sample bitstring')
# circuit file identity
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
check(h(os.path.join(ROOT, 'nq70_depth70_checks27_doped.qasm')) == h(os.path.join(ROOT, 'sutd/data-for-doped-clifford-tn-simulation/circuit/nq70_depth70_checks27_doped.qasm')),
      'workspace circuit qasm == SUTD copy of the IBM circuit (sha256)')
print('sha256 out_final74.jsonl', h(os.path.join(ROOT, 'production/session2/out_final74.jsonl')))
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
