#!/usr/bin/env python3
"""Job lines for the noise-oracles campaign.

Line format: `name  CONC  noise_oracles-args...` (one CSV per line).
Instances = research/data/shor-noise/instances.txt (same N, a as shor-noise).
cap = 4r (= 8 x the noiseless peak support r/2, the v-cap rule of shor-noise),
except the calibration runs (no binding cap).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
inst = {}
for l in open(os.path.join(HERE, '../../shor-noise/instances.txt')):
    if l.startswith('#'):
        continue
    b, N, p, q, a, r = l.split()[:6]
    inst[int(b)] = (int(N), int(a), int(r))

def jobs(oracles, ns, k0=150, k1=800, k23=300, tag='', kind='depol', extra_env=''):
    out = []
    for o in oracles:
        oname = o.replace(':', '')
        for n in ns:
            N, a, r = inst[n]
            cap = 4 * r
            base = 100000 * (1 + ['opt:4', 'mbul:4', 'mbu:4', 'windowed:4'].index(o)) + 100 * n
            s = base + (50 if tag else 0)
            pre = f'{oname}{tag}_{kind}_{n}'
            out.append(f'{pre}_k0 QSIM_NOISE_KMIN=0 {extra_env} strat {o} {N} {a} {kind} 0 {k0} {s+1} {cap}')
            out.append(f'{pre}_k1 QSIM_NOISE_KMIN=1 {extra_env} strat {o} {N} {a} {kind} 1 {k1} {s+2} {cap}')
            out.append(f'{pre}_k23 QSIM_NOISE_KMIN=2 {extra_env} strat {o} {N} {a} {kind} 3 {k23} {s+3} {cap}')
    return out

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'vps':
        ns = [10, 12, 14, 16]
        L = jobs(['opt:4', 'mbul:4', 'mbu:4'], ns)
    elif which == 'cal':
        # calibration: no binding cap (support may reach 2^(2n)), k = 1..3
        L = []
        for o in ['opt:4', 'mbul:4', 'mbu:4']:
            for n in [10, 11]:
                N, a, r = inst[n]
                oname = o.replace(':', '')
                s = 900000 + 1000 * ['opt:4', 'mbul:4', 'mbu:4'].index(o) + 10 * n
                L.append(f'{oname}cal_depol_{n}_k123 QSIM_NOISE_KMIN=1 strat {o} {N} {a} depol 3 300 {s} {1 << 26}')
    elif which == 'mac':
        L = jobs(['opt:4', 'mbul:4', 'mbu:4'], [18, 20, 22, 24])
    else:
        raise SystemExit(which)
    print('\n'.join(L))
