#!/usr/bin/env python3
"""Extract the Kim et al. (Nature 618, 500 (2023)) experiment values from the
authors' data repository (github.com/youngseok-kim1/Evidence-for-the-utility-
of-quantum-computing-before-fault-tolerance, also figshare 22500355), using the
exact selection rule of their plotting notebooks: the ZNE point estimate is the
last of (linear, exponential) fits whose fit uncertainty is < 0.5, else the
unmitigated value; error bars are the 68% bootstrap interval.
Usage: extract_kim.py <path-to-kimrepo>   -> kim_fig*_experiment.csv here.
Note: fig3c and fig4a are stored with the opposite sign to the paper's plots
(the fig4a notebook multiplies by ysign = -1); values here are as stored,
which match this repo's convention (the stabilizer value at θ = π/2 is −1)."""
import numpy as np, csv, copy, os, sys
repo = sys.argv[1]
out = os.path.dirname(os.path.abspath(__file__))
crit = 0.5; pct = 50 + 68.2 / 2
def load(fn):
    rows = {}
    for d in csv.reader(open(os.path.join(repo, 'data', fn))):
        rows[float(d[0])] = np.array([float(x) for x in d[1:]])
    return rows
def single(fig, nsf=3):
    bu = load(f'{fig}_bootstrap_unmit.txt'); bm = load(f'{fig}_bootstrap_mit.txt')
    eu = load(f'{fig}_experiment_unmit.txt'); em = load(f'{fig}_experiment_mit.txt')
    res = []
    for a in sorted(eu):
        unmit = bu[a].reshape(nsf, 100); mit = bm[a].reshape(100, 2, 2); me = em[a].reshape(2, 2)
        best = copy.copy(unmit[0, :])
        for k in range(2):
            t = np.where(mit[:, k, 1] < crit)[0]; best[t] = mit[t, k, 0]
        med = np.median(best); lo = np.percentile(best, 100 - pct); hi = np.percentile(best, pct)
        y = eu[a][0]
        for k in range(2):
            if me[k, 1] < crit: y = me[k, 0]
        res.append((a, eu[a][0], y, med, lo, hi))
    return res
def mz(fig='fig3a', n=127, nsf=3):
    bu = load(f'{fig}_bootstrap_unmit.txt'); bm = load(f'{fig}_bootstrap_mit.txt')
    eu = load(f'{fig}_experiment_unmit.txt'); em = load(f'{fig}_experiment_mit.txt')
    res = []
    for a in sorted(eu):
        unmit = bu[a].reshape(n, nsf, 100); mit = bm[a].reshape(n, 100, 2, 2)
        me = em[a].reshape(n, 2, 2); e = eu[a].reshape(n, nsf)
        best = copy.copy(unmit[:, 0, :])
        for nb in range(100):
            for k in range(2):
                t = np.where(mit[:, nb, k, 1] < crit)[0]; best[t, nb] = mit[t, nb, k, 0]
        m = best.mean(axis=0)
        tmp = copy.copy(e[:, 0])
        for k in range(2):
            mk = np.where(me[:, k, 1] < crit)[0]; tmp[mk] = me[mk, k, 0]
        res.append((a, e[:, 0].mean(), tmp.mean(), np.median(m), np.percentile(m, 100 - pct), np.percentile(m, pct)))
    return res
for fig, f in [('fig3a', mz), ('fig3b', lambda: single('fig3b')), ('fig3c', lambda: single('fig3c')),
               ('fig4a', lambda: single('fig4a')), ('fig4b', lambda: single('fig4b'))]:
    with open(f'{out}/kim_{fig}_experiment.csv', 'w') as fh:
        fh.write('theta_h,unmitigated,mitigated,boot_median,boot_lo68,boot_hi68\n')
        for row in f(): fh.write(','.join('%.6g' % v for v in row) + '\n')
