"""Full-system lam-scan (TEBD chi=64, steps<=10): residual R(lam)=[E(lam)-E(0)] - a2 lam^2, fit a4 lam^4 + a6 lam^6;
odd-order check E(1)-E(-1).  Uses TEBD(lam)-TEBD(0) differences so truncation error largely cancels."""
import json, os, numpy as np
pt = json.load(open('pt2.json'))
def ld(c, l, chi=64):
    f = f'tebd_runs/{c}_chi{chi}_lam{l}.json'; return json.load(open(f)) if os.path.exists(f) else None
res = {}
for c in ('SCV', 'meson'):
    E = {l: ld(c, l) for l in (0, 1, -1, 2, 4, 8)}
    lams = [l for l in (1, 2, 4, 8) if E[l] is not None]
    print(c, 'lams', lams)
    rows = []
    for k in range(10):
        a2 = pt[c][k][2]
        R = {l: E[l][k]['stag'] - E[0][k]['stag'] - a2 * l * l for l in lams}
        odd = (E[1][k]['stag'] - E[-1][k]['stag']) if E[-1] is not None else float('nan')
        fit = [np.nan, np.nan]
        if len([l for l in lams if l >= 2]) >= 2:
            ls = np.array([l for l in lams if l >= 2], float); y = np.array([R[l] for l in lams if l >= 2])
            fit = np.linalg.lstsq(np.vstack([ls**4, ls**6]).T, y, rcond=None)[0]
        rows.append(dict(step=k + 1, a2=a2, R=R, odd=odd, a4=fit[0], a6=fit[1]))
        print(f"{c} step {k+1:2d} a2 {a2:+.3e} R(1) {R[1]:+.2e} " + ' '.join(f'R({l}) {R[l]:+.2e}' for l in lams if l > 1) +
              f" | fit a4 {fit[0]:+.2e} a6 {fit[1]:+.2e} | E(1)-E(-1) {odd:+.1e}")
    res[c] = rows
json.dump(res, open('lam_scan.json', 'w'), default=float)
