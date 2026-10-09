"""Final centre-site (29) table: forward TEBD chi=3072 for t<=3, meet-in-the-middle (MIM) for t=4..6; error bars from chi_O/chi_psi spreads."""
import json, glob, pickle, numpy as np
D = '/tmp/fh-231/'
t1 = pickle.load(open(D + 'tdvp_data_1_pt.pkl', 'rb')); t2 = pickle.load(open(D + 'tdvp_data_2_pt.pkl', 'rb'))
hw1 = pickle.load(open(D + 'mm_and_dr_data_1_pt.pkl', 'rb')); hw2 = pickle.load(open(D + 'mm_and_dr_data_2_pt.pkl', 'rb'))
f3 = json.load(open('/tmp/fh-231-t/runs/tebd_L60_chi3072.json'))['rec']; f2 = json.load(open('/tmp/fh-231-t/runs/tebd_L60_chi2048.json'))['rec']
mim = {}
for f in glob.glob('/tmp/fh-231-t/mim/mim_*.json'):
    r = json.load(open(f)); mim[(r['obs'], r['k'], r['k0'], r['chiPsi'], r['chiO'])] = r['val']
def best_mim(obs, k):
    keys = [kk for kk in mim if kk[0] == obs and kk[1] == k]
    chiO = max(kk[4] for kk in keys)
    cands = [kk for kk in keys if kk[4] == chiO]
    # prefer the larger chiPsi at the largest chiO
    top = max(cands, key=lambda kk: kk[3])
    v = mim[top]
    # chi_O spread: compare with next lower chi_O at same chiPsi (or any chiPsi if absent)
    lower = sorted({kk[4] for kk in keys if kk[4] < chiO})
    dO = np.nan
    if lower:
        cl = [kk for kk in keys if kk[4] == lower[-1]]
        same = [kk for kk in cl if kk[3] == top[3]] or cl
        dO = abs(v - mim[same[0]])
    # chi_psi effect at matched chi_O (any chiO where both 256 and 512 exist)
    dP = 0.
    for co in sorted({kk[4] for kk in keys}):
        a = [mim[kk] for kk in keys if kk[4] == co and kk[3] == 256]; b = [mim[kk] for kk in keys if kk[4] == co and kk[3] == 512]
        if a and b: dP = max(dP, abs(a[0] - b[0]))
    return v, dO, dP, top
tdv = lambda obs, k: (np.nan if k >= 30 else (t1[(29, 'up')][k] if obs == 'nu' else t2[(29, 'down', 29, 'up')][k]))
hwv = lambda obs, k: (hw1[(29, 'up')][k] if obs == 'nu' else hw2[(29, 'down', 29, 'up')][k])
rows = []
P = rows.append
P('| t | quantity | method | value | error bar | TDVP (paper, cont.) | HW mm+dr | HW − ours | TDVP − ours |')
P('|---|---|---|---|---|---|---|---|---|')
summary = {}
for t in range(1, 7):
    k = 5 * t
    for obs, name in (('nu', '⟨n↑⟩'), ('dd', '⟨n↑n↓⟩')):
        if t <= 3:
            r = f3[k - 1]; v = r[obs][29]; nxt = json.load(open('/tmp/fh-231-t/runs/tebd_L60_chi2048.json'))['rec'][k - 1][obs][29]
            err = max(abs(v - nxt), 1e-13); meth = 'forward TEBD χ=3072 (spread vs χ=2048)'
        else:
            v, dO, dP, top = best_mim(obs, k)
            err = (0 if np.isnan(dO) else dO) + dP
            err = max(err, 1e-4)
            meth = f'MIM k0={top[2]} χψ={top[3]} χO={top[4]}'
        td = tdv(obs, k); hw = hwv(obs, k)
        summary[(obs, t)] = (v, err)
        P(f'| {t} | {name} | {meth} | {v:.6f} | {err:.1e} | {td:.6f} | {hw:.6f} | {hw - v:+.4f} | {td - v:+.4f} |')
open('/tmp/fh-231-t/FINAL_TABLE.md', 'w').write('\n'.join(rows)); print('\n'.join(rows))
