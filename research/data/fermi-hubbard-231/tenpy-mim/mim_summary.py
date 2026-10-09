import json, glob, re, pickle, numpy as np
D = '/tmp/fh-231/'
t1 = pickle.load(open(D + 'tdvp_data_1_pt.pkl', 'rb')); t2 = pickle.load(open(D + 'tdvp_data_2_pt.pkl', 'rb'))
hw1 = pickle.load(open(D + 'mm_and_dr_data_1_pt.pkl', 'rb')); hw2 = pickle.load(open(D + 'mm_and_dr_data_2_pt.pkl', 'rb'))
res = {}
for f in glob.glob('/tmp/fh-231-t/mim/mim_*.json'):
    r = json.load(open(f)); res.setdefault((r['obs'], r['k']), []).append(r)
out = []
P = out.append
P('| obs | t (k) | k0 | chi_psi | chi_O | value | cum. discarded wt (psi / O) | time Heis (s) |')
P('|---|---|---|---|---|---|---|---|')
for (obs, k) in sorted(res):
    for r in sorted(res[(obs, k)], key=lambda r: (r['k0'], r['chiPsi'], r['chiO'])):
        P(f"| {obs} | {k*0.2:.1f} ({k}) | {r['k0']} | {r['chiPsi']} | {r['chiO']} | {r['val']:.6f} | {r['epsPsi']:.1e} / {r['epsO']:.1e} | {r['tH']:.0f} |")
open('/tmp/fh-231-t/mim_table.md', 'w').write('\n'.join(out)); print('\n'.join(out))
