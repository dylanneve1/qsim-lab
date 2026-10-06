"""Post-hoc comparison of solver peaks with reference peaks (run only AFTER solving)."""
import json, glob, sys
ref = json.load(open('/tmp/peaked-generic/refs/reference_peaks.json'))
key = {'heavy_hex_49x4020': 'heavy_hex_49x4020', 'heavy_hex_49x5072': 'heavy_hex_49x5072', 'P9_Hqap': 'P9_Hqap_56x1917',
       'P11_Hqap': 'P11', 'P12_Hqap': 'P12'}
for f in sorted(sys.argv[1:] or glob.glob('/tmp/peaked-generic/results/*.json') + glob.glob('/tmp/peaked-generic/private/*.json')):
    r = json.load(open(f)); name = r.get('file', f)
    k = [v for s, v in key.items() if s in name]
    if not k: continue
    R = ref[k[0]]
    if not R['bits']:
        print(f"{f.split('/')[-1]}: peak p={r['p']:.4f}; no public reference ({R['source'][:80]})"); continue
    h = sum(a != b for a, b in zip(r['peak'], R['bits'])); hr = sum(a != b for a, b in zip(r['peak'], R['bits'][::-1]))
    print(f"{f.split('/')[-1]}: p={r['p']:.4f}; Hamming to reference {h} (reversed order {hr}) / {len(R['bits'])}; ref: {R['source'][:90]}")
