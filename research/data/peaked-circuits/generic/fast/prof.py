"""Profile: cProfile of solve_generic.solve (P4/P10) or of tnoq.grow / tnos.grow on P9.
usage: prof.py solve FILE [CENTRE] | prof.py grow FILE CENTRE MAX_ELEMS [tnoq|tnos] [TMAX]"""
import sys, time, cProfile, pstats, io, collections
import numpy as np
import gparse as G, struct_probe as SP, tnoq, tnos, solve_generic as SG

def categorize(st):
    """sum tottime by category of leaf function"""
    cats = collections.Counter()
    for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
        key = f"{fn}:{name}"
        if 'svd' in name.lower() or 'gesdd' in name or 'svd' in fn.split('/')[-1] and 'decomp' in fn: c = 'svd'
        elif 'qr' in name.lower(): c = 'qr'
        elif 'eigh' in name.lower(): c = 'eigh'
        elif name in ('tensordot', 'einsum', 'matmul', 'dot') or 'c_einsum' in name or name == '<built-in method numpy.core._multiarray_umath.c_einsum>': c = 'contract'
        elif 'cotengra' in fn or 'opt_einsum' in fn: c = 'cotengra/path'
        elif 'numpy' in fn or 'built-in method numpy' in name: c = 'numpy-other'
        elif 'quimb' in fn: c = 'quimb-python'
        elif 'autoray' in fn: c = 'autoray'
        else: c = 'other:' + (fn.split('/')[-1] if '/' in fn else name)[:40]
        cats[c] += tt
    return cats

mode, f = sys.argv[1], sys.argv[2]
pr = cProfile.Profile(); t0 = time.time()
if mode == 'solve':
    c = float(sys.argv[3]) if len(sys.argv) > 3 else None
    pr.enable(); r = SG.solve(f, c, log=lambda s: print(s, flush=True)); pr.disable()
    import hashlib
    print('peak hash', hashlib.sha1(r['peak'].encode()).hexdigest()[:10], 'p', r['p'], 'secs', r['seconds'])
else:
    c, me = float(sys.argv[3]), float(sys.argv[4]); which = sys.argv[5] if len(sys.argv) > 5 else 'tnoq'
    n, units, tail = G.parse(f); L = SP.layers(n, units)
    pr.enable()
    if which == 'tnoq':
        W, lo, hi, h = tnoq.grow(n, units, L, c, max_elems=me, log=True)
    else:
        W, lo, hi, h = tnos.grow(n, units, L, c, max_elems=me, log=True, unswap=True)
    pr.disable()
print('wall', round(time.time() - t0, 1))
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats('tottime').print_stats(30); print(s.getvalue())
s = io.StringIO(); st.stream = s; st.sort_stats('cumulative').print_stats(45); print(s.getvalue())
tot = sum(v[2] for v in st.stats.values())
for k, v in categorize(st).most_common(25):
    print(f"  {k:45s} {v:8.2f} s  {100*v/tot:5.1f}%")
