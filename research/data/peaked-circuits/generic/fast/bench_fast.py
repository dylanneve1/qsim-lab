"""VPS benchmarks of tnoq_fast vs tnoq (quimb).
  bench_fast.py solve FILE IMPL [CENTRE] [--c64] [--batch]    IMPL = quimb | fast
  bench_fast.py grow  FILE CENTRE IMPL MAX_ELEMS TMAX [--c64] [--batch]
Prints a RESULT json line (peak only as sha1)."""
import sys, os, time, json, hashlib, resource
import numpy as np
args = [a for a in sys.argv[1:] if not a.startswith('--')]
flags = [a for a in sys.argv[1:] if a.startswith('--')]
import tnoq_fast
if '--c64' in flags:
    tnoq_fast.DTYPE = np.complex64
if '--batch' in flags:
    tnoq_fast.BATCH_QR = True
if '--lapack' in flags:
    tnoq_fast.QR_IMPL = 'lapack'
tag = ('c64' if '--c64' in flags else 'c128') + ('+batch' if '--batch' in flags else '') + ('+lapack' if '--lapack' in flags else '')
mode, f = args[0], args[1]
def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
if mode == 'solve':
    impl = args[2]; c = float(args[3]) if len(args) > 3 else None
    SG = __import__('solve_f' if impl == 'fast' else 'solve_generic')
    logs = []
    def log(s): logs.append(s); print(s, flush=True)
    c0 = cpu(); t0 = time.time(); r = SG.solve(f, c, log=log)
    res = dict(mode='solve', file=os.path.basename(f), impl=impl, opts=tag if impl == 'fast' else 'c128', centre=r['centre'],
               window=r['window'], W_max_bond=r['W_max_bond'], width=r['width'],
               peak_sha1=hashlib.sha1(r['peak'].encode()).hexdigest(), p=r['p'], norm=r['norm'],
               seconds=round(time.time() - t0, 2), cpu_s=round(cpu() - c0, 2), rss_mb=rss())
    for s in logs:
        if 'centre layer' in s and 'scan' in s: res['t_scan'] = float(s.split('scan ')[1].split('s')[0])
        if 'gates absorbed' in s: res['t_grow_main'] = float(s.rsplit('(', 1)[1].split('s)')[0])
    print("RESULT", json.dumps(res), flush=True)
else:
    c, impl, me, tmax = float(args[2]), args[3], float(args[4]), float(args[5])
    import gparse as G, struct_probe as SP
    n, units, tail = G.parse(f); L = SP.layers(n, units)
    t0 = time.time(); c0 = cpu()
    if impl == 'fast':
        W, lo, hi, h = tnoq_fast.grow(n, units, L, c, max_elems=me, tmax=tmax, log=True)
    else:
        import tnob
        W, lo, hi, h = tnob.grow(n, units, L, c, max_elems=me, tmax=tmax, log=True)
    res = dict(mode='grow', file=os.path.basename(f), impl=impl, opts=tag if impl == 'fast' else 'c128', centre=c,
               steps=len(h), lo=lo, hi=hi, wall=round(time.time() - t0, 2), cpu_s=round(cpu() - c0, 2), elems=[x['elems'] for x in h],
               max_bond=[x['max_bond'] for x in h], nbonds=[x['nbonds'] for x in h], step_times=[x['step'] for x in h],
               rss_mb=rss())
    print("RESULT", json.dumps(res), flush=True)
