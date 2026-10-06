"""summarise bench_fast.py RESULT lines (logs/fast_grow_*_t1.log, logs/fsolve_*.log)"""
import json, glob
def rows(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.startswith('RESULT {'):
                r = json.loads(l[7:]); r['log'] = f; out.append(r)
    return out
G = rows('logs/fast_grow_P9_*_t1.log')
ref = next((r for r in G if r['impl'] == 'quimb'), None)
print("P9 tnoq growth c=50 -> 1.2e6 elems (single-thread BLAS):  impl opts | cpu s | wall s | trajectory == quimb | last step s")
for r in G:
    same = ref is not None and all(r[k] == ref[k] for k in ('elems', 'max_bond', 'nbonds'))
    print(f"  {r['impl']:5s} {r['opts']:11s} | {r['cpu_s']:7.2f} | {r['wall']:7.2f} | {same} | {r['step_times'][-1]:.2f}")
S = rows('logs/fsolve_*.log')
print("solve:  file impl opts | cpu s | wall s | scan s | main grow s | peak == quimb | p | |dp|")
for r in S:
    b = next(x for x in S if x['file'] == r['file'] and x['impl'] == 'quimb')
    print(f"  {r['file'][:4]} {r['impl']:5s} {r['opts']:11s} | {r['cpu_s']:7.2f} | {r['seconds']:7.2f} | {r.get('t_scan', 0):6.1f} | "
          f"{r.get('t_grow_main', 0):5.1f} | {r['peak_sha1'] == b['peak_sha1']} | {r['p']:.6f} | {abs(r['p'] - b['p']):.1e}")
