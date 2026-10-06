import json, glob, sys
rows = []
for f in sorted(glob.glob('logs/*.log')):
    for line in open(f):
        if line.startswith('RESULT {'):
            r = json.loads(line[7:]); r['log'] = f; rows.append(r)
S = [r for r in rows if r.get('mode') == 'solve']
ref = {r['file']: r for r in S if r['backend'] == 'np128'}
print("SOLVE: file backend | total s | scan s | main grow s | t_compress | t_gate | peak==np128 | p | |dp| | W bond | width | rss MB")
for r in S:
    b = ref[r['file']]
    print(f"{r['file'][:4]:5s} {r['backend']:7s} | {r['seconds']:7.1f} | {r.get('t_scan',0):6.1f} | {r.get('t_grow_main',0):5.1f} | {r['t_compress']:7.1f} | {r['t_gate']:5.2f} | "
          f"{r['peak_sha1']==b['peak_sha1']} | {r['p']:.6f} | {abs(r['p']-b['p']):.1e} | {r['W_max_bond']} | {r['width']} | {r['rss_mb']}")
G = [r for r in rows if r.get('mode') == 'grow']
print("GROW: log | backend | steps | final elems | wall s | compress s | sync s | last step s | step times")
for r in G:
    print(f"{r['log']:32s} {r['backend']:7s} {r['steps']:3d} {r['final']['elems']:8d} {r['wall']:7.1f} {r['t_compress']:7.1f} {r['t_sync']:5.1f} {r['step_times'][-1]:6.2f}  {[round(x,1) for x in r['step_times'][-5:]]}")
