"""Robustness sweep: same circuit, several cutoffs and centres; peak must not move. Writes the full results (with
peak strings) to private/, and prints a table with a short hash of each peak instead of the string."""
import sys, json, hashlib, solve_generic as SG
f = sys.argv[1]; tag = sys.argv[2]; centres = [float(x) for x in sys.argv[3].split(',')]; cutoffs = [float(x) for x in sys.argv[4].split(',')]
rows = []
for c in centres:
    for cu in cutoffs:
        try:
            r = SG.solve(f, centre=c, cutoff=cu, log=lambda s: None)
        except Exception as e:
            print(f"centre {c} cutoff {cu}: FAILED {e}", flush=True); continue
        r['peak_hash'] = 'peak#%d' % (1 + sorted(set(x['peak'] for x in rows + [r]), key=[x['peak'] for x in rows + [r]].index).index(r['peak']))
        rows.append(r)
        print(f"centre {c:6} cutoff {cu:7.0e}: {r['peak_hash']} p {r['p']:.4f} norm {r['norm']:.4f} min|Z| {r['min_abs_z']:.3f} "
              f"W bond {r['W_max_bond']} width {r['width']} log2flops {r['log2_flops']} {r['seconds']}s RSS {r['rss_mb']}MB", flush=True)
json.dump(rows, open(f"/tmp/peaked-generic/private/robust_{tag}.json", "w"), indent=1)
print("distinct peaks:", len(set(r['peak'] for r in rows)))
