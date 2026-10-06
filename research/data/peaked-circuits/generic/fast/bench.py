"""Benchmarks for the backend port.
  bench.py solve FILE BACKEND [CENTRE]                 full solve_generic pipeline, growth on BACKEND,
                                                       final exact contraction in numpy complex128
  bench.py grow  FILE CENTRE BACKEND MAX_ELEMS TMAX [prof]   growth only (tnob.grow), per-step log
Prints one 'RESULT {json}' line. The peak string itself is never printed: only sha1 hash."""
import sys, os, time, json, threading, hashlib, resource
import numpy as np


def mem_guard(limit_gb=4.0):
    import psutil
    p = psutil.Process()
    def run():
        while True:
            rss = p.memory_info().rss / 2**30
            gpu = 0.0
            try:
                import backend as B
                if B._mx is not None:
                    gpu = B._mx.get_active_memory() / 2**30
            except Exception:
                pass
            if rss + gpu > limit_gb:
                print(f"MEM GUARD: rss {rss:.2f} GB + mlx {gpu:.2f} GB > {limit_gb} GB, aborting", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r // 2**20 if sys.platform == 'darwin' else r // 1024


if __name__ == '__main__':
    mem_guard()
    mode, f = sys.argv[1], sys.argv[2]
    import backend as B
    if mode == 'solve':
        be = sys.argv[3]; c = float(sys.argv[4]) if len(sys.argv) > 4 else None
        B.setup(be)
        import solve_b as SG, tnob
        logs = []
        def log(s):
            logs.append(s); print(s, flush=True)
        t0 = time.time()
        r = SG.solve(f, c, log=log)
        res = dict(mode=mode, file=os.path.basename(f), backend=be, centre=r['centre'], window=r['window'],
                   absorbed=r['absorbed'], W_max_bond=r['W_max_bond'], width=r['width'],
                   peak_sha1=hashlib.sha1(r['peak'].encode()).hexdigest(), p=r['p'], norm=r['norm'],
                   min_abs_z=r['min_abs_z'], seconds=round(time.time() - t0, 2),
                   t_gate=round(tnob.TIMERS['gate'], 2), t_compress=round(tnob.TIMERS['compress'], 2),
                   t_sync=round(tnob.TIMERS['sync'], 2), rss_mb=peak_rss_mb())
        for s in logs:
            if 'scan' in s and 'centre layer' in s:
                res['t_scan'] = float(s.split('scan ')[1].split('s')[0])
            if 'gates absorbed' in s:
                res['t_grow_main'] = float(s.rsplit('(', 1)[1].split('s)')[0])
        print("RESULT", json.dumps(res), flush=True)
    else:
        c, be, me, tmax = float(sys.argv[3]), sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
        prof = len(sys.argv) > 7 and sys.argv[7] == 'prof'
        B.setup(be)
        import gparse as G, struct_probe as SP, tnob
        n, units, tail = G.parse(f); L = SP.layers(n, units)
        if prof:
            import cProfile, pstats, io
            pr = cProfile.Profile(); pr.enable()
        t0 = time.time()
        W, lo, hi, hist = tnob.grow(n, units, L, c, max_elems=me, tmax=tmax, log=True)
        wall = time.time() - t0
        if prof:
            pr.disable()
            s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats('tottime').print_stats(35); print(s.getvalue())
            s = io.StringIO(); st.stream = s; st.sort_stats('cumulative').print_stats(40); print(s.getvalue())
        res = dict(mode=mode, file=os.path.basename(f), backend=be, centre=c, lo=lo, hi=hi, steps=len(hist),
                   final=hist[-1], wall=round(wall, 2), t_gate=round(tnob.TIMERS['gate'], 2),
                   t_compress=round(tnob.TIMERS['compress'], 2), t_sync=round(tnob.TIMERS['sync'], 2),
                   step_times=[h['step'] for h in hist], elems=[h['elems'] for h in hist], rss_mb=peak_rss_mb())
        print("RESULT", json.dumps(res), flush=True)
