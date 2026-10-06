"""cProfile of the swap-aware growth (tnos.grow_adaptive, unswap=True) on P9 from c=50, stopped after TMAX s.
usage: prof_tnos.py FILE CENTRE TMAX"""
import sys, time, signal, cProfile, pstats, io
import gparse as G, struct_probe as SP, tnos
if len(sys.argv) > 4 and sys.argv[4] == "fast":
    import tno_fast; tno_fast.install()
f, c, tmax = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
n, units, tail = G.parse(f); L = SP.layers(n, units)
class Stop(Exception): pass
def h(*a): raise Stop()
signal.signal(signal.SIGALRM, h); signal.alarm(tmax)
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
try:
    tnos.grow_adaptive(n, units, L, c, log=True, unswap=True, max_elems=4e5)
except Stop:
    print("STOP at", round(time.time() - t0, 1), "s", flush=True)
pr.disable()
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats('tottime').print_stats(30); print(s.getvalue())
s = io.StringIO(); st.stream = s; st.sort_stats('cumulative').print_stats(35); print(s.getvalue())
