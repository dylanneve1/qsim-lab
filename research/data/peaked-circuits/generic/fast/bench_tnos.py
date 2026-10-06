"""Swap-aware P9 growth (tnos.grow_adaptive, unswap=True, local_cutoff 1e-8, cutoff 1e-3) from CENTRE:
  orig    : tno.py as is (complex128)
  orig64  : tno.py as is, tensors complex64
  fast1   : tno_fast level 1 (QR-reduced split SVDs), complex128
  fast    : tno_fast level 2 (QR-reduced + singular-values-only choice of the leg assignment), complex128
  fast64  : tno_fast level 2, complex64
Stops after TMAX s (SIGALRM) or MAX_ELEMS. Prints per-step dicts and a RESULT line.
usage: bench_tnos.py FILE CENTRE VARIANT TMAX MAX_ELEMS [LOCAL_CUTOFF=1e-8]"""
import sys, time, signal, json, io, contextlib
import numpy as np
from bench import mem_guard, peak_rss_mb
mem_guard()
f, c, var, tmax, me = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
lc = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-8
import gparse as G, struct_probe as SP, tno as TN, tnos, tno_fast
if var.startswith("fast1"):
    tno_fast.SVALS_ONLY = False
if var.startswith("fast"):
    tno_fast.install()
if var.endswith('64'):
    tno_fast.set_dtype(np.complex64)
n, units, tail = G.parse(f); L = SP.layers(n, units)
class Stop(Exception): pass
def h(*a): raise Stop()
signal.signal(signal.SIGALRM, h); signal.alarm(tmax)
buf = io.StringIO(); hist = []
class Tee(io.TextIOBase):
    def write(self, s):
        sys.__stdout__.write(s); sys.__stdout__.flush(); buf.write(s); return len(s)
t0 = time.time(); stopped = False
try:
    with contextlib.redirect_stdout(Tee()):
        tnos.grow_adaptive(n, units, L, c, log=True, unswap=True, max_elems=me, local_cutoff=lc)
except Stop:
    stopped = True
wall = time.time() - t0
import ast
for line in buf.getvalue().splitlines():
    line = line.strip()
    if line.startswith('{'):
        hist.append(ast.literal_eval(line))
print("RESULT", json.dumps(dict(variant=var, local_cutoff=lc, centre=c, stopped_by_tmax=stopped, wall=round(wall, 1), steps=len(hist),
                                hist=hist, rss_mb=peak_rss_mb())), flush=True)
