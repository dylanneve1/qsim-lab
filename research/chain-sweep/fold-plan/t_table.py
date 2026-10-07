"""For each k (group size) and local-bit budget w: the fewest-pass legal group-fold schedule.
python3 t_table.py n D k1,k2 w1,w2 [exec]"""
from fold import *
import random, sys, time
n0, ops = parse()
n, D = int(sys.argv[1]), int(sys.argv[2])
ks = [int(v) for v in sys.argv[3].split(',')]; ws = [int(v) for v in sys.argv[4].split(',')]
STRIDES = (1,) if n >= 40 else (1, 2, 3)
exe = len(sys.argv) > 5 and sys.argv[5] == 'exec'
w_ = window(n0, ops, n, 70-D, 70); lines, be = chain(n, w_)
x = random.Random(7).getrandbits(n)
if exe: e = sv_amp(n, w_, x); rms = 2**(-n/2)
p0 = compile_plan(lines, x); W = max(p0['cut_width'])
print(f"n={n} D={D} register width W={W} (sequential plan ops {len(p0['ops'])})", flush=True)
for k in ks:
    cands = []
    for first_dir in (False, True):
        b, groups = build_groups(lines, x, k, first_dir)
        for stride in STRIDES:
            for T0 in range(D//2 - 6, D//2 + 6 + 2*k):
                passes = assign_passes(b, groups, k, T0, stride)
                if check_legal(b.ops, passes): continue
                nloc, stored, local = pass_stats(b, passes)
                cands.append((max(nloc), max(stored), max(passes)+1, first_dir, stride, T0, b, passes, local, nloc, stored))
    for wl in ws:
        ok = [c for c in cands if c[0] <= wl]
        if not ok: print(f"k={k} w={wl}: no schedule fits"); continue
        c = min(ok, key=lambda c: (c[1], c[2]))      # least stored first, then fewest passes
        ml, ms, P, fd, stride, T0, b, passes, local, nloc, stored = c
        line = (f"k={k} w={wl}: passes={P} stored_max={ms} (+{ms-W} bits, x{2**(ms-W)}) max_local={ml} "
                f"dir0={'B' if fd else 'f'} stride={stride} T0={T0}")
        if exe:
            t = time.time(); a = block_execute(b, passes, local)
            line += f" | block-exec err/rms={abs(a-e)/rms:.1e} ({time.time()-t:.0f}s)"
        print(line, flush=True)
