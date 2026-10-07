from chain import *
import random
n0, ops = parse()
for (n, lo, hi) in [(12, 0, 12), (14, 56, 70), (16, 50, 70)]:
    w = window(n0, ops, n, lo, hi); lines, be = chain(n, w)
    for s in range(3):
        x = random.Random(s).getrandbits(n)
        p = compile_plan(lines, x); a = run(p); e = sv_amp(n, w, x)
        print(n, lo, hi, p['width'], len(p['ops']), sum(p['backward']), abs(a-e)/2**(-n/2))
