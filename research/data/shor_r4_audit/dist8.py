import sys, random, numpy as np
from math import gcd
sys.argv = [sys.argv[0], sys.argv[1], "A"]
from dist_check import rust_dist, textbook
random.seed(4242)
worst = 0
for N in (143, 187, 221, 247, 253, 129, 133, 205):
    a = random.randrange(2, N - 1)
    while gcd(a, N) != 1: a = random.randrange(2, N - 1)
    w = random.choice((2, 3, 4))
    t, pr = rust_dist(N, a, w); pt = textbook(N, a, t)
    d = np.abs(pr - pt).max(); worst = max(worst, d)
    print(f"N={N} a={a} w={w} t={t} sum={pr.sum():.15f} max|dp|={d:.3e}", flush=True)
print("WORST 8-bit:", worst)
