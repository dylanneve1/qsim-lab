"""Independent audit of the support law B_i = min(2^i, r/gcd(r, 2^(t-i))) and
the work counter W = sum_i 2*|S_i|*G_i on fresh seeded instances (my own
generator, seed 2026), including deliberately odd r (base = b^(2^s)) and
high nu2(r) (primes p = 2^k m + 1)."""
import subprocess, sys, random
from math import gcd
EX = sys.argv[1]
def isprime(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True
def order(a, N):
    r, x = 1, a % N
    while x != 1: x = x * a % N; r += 1
    return r
def nu2(r): return (r & -r).bit_length() - 1
def lam(p, q):
    return (p - 1) * (q - 1) // gcd(p - 1, q - 1)
random.seed(2026)
cases = []
def randprime(lo, hi, cond=lambda p: True):
    while True:
        p = random.randrange(lo, hi) | 1
        if isprime(p) and cond(p): return p
for bits in (12, 14, 16, 18, 20, 22):
    h = bits // 2
    p = randprime(1 << (h - 1), 1 << h); q = randprime(1 << (bits - h - 1), 1 << (bits - h))
    while q == p: q = randprime(1 << (bits - h - 1), 1 << (bits - h))
    N = p * q
    a = random.randrange(2, N - 1)
    while gcd(a, N) != 1: a = random.randrange(2, N - 1)
    cases.append((N, a, "random base"))
    # odd order: a = b^(2^nu2(lambda))
    L = lam(p, q); s = nu2(L)
    b = random.randrange(2, N - 1)
    while gcd(b, N) != 1: b = random.randrange(2, N - 1)
    ao = pow(b, 1 << s, N)
    if ao not in (0, 1): cases.append((N, ao, "odd r"))
for bits in (14, 18, 22):
    h = bits // 2
    # high 2-adic valuation: p-1, q-1 divisible by 2^(h-4)
    k = h - 4
    p = randprime(1 << (h - 1), 1 << h, lambda p: (p - 1) % (1 << k) == 0)
    q = randprime(1 << (bits - h - 1), 1 << (bits - h), lambda q: (q - 1) % (1 << k) == 0 and q != p)
    N = p * q
    a = random.randrange(2, N - 1)
    while gcd(a, N) != 1: a = random.randrange(2, N - 1)
    cases.append((N, a, "high nu2"))
tot = eq = 0; bad = []
for N, a, kind in cases:
    r = order(a, N)
    out = subprocess.run([EX, "trace", str(N), str(a), "4", "7"], capture_output=True, text=True, check=True).stdout.split("\n")
    t = int(out[0].split()[1]); ops = int(out[0].split()[5])
    S = eval(out[1].split(" ", 1)[1]); G = eval(out[2].split(" ", 1)[1])
    B = [min(1 << i, r // gcd(r, 1 << (t - i))) for i in range(t)]
    neq = [(i, S[i], B[i]) for i in range(t) if S[i] != B[i]]
    over = [x for x in neq if x[1] > x[2]]
    W = sum(2 * S[i] * G[i] for i in range(t)); WB = sum(2 * B[i] * G[i] for i in range(t))
    tot += t; eq += t - len(neq)
    if over: bad.append((N, a, over))
    print(f"N={N} ({N.bit_length()} b) a={a} [{kind}] r={r} nu2={nu2(r)} t={t} |S_i|=B_i in {t-len(neq)}/{t} rounds; "
          f"exceed={len(over)}; W_counter={ops} sum2SG={W} match={ops==W}; W(B)/W={WB/ops:.4f}; peak={max(S)} vs max(r_odd,r/2)={max(r>>nu2(r), r//2)}" + (f" diffs={neq[:4]}" if neq else ""), flush=True)
print(f"TOTAL |S_i|=B_i in {eq}/{tot} rounds; rounds exceeding the bound: {bad}")
