# Generates jobs.txt for the Mac campaign (one tt invocation per line).
P1 = [0.13, 0.14, 0.15, 0.155, 0.16, 0.165, 0.17, 0.18, 0.20, 0.22]
NS = {64: 200, 128: 120, 256: 64, 512: 32, 1024: 12}
jobs = []
def steady(tag, n, p, eta, s, pattern="poisson", init="zero", depth=None, window=None, seed0=0):
    depth = depth or 2 * n
    window = window or n // 2
    jobs.append(f"steady tag={tag} n={n} pm={p} eta={eta} samples={s} pattern={pattern} init={init} depth={depth} window={window} seed0={seed0} header=0")
# S: eta families (poisson)
for eta in [0.5, 2, 4, 1]:
    for n, s in sorted(NS.items(), reverse=True):
        for p in P1:
            steady("S", n, p, eta, s)
# F: fixed-site line defect, exact count
for n, s in sorted(NS.items(), reverse=True):
    for p in P1:
        steady("F", n, p, 1, s, pattern="fixed")
for n in [128, 256, 512]:
    for p in [0.14, 0.16, 0.18]:
        steady("X", n, p, 1, NS[n], pattern="exact")
# V: single-injection survival
for p in [0.155, 0.16, 0.165]:
    for k in range(4):
        jobs.append(f"survival n=1024 pm={p} burn=256 tmax=512 gap=32 samples=10 seed0={1000*k} header=0")
    jobs.append(f"survival n=256 pm={p} burn=128 tmax=128 gap=16 samples=100 seed0=50000 header=0")
# D: decay of the maximally mixed state
for p in [0.155, 0.16, 0.165]:
    for n, s in [(256, 40), (512, 20), (1024, 10)]:
        jobs.append(f"decay n={n} pm={p} depth={2*n} samples={s} header=0")
# H: small h near p_c
for p in [0.15, 0.155, 0.16, 0.165, 0.17]:
    steady("H", 1024, p, 0.25, 24)
    steady("H", 1024, p, 0.0625, 24)
    steady("H", 2048, p, 0.03125, 8)
# M: stationarity, init mixed vs zero, deep circuits
for p in [0.10, 0.13, 0.14]:
    for n, s in [(256, 16), (512, 8)]:
        for init in ["zero", "mixed"]:
            steady("M", n, p, 1, s, init=init, depth=8 * n, window=n)
# L: n = 2048 for eta = 1, 4
for eta in [1, 4]:
    for p in [0.15, 0.16, 0.17]:
        steady("S", 2048, p, eta, 6)
open("jobs.txt", "w").write("\n".join(jobs) + "\n")
print(len(jobs))
