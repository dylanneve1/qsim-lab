#!/usr/bin/env python3
"""Magic-atlas sweep driver (research/magic-atlas.md).

  python3 driver.py atlas BIN OUTDIR       # cheap invariants at scale -> OUTDIR/atlas.csv, OUTDIR/profiles/*.csv
  python3 driver.py magic BIN OUTDIR       # ground-truth nullity/SRE at n<=12 -> OUTDIR/magic.csv, OUTDIR/magic/*.csv
Deterministic (no timing claims); runs anywhere. Use nice / -j on shared hosts.
"""
import csv, json, math, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

PI = math.pi

def atlas_specs():
    S = []
    for inp in ["basis", "plus", "graph"]:
        for n in [16, 32, 64, 128, 256, 512, 1024]:
            S.append(("qft", f"qft:n={n},in={inp}"))
    for inp in ["basis", "graph"]:
        for n in [64, 256]:
            for cut in [1, 2, 3, 4, 6, 8, 12, 16]:
                S.append(("aqft", f"qft:n={n},in={inp},cut={cut}"))
    for fam in ["cuccaro", "gidney", "draper"]:
        for inp in ["basis", "plusa", "plusab"]:
            for bits in [8, 16, 32, 64, 128, 256]:
                S.append((fam, f"{fam}:bits={bits},in={inp}"))
    for cut in [2, 3, 4, 8]:
        for inp in ["basis", "plusab"]:
            S.append(("draper", f"draper:bits=64,in={inp},cut={cut}"))
    for nb in [4, 8, 12, 16, 24, 32]:
        for w in [1, 2, 4]:
            for inp in ["one", "half"]:
                S.append(("shorwin", f"shorwin:nbits={nb},w={w},in={inp}"))
    for nb in [4, 5, 6, 7, 8, 10]:
        S.append(("shor", f"shor:nbits={nb},w=2"))
    for n in [8, 16, 32, 64, 128]:
        for it in [1, 2, 4, 8]:
            S.append(("grover", f"grover:n={n},it={it}"))
    for n in [16, 32, 64, 128, 256]:
        for st in [1, 2, 4, 8, 16]:
            S.append(("ising", f"ising:n={n},steps={st},dt=0.1"))
            S.append(("heis", f"heis:n={n},steps={st},dt=0.1"))
    # step-size sweep through the Clifford point J dt = h dt = pi/4
    for k in range(0, 17):
        dt = PI / 4 * k / 8
        if k == 0:
            continue
        S.append(("ising-dt", f"ising:n=64,steps=4,dt={dt:.15f}"))
    # dual-unitary / Clifford-ZZ lines: J dt = pi/4 with generic h, and vice versa
    for h in [0.1, 0.3, 0.5]:
        S.append(("ising-du", f"ising:n=64,steps=4,dt=1,J={PI/4:.15f},h={h}"))
        S.append(("ising-du", f"ising:n=64,steps=4,dt=1,J={h},h={PI/4:.15f}"))
    for n in [16, 32, 64, 128, 256]:
        for p in [1, 2, 3]:
            for g in ["ring", "reg3"]:
                S.append(("qaoa", f"qaoa:n={n},p={p},graph={g}"))
        for L in [1, 2, 4]:
            S.append(("hea", f"hea:n={n},layers={L}"))
    for t in [4, 8, 12, 16, 20, 24]:
        for s in [16, 64, 256]:
            S.append(("qpe", f"qpe:t={t},s={s},kind=stab"))
    for t in [2, 3, 4, 5]:
        for s in [8, 16, 32]:
            S.append(("qpe-trotter", f"qpe:t={t},s={s},kind=trotter"))
    for m in [4, 8, 16, 32]:
        for st in [1, 2, 4, 8]:
            S.append(("walk", f"walk:m={m},steps={st}"))
    for t in [3, 4, 5, 6, 7, 8]:
        for m in sorted({2, t - 1}):
            S.append(("hhl", f"hhl:t={t},m={m}"))
    for n in [32, 64, 128, 256]:
        for t in [n // 4, n // 2, n, 2 * n]:
            S.append(("rct", f"rct:n={n},L={n // 2},t={t}"))
    return S

# instances whose per-checkpoint profile is written (the "when" plots)
PROFILE = {
    "qft:n=256,in=basis", "qft:n=256,in=graph", "qft:n=256,in=graph,cut=4",
    "cuccaro:bits=64,in=plusa", "gidney:bits=64,in=plusa", "draper:bits=64,in=plusa",
    "shorwin:nbits=16,w=4,in=one", "shor:nbits=8,w=2", "grover:n=64,it=4",
    "ising:n=128,steps=8,dt=0.1", "heis:n=128,steps=8,dt=0.1", "qaoa:n=128,p=2,graph=reg3",
    "hea:n=128,layers=2", "qpe:t=24,s=256,kind=stab", "walk:m=16,steps=4", "hhl:t=8,m=7",
    "rct:n=128,L=64,t=128",
}

def magic_specs():
    S = []
    for n in [6, 8, 10, 12]:
        S += [("qft", f"qft:n={n},in=basis"), ("qft", f"qft:n={n},in=graph"), ("qft", f"qft:n={n},in=plus")]
        S += [("hea", f"hea:n={n},layers=1"), ("qaoa", f"qaoa:n={n},p=1,graph=reg3"),
              ("ising", f"ising:n={n},steps=3,dt=0.1"), ("heis", f"heis:n={n},steps=2,dt=0.1"),
              ("rct", f"rct:n={n},L={n},t={n}")]
    for b in [2, 3, 4, 5]:
        for inp in ["basis", "plusa", "plusab"]:
            S.append(("cuccaro", f"cuccaro:bits={b},in={inp}"))
            S.append(("draper", f"draper:bits={b},in={inp}"))
        for inp in ["basis", "plusa"]:
            if 3 * b - 1 <= 12:
                S.append(("gidney", f"gidney:bits={b},in={inp}"))
    S.append(("shorwin", "shorwin:nbits=2,w=1,in=one"))
    S.append(("shorwin", "shorwin:nbits=2,w=1,in=half"))
    for n in [4, 5, 6, 7]:
        S.append(("grover", f"grover:n={n},it=3"))
    for t, s in [(3, 3), (4, 4), (5, 5), (6, 6)]:
        S.append(("qpe", f"qpe:t={t},s={s},kind=stab"))
    S.append(("qpe-trotter", "qpe:t=3,s=6,kind=trotter"))
    for m in [3, 4, 5]:
        S.append(("walk", f"walk:m={m},steps=4"))
    for t, m in [(3, 2), (4, 2), (4, 3), (5, 3)]:
        S.append(("hhl", f"hhl:t={t},m={m}"))
    return S

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr else "fail"}
    return json.loads(r.stdout.strip().splitlines()[-1])

def main():
    mode, binp, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    workers = int(os.environ.get("WORKERS", "2"))
    if mode == "atlas":
        os.makedirs(f"{out}/profiles", exist_ok=True)
        specs = atlas_specs()
        def job(fs):
            fam, spec = fs
            cmd = ["nice", "-n", "15", binp, "profile", spec, "1"]
            if spec in PROFILE:
                cmd.append(f"{out}/profiles/{spec.replace(':','_').replace(',','_').replace('=','')}.csv")
            r = run(cmd)
            r["family"] = fam
            r.setdefault("spec", spec)
            return r
        rows = list(ThreadPoolExecutor(workers).map(job, specs))
        fn = f"{out}/atlas.csv"
    else:
        os.makedirs(f"{out}/magic", exist_ok=True)
        specs = magic_specs()
        def job(fs):
            fam, spec = fs
            path = f"{out}/magic/{spec.replace(':','_').replace(',','_').replace('=','')}.csv"
            r = run(["nice", "-n", "15", binp, "magic", spec, "1", path, "150"])
            r["family"] = fam
            r.setdefault("spec", spec)
            return r
        rows = list(ThreadPoolExecutor(workers).map(job, specs))
        fn = f"{out}/magic.csv"
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    bad = [r for r in rows if "error" in r]
    print(f"{len(rows)} rows -> {fn}; {len(bad)} errors")
    for r in bad:
        print(r)

if __name__ == "__main__":
    main()
