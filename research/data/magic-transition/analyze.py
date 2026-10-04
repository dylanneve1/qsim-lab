#!/usr/bin/env python3
"""Aggregation, finite-size scaling and figures for research/magic-transition.md.

usage: analyze.py <raw.csv> <outdir>
"""
import sys, json, math
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(12345)

cols = None
rows = []
for line in open(raw):
    line = line.strip()
    if not line:
        continue
    if line.startswith("tag,"):
        cols = line.split(",")
        continue
    rows.append(line.split(","))
cols = cols or "tag,n,depth,p_m,p_t,seed,mode,d_final,d_avg,d_max,s_lo,s_hi,s_exact,i3_lo,i3_hi,i3_exact,nu,m2,t_gates,t_act,t_reg,m_frame,m_reg,m_det,elem_ops,secs,failed".split(",")
ci = {c: i for i, c in enumerate(cols)}


def f(r, c):
    v = r[ci[c]]
    try:
        return float(v)
    except ValueError:
        return float("nan")


def family(r):
    """Family from (n, p_t); phase-1 rows lost their tag (driver-script bug)."""
    tag = r[ci["tag"]]
    n = int(r[ci["n"]])
    pt = f(r, "p_t")
    mode = r[ci["mode"]]
    pre = "exact:" if mode == "exact" else ""
    if tag in ("E6", "E7"):
        return f"{tag}:depth={r[ci['depth']]}:" + (f"pt={round(pt * n)}/n")
    if pt == 0:
        return pre + "pt=0"
    for eta in (1, 2):
        if abs(pt - eta / n) < 1e-12:
            return pre + (f"pt={eta}/n" if eta > 1 else "pt=1/n")
    return pre + f"pt={pt:g}"


# cells: (family, n, p_m) -> arrays
cells = {}
for r in rows:
    if r[ci["failed"]] != "0":
        continue
    key = (family(r), int(r[ci["n"]]), round(f(r, "p_m"), 4))
    cells.setdefault(key, []).append(r)

def arr(key, c):
    return np.array([f(r, c) for r in cells[key]])

# ---------------------------------------------------------------- aggregate
agg = []
for key in sorted(cells):
    fam, n, pm = key
    d = arr(key, "d_avg") / n
    dfin = arr(key, "d_final")
    slo, shi = arr(key, "s_lo"), arr(key, "s_hi")
    i3l, i3h = arr(key, "i3_lo"), arr(key, "i3_hi")
    rec = dict(family=fam, n=n, p_m=pm, samples=len(d),
               phi=d.mean(), phi_err=d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan"),
               d_final=dfin.mean(),
               P_d_ge_0p1n=(dfin >= 0.1 * n).mean(), P_d_ge_0p05n=(dfin >= 0.05 * n).mean(),
               s_half_lo=np.nanmean(slo), s_half_hi=np.nanmean(shi),
               i3_lo=np.nanmean(i3l) if np.isfinite(i3l).any() else float("nan"),
               i3_hi=np.nanmean(i3h) if np.isfinite(i3h).any() else float("nan"),
               i3_err=np.nanstd(i3l, ddof=1) / math.sqrt(len(i3l)) if np.isfinite(i3l).any() else float("nan"),
               t_act_frac=arr(key, "t_act").sum() / max(1, arr(key, "t_gates").sum()),
               secs=arr(key, "secs").mean())
    agg.append(rec)
with open(f"{out}/aggregate.csv", "w") as fo:
    ks = list(agg[0].keys())
    fo.write(",".join(ks) + "\n")
    for a in agg:
        fo.write(",".join(f"{a[k]:.6g}" if isinstance(a[k], float) else str(a[k]) for k in ks) + "\n")

# ---------------------------------------------------------------- FSS
EFLOOR = {"v": 1e-3}


def collapse_cost(theta, data, beta):
    pc, nu = theta[0], theta[1]
    bnu = theta[2] if beta else 0.0
    if nu <= 0.3 or nu > 5:
        return 1e9
    xs, ys, ws = [], [], []
    for (n, p, y, e) in data:
        xs.append((p - pc) * n ** (1 / nu))
        ys.append(y * n ** bnu)
        ws.append(1.0 / max(e, EFLOOR["v"], 0.01 * abs(y)) ** 2 / n ** (2 * bnu))
    xs, ys, ws = map(np.array, (xs, ys, ws))
    # master curve: weighted polynomial
    deg = 4
    if len(xs) <= deg + 3:
        return 1e9
    try:
        cf = np.polyfit(xs, ys, deg, w=np.sqrt(ws))
    except Exception:
        return 1e9
    res = (np.polyval(cf, xs) - ys) ** 2 * ws
    return res.sum() / (len(xs) - deg - 1 - len(theta))


def fit(data, beta, x0):
    best = None
    starts = [x0]
    for dp in (-0.02, 0.0, 0.02):
        for nu in (0.9, 1.3, 2.0):
            for b in ((-0.3, 0.0, 0.3) if beta else (0.0,)):
                starts.append([x0[0] + dp, nu] + ([b] if beta else []))
    for s in starts:
        r = minimize(collapse_cost, s, args=(data, beta), method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-7, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return best


def cell_data(fam, ns, pwin, ycol, transform):
    data = []
    for (fm, n, pm), rs in cells.items():
        if fm != fam or n not in ns or not (pwin[0] <= pm <= pwin[1]):
            continue
        y = transform(n, rs)
        if y is None:
            continue
        data.append((n, pm, *y))
    return data


def boot_fit(fam, ns, pwin, transform, beta, x0, nboot=60):
    import os
    nboot = int(os.environ.get('NBOOT', nboot))
    data = cell_data(fam, ns, pwin, None, lambda n, rs: transform(n, rs, None))
    best = fit(data, beta, x0)
    bs = []
    for b in range(nboot):
        dat = cell_data(fam, ns, pwin, None, lambda n, rs: transform(n, rs, rng))
        r = fit(dat, beta, best.x)
        bs.append(r.x)
    bs = np.array(bs) if bs else np.full((1, len(best.x)), np.nan)
    return best, bs.std(axis=0), data


def phi_tr(n, rs, rg):
    v = np.array([float(r[ci["d_avg"]]) for r in rs]) / n
    if rg is not None:
        v = rg.choice(v, len(v))
    return (v.mean(), v.std(ddof=1) / math.sqrt(len(v)))


def i3_tr(n, rs, rg):
    v = np.array([float(r[ci["i3_lo"]]) for r in rs])
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return None
    if rg is not None:
        v = rg.choice(v, len(v))
    return (v.mean(), v.std(ddof=1) / math.sqrt(len(v)))


def pd_tr(alpha):
    def tr(n, rs, rg):
        v = (np.array([float(r[ci["d_final"]]) for r in rs]) >= alpha * n).astype(float)
        if rg is not None:
            v = rg.choice(v, len(v))
        m = v.mean()
        return (m, max(math.sqrt(m * (1 - m) / len(v)), 0.5 / len(v)))
    return tr


results = {}
NS_ALL = [16, 32, 64, 128, 256, 512, 1024]
for label, fam, ns, pwin, tr, beta, x0, fl in [
    ("EE_I3_pt0", "pt=0", [32, 64, 128, 256], (0.13, 0.19), i3_tr, False, [0.16, 1.3], 0.03),
    ("EE_I3_pt0_n64+", "pt=0", [64, 128, 256], (0.13, 0.19), i3_tr, False, [0.16, 1.3], 0.03),
    ("d_over_n_pt1n", "pt=1/n", [32, 64, 128, 256, 512, 1024], (0.10, 0.22), phi_tr, True, [0.16, 1.3, 0.4], 1e-3),
    ("d_over_n_pt1n_n64+", "pt=1/n", [64, 128, 256, 512, 1024], (0.10, 0.22), phi_tr, True, [0.16, 1.3, 0.4], 1e-3),
    ("d_over_n_pt1n_n128+", "pt=1/n", [128, 256, 512, 1024], (0.10, 0.22), phi_tr, True, [0.16, 1.3, 0.4], 1e-3),
    ("d_over_n_pt2n", "pt=2/n", [64, 128, 256, 512, 1024], (0.14, 0.25), phi_tr, True, [0.16, 1.3, 0.4], 1e-3),
    ("d_over_n_pt2n_n128+", "pt=2/n", [128, 256, 512, 1024], (0.14, 0.25), phi_tr, True, [0.16, 1.3, 0.4], 1e-3),
]:
    EFLOOR["v"] = fl
    try:
        best, err, data = boot_fit(fam, ns, pwin, tr, beta, x0)
        results[label] = dict(params=list(map(float, best.x)), err=list(map(float, err)),
                              chi2dof=float(best.fun), sizes=ns, window=pwin, points=len(data))
        print(label, results[label], flush=True)
    except Exception as e:
        print("fit failed", label, e)

# constant p_t: same window fits (expected to be poor: no transition)
for pt in ["pt=0.01", "pt=0.05", "pt=0.2"]:
    try:
        EFLOOR["v"] = 1e-3
        best, err, data = boot_fit(pt, [64, 128, 256, 512], (0.05, 0.30), phi_tr, True, [0.16, 1.3, 0.0], nboot=10)
        results[f"d_over_n_{pt}"] = dict(params=list(map(float, best.x)), err=list(map(float, err)),
                                          chi2dof=float(best.fun), sizes=[64, 128, 256, 512], window=(0.05, 0.3))
        print(pt, results[f"d_over_n_{pt}"], flush=True)
    except Exception as e:
        print("fit failed", pt, e)

# ---------------------------------------------------------------- model-free crossings
def cell_vals(fam, n, pm, kind):
    rs = cells.get((fam, n, pm))
    if not rs:
        return None
    if kind == "dbar":
        return np.array([float(r[ci["d_avg"]]) for r in rs])
    v = np.array([float(r[ci["i3_lo"]]) for r in rs])
    return v[np.isfinite(v)]


def curve(fam, n, kind, rg, pms):
    out_ = []
    for pm in pms:
        v = cell_vals(fam, n, pm, kind)
        if v is None or len(v) == 0:
            return None
        if rg is not None:
            v = rg.choice(v, len(v))
        out_.append(v.mean())
    return np.array(out_)


def first_cross(pms, a, b):
    """p where a - b changes sign (linear interpolation), first from the left."""
    dlt = a - b
    for i in range(len(pms) - 1):
        if dlt[i] == 0:
            return pms[i]
        if dlt[i] * dlt[i + 1] < 0:
            t = dlt[i] / (dlt[i] - dlt[i + 1])
            return pms[i] + t * (pms[i + 1] - pms[i])
    return float("nan")


def common_pms(fam, ns, kind):
    sets = [set(pm for (fm, n, pm) in cells if fm == fam and n == nn) for nn in ns]
    return sorted(set.intersection(*sets)) if sets else []


crossings = {}
# EE: I3(n) = I3(2n)
for (n1, n2) in [(32, 64), (64, 128), (128, 256)]:
    pms = [pm for pm in common_pms("pt=0", [n1, n2], "i3") if 0.12 <= pm <= 0.2]
    vals = []
    for b in range(201):
        rg = None if b == 0 else rng
        a1, a2 = curve("pt=0", n1, "i3", rg, pms), curve("pt=0", n2, "i3", rg, pms)
        vals.append(first_cross(pms, a1, a2))
    vals = np.array(vals)
    crossings[f"I3 {n1}x{n2}"] = (float(vals[0]), float(np.nanstd(vals[1:])))
# d: local exponent kappa(n) = log2(dbar(2n)/dbar(n)); crossing of kappa(n,2n) and kappa(2n,4n)
kap = {}
for fam, ns in [("pt=1/n", [16, 32, 64, 128, 256, 512, 1024]), ("pt=2/n", [32, 64, 128, 256, 512, 1024])]:
    for i in range(len(ns) - 2):
        n1, n2, n3 = ns[i], ns[i + 1], ns[i + 2]
        pms = [pm for pm in common_pms(fam, [n1, n2, n3], "dbar") if 0.08 <= pm <= 0.25]
        if len(pms) < 3:
            continue
        vals, kv = [], []
        for b in range(201):
            rg = None if b == 0 else rng
            c1, c2, c3 = (curve(fam, nn, "dbar", rg, pms) for nn in (n1, n2, n3))
            k1, k2 = np.log2(c2 / c1), np.log2(c3 / c2)
            x = first_cross(pms, k2, k1)
            vals.append(x)
            # kappa at the crossing
            kv.append(np.interp(x, pms, k2) if np.isfinite(x) else np.nan)
        vals, kv = np.array(vals), np.array(kv)
        crossings[f"kappa {fam} {n1},{n2},{n3}"] = (float(vals[0]), float(np.nanstd(vals[1:])), float(kv[0]), float(np.nanstd(kv[1:])))
    for i in range(len(ns) - 1):
        n1, n2 = ns[i], ns[i + 1]
        pms = sorted(set(pm for (fm, n, pm) in cells if fm == fam and n == n1) & set(pm for (fm, n, pm) in cells if fm == fam and n == n2))
        if pms:
            c1, c2 = curve(fam, n1, "dbar", None, pms), curve(fam, n2, "dbar", None, pms)
            kap[(fam, n1, n2)] = (np.array(pms), np.log2(c2 / c1))
for k, v in crossings.items():
    print("crossing", k, v)
results["crossings"] = {k: list(v) for k, v in crossings.items()}
json.dump(results, open(f"{out}/fss.json", "w"), indent=1)

# ---------------------------------------------------------------- figures
cmap = plt.get_cmap("viridis")


def ncol(n):
    return cmap((math.log2(n) - 4) / 6.2)


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, fam in enumerate(["pt=1/n", "pt=2/n"]):
    for (fm, n1, n2), (pms_, kk) in sorted(kap.items()):
        if fm == fam:
            ax[k].plot(pms_, kk, marker="o", ms=3, color=ncol(n2), label=f"n={n1}→{n2}")
    ax[k].axhline(1, c="k", lw=0.5); ax[k].axhline(0, c="k", lw=0.5); ax[k].axvline(0.16, ls=":", c="gray")
    ax[k].set_xlabel("$p_m$"); ax[k].set_ylabel("local exponent $\\kappa = \\log_2[\\bar d(2n)/\\bar d(n)]$")
    ax[k].set_title(f"{fam.replace('pt', '$p_T$')}: $\\bar d \\sim n^\\kappa$"); ax[k].legend(fontsize=7); ax[k].set_xlim(0.03, 0.42)
fig.tight_layout(); fig.savefig(f"{out}/local_exponent.png", dpi=130); plt.close(fig)

def series(fam, n, ycol="phi"):
    pts = sorted((a["p_m"], a[ycol], a.get(ycol + "_err", 0.0)) for a in agg if a["family"] == fam and a["n"] == n)
    return np.array(pts) if pts else np.zeros((0, 3))


# 1. E1 dilute: d/n with collapse
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for n in NS_ALL:
    s = series("pt=1/n", n)
    if len(s):
        ax[0].errorbar(s[:, 0], s[:, 1], s[:, 2], marker="o", ms=3, color=ncol(n), label=f"n={n}")
ax[0].axvline(0.16, ls=":", c="gray")
ax[0].set_xlabel("$p_m$"); ax[0].set_ylabel("$\\bar d/n$ (steady state)"); ax[0].set_title("dilute T: $p_T=1/n$ (one T per layer)")
ax[0].legend(fontsize=8)
for n in NS_ALL:
    pts = sorted((a["p_m"], a["P_d_ge_0p05n"]) for a in agg if a["family"] == "pt=1/n" and a["n"] == n)
    if pts:
        pts = np.array(pts)
        ax[1].plot(pts[:, 0], pts[:, 1], marker="o", ms=3, color=ncol(n), label=f"n={n}")
ax[1].axvline(0.16, ls=":", c="gray")
ax[1].set_xlabel("$p_m$"); ax[1].set_ylabel("$P(d \\geq 0.05 n)$"); ax[1].set_title("probability of an extensive register")
r = results.get("d_over_n_pt1n")
if r:
    pc, nu, bnu = r["params"]
    for n in [32, 64, 128, 256, 512]:
        s = series("pt=1/n", n)
        if len(s):
            m = (s[:, 0] >= 0.06) & (s[:, 0] <= 0.3)
            ax[2].errorbar((s[m, 0] - pc) * n ** (1 / nu), s[m, 1] * n ** bnu, s[m, 2] * n ** bnu, marker="o", ms=3, ls="", color=ncol(n), label=f"n={n}")
    e = r["err"]
    ax[2].set_title(f"collapse: $p_c$={pc:.4f}±{e[0]:.4f}, $\\nu$={nu:.2f}±{e[1]:.2f}, $\\beta/\\nu$={bnu:.3f}±{e[2]:.3f}", fontsize=9)
    ax[2].set_xlabel("$(p_m-p_c) n^{1/\\nu}$"); ax[2].set_ylabel("$(\\bar d/n)\\, n^{\\beta/\\nu}$")
    ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/collapse_dilute.png", dpi=130); plt.close(fig)

# 2. EE reference: I3 at p_T = 0
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for n in [16, 32, 64, 128, 256]:
    pts = sorted((a["p_m"], a["i3_lo"], a["i3_err"]) for a in agg if a["family"] == "pt=0" and a["n"] == n)
    if pts:
        pts = np.array(pts)
        ax[0].errorbar(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", ms=3, color=ncol(n), label=f"n={n}")
ax[0].set_xlabel("$p_m$"); ax[0].set_ylabel("$I_3$ (bits)"); ax[0].set_title("Clifford brickwork ($p_T=0$): tripartite MI")
ax[0].legend(fontsize=8)
r = results.get("EE_I3_pt0")
if r:
    pc, nu = r["params"]
    for n in [32, 64, 128, 256]:
        pts = sorted((a["p_m"], a["i3_lo"], a["i3_err"]) for a in agg if a["family"] == "pt=0" and a["n"] == n)
        if pts:
            pts = np.array(pts)
            ax[1].errorbar((pts[:, 0] - pc) * n ** (1 / nu), pts[:, 1], pts[:, 2], marker="o", ms=3, ls="", color=ncol(n), label=f"n={n}")
    e = r["err"]
    ax[1].set_title(f"collapse: $p_c^{{EE}}$={pc:.4f}±{e[0]:.4f}, $\\nu$={nu:.2f}±{e[1]:.2f}", fontsize=9)
    ax[1].set_xlabel("$(p_m-p_c) n^{1/\\nu}$"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/collapse_ee.png", dpi=130); plt.close(fig)

# 3. constant p_T: no crossing
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, pt in enumerate(["pt=0.01", "pt=0.05", "pt=0.2"]):
    for n in NS_ALL:
        s = series(pt, n)
        if len(s):
            ax[k].errorbar(s[:, 0], s[:, 1], s[:, 2], marker="o", ms=3, color=ncol(n), label=f"n={n}")
    ax[k].axvline(0.16, ls=":", c="gray")
    ax[k].set_title(f"constant {pt.replace('pt', '$p_T$')}"); ax[k].set_xlabel("$p_m$"); ax[k].set_ylabel("$\\bar d/n$")
    ax[k].set_yscale("log")
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/constant_pt.png", dpi=130); plt.close(fig)

# 4. phase diagram at the largest n
fams = ["pt=0", "pt=1/n", "pt=0.01", "pt=0.05", "pt=0.2"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
nbig = 512
for fam, c in zip(fams[1:], ["C0", "C1", "C2", "C3"]):
    s = series(fam, nbig)
    if len(s):
        ax[0].errorbar(s[:, 0], s[:, 1], s[:, 2], marker="o", ms=3, color=c, label=fam.replace("pt", "$p_T$"))
ax[0].axvline(0.16, ls=":", c="gray", label="$p_c^{EE}$≈0.16")
ax[0].set_xlabel("$p_m$"); ax[0].set_ylabel("$\\bar d/n$"); ax[0].set_title(f"steady-state active dimension, n={nbig}")
ax[0].legend(fontsize=8)
pms = sorted({a["p_m"] for a in agg if a["family"] == "pt=0.05" and a["n"] == nbig})
grid = np.full((4, len(pms)), np.nan)
for i, fam in enumerate(fams[1:]):
    for j, pm in enumerate(pms):
        v = [a["phi"] for a in agg if a["family"] == fam and a["n"] == nbig and abs(a["p_m"] - pm) < 1e-9]
        if v:
            grid[i, j] = v[0]
im = ax[1].imshow(grid, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
ax[1].set_xticks(range(len(pms))); ax[1].set_xticklabels([f"{p:g}" for p in pms], rotation=60, fontsize=7)
ax[1].set_yticks(range(4)); ax[1].set_yticklabels([f"1/n", "0.01", "0.05", "0.2"])
ax[1].set_xlabel("$p_m$"); ax[1].set_ylabel("$p_T$"); ax[1].set_title(f"$\\bar d/n$ (exact cost $2^d$), n={nbig}")
fig.colorbar(im, ax=ax[1])
fig.tight_layout(); fig.savefig(f"{out}/phase_diagram.png", dpi=130); plt.close(fig)

# 5. eta = 2: compare with Fux et al. (their Fig. 3 axes: steady state vs N, log-linear)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
pms3 = sorted({a["p_m"] for a in agg if a["family"] == "pt=2/n"})
cm2 = plt.get_cmap("plasma")
for i, pm in enumerate(pms3):
    pts = sorted((a["n"], a["phi"] * a["n"], a["phi_err"] * a["n"]) for a in agg if a["family"] == "pt=2/n" and abs(a["p_m"] - pm) < 1e-9)
    if pts:
        pts = np.array(pts)
        ax[0].errorbar(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", ms=3, color=cm2(i / max(1, len(pms3) - 1)), label=f"$p_m$={pm:g}")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xlabel("n"); ax[0].set_ylabel("steady-state $\\bar d$  ($\\geq \\nu \\geq M_2$)")
ax[0].axvline(184, ls=":", c="gray"); ax[0].text(190, ax[0].get_ylim()[0] * 1.5, "Fux et al. max N", fontsize=7)
ax[0].set_title("$p_T = 2/n$ (Fux et al. $\\eta=2$, $\\beta=1$)"); ax[0].legend(fontsize=7)
for n in [32, 64, 128, 256, 512, 1024]:
    s3 = series("pt=2/n", n)
    if len(s3):
        ax[1].errorbar(s3[:, 0], s3[:, 1], s3[:, 2], marker="o", ms=3, color=ncol(n), label=f"n={n}")
ax[1].axvline(0.16, ls=":", c="gray"); ax[1].axvline(0.22, ls="--", c="red", lw=0.8)
ax[1].text(0.222, 0.3, "Fux et al.\n$p_c^{magic}$≈0.22", color="red", fontsize=7)
ax[1].set_xlabel("$p_m$"); ax[1].set_ylabel("$\\bar d/n$"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{out}/eta2_vs_fux.png", dpi=130); plt.close(fig)

# 6. exact runs: nu and M2 against d
ex = [r for r in rows if r[ci["mode"]] == "exact" and r[ci["failed"]] == "0" and np.isfinite(f(r, "nu"))]
if ex:
    d = np.array([f(r, "d_final") for r in ex]); nu = np.array([f(r, "nu") for r in ex]); m2 = np.array([f(r, "m2") for r in ex])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    jit = (rng.random(len(d)) - 0.5) * 0.3
    ax[0].scatter(d + jit, nu + jit[::-1], s=6, alpha=0.4)
    ax[0].plot([0, 12], [0, 12], "k:", lw=0.8); ax[0].set_xlabel("d (register size)"); ax[0].set_ylabel("stabilizer nullity ν")
    ax[1].scatter(nu + jit, m2, s=6, alpha=0.4)
    ax[1].plot([0, 12], [0, 12 * math.log2(4 / 3)], "r--", lw=0.8, label="ν·log2(4/3) (product of T states)")
    ax[1].plot([0, 12], [0, 12], "k:", lw=0.8, label="M2 = ν")
    ax[1].set_xlabel("ν"); ax[1].set_ylabel("$M_2$"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/magic_exact.png", dpi=130); plt.close(fig)
    summ = dict(runs=len(ex), frac_nu_eq_d=float((nu == d).mean()), mean_d_minus_nu=float((d - nu).mean()),
                frac_m2_product=float((np.abs(m2 - nu * math.log2(4 / 3)) < 1e-6).mean()),
                max_n=int(max(int(r[ci["n"]]) for r in ex)))
    results["exact_magic_summary"] = summ
    print("exact magic", summ)
    json.dump(results, open(f"{out}/fss.json", "w"), indent=1)
print("done")
