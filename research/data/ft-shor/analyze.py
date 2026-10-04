#!/usr/bin/env python3
"""Analyse ft_shor campaign output (res*.txt): tables, crossovers, plots.

Estimators. For every run, the shots split into 'clean' runs (no logical-level
fault: encoded = no decoded logical error anywhere; unencoded = no physical
fault at all) and 'faulty' runs. Clean runs follow the ideal distribution
exactly (location structure is outcome-independent; tested in
tests/ft_shor.rs::clean_runs_follow_ideal_distribution), so
    P(y) = (1-q) ideal(y) + q P(y | faulty),    q = P(faulty),
and every output metric is q x (its value on the faulty runs). Errors:
parametric bootstrap over (q, faulty histogram); TVD is bias-corrected.
"""
import glob, math, random, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXREC = []
IDEAL = np.array([.25, 0, .25, 0, .25, 0, .25, 0])
PEAK = IDEAL > 0
ORDER = np.array([0, 0, 1, 0, 0, 0, 1, 0], bool)  # y/8 = 1/4, 3/4


def parse(files):
    rows = []
    for f in files:
        for line in open(f):
            if line.startswith("level=") and "gadget=" in line:
                line = "kind=exrec " + line
            if not line.startswith("kind="):
                continue
            d = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d[k] = v
            rows.append(d)
    return rows


def wilson(k, n, z=1.0):
    if n == 0:
        return (0, 0, 1)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def peakmask(r):
    return np.array([any(abs(y / 8 - s / r) < 1 / (2 * r * r) for s in range(r)) for y in range(8)])


def plain_metrics(N, hist, ideal, pk, nf):
    f = hist / N
    return dict(q=nf / N, tvd=0.5 * np.abs(f - ideal).sum(), fpeak=f[~pk].sum(), forder=0.0)


def metrics(N, hist_f, ideal=IDEAL, pk=PEAK):
    nf = hist_f.sum()
    q = nf / N
    if nf == 0:
        return dict(q=0.0, tvd=0.0, fpeak=0.0, forder=0.0)
    f = hist_f / nf
    return dict(q=q, tvd=q * 0.5 * np.abs(f - ideal).sum(),
                fpeak=q * f[~pk].sum(),
                forder=q * (0.5 - f[ORDER].sum()))


def boot(N, hist_f, B=400, rng=np.random.default_rng(1)):
    nf = hist_f.sum()
    q = nf / N
    out = []
    for _ in range(B):
        nfb = rng.binomial(N, q) if q > 0 else 0
        if nfb == 0:
            out.append(metrics(N, np.zeros(8)))
            continue
        hb = rng.multinomial(nfb, hist_f / nf)
        out.append(metrics(N, hb))
    return out


def summarize(r):
    N = int(r["shots"])
    hist = np.array([int(x) for x in r["hist"].split(",")])
    hf = np.array([int(x) for x in r["hist_faulty"].split(",")])
    n21 = r.get("inst") == "21c"
    ideal = np.array([float(x) for x in r["ideal"].split(",")]) if "ideal" in r else IDEAL
    pk = peakmask(3) if n21 else PEAK
    rng = np.random.default_rng(1)
    res = dict(N=N, nf=int(hf.sum()), estimator="plain" if n21 else "clean-run")
    if n21:
        # outcome-dependent T-dagger corrections occur in error-free runs (r = 3):
        # the clean-run estimator does not apply; plain multinomial estimates.
        m = plain_metrics(N, hist, ideal, pk, hf.sum())
        bs = []
        for _ in range(400):
            hb = rng.multinomial(N, hist / N)
            bs.append(plain_metrics(N, hb, ideal, pk, rng.binomial(N, hf.sum() / N)))
    else:
        m = metrics(N, hf, ideal, pk)
        bs = boot(N, hf)
    for k in m:
        vals = np.array([b[k] for b in bs])
        est = m[k]
        if k == "tvd":
            est = max(2 * m[k] - vals.mean(), 0.0)  # bias correction
        res[k] = est
        res[k + "_lo"], res[k + "_hi"] = np.percentile(vals - vals.mean() + est, [16, 84])
    if res["nf"] == 0:
        ub = 1.14 / N
        for k in ("q",) + (() if n21 else ("tvd", "fpeak", "forder")):
            res[k + "_hi"] = ub
    res["P_peak_plain"] = hist[pk].sum() / N
    res["tvd_plain"] = 0.5 * np.abs(hist / N - ideal).sum()
    return res


def key(r):
    pre = "N21:" if r.get("inst") == "21c" else ""
    return pre + key0(r)


def key0(r):
    if r["mode"] == "enc":
        return f"L{r['level']}-{r['magic']}" + ("" if r.get("mask", "31") == "31" else f"-m{r['mask']}")
    return r["mode"]


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "res*.txt")))
    rows = parse(files)
    shor = [r for r in rows if r["kind"] == "shor"]
    inj = [r for r in rows if r["kind"] == "inject"]
    global EXREC
    EXREC = [r for r in rows if r["kind"] == "exrec"]
    series = {}
    for r in shor:
        k = key(r)
        s = summarize(r)
        s.update(p=float(r["p"]), locs=float(r["locs_per_shot"]), qubits=int(r["phys_qubits"]),
                 g2=float(r["g2_per_shot"]), g1=float(r["g1_per_shot"]), prep=float(r["prep_per_shot"]),
                 meas=float(r["meas_per_shot"]), raw=r)
        # merge repeated (k, p) runs by pooling histograms
        series.setdefault(k, {}).setdefault(s["p"], []).append((r, s))
    pooled = {}
    for k, d in series.items():
        for p, lst in d.items():
            if len(lst) == 1:
                pooled.setdefault(k, {})[p] = lst[0][1]
                continue
            N = sum(int(r["shots"]) for r, _ in lst)
            hf = sum(np.array([int(x) for x in r["hist_faulty"].split(",")]) for r, _ in lst)
            hist = sum(np.array([int(x) for x in r["hist"].split(",")]) for r, _ in lst)
            r0 = dict(lst[0][0]); r0["shots"] = str(N)
            r0["hist_faulty"] = ",".join(map(str, hf)); r0["hist"] = ",".join(map(str, hist))
            s = summarize(r0); s.update({k2: lst[0][1][k2] for k2 in ("p", "locs", "qubits", "g2", "g1", "prep", "meas")})
            s["raw"] = r0
            pooled.setdefault(k, {})[p] = s
    return pooled, inj


def fmt(x):
    if x == 0:
        return "0"
    return f"{x:.2e}"


def table(pooled, inj, out):
    lines = []
    order = [pre + o for pre in ("", "N21:") for o in ("unenc", "unenc-ccx", "L1-raw", "L1-ideal", "L2-raw", "L2-ideal")]
    lines.append("| series | p | runs | faulty runs | q = P(logical fault) | TVD to ideal | 1 − P_peak | 0.5 − P_order |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in order:
        if k not in pooled:
            continue
        for p in sorted(pooled[k]):
            s = pooled[k][p]
            lines.append(f"| {k} | {p:.0e} | {s['N']} | {s['nf']} | {fmt(s['q'])} [{fmt(s['q_lo'])}, {fmt(s['q_hi'])}] | "
                         f"{fmt(s['tvd'])} [{fmt(s['tvd_lo'])}, {fmt(s['tvd_hi'])}] | {fmt(s['fpeak'])} | {fmt(s['forder'])} |")
    open(os.path.join(out, "table_main.md"), "w").write("\n".join(lines) + "\n")
    # components
    lines = ["| p | component | q (only this component noisy) | TVD |", "|---|---|---|---|"]
    names = {1: "prep", 2: "gate", 4: "ec", 8: "inject", 16: "meas"}
    for k in sorted(pooled):
        if "-m" not in k:
            continue
        m = int(k.split("-m")[1])
        for p in sorted(pooled[k]):
            s = pooled[k][p]
            lines.append(f"| {p:.0e} | {names.get(m, m)} | {fmt(s['q'])} [{fmt(s['q_lo'])}, {fmt(s['q_hi'])}] | {fmt(s['tvd'])} |")
    open(os.path.join(out, "table_components.md"), "w").write("\n".join(lines) + "\n")
    lines = ["| level | p | post-select | pX | pY | pZ | ε (twirled) | ε/p | accept | 35ε³ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(inj, key=lambda r: (int(r["level"]), r["postselect"], float(r["p"]))):
        p = float(r["p"]); e = float(r["eps"])
        lines.append(f"| {r['level']} | {p:.0e} | {r['postselect']} | {fmt(float(r['pX']))} | {fmt(float(r['pY']))} | {fmt(float(r['pZ']))} | {fmt(e)} | {e/p:.2f} | {float(r['accept']):.4f} | {fmt(35*e**3)} |")
    open(os.path.join(out, "table_inject.md"), "w").write("\n".join(lines) + "\n")
    # overhead
    lines = ["| series | physical qubits | locations / run | prep | 1q gates | CNOTs | measurements |", "|---|---|---|---|---|---|---|"]
    for k in order:
        if k not in pooled:
            continue
        p = min(pooled[k])
        s = pooled[k][p]
        lines.append(f"| {k} | {s['qubits']} | {s['locs']:.3g} | {s['prep']:.3g} | {s['g1']:.3g} | {s['g2']:.3g} | {s['meas']:.3g} |")
    open(os.path.join(out, "table_overhead.md"), "w").write("\n".join(lines) + "\n")
    lines = ["| level | p | gadget | trials | failures | rate (1-exRec) | rate / p |", "|---|---|---|---|---|---|---|"]
    for r in sorted(EXREC, key=lambda r: (int(r["level"]), float(r["p"]))):
        n = int(r["trials"]); f = int(r["fail"]); p = float(r["p"])
        ph, lo, hi = wilson(f, n)
        lines.append(f"| {r['level']} | {p:.0e} | {r['gadget']} | {n} | {f} | {fmt(ph)} [{fmt(lo)}, {fmt(hi)}] | {ph/p:.3f} |")
    open(os.path.join(out, "table_exrec.md"), "w").write("\n".join(lines) + "\n")


def crossover(pooled, a, b, metric):
    """p where series a and b cross (log-log interpolation), else None."""
    if a not in pooled or b not in pooled:
        return None
    ps = sorted(set(pooled[a]) & set(pooled[b]))
    prev = None
    res = []
    for p in ps:
        va, vb = pooled[a][p][metric], pooled[b][p][metric]
        if va <= 0 or vb <= 0:
            prev = None
            continue
        d = math.log(va) - math.log(vb)
        if prev is not None and (d > 0) != (prev[1] > 0):
            p0, d0 = prev
            t = d0 / (d0 - d)
            res.append(math.exp(math.log(p0) + t * (math.log(p) - math.log(p0))))
        prev = (p, d)
    return res


def plots(pooled, out):
    plots1(pooled, out, "", "Shor N = 15 (a = 7, 3 semiclassical rounds)", "")
    plots1(pooled, out, "N21:", "Shor N = 21 (a = 4, compiled, 3 rounds)", "n21_")
    plot_exrec(out)


def plot_exrec(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for lev, c in (("1", "C0"), ("2", "C3")):
        rs = sorted([r for r in EXREC if r["level"] == lev and r["gadget"] == "cnot"], key=lambda r: float(r["p"]))
        if not rs:
            continue
        ps = np.array([float(r["p"]) for r in rs])
        w = [wilson(int(r["fail"]), int(r["trials"])) for r in rs]
        v = np.array([x[0] for x in w]); lo = np.array([x[1] for x in w]); hi = np.array([x[2] for x in w])
        pos = v > 0
        ax.errorbar(ps[pos], v[pos], yerr=[(v - lo)[pos], (hi - v)[pos]], color=c, marker="o", capsize=2, label=f"level {lev} CNOT 1-exRec")
        if (~pos).any():
            ax.plot(ps[~pos], hi[~pos], color=c, marker="v", ls="none", mfc="none")
    xs = np.logspace(-5, -2.3, 10)
    ax.plot(xs, xs, "k:", label="p (break-even)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical failure per CNOT exRec")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "exrec_cnot.png"), dpi=130); plt.close(fig)


def plots1(pooled, out, pre, title, fprefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"unenc": ("k", "o", "unencoded (Clifford+T)"), "unenc-ccx": ("0.5", "s", "unencoded (native CCX)"),
             "L1-raw": ("C0", "^", "Steane L1, raw injection"), "L1-ideal": ("C0", "v", "Steane L1, ideal magic"),
             "L2-raw": ("C3", "^", "Steane L2 [[49,1,9]], raw injection"), "L2-ideal": ("C3", "v", "Steane L2, ideal magic")}
    for metric, ylabel, fn in [("tvd", "TVD of output distribution from ideal", "tvd_vs_p.png"),
                               ("q", "P(run has a logical-level fault)", "fault_vs_p.png"),
                               ("fpeak", "1 − P_peak (y off the peaks)", "peak_vs_p.png")]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for k0, (c, mk, lab) in style.items():
            k = pre + k0
            if k not in pooled:
                continue
            ps = sorted(pooled[k])
            v = np.array([pooled[k][p][metric] for p in ps])
            lo = np.array([pooled[k][p][metric + "_lo"] for p in ps])
            hi = np.array([pooled[k][p][metric + "_hi"] for p in ps])
            ls = "-" if ("raw" in k0 or k0.startswith("unenc")) else "--"
            pos = v > 0
            ax.errorbar(np.array(ps)[pos], v[pos], yerr=[np.maximum(v - lo, 0)[pos], np.maximum(hi - v, 0)[pos]],
                        color=c, marker=mk, ls=ls, label=lab, capsize=2, ms=4)
            z = ~pos
            if z.any():
                ax.plot(np.array(ps)[z], hi[z], color=c, marker="v", ls="none", mfc="none")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("physical error rate p (per location)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fprefix + fn), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    pooled, inj = main()
    table(pooled, inj, HERE)
    plots(pooled, HERE)
    lines = []
    for pre in ("", "N21:"):
        for m in ("tvd", "q", "fpeak"):
            for a in ("L1-raw", "L1-ideal", "L2-raw", "L2-ideal"):
                for b in ("unenc", "unenc-ccx"):
                    lines.append(f"crossover {pre}{m} {a} vs {b}: {crossover(pooled, pre+a, pre+b, m)}")
            lines.append(f"crossover {pre}{m} L2-ideal vs L1-ideal: {crossover(pooled, pre+'L2-ideal', pre+'L1-ideal', m)}")
            lines.append(f"crossover {pre}{m} L2-raw vs L1-raw: {crossover(pooled, pre+'L2-raw', pre+'L1-raw', m)}")
    open(os.path.join(HERE, "crossovers.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def harm_table(pooled, out, qmax=0.05):
    """Harm per faulty run h = TVD(P(y | faulty), ideal), pooled over the
    single-logical-fault regime (q <= qmax, where P(y | faulty) does not
    depend on p), with a bootstrap bias correction. Low-p output error is then
    TVD(p) = q(p) h."""
    rng = np.random.default_rng(7)
    lines = ["| series | p range pooled | faulty runs | h = TVD(faulty runs, ideal) | h_peak = P(off-peak \\| faulty) |", "|---|---|---|---|---|"]
    H = {}
    for k in sorted(pooled):
        if "-m" in k:
            continue
        ps = [p for p in sorted(pooled[k]) if pooled[k][p]["q"] <= qmax and pooled[k][p]["nf"] > 0]
        if not ps:
            continue
        hf = sum(np.array([int(x) for x in pooled[k][p]["raw"]["hist_faulty"].split(",")]) for p in ps)
        r0 = pooled[k][ps[0]]["raw"]
        ideal = np.array([float(x) for x in r0["ideal"].split(",")]) if "ideal" in r0 else IDEAL
        pk = peakmask(3) if k.startswith("N21") else PEAK
        n = hf.sum()
        f = hf / n
        h = 0.5 * np.abs(f - ideal).sum()
        bs = np.array([0.5 * np.abs(rng.multinomial(n, f) / n - ideal).sum() for _ in range(1000)])
        hc = max(2 * h - bs.mean(), 0)
        lo, hi = np.percentile(bs - bs.mean() + hc, [16, 84])
        hp = f[~pk].sum()
        H[k] = (hc, max(lo, 0), hi)
        lines.append(f"| {k} | {ps[0]:.0e}–{ps[-1]:.0e} | {n} | {hc:.3f} [{max(lo,0):.3f}, {hi:.3f}] | {hp:.3f} |")
    open(os.path.join(out, "table_harm.md"), "w").write("\n".join(lines) + "\n")
    # q*h crossovers (low-p output error)
    res = []
    for pre in ("", "N21:"):
        for a in ("L1-raw", "L1-ideal", "L2-raw", "L2-ideal"):
            for b in ("unenc", "unenc-ccx"):
                A, B = pre + a, pre + b
                if A not in H or B not in H:
                    continue
                # fit q ~ c p^k on each, find where qA hA = qB hB
                def fit(s):
                    ps = np.array([p for p in sorted(pooled[s]) if 0 < pooled[s][p]["q"] <= 0.3])
                    qs = np.array([pooled[s][p]["q"] for p in ps])
                    return ps, qs
                pa, qa = fit(A); pb, qb = fit(B)
                common = sorted(set(pa) & set(pb))
                prev = None
                xs = []
                for p in common:
                    d = math.log(pooled[A][p]["q"] * H[A][0] + 1e-300) - math.log(pooled[B][p]["q"] * H[B][0] + 1e-300)
                    if prev is not None and (d > 0) != (prev[1] > 0):
                        t = prev[1] / (prev[1] - d)
                        xs.append(math.exp(math.log(prev[0]) + t * (math.log(p) - math.log(prev[0]))))
                    prev = (p, d)
                res.append(f"q*h crossover {A} vs {B}: {xs}")
    open(os.path.join(out, "crossovers_qh.txt"), "w").write("\n".join(res) + "\n")
    return H


if __name__ == "__main__":
    pooled, inj = main()
    H = harm_table(pooled, HERE)
    print(open(os.path.join(HERE, "table_harm.md")).read())
    print(open(os.path.join(HERE, "crossovers_qh.txt")).read())
