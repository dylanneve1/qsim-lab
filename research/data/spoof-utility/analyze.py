#!/usr/bin/env python3
"""Analysis for research/simulability/spoof-utility.md.

Inputs (all in this directory):
  campaign*.jsonl              SPD runs (examples/spoof_utility.rs output)
  kim/*.txt                    Kim et al. exact / MPS curves (authors' data repo)
  kim_fig*_experiment.csv      experiment values (extract_kim.py)
  tindall/*.csv                BP-TNS reference data of Tindall et al. (github.com/JoeyT1994/BP-TNS-Data)
Outputs: tables.md, figures_kim.png, convergence.png, depth_scan.png, noise.png
"""
import glob, json, math, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# reference categorical palette (light), fixed order; grey for references
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
C_REF = "#52514e"

W17_SIGN = -1.0  # Kim et al. store fig3c/fig4a with the sign of our convention; Tindall's W17 has the opposite sign

FIGS = {
    "3a": "$M_z$, 5 steps",
    "3b": "weight-10 $X_{13,29,31}Y_{9,30}Z_{8,12,17,28,32}$, 5 steps",
    "3c": "weight-17 $X_{\\{8\\}}Y_{75}Z_{\\{8\\}}$, 5 steps",
    "4a": "weight-17 $X_{\\{8\\}}Y_{\\{8\\}}Z_{75}$, 5 steps + RX",
    "4b": "$\\langle Z_{62}\\rangle$, 20 steps",
}


def load_runs():
    runs = []
    for f in sorted(glob.glob("campaign*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                r.setdefault("branch_factor", 1)
                r.setdefault("stream", 1)
                r["src"] = f
                runs.append(r)
    return runs


def load_xy(path, col=1, scale=1.0):
    d = {}
    for line in open(path):
        if line[0].isalpha():
            continue
        p = line.strip().split(",")
        d[round(float(p[0]), 4)] = scale * float(p[col])
    return d


def select(runs, fig, lattice=127, steps=None, depol=0.0, max_weight=-1):
    """theta -> [(delta, value, seconds, run)] by decreasing delta (latest run per delta)."""
    by = defaultdict(dict)
    for r in runs:
        if r["fig"] != fig or r["lattice"] != lattice or r["aborted"]:
            continue
        if abs(r["depol"] - depol) > 1e-12 or r.get("max_weight", -1) != max_weight:
            continue
        if steps is not None and r["steps"] != steps:
            continue
        if r["branch_factor"] != 1:
            continue
        by[round(r["theta"], 4)][r["delta"]] = (r["delta"], r["value"], r["seconds"], r)
    return {k: sorted(v.values(), key=lambda t: -t[0]) for k, v in sorted(by.items())}


def deficit(r):
    """1 − ‖O‖²/‖O_0‖² (norm entering the last layer, relative to the observable's)."""
    n0 = 1.0 / r["lattice"] if r["fig"] in ("3a", "mz") else 1.0
    return 1.0 - r["norm2"] / n0


def references(fig):
    """theta -> reference value, and its label."""
    if fig in ("3a", "3b", "3c"):
        return load_xy(f"kim/fig{fig}_exact.txt"), "exact (Kim et al.)"
    if fig == "4a":
        # χ = 500 column: SPD at δ = 1e-5 agrees with it to ≤ 3e-4, while their
        # 1/χ → 0 extrapolation (column 1) deviates by up to 3e-3 (see the doc).
        return load_xy("tindall/w17_6layers_bptns.csv", 2, W17_SIGN), "BP-TNS χ=500 (Tindall et al.)"
    if fig == "4b":
        return load_xy("tindall/z62_20steps_bptns.csv", 1), "BP-TNS χ→∞ (Tindall et al.)"
    return {}, ""


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    runs = load_runs()
    out = []

    # ---------------- Kim et al. figures ----------------
    fig_all, axes = plt.subplots(1, 5, figsize=(23, 4.6))
    for ax, (fig, title) in zip(axes, FIGS.items()):
        steps = 20 if fig == "4b" else 5
        conv = select(runs, fig, steps=steps)
        ref, reflabel = references(fig)
        exp = np.genfromtxt(f"kim_fig{fig}_experiment.csv", delimiter=",", names=True)
        expmap = {round(float(r["theta_h"]), 4): r for r in exp}
        out.append(f"\n### Fig. {fig}: {title}\n")
        out.append(f"Reference: {reflabel}.\n")
        out.append("| θ_h | SPD (δ_min) | δ-ladder diff | δ_min | 1−‖O‖² | time/pt (s) | reference | SPD−ref | ZNE exp. | exp. 68% CI | exp−ref |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        xs, ys, es = [], [], []
        maxerr, maxexp = 0.0, 0.0
        for th, lst in conv.items():
            d, v, sec, r = lst[-1]
            e = abs(lst[-1][1] - lst[-2][1]) if len(lst) > 1 else float("nan")
            xs.append(th)
            ys.append(v)
            es.append(0 if math.isnan(e) else e)
            rv = ref.get(th)
            rcell = f"{rv:+.4f}" if rv is not None else "–"
            dcell = f"{v - rv:+.1e}" if rv is not None else "–"
            if rv is not None:
                maxerr = max(maxerr, abs(v - rv))
            er = expmap.get(th)
            if er is not None:
                ecell = f"{er['mitigated']:+.3f}"
                ci = f"[{er['boot_lo68']:+.3f}, {er['boot_hi68']:+.3f}]"
                base = rv if rv is not None else v
                ediff = f"{er['mitigated'] - base:+.3f}"
                maxexp = max(maxexp, abs(er["mitigated"] - base))
            else:
                ecell = ci = ediff = "–"
            out.append(
                f"| {th:.4f} | {v:+.5f} | {e:.1e} | {d:.0e} | {deficit(r):+.1e} | {sec:.2f} | {rcell} | {dcell} | {ecell} | {ci} | {ediff} |"
            )
        out.append(f"\nmax |SPD − reference| = **{maxerr:.1e}**; max |ZNE experiment − reference| = **{maxexp:.3f}**\n")
        if ref:
            rx = sorted(ref)
            ax.plot(rx, [ref[x] for x in rx], color=C_REF, lw=2.0, label=reflabel, zorder=1)
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5, color=C1, label="SPD, this work (M1)", zorder=3)
        lo = np.abs(exp["mitigated"] - exp["boot_lo68"])
        hi = np.abs(exp["boot_hi68"] - exp["mitigated"])
        ax.errorbar(exp["theta_h"], exp["mitigated"], yerr=[lo, hi], fmt="s", ms=5, color=C2, capsize=2, label="experiment, ZNE", zorder=2)
        ax.plot(exp["theta_h"], exp["unmitigated"], "^", ms=5, mfc="none", color=C3, label="experiment, unmitigated", zorder=2)
        ax.set_title(f"Kim et al. Fig. {fig}: " + title, fontsize=9)
        ax.set_xlabel("$\\theta_h$")
        ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig_all.tight_layout()
    fig_all.savefig("figures_kim.png", dpi=120)

    # ---------------- convergence ----------------
    fc, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for fig, col in zip(["3a", "3b", "3c", "4a"], [C1, C2, C3, C4]):
        conv = select(runs, fig, steps=5)
        ref, _ = references(fig)
        errs = defaultdict(float)
        for th, lst in conv.items():
            if th not in ref:
                continue
            for d, v, s, r in lst:
                errs[d] = max(errs[d], abs(v - ref[th]))
        ds = sorted(errs)
        if ds:
            ax[0].loglog(ds, [max(errs[d], 1e-12) for d in ds], "o-", color=col, label=f"Fig. {fig}")
    ax[0].set_xlabel("threshold δ")
    ax[0].set_ylabel("max over θ of |SPD − reference|")
    ax[0].set_title("5 steps: error vs threshold", fontsize=10)
    ax[0].invert_xaxis()
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3, which="both")
    conv = select(runs, "4b", steps=20)
    ref, _ = references("4b")
    for th, col in zip([0.3, 0.5, 0.6, 0.7, 0.8, 1.0], [C1, C2, C3, C4, C5, C6]):
        if th in conv:
            lst = conv[th]
            ax[1].semilogx([t[0] for t in lst], [t[1] for t in lst], "o-", color=col, label=f"θ={th}")
            if th in ref:
                ax[1].axhline(ref[th], color=col, ls=":", lw=1)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("threshold δ")
    ax[1].set_ylabel("$\\langle Z_{62}\\rangle$, 20 steps")
    ax[1].set_title("20 steps: SPD vs δ (dotted: BP-TNS)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3, which="both")
    # error vs norm deficit, 20 steps
    pts = []
    for th, lst in conv.items():
        if th not in ref:
            continue
        for d, v, s, r in lst:
            pts.append((abs(deficit(r)), abs(v - ref[th]), abs(lst[-1][1] - lst[-2][1]) if len(lst) > 1 else np.nan))
    if pts:
        p = np.array(pts)
        ax[2].loglog(np.maximum(p[:, 0], 1e-8), np.maximum(p[:, 1], 1e-8), "o", color=C1, ms=4, label="|SPD − BP-TNS|")
        g = np.linspace(-8, 0, 10)
        ax[2].loglog(10**g, 10**g, "-", color=C_REF, lw=1, label="error = 1 − ‖O‖²")
        ax[2].set_xlabel("norm deficit 1 − ‖O‖² (discarded weight)")
        ax[2].set_ylabel("true error vs BP-TNS")
        ax[2].set_title("20 steps: the norm deficit tracks the error", fontsize=10)
        ax[2].legend(frameon=False, fontsize=8)
        ax[2].grid(alpha=0.3, which="both")
    fc.tight_layout()
    fc.savefig("convergence.png", dpi=120)

    # ---------------- depth scan ----------------
    fd, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    out.append("\n### Depth scan: ⟨Z62⟩ vs Trotter steps (BP-TNS reference: Tindall et al. χ→∞)\n")
    out.append("| θ_h | steps | SPD δ=1e-4 | SPD δ=3e-5 | 1−‖O‖² (3e-5) | BP-TNS | err (3e-5) |")
    out.append("|---|---|---|---|---|---|---|")
    for a, th in zip(ax, [0.6, 0.8, 1.0]):
        dyn = load_xy(f"tindall/z62_dynamics_theta{th}_bptns.csv")
        sx = sorted(dyn)
        a.plot(sx, [dyn[s] for s in sx], "-", color=C_REF, lw=2, label="BP-TNS (Tindall et al.)")
        for d, col in [(1e-4, C2), (3e-5, C1)]:
            pts = []
            for r in runs:
                if r["fig"] == "4b" and abs(r["theta"] - th) < 1e-6 and r["delta"] == d and not r["aborted"] and r["depol"] == 0 and r.get("max_weight", -1) == -1:
                    pts.append((r["steps"], r["value"], r["norm2"]))
            pts = sorted(set(pts))
            if pts:
                a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, ms=4, label=f"SPD δ={d:g}")
        rows = defaultdict(dict)
        for r in runs:
            if r["fig"] == "4b" and abs(r["theta"] - th) < 1e-6 and not r["aborted"] and r["depol"] == 0 and r.get("max_weight", -1) == -1 and r["delta"] in (1e-4, 3e-5):
                rows[r["steps"]][r["delta"]] = r
        for s in sorted(rows):
            r4, r3 = rows[s].get(1e-4), rows[s].get(3e-5)
            ref = dyn.get(s)
            out.append(
                f"| {th} | {s} | {r4['value'] if r4 else float('nan'):+.4f} | {r3['value'] if r3 else float('nan'):+.4f} | {deficit(r3) if r3 else float('nan'):.2e} | {ref:+.4f} | {(r3['value'] - ref) if r3 else float('nan'):+.4f} |"
            )
        a.set_title(f"$\\langle Z_{{62}}\\rangle$ vs steps, θ={th}", fontsize=10)
        a.set_xlabel("Trotter steps")
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8)
    fd.tight_layout()
    fd.savefig("depth_scan.png", dpi=120)

    # ---------------- noise ----------------
    noisy = [r for r in runs if r["depol"] > 0 and not r["aborted"]]
    if noisy:
        fn, ax = plt.subplots(1, 4, figsize=(20, 4.2))
        out.append("\n### Noise-aware SPD vs the unmitigated experiment (p fitted at θ=0 only)\n")
        for a, (fig, p0) in zip(ax, [("3a", 0.0266), ("3b", 0.0266), ("3c", 0.0266), ("4b", 0.0209)]):
            exp = np.genfromtxt(f"kim_fig{fig}_experiment.csv", delimiter=",", names=True)
            ideal = select(runs, fig, steps=20 if fig == "4b" else 5)
            ref, _ = references(fig)
            curves = {}
            for G in (1.0, 1.2, 1.6):
                c = select(runs, fig, steps=20 if fig == "4b" else 5, depol=round(p0 * G, 6))
                if not c:
                    c = {}
                    for r in noisy:
                        if r["fig"] == fig and abs(r["depol"] - p0 * G) < 1e-6:
                            c[round(r["theta"], 4)] = [(r["delta"], r["value"], r["seconds"], r)]
                curves[G] = {th: lst[-1][1] for th, lst in c.items()}
            if not curves[1.0]:
                continue
            th = sorted(curves[1.0])
            a.plot(th, [curves[1.0][t] for t in th], "o-", color=C3, ms=4, label=f"SPD + depolarizing p={p0}")
            a.plot(exp["theta_h"], exp["unmitigated"], "^", mfc="none", color=C3, ms=6, label="experiment, unmitigated")
            # our own ZNE: exponential fit through G = 1, 1.2, 1.6
            zne = {}
            for t in th:
                if all(t in curves[G] for G in (1.0, 1.2, 1.6)):
                    ys = np.array([curves[G][t] for G in (1.0, 1.2, 1.6)])
                    gs = np.array([1.0, 1.2, 1.6])
                    if np.all(ys > 1e-9) or np.all(ys < -1e-9):
                        sgn = np.sign(ys[0])
                        b, la = np.polyfit(gs, np.log(np.abs(ys)), 1)
                        zne[t] = sgn * math.exp(la)
                    else:
                        zne[t] = np.polyval(np.polyfit(gs, ys, 1), 0.0)
            if zne:
                a.plot(sorted(zne), [zne[t] for t in sorted(zne)], "s--", color=C2, ms=4, label="exp. ZNE of the noisy SPD model")
            if ref:
                rx = sorted(ref)
                a.plot(rx, [ref[x] for x in rx], "-", color=C_REF, lw=1.5, label="noise-free reference")
            a.set_title(f"Fig. {fig}: noise-aware SPD", fontsize=10)
            a.set_xlabel("$\\theta_h$")
            a.grid(alpha=0.3)
            a.legend(frameon=False, fontsize=7)
            out.append(f"\nFig. {fig} (p = {p0}): θ, noisy SPD, unmitigated exp., model-ZNE, noise-free ref")
            em = {round(float(r['theta_h']), 4): r for r in exp}
            out.append("| θ_h | noisy SPD (G=1) | unmitigated exp. | diff | model ZNE | noise-free |")
            out.append("|---|---|---|---|---|---|")
            for t in th:
                u = em.get(t)
                rv = ref.get(t)
                out.append(
                    f"| {t:.4f} | {curves[1.0][t]:+.4f} | {u['unmitigated'] if u is not None else float('nan'):+.4f} | {(curves[1.0][t] - u['unmitigated']) if u is not None else float('nan'):+.4f} | {zne.get(t, float('nan')):+.4f} | {rv if rv is not None else float('nan'):+.4f} |"
                )
        fn.tight_layout()
        fn.savefig("noise.png", dpi=120)

    # ---------------- larger lattices ----------------
    out.append("\n### Larger heavy-hex lattices\n")
    out.append("| observable | lattice | steps | θ_h | δ | value | 1−‖O‖² | light cone | words | peak terms | time (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["fig"], r["lattice"], r["steps"], r["theta"], -r["delta"])):
        if (r["fig"] in ("mz", "z215", "z559") or (r["fig"] == "z62")) and r["depol"] == 0 and r.get("max_weight", -1) == -1:
            out.append(
                f"| {r['fig']} | {r['lattice']} | {r['steps']} | {r['theta']:.4f} | {r['delta']:.0e} | {r['value']:+.5f} | {deficit(r):+.1e} | {r['cone']} | {r['words']} | {r['peak_terms']} | {r['seconds']:.2f} |"
            )

    # ---------------- M_z finite size ----------------
    out.append("\n### M_z after 5 steps vs lattice size (smallest δ per point; δ is absolute, per-site coefficients are 1/n)\n")
    out.append("| θ_h | 127 (exact, Kim et al.) | 127 SPD | 433 SPD | 1121 SPD | 1121 − 127 |")
    out.append("|---|---|---|---|---|---|")
    mzs = {L: select(runs, "3a" if L == 127 else "mz", lattice=L, steps=5) for L in (127, 433, 1121)}
    ex = load_xy("kim/fig3a_exact.txt")
    for th in sorted(mzs[1121]):
        v = {L: (mzs[L][th][-1][1] if th in mzs[L] else float("nan")) for L in mzs}
        dd = {L: (mzs[L][th][-1][0] if th in mzs[L] else float("nan")) for L in mzs}
        out.append(
            f"| {th:.4f} | {ex.get(th, float('nan')):+.5f} | {v[127]:+.5f} (δ {dd[127]:.0e}) | {v[433]:+.5f} (δ {dd[433]:.0e}) | {v[1121]:+.5f} (δ {dd[1121]:.0e}) | {v[1121] - v[127]:+.4f} |"
        )

    # ---------------- 1121-qubit reliability map ----------------
    mp = {}
    for r in runs:
        if r["fig"] == "z559" and r["lattice"] == 1121 and r["depol"] == 0 and r.get("max_weight", -1) == -1:
            key = (r["steps"], round(r["theta"], 4))
            if r["aborted"]:
                mp.setdefault(key, None)
                continue
            old = mp.get(key)
            if old is None or r["delta"] < old["delta"]:
                mp[key] = r
    if mp:
        steps_l = sorted({k[0] for k in mp})
        th_l = sorted({k[1] for k in mp})
        M = np.full((len(steps_l), len(th_l)), np.nan)
        out.append("\n### Bulk ⟨Z_559⟩ on the 1121-qubit lattice: value (norm deficit, δ) per depth and θ\n")
        out.append("| steps \\ θ_h | " + " | ".join(f"{t}" for t in th_l) + " |")
        out.append("|---" * (len(th_l) + 1) + "|")
        for i, st in enumerate(steps_l):
            cells = []
            for j, t in enumerate(th_l):
                r = mp.get((st, t))
                if r is None:
                    cells.append("–")
                    continue
                d = abs(deficit(r))
                M[i, j] = d
                cells.append(f"{r['value']:+.3f} ({d:.0e}, {r['delta']:.0e})")
            out.append(f"| {st} | " + " | ".join(cells) + " |")
        fm, ax = plt.subplots(figsize=(9, 3.8))
        im = ax.imshow(np.log10(np.maximum(M, 1e-8)), cmap="Blues", vmin=-8, vmax=0, aspect="auto", origin="lower")
        ax.set_xticks(range(len(th_l)))
        ax.set_xticklabels([str(t) for t in th_l])
        ax.set_yticks(range(len(steps_l)))
        ax.set_yticklabels([str(s) for s in steps_l])
        ax.set_xlabel("$\\theta_h$")
        ax.set_ylabel("Trotter steps")
        for i in range(len(steps_l)):
            for j in range(len(th_l)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0e}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 1e-3 else "#0b0b0b")
        cb = fm.colorbar(im, ax=ax)
        cb.set_label("log10 norm deficit $|1-\\|O\\|^2|$")
        ax.set_title("1121-qubit heavy hex, bulk $\\langle Z_{559}\\rangle$: discarded weight at the smallest δ that fits in 3 GB", fontsize=9)
        fm.tight_layout()
        fm.savefig("map_1121.png", dpi=120)

    # ---------------- weight cap ----------------
    wc = [r for r in runs if r.get("max_weight", -1) != -1]
    if wc:
        ref, _ = references("4b")
        ws = sorted({r["max_weight"] for r in wc})
        ths = sorted({round(r["theta"], 4) for r in wc})
        out.append("\n### 20-step ⟨Z62⟩ (127 qubits) with a Pauli-weight cap, δ = 1e-5: value (error vs BP-TNS)\n")
        out.append("| θ_h | BP-TNS | uncapped (best δ) | " + " | ".join(f"w ≤ {w}" for w in ws) + " | ZNE exp. |")
        out.append("|---" * (len(ws) + 4) + "|")
        exp4 = {round(float(r["theta_h"]), 4): r for r in np.genfromtxt("kim_fig4b_experiment.csv", delimiter=",", names=True)}
        unc = select(runs, "4b", steps=20)
        for t in ths:
            rv = ref.get(t, float("nan"))
            cells = []
            for w in ws:
                c = [r for r in wc if round(r["theta"], 4) == t and r["max_weight"] == w and r["delta"] == 1e-5 and r["steps"] == 20 and not r["aborted"]]
                cells.append(f"{c[-1]['value']:+.3f} ({c[-1]['value'] - rv:+.3f})" if c else "–")
            u = unc.get(t)
            ucell = f"{u[-1][1]:+.3f} ({u[-1][1] - rv:+.3f})" if u else "–"
            e = exp4.get(t)
            out.append(f"| {t} | {rv:+.3f} | {ucell} | " + " | ".join(cells) + f" | {e['mitigated'] if e is not None else float('nan'):+.3f} |")
        fw, axw = plt.subplots(figsize=(7.5, 4.4))
        rx = sorted(ref)
        axw.plot(rx, [ref[x] for x in rx], "-", color=C_REF, lw=2, label="BP-TNS χ→∞ (Tindall et al.)")
        ux = sorted(unc)
        axw.plot(ux, [unc[x][-1][1] for x in ux], "o-", color=C1, ms=4, label="SPD, δ only (best δ in 3 GB)")
        for w, col in zip([8, 10], [C3, C2]):
            pts = sorted((round(r["theta"], 4), r["value"]) for r in wc if r["max_weight"] == w and r["delta"] == 1e-5 and r["steps"] == 20 and not r["aborted"])
            if pts:
                axw.plot([p[0] for p in pts], [p[1] for p in pts], "s--", color=col, ms=4, label=f"SPD, δ = 1e-5 and weight ≤ {w}")
        exp = np.genfromtxt("kim_fig4b_experiment.csv", delimiter=",", names=True)
        axw.errorbar(exp["theta_h"], exp["mitigated"], yerr=[np.abs(exp["mitigated"] - exp["boot_lo68"]), np.abs(exp["boot_hi68"] - exp["mitigated"])], fmt="^", color=C4, ms=5, capsize=2, label="experiment, ZNE")
        axw.set_xlabel("$\\theta_h$")
        axw.set_ylabel("$\\langle Z_{62}\\rangle$, 20 steps")
        axw.grid(alpha=0.3)
        axw.legend(frameon=False, fontsize=8)
        fw.tight_layout()
        fw.savefig("weight_cap.png", dpi=120)
        ref, _ = references("4b")
        out.append("\n### 20-step ⟨Z62⟩ with an added Pauli-weight cap\n")
        out.append("| θ_h | max weight | δ | value | BP-TNS | err | 1−‖O‖² | peak terms | time (s) | aborted |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(wc, key=lambda r: (r["theta"], r["max_weight"], -r["delta"])):
            rv = ref.get(round(r["theta"], 4), float("nan"))
            out.append(
                f"| {r['theta']:.2f} | {r['max_weight']} | {r['delta']:.0e} | {r['value']:+.4f} | {rv:+.4f} | {r['value'] - rv:+.4f} | {deficit(r):+.2e} | {r['peak_terms']} | {r['seconds']:.1f} | {r['aborted']} |"
            )

    # ---------------- exact patches at depth ----------------
    prs = []
    for f in sorted(glob.glob("patch*.jsonl")):
        for line in open(f):
            if line.startswith("{"):
                prs.append(json.loads(line))
    if prs:
        out.append("\n### Heavy-hex patches (BFS ball around Eagle qubit 62), exact state vector vs SPD\n")
        out.append("| qubits | θ_h | steps | exact ⟨Z⟩ | δ | SPD | error | 1−‖O‖² | peak terms | time (s) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(prs, key=lambda r: (r["k"], r["theta"], r["steps"], -r["delta"])):
            if r["aborted"] or r.get("branch_factor", 1) != 1 or r.get("max_weight", -1) != -1:
                continue
            out.append(
                f"| {r['k']} | {r['theta']} | {r['steps']} | {r['exact']:+.4f} | {r['delta']:.0e} | {r['spd']:+.4f} | {r['err']:+.4f} | {1 - r['norm2']:.3f} | {r['peak_terms']} | {r['seconds']:.1f} |"
            )

    wp = [r for r in prs if r.get("max_weight", -1) != -1 and not r["aborted"]]
    if wp:
        out.append("\n### Weight-capped SPD on the exact 24-qubit patch, 20 steps\n")
        ths = sorted({r["theta"] for r in wp})
        ws = sorted({r["max_weight"] for r in wp})
        unc = {}
        for r in prs:
            if r.get("max_weight", -1) == -1 and r["k"] == 24 and r["steps"] == 20 and not r["aborted"] and r.get("branch_factor", 1) == 1:
                if r["theta"] not in unc or r["delta"] < unc[r["theta"]]["delta"]:
                    unc[r["theta"]] = r
        out.append("| θ_h | exact | " + " | ".join(f"w={w} err" for w in ws) + " | uncapped err (δ) |")
        out.append("|---" * (len(ws) + 3) + "|")
        for t in ths:
            cells = []
            ex = None
            for w in ws:
                c = [r for r in wp if r["theta"] == t and r["max_weight"] == w and r["delta"] == 1e-5 and r["steps"] == 20]
                if c:
                    cells.append(f"{c[-1]['err']:+.4f}")
                    ex = c[-1]["exact"]
                else:
                    cells.append("–")
            u = unc.get(t)
            out.append(f"| {t} | {ex if ex is not None else float('nan'):+.4f} | " + " | ".join(cells) + f" | {u['err'] if u else float('nan'):+.4f} ({u['delta'] if u else float('nan'):.0e}) |")

    # ---------------- locked timings ----------------
    bt = defaultdict(list)
    loads = []
    for f in sorted(glob.glob("bench*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "fig" in r:
                bt[(r["fig"], r["theta"], r["delta"], r["threads"])].append(r)
            else:
                loads.append(r)
    if bt:
        out.append("\n### Locked timings (min of 3, interleaved)\n")
        out.append(f"load: {loads}\n")
        out.append("| figure | θ_h | δ | threads | min time (s) | all (s) | peak stored terms | value |")
        out.append("|---|---|---|---|---|---|---|---|")
        for k in sorted(bt):
            v = bt[k]
            out.append(
                f"| {k[0]} | {k[1]} | {k[2]:.0e} | {k[3]} | {min(x['seconds'] for x in v):.3f} | {', '.join('%.3f' % x['seconds'] for x in v)} | {v[0]['peak_terms']} | {v[0]['value']:+.6f} |"
            )

    open("tables.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
