#!/usr/bin/env python3
"""Render a circuit exported by `qsim export` (ONNX) as a poster-style PNG.

The picture is drawn from the .onnx file itself, so it shows exactly the graph
a viewer such as Netron loads: one card per node, one coloured wire per qubit
tensor, dashed edges for classical tensors (measurement records feeding
classically controlled gates, detectors and observables).

Layout: a node's depth is its longest path from the graph inputs (ASAP
layering); its lane is the mean of its qubits (from the `qubits` attribute).
Small graphs flow top to bottom like Netron; large graphs flow left to right
and wrap into rows (`--wrap`).

    python3 tools/onnx_gallery.py qft4.onnx -o qft4.png
    python3 tools/onnx_gallery.py shor15_ripple.onnx -o shor.png --wrap 6

Needs `onnx`, `numpy` and `matplotlib`.
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"

# node category -> (header colour, label)
CATS = {
    "clifford1": ("#2f81f7", "1-qubit Clifford"),
    "rotation": ("#a371f7", "rotation / non-Clifford"),
    "entangle": ("#3fb950", "2-qubit gate"),
    "toffoli": ("#f0883e", "Toffoli"),
    "measure": ("#f85149", "measurement"),
    "reset": ("#6e7681", "reset"),
    "classical": ("#d29922", "classically controlled"),
    "noise": ("#db61a2", "noise channel"),
    "detector": ("#56d4dd", "detector / observable"),
}


def category(op):
    if op.startswith("IF_"):
        return "classical"
    if op in ("H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "I"):
        return "clifford1"
    if op in ("T", "T_DAG", "RX", "RY", "RZ", "P", "U"):
        return "rotation"
    if op in ("CX", "CZ", "SWAP", "ISWAP", "ISWAP_DAG", "CP"):
        return "entangle"
    if op == "CCX":
        return "toffoli"
    if op == "Measure":
        return "measure"
    if op == "Reset":
        return "reset"
    if op.endswith("_ERROR") or op.startswith("DEPOLARIZE"):
        return "noise"
    if op in ("DETECTOR", "OBSERVABLE_INCLUDE"):
        return "detector"
    return "clifford1"


def pi_label(x):
    if abs(x) < 1e-12:
        return "0"
    r = x / math.pi
    for q in [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512, 1024]:
        p = round(r * q)
        if p != 0 and abs(r * q - p) < 1e-5 * q:
            g = math.gcd(abs(p), q)
            p, q = p // g, q // g
            num = ("-" if p < 0 else "") + ("π" if abs(p) == 1 else f"{abs(p)}π")
            return num if q == 1 else f"{num}/{q}"
    return f"{x:.3g}"


def load(path):
    m = onnx.load(path)
    onnx.checker.check_model(m)
    g = m.graph
    meta = {p.key: p.value for p in m.metadata_props}
    nodes = []
    for n in g.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        nodes.append(
            dict(
                op=n.op_type,
                inputs=list(n.input),
                outputs=list(n.output),
                qubits=list(attrs.get("qubits", [])),
                attrs=attrs,
            )
        )
    n_qubits = len(g.input)
    return g.name, meta, nodes, n_qubits


def layout(nodes, n_qubits):
    """ASAP depth per node and a lane coordinate (qubit index scale)."""
    producer = {}
    depth = []
    for i, n in enumerate(nodes):
        d = 0
        for t in n["inputs"]:
            if t in producer:
                d = max(d, depth[producer[t]] + 1)
        depth.append(d + 1)
        for t in n["outputs"]:
            producer[t] = i
    lane = []
    det_col = 0
    for i, n in enumerate(nodes):
        if n["qubits"]:
            lane.append(float(np.mean(n["qubits"])))
        else:  # detectors / observables: a column beside the register
            lane.append(n_qubits + 0.8 + 0.55 * (det_col % 4))
            det_col += 1
    # de-overlap nodes that share a depth (cards are ~0.8 lanes wide)
    by_depth = {}
    for i, d in enumerate(depth):
        by_depth.setdefault(d, []).append(i)
    for d, idx in by_depth.items():
        idx.sort(key=lambda i: lane[i])
        for a, b in zip(idx, idx[1:]):
            if lane[b] - lane[a] < 0.9:
                lane[b] = lane[a] + 0.9
    return depth, lane, producer


def bezier(ax, p0, p1, color, lw, alpha, dashed=False, vertical=True, z=1):
    (x0, y0), (x1, y1) = p0, p1
    if vertical:
        dy = (y1 - y0) * 0.5
        verts = [(x0, y0), (x0, y0 + dy), (x1, y1 - dy), (x1, y1)]
    else:
        dx = (x1 - x0) * 0.5
        verts = [(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)]
    path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            ec=color,
            lw=lw,
            alpha=alpha,
            ls=(0, (3, 2)) if dashed else "-",
            capstyle="round",
            zorder=z,
        )
    )


def qubit_colors(n):
    cmap = plt.get_cmap("turbo")
    return [cmap(0.08 + 0.84 * (q / max(1, n - 1))) for q in range(n)]


def wire_qubit(tensor):
    # q{i} or q{i}_{k}
    if tensor.startswith("q"):
        head = tensor[1:].split("_")[0]
        if head.isdigit():
            return int(head)
    return None


def node_text(n):
    op = n["op"]
    a = n["attrs"]
    sub = []
    if "theta" in a and "phi" not in a:
        sub.append(pi_label(a["theta"]))
    elif "theta" in a:
        sub.append(",".join(pi_label(a[k]) for k in ("theta", "phi", "lambda")))
    if "p" in a:
        sub.append(f"p={a['p']:.2g}")
    if op == "Measure":
        sub.append(f"m{a.get('record', '')}")
    if op.startswith("IF_"):
        sub.append(f"if m{a.get('record', '')}")
    return op.replace("OBSERVABLE_INCLUDE", "OBS").replace("DETECTOR", "DET"), " ".join(sub)


def draw_small(path, out, title, subtitle):
    name, meta, nodes, nq = load(path)
    depth, lane, producer = layout(nodes, nq)
    qc = qubit_colors(nq)
    max_d = max(depth) + 1
    det_lanes = [lane[i] for i, n in enumerate(nodes) if not n["qubits"]]
    width_lanes = max([nq - 1] + det_lanes) + 1.2
    sx, sy = 1.0, 1.0  # lane spacing, depth spacing
    fig_w = max(6.0, 1.25 * width_lanes + 2.0)
    fig_h = max(5.0, 0.85 * (max_d + 1) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.2, width_lanes + 0.2)
    ax.set_ylim(max_d + 0.9, -1.6)
    ax.axis("off")

    def pos(i):
        return lane[i] * sx, depth[i] * sy

    # faint lane guides
    for q in range(nq):
        ax.plot([q, q], [-0.35, max_d + 0.35], color=qc[q], lw=0.6, alpha=0.12, zorder=0)
    # graph inputs / outputs
    for q in range(nq):
        ax.scatter([q], [-0.45], s=90, color=qc[q], zorder=4, edgecolors=BG, linewidths=1.5)
        ax.text(q, -0.95, f"q{q}", color=FG, ha="center", va="center", fontsize=8, family="monospace")
    # edges
    final_wire = {}
    for i, n in enumerate(nodes):
        x1, y1 = pos(i)
        k = len(n["inputs"])
        for j, t in enumerate(n["inputs"]):
            off = (j - (k - 1) / 2) * 0.16 if k > 1 else 0.0
            q = wire_qubit(t)
            if t in producer:
                x0, y0 = pos(producer[t])
                src = (x0, y0 + 0.27)
            elif q is not None:
                src = (q, -0.45)
            else:
                continue
            if q is not None:
                bezier(ax, src, (x1 + off, y1 - 0.27), qc[q], 2.0, 0.95, z=2)
            else:
                bezier(ax, src, (x1 + off, y1 - 0.27), "#d29922", 1.2, 0.85, dashed=True, z=2)
        for t in n["outputs"]:
            q = wire_qubit(t)
            if q is not None:
                final_wire[q] = i
    for q in range(nq):
        if q in final_wire:
            x0, y0 = pos(final_wire[q])
            bezier(ax, (x0, y0 + 0.27), (q, max_d + 0.4), qc[q], 2.0, 0.95, z=2)
        else:
            bezier(ax, (q, -0.45), (q, max_d + 0.4), qc[q], 2.0, 0.6, z=2)
        ax.scatter([q], [max_d + 0.45], s=60, marker="v", color=qc[q], zorder=4)
    # node cards
    for i, n in enumerate(nodes):
        x, y = pos(i)
        col = CATS[category(n["op"])][0]
        w, h = 0.78, 0.54
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                fc="#161b22", ec=col, lw=1.6, zorder=5,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, 0.22,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                fc=col, ec=col, lw=0, zorder=6,
            )
        )
        label, sub = node_text(n)
        ax.text(x, y - h / 2 + 0.115, label, color="white", ha="center", va="center",
                fontsize=6.6 if len(label) > 5 else 7.5, fontweight="bold", zorder=7, family="DejaVu Sans")
        qs = n["qubits"]
        detail = sub if sub else (",".join(f"q{q}" for q in qs) if qs else "")
        if qs and sub:
            detail = sub
        ax.text(x, y + 0.11, detail, color=MUTED if not sub else FG, ha="center", va="center",
                fontsize=5.6, zorder=7, family="DejaVu Sans")
    # title
    ax.text(-1.1, -1.45, title, color=FG, fontsize=13, fontweight="bold", va="center", ha="left")
    ax.text(-1.1, -1.12, subtitle, color=MUTED, fontsize=7.5, va="center", ha="left")
    legend(ax, nodes, fig)
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def legend(ax, nodes, fig):
    used = []
    for n in nodes:
        c = category(n["op"])
        if c not in used:
            used.append(c)
    handles = [
        plt.Line2D([0], [0], marker="s", color=BG, markerfacecolor=CATS[c][0], markersize=7, label=CATS[c][1])
        for c in CATS if c in used
    ]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.06), ncol=min(4, len(handles)),
                    frameon=False, fontsize=6.5, labelcolor=FG, handletextpad=0.3, columnspacing=1.0)
    return leg


def draw_poster(path, out, title, subtitle, rows):
    """Large graphs: time runs left to right and wraps into `rows` bands."""
    name, meta, nodes, nq = load(path)
    depth, lane, producer = layout(nodes, nq)
    qc = qubit_colors(nq)
    max_d = max(depth) + 1
    per_row = math.ceil(max_d / rows)
    band = nq + 3.0
    fig_w = 22.0
    fig_h = max(6.0, rows * band * 0.23 + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-per_row * 0.02, per_row * 1.01)
    ax.set_ylim(rows * band + 0.5, -3.2)
    ax.axis("off")

    def pos(i):
        d = depth[i]
        r = (d - 1) // per_row
        return (d - 1) % per_row + 0.5, r * band + lane[i] + 1.0, r

    # node dots and wires (only wires within one band are drawn)
    seg_x, seg_y, seg_c = [], [], []
    for i, n in enumerate(nodes):
        x1, y1, r1 = pos(i)
        for t in n["inputs"]:
            q = wire_qubit(t)
            if t in producer:
                x0, y0, r0 = pos(producer[t])
                if r0 != r1:
                    continue
                if q is None:
                    continue
                bezier(ax, (x0, y0), (x1, y1), qc[q], 0.45, 0.8, vertical=False, z=1)
    xs, ys, cs, ss = [], [], [], []
    for i, n in enumerate(nodes):
        x, y, _ = pos(i)
        xs.append(x)
        ys.append(y)
        cs.append(CATS[category(n["op"])][0])
        ss.append(9 if n["op"] in ("CCX", "Measure", "IF_P", "IF_X") else 4)
    ax.scatter(xs, ys, s=ss, c=cs, zorder=3, linewidths=0)
    for r in range(rows):
        for q in range(nq):
            ax.text(-per_row * 0.012, r * band + q + 1.0, f"q{q}", color=qc[q], fontsize=4.2, ha="right",
                    va="center", family="monospace")
    ax.text(0, -2.4, title, color=FG, fontsize=15, fontweight="bold", va="center")
    ax.text(0, -1.3, subtitle, color=MUTED, fontsize=8.5, va="center")
    used = []
    for n in nodes:
        c = category(n["op"])
        if c not in used:
            used.append(c)
    handles = [
        plt.Line2D([0], [0], marker="o", color=BG, markerfacecolor=CATS[c][0], markersize=6, label=CATS[c][1])
        for c in CATS if c in used
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=len(handles), frameon=False,
              fontsize=7, labelcolor=FG)
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title")
    ap.add_argument("--subtitle")
    ap.add_argument("--wrap", type=int, default=0, help="poster mode: number of rows (large graphs)")
    a = ap.parse_args()
    name, meta, nodes, nq = load(a.model)
    title = a.title or name
    parts = [f"{meta.get('qubits', nq)} qubits", f"{len(nodes)} nodes", f"depth {meta.get('depth', '?')}"]
    for key, label in [("gates_2q", "two-qubit gates"), ("gates_3q", "Toffolis"), ("t_count", "T gates"),
                       ("measurements", "measurements"), ("detectors", "detectors")]:
        if meta.get(key, "0") != "0":
            parts.append(f"{meta[key]} {label}")
    sub = a.subtitle or " · ".join(parts)
    if a.wrap:
        draw_poster(a.model, a.out, title, sub, a.wrap)
    else:
        draw_small(a.model, a.out, title, sub)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
