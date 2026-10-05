#!/usr/bin/env python3
"""cotengra + quimb contraction-path costs for the single amplitude <0^n|C|0^n> of circuit files
written by sycamore.py (format documented there).

Environment (private install, shared venv untouched; numba is a pass-through stub, see the report):

    export PYTHONPATH=/dev/shm/qsim/ext/tn-py OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    PY=/dev/shm/qsim/venv/bin/python

Usage:

    # check the quimb builder against cirq's exact amplitudes (12-qubit circuits)
    nice -n 15 $PY quimb_paths.py --validate circuits/sycsub12_m8_s0.txt circuits/sycsub12_m8_s1.txt

    # path searches; appends one JSON line per (circuit, config, budget) to --out
    nice -n 15 prlimit --as=3000000000 $PY quimb_paths.py circuits/syc53_m10_s0.txt \
        --configs unsliced sliced_sl --budget std --parallel 2 --out cotengra_paths.jsonl \
        --save-dir /dev/shm/qsim/ext/tn-py/work          (or: bash run_paths.sh)

Method. Each op is applied with quimb.tensor.Circuit.apply_gate_raw(matrix, qubits) using the
matrices below (independent re-implementation of the ones in sycamore.py; --validate checks them,
the parser and the qubit mapping against cirq). File qubit q is quimb qubit q. The amplitude network
is quimb's: Circuit.amplitude_rehearse('0'*n, simplify_sequence='ADCRS', optimize=opt) (full
simplification of the state, project every output index on 0, simplify again), and the tree is the
one cotengra's HyperOptimizer returns for that simplified network:

    unsliced: HyperOptimizer(methods=['kahypar', 'greedy'], minimize='flops', reconf_opts={},
                             max_repeats=R, max_time=T, parallel=P, optlib='optuna')
              (+ post_slice_*: the best tree sliced afterwards with tree.slice(target_size=2**27))
    sliced:   the same plus slicing_reconf_opts={'target_size': 2**27}  (as specified; every trial
              alternates 2x slicing steps with full subtree reconfigurations; one trial took > 6 min at
              m=12 on the loaded machine, so it was only run for m=10 and m=12)
    sliced_sl: the same as unsliced plus slicing_opts={'target_size': 2**27} (each trial is sliced
              greedily to 2^27, then subtree-reconfigured once; cotengra's cheaper slicing mode).
              From 12:40 run in-process (parallel=False) with SliceFinder max_repeats=4 (default 16)
              because per-trial slicing of 2^50-wide trial trees exceeded the 3 GB per-process cap;
              still not viable for m >= 16 (MemoryError at m=20), so used for m <= 14 (+ one m=18 run).
    unsliced+slice_reconf (--slice-reconf): best unsliced tree of a run, then one
              tree.slice_and_reconfigure(target_size=2**27), cotengra defaults, single process.
    Every sliced tree still wider than 2^27 after cotengra's final reconfiguration is re-sliced
    (enforce_target; flag post_sliced_to_target, pre-fix numbers kept).
    budgets:  standard R=512, T=120 s with P=8 (runs before 12:35);  std R=256, T=120 s and long
              R=2048, T=600 s with P=2 and prlimit --as=3e9 (after the coordinator's process-pool rule)

Recorded per run (see FIELDS at the bottom of this docstring): simplified network size, the tree's
contraction_cost() = sum over pairwise contractions of the product of the dimensions of every index
involved (= number of scalar complex multiply-adds, times the number of slices for a sliced tree),
total_flops() (cotengra 0.8.2: identical to contraction_cost() when no dtype is given; with
dtype='complex' it is 4x that, cotengra's convention; one complex multiply-add is 8 real flops),
log2 of the largest intermediate (per slice), number of slices, search time and trials.

    FIELDS: circuit m seed config budget n_tensors n_inds n_hyper_inds cost log10_cost
            total_flops total_flops_complex log2_max_size contraction_width nslices
            log10_cost_per_slice log2_peak_size log10_total_write search_time_s
            rehearse_time_s simplify_time_*_s trials best_method optimizer (all kwargs)
            cpu_share search_processes worker_cpu_s main_cpu_s rlimit_as_bytes main_maxrss_gb
            worker_maxrss_gb post_slice_* load_before load_after versions date ...
"""

import argparse
import datetime
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time

import numpy as np

S2 = 1 / np.sqrt(2)
MAT_1Q = {
    "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2,
    "sy": np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]], dtype=complex) / 2,
    "sw": np.array([[(1 + 1j) / 2, -1j * S2], [S2, (1 + 1j) / 2]], dtype=complex),
}
BUDGETS = {"standard": (512, 120), "std": (256, 120), "long": (2048, 600), "smoke": (16, 20)}
# 'standard' (512 repeats, 120 s) ran with parallel=8 before 12:35; the coordinator then capped process
# pools at 2 workers x prlimit --as=3e9 (8 workers had reached ~2 GB RSS each), so later runs use
# 'std' (256 repeats, 120 s) and 'long' (2048 repeats, 600 s) with parallel=2. Every JSON line records
# its own max_repeats/max_time/parallel, RLIMIT_AS and peak RSS of the main process and the workers.
TARGET_SIZE = 2**27
SLICE_REPEATS = 4


def fsim(theta, phi):
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]],
        dtype=complex,
    )


def read_circuit(path):
    """Parse a circuit file -> (n_qubits, [(name, qubits, params)])."""
    n, ops = None, []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            t = line.split()
            if t[0] == "qubits":
                n = int(t[1])
            elif t[0] in MAT_1Q:
                ops.append((t[0], (int(t[1]),), ()))
            elif t[0] == "fsim":
                ops.append(("fsim", (int(t[1]), int(t[2])), (float(t[3]), float(t[4]))))
            else:
                raise ValueError(f"{path}: unknown op {t[0]!r}")
    if n is None:
        raise ValueError(f"{path}: no 'qubits' line")
    return n, ops


def build_circuit(path):
    """quimb Circuit for a circuit file (file qubit q = quimb qubit q)."""
    import quimb.tensor as qtn

    n, ops = read_circuit(path)
    circ = qtn.Circuit(n)
    for name, qs, params in ops:
        mat = fsim(*params) if name == "fsim" else MAT_1Q[name]
        circ.apply_gate_raw(mat, qs)
    return circ, n, ops


def versions():
    import autoray
    import cotengra
    import kahypar
    import optuna
    import quimb
    import scipy

    try:
        import cotengrust

        ctr = getattr(cotengrust, "__version__", "0.2.1 (wheel)")
    except ImportError:
        ctr = None
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "quimb": quimb.__version__,
        "cotengra": cotengra.__version__,
        "kahypar": getattr(kahypar, "__version__", "1.3.7 (wheel)"),
        "optuna": optuna.__version__,
        "autoray": autoray.__version__,
        "cotengrust": ctr,
        "numba": numba.__version__ + (" (pass-through stub)" if getattr(numba, "__stub__", False) else ""),
    }


def make_opt(config, budget, parallel):
    import cotengra as ctg

    reps, tmax = BUDGETS[budget]
    kw = dict(
        methods=["kahypar", "greedy"],
        minimize="flops",
        reconf_opts={},
        max_repeats=reps,
        max_time=tmax,
        parallel=parallel,
        optlib="optuna",
    )
    if config == "sliced":
        kw["slicing_reconf_opts"] = {"target_size": TARGET_SIZE}
    elif config == "sliced_sl":
        # in-process trials (parallel=False): a trial that hits the prlimit memory cap then fails alone
        # (cotengra scores it inf and goes on) instead of killing a worker pool (BrokenProcessPool).
        kw["parallel"] = False
        # max_repeats=4 (cotengra default 16): SliceFinder caches a cost-table copy per tried index
        # set, which reached > 2.5 GB per worker for m >= 16 trees (2^50 wide) under the 3 GB cap.
        # The m=12/14 'standard' sliced_sl rows (before 12:40) used the default 16.
        kw["slicing_opts"] = {"target_size": TARGET_SIZE, "max_repeats": SLICE_REPEATS}
    elif config != "unsliced":
        raise ValueError(config)
    return ctg.HyperOptimizer(**kw), kw


def enforce_target(tree, config):
    """cotengra applies reconf_opts AFTER slicing, and a flops-only subtree reconfiguration can regrow
    the largest intermediate past target_size. For the sliced configs, re-slice such a tree greedily
    (ContractionTree.slice(target_size=2**27)) so every reported sliced tree meets the target; the
    pre-fix numbers are kept in the record. Returns (tree, fix_info or None)."""
    if config == "unsliced" or tree.max_size() <= TARGET_SIZE:
        return tree, None
    pre = tree_stats(tree)
    fixed = tree.slice(target_size=TARGET_SIZE, max_repeats=SLICE_REPEATS)
    return fixed, {"post_sliced_to_target": True, "pre_fix": pre}


def kill_pool():
    """Terminate cotengra's worker processes. When max_time expires cotengra stops waiting but
    trials already running keep their workers busy, and interpreter exit would wait for them; kill
    them so the next search starts on idle workers. Returns nothing."""
    from cotengra.parallel import ProcessPoolHandler

    pool = ProcessPoolHandler._pool
    if pool is None:
        return
    procs = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    for p in procs:
        try:
            p.join(timeout=10)
        except Exception:  # noqa: BLE001
            pass
    ProcessPoolHandler._pool = None
    ProcessPoolHandler._n_workers = -1
    ProcessPoolHandler._pid = None


def resource_info():
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    return {
        "rlimit_as_bytes": None if soft == resource.RLIM_INFINITY else int(soft),
        "main_maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "worker_maxrss_gb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20,
    }


def children_cpu_s():
    import resource

    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def network_stats(tn):
    from collections import Counter

    counts = Counter(ix for t in tn for ix in t.inds)
    sizes = Counter(tn.ind_size(ix) for ix in counts)
    return {
        "n_tensors": tn.num_tensors,
        "n_inds": len(counts),
        "n_hyper_inds": sum(1 for c in counts.values() if c > 2),
        "max_ind_appearances": max(counts.values()),
        "ind_size_hist": {str(k): v for k, v in sorted(sizes.items())},
        "max_tensor_rank": max(t.ndim for t in tn),
        "log2_max_tensor_size": max(math.log2(t.size) for t in tn),
    }


def tree_stats(tree):
    cost = tree.contraction_cost()
    nsl = tree.multiplicity
    return {
        "cost": float(cost),
        "log10_cost": math.log10(max(cost, 1)),
        "total_flops": float(tree.total_flops()),
        "total_flops_complex": float(tree.total_flops(dtype="complex")),
        "log2_max_size": math.log2(tree.max_size()),
        "contraction_width": float(tree.contraction_width()),
        "nslices": int(nsl),
        "n_sliced_inds": len(tree.sliced_inds),
        "log10_cost_per_slice": math.log10(max(cost / nsl, 1)),
        "log2_peak_size": math.log2(tree.peak_size()),
        "log10_total_write": math.log10(max(tree.total_write(), 1)),
    }


def load_snapshot():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"

    return {"uptime": run(["uptime"]), "free_g": run(["free", "-g"]).splitlines()[1] if run(["free", "-g"]) else ""}


def validate(paths):
    """Compare quimb amplitudes (complex128, greedy tree) with the cirq values in *.amps.json."""
    worst = 0.0
    for path in paths:
        circ, n, _ = build_circuit(path)
        ref = json.load(open(path[: -len(".txt")] + ".amps.json"))
        for e in [ref["zeros"], ref["ones"]] + ref["extra"]:
            bits = e["bits_q0_first"]  # quimb bitstring position i = qubit i
            a = complex(circ.amplitude(bits, optimize="greedy", dtype="complex128"))
            b = complex(e["re"], e["im"])
            err = abs(a - b)
            worst = max(worst, err)
            print(f"{os.path.basename(path)} idx={e['index']:5d} bits={bits} quimb={a:.15e} cirq={b:.15e} |diff|={err:.2e}")
    print(f"max |quimb - cirq| = {worst:.3e}")
    if worst > 1e-10:
        raise SystemExit("validation FAILED")
    print("validation OK")


def run_one(path, config, budget, parallel, save_dir, out):
    import quimb.tensor as qtn  # noqa: F401

    stem = os.path.basename(path)[: -len(".txt")]
    m = int(stem.split("_m")[1].split("_")[0])
    seed = int(stem.split("_s")[-1])
    circ, n, ops = build_circuit(path)
    zeros = "0" * n

    # simplification only (also warms quimb's cache of the simplified state)
    t0 = time.perf_counter()
    tn_simpl = circ.amplitude_tn(zeros, simplify_sequence="ADCRS")
    t1 = time.perf_counter()
    simplify_time_first = t1 - t0
    t0 = time.perf_counter()
    tn_simpl = circ.amplitude_tn(zeros, simplify_sequence="ADCRS")
    t1 = time.perf_counter()
    simplify_time_cached = t1 - t0

    opt, kw = make_opt(config, budget, parallel)
    before = load_snapshot()
    import resource

    def self_cpu():
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    cpu0, scpu0 = children_cpu_s(), self_cpu()
    t0 = time.perf_counter()
    rehearse = circ.amplitude_rehearse(zeros, simplify_sequence="ADCRS", optimize=opt)
    t1 = time.perf_counter()
    after = load_snapshot()
    kill_pool()
    worker_cpu = children_cpu_s() - cpu0
    main_cpu = self_cpu() - scpu0
    nproc = kw["parallel"] if kw["parallel"] else 1
    tn, tree = rehearse["tn"], rehearse["tree"]
    assert tn.num_tensors == tn_simpl.num_tensors
    tree, fix = enforce_target(tree, config)
    post = None
    if config == "unsliced":
        # cheap post-hoc sliced figure for every circuit: this tree sliced greedily to 2^27 (no reconf)
        st = tree.slice(target_size=TARGET_SIZE, max_repeats=16)
        post = {
            "post_slice_log10_cost": math.log10(max(st.contraction_cost(), 1)),
            "post_slice_log2_max_size": math.log2(st.max_size()),
            "post_slice_nslices": int(st.multiplicity),
            "post_slice_note": "unsliced best tree + ContractionTree.slice(target_size=2**27, max_repeats=16)",
        }

    rec = {
        "circuit": stem,
        "file": os.path.relpath(path, os.path.dirname(os.path.abspath(__file__))),
        "m": m,
        "seed": seed,
        "n_qubits": n,
        "n_ops": len(ops),
        "amplitude": "<0^n|C|0^n>",
        "config": config,
        "budget": budget,
        "optimizer": {k: v for k, v in kw.items()},
        "simplify_sequence": "ADCRS",
        **network_stats(tn),
        **tree_stats(tree),
        **(post or {}),
        "post_sliced_to_target": bool(fix),
        **({"pre_fix": fix["pre_fix"]} if fix else {}),
        "rehearse_W": rehearse["W"],
        "rehearse_C": rehearse["C"],
        "search_time_s": (t1 - t0) - simplify_time_cached,
        "rehearse_time_s": t1 - t0,
        "simplify_time_first_s": simplify_time_first,
        "simplify_time_cached_s": simplify_time_cached,
        "trials": len(opt.scores),
        **resource_info(),
        "worker_cpu_s": worker_cpu,
        "worker_cpu_share": worker_cpu / max(parallel * (t1 - t0), 1e-9),
        "main_cpu_s": main_cpu,
        "search_processes": nproc,
        "cpu_share": (worker_cpu + (main_cpu if not kw["parallel"] else 0)) / max(nproc * (t1 - t0), 1e-9),
        "best_method": opt.best.get("params", {}).get("method"),
        "best_params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                        for k, v in opt.best.get("params", {}).items()},
        "cost_units": (
            "cost = contraction_cost() = scalar complex multiply-adds (incl. all slices); "
            "total_flops = cotengra total_flops() (== cost); total_flops_complex = "
            "total_flops(dtype='complex') = 4*cost (cotengra convention; 1 complex MAC = 8 real flops)"
        ),
        "versions": versions(),
        "load_before": before,
        "load_after": after,
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pk = os.path.join(save_dir, f"{stem}_{config}_{budget}.pkl")
        with open(pk, "wb") as f:
            pickle.dump({"tn": tn, "tree": tree, "record": rec}, f)
        rec["pickle"] = pk
    line = json.dumps(rec)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")
    print(
        f"{stem} {config:9s} {budget:8s}: tensors={rec['n_tensors']} log10cost={rec['log10_cost']:.3f} "
        f"log2max={rec['log2_max_size']:.1f} slices={rec['nslices']} trials={rec['trials']} "
        f"search={rec['search_time_s']:.1f}s cpu_share={rec['cpu_share']:.2f} procs={rec['search_processes']} "
        f"load1={before['uptime'].split('load average:')[1].split(',')[0].strip()} method={rec['best_method']}",
        flush=True,
    )
    return rec, tn, tree


def fixup(pickles, out):
    """Apply enforce_target to saved (tn, tree, record) pickles; rewrite the pickle and replace the
    matching line of the JSONL file `out` (same circuit, config, budget and date)."""
    for pk in pickles:
        with open(pk, "rb") as f:
            d = pickle.load(f)
        rec = d["record"]
        tree, fix = enforce_target(d["tree"], rec["config"])
        new = dict(rec)
        if rec["config"] == "unsliced" and "post_slice_log10_cost" not in rec:
            st = tree.slice(target_size=TARGET_SIZE, max_repeats=16)
            new.update({
                "post_slice_log10_cost": math.log10(max(st.contraction_cost(), 1)),
                "post_slice_log2_max_size": math.log2(st.max_size()),
                "post_slice_nslices": int(st.multiplicity),
                "post_slice_note": "unsliced best tree + ContractionTree.slice(target_size=2**27, max_repeats=16)"
                                   " (added by --fixup)",
            })
        elif fix is None:
            print(f"{pk}: ok, nothing to do")
            continue
        else:
            new.update(tree_stats(tree))
            new["post_sliced_to_target"] = True
            new["pre_fix"] = fix["pre_fix"]
        new["fixup_date"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with open(pk, "wb") as f:
            pickle.dump({"tn": d["tn"], "tree": tree, "record": new}, f)
        if out and os.path.exists(out):
            lines = open(out).read().splitlines()
            key = (rec["circuit"], rec["config"], rec["budget"], rec["date"])
            with open(out, "w") as f:
                for line in lines:
                    r = json.loads(line)
                    same = (r["circuit"], r["config"], r["budget"], r["date"]) == key
                    f.write((json.dumps(new) if same else line) + "\n")
        if fix:
            print(f"{pk}: log2 max {fix['pre_fix']['log2_max_size']:.1f} -> {new['log2_max_size']:.1f}, "
                  f"log10 cost {fix['pre_fix']['log10_cost']:.3f} -> {new['log10_cost']:.3f}, slices {new['nslices']}")
        else:
            print(f"{pk}: post-slice log10 cost {new['post_slice_log10_cost']:.3f}, "
                  f"slices {new['post_slice_nslices']}")


def slice_reconf(pickles, out, save_dir):
    """config 'unsliced+slice_reconf': the best unsliced tree of a saved run, then one
    ContractionTree.slice_and_reconfigure(target_size=2**27) (cotengra defaults: step_size=2, subtree
    reconfiguration after every 2x slicing step), single process. Appends one JSON line per pickle."""
    for pk in pickles:
        with open(pk, "rb") as f:
            d = pickle.load(f)
        rec = d["record"]
        assert rec["config"] == "unsliced", pk
        t0 = time.perf_counter()
        st = d["tree"].slice_and_reconfigure(target_size=TARGET_SIZE)
        t1 = time.perf_counter()
        st, fix = enforce_target(st, "slice_reconf")
        new = {k: v for k, v in rec.items() if not k.startswith("post_slice")}
        new.update(tree_stats(st))
        new.update({
            "config": "unsliced+slice_reconf",
            "source_record_date": rec["date"],
            "slice_reconf_time_s": t1 - t0,
            "slice_reconf_note": "best unsliced tree of the same circuit/budget, then "
                                 "slice_and_reconfigure(target_size=2**27), cotengra defaults, 1 process",
            "post_sliced_to_target": bool(fix),
            **resource_info(),
            "load_after": load_snapshot(),
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if save_dir:
            pko = os.path.join(save_dir, f"{rec['circuit']}_unsliced+slice_reconf_{rec['budget']}.pkl")
            with open(pko, "wb") as f:
                pickle.dump({"tn": d["tn"], "tree": st, "record": new}, f)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(new) + "\n")
        print(f"{rec['circuit']} {rec['budget']}: unsliced {rec['log10_cost']:.3f} (2^{rec['log2_max_size']:.0f}) -> "
              f"slice_reconf {new['log10_cost']:.3f} (2^{new['log2_max_size']:.0f}, {new['nslices']} slices) "
              f"in {t1 - t0:.0f} s, maxrss {new['main_maxrss_gb']:.2f} GB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="cotengra/quimb single-amplitude path costs")
    ap.add_argument("circuits", nargs="+")
    ap.add_argument("--validate", action="store_true", help="compare with *.amps.json and exit")
    ap.add_argument("--fixup", action="store_true", help="positional args are pickles: enforce 2^27 on sliced trees")
    ap.add_argument("--slice-reconf", action="store_true",
                    help="positional args are unsliced pickles: slice_and_reconfigure the tree to 2^27")
    ap.add_argument("--configs", nargs="+", default=["unsliced", "sliced"])
    ap.add_argument("--budget", default="standard", choices=sorted(BUDGETS))
    ap.add_argument("--parallel", type=int, default=2, help="cotengra worker processes (machine rule: <= 2)")
    ap.add_argument("--out", default=None, help="JSONL file to append to")
    ap.add_argument("--save-dir", default=None, help="pickle (tn, tree, record) here")
    args = ap.parse_args(argv)

    if args.validate:
        validate(args.circuits)
        return 0
    if args.fixup:
        fixup(args.circuits, args.out)
        return 0
    if args.slice_reconf:
        slice_reconf(args.circuits, args.out, args.save_dir)
        return 0
    for path in args.circuits:
        for config in args.configs:
            run_one(path, config, args.budget, args.parallel, args.save_dir, args.out)
            kill_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
