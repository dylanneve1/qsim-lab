#!/usr/bin/env python3
"""Timed quimb/cotengra contractions of the single amplitude <0^53|C|0^53> (step 5 of the TN baseline).

The trees come from quimb_paths.py (pickles of the ADCRS-simplified network + cotengra tree in
--work); for each circuit the cheapest tree with largest intermediate <= 2^27 elements is used.
Only the contraction is timed: tn.astype(dtype) is done once, then
tn.contract(all, output_inds=(), optimize=tree) (quimb's own call inside Circuit.amplitude) is
timed --reps times; the minimum is the headline. Mode e2e times one fresh
Circuit.amplitude('0'*53, simplify_sequence='ADCRS', optimize='auto-hq', dtype=...) (quimb's default
optimizer) from the imports and parsing of the circuit file onwards (path search included).

Run under the bench lock with 8 BLAS threads on the 8 physical cores (even logical CPUs):

    export PYTHONPATH=/dev/shm/qsim/ext/tn-py OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    flock /dev/shm/qsim/bench.lock taskset -c 0,2,4,6,8,10,12,14 nice -n 15 prlimit --as=$((12*2**30)) \
        /dev/shm/qsim/venv/bin/python quimb_timings.py contract syc53_m10_s0 --dtypes complex64 complex128 \
        --reps 3 --out quimb_timings.jsonl
    (same prefix) ... quimb_timings.py e2e syc53_m10_s0 --dtypes complex64 --out quimb_timings.jsonl
    (same prefix) ... quimb_timings.py slices syc53_m12_s0 --dtypes complex64 --reps 3 --out quimb_timings.jsonl
Or simply: bash run_timings.sh <stem> contract|slices|e2e [dtypes...]
"""

import argparse
import ctypes
import datetime
import glob
import json
import math
import os
import pickle
import platform
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import quimb_paths as qp  # noqa: E402

MAX_LOG2_SIZE = 27


def snapshot():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"

    up = run(["uptime"])
    free = run(["free", "-g"]).splitlines()
    load1 = float(up.split("load average:")[1].split(",")[0]) if "load average:" in up else None
    return {"uptime": up, "load1": load1, "free_g": free[1] if len(free) > 1 else ""}


def blas_info():
    info = {"env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        d = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info["numpy_blas"] = f"{d['name']} {d['version']} ({d.get('openblas configuration', '')})"
    except Exception as e:  # noqa: BLE001
        info["numpy_blas"] = f"unknown ({e})"
    try:  # runtime kernel and thread count of numpy's bundled OpenBLAS
        import numpy.linalg  # noqa: F401

        libs = [p for p in open("/proc/self/maps").read().split() if "openblas" in p and p.endswith(".so")]
        lib = ctypes.CDLL(sorted(set(libs))[0])
        for name in ("scipy_openblas_get_corename64_", "openblas_get_corename64_", "openblas_get_corename"):
            if hasattr(lib, name):
                f = getattr(lib, name)
                f.restype = ctypes.c_char_p
                info["openblas_core"] = f().decode()
                break
        for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
            if hasattr(lib, name):
                info["openblas_threads"] = int(getattr(lib, name)())
                break
    except Exception as e:  # noqa: BLE001
        info["openblas_runtime"] = f"unknown ({e})"
    try:
        info["affinity"] = sorted(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        pass
    return info


def pick_tree(stem, work):
    best = None
    for pk in sorted(glob.glob(os.path.join(work, f"{stem}_*.pkl"))):
        if pk.endswith("_smoke.pkl"):
            continue
        with open(pk, "rb") as f:
            d = pickle.load(f)
        rec = d["record"]
        if rec["log2_max_size"] > MAX_LOG2_SIZE + 1e-9:
            continue
        if best is None or rec["cost"] < best[0]["record"]["cost"]:
            best = (d, pk)
    if best is None:
        raise SystemExit(f"no usable tree for {stem} in {work}")
    return best


def contract_mode(stem, dtypes, reps, work, out, max_gb):
    import quimb  # noqa: F401

    d, pk = pick_tree(stem, work)
    tn, tree, rec = d["tn"], d["tree"], d["record"]
    for dtype in dtypes:
        itemsize = np.dtype(dtype).itemsize
        est_gb = tree.peak_size() * itemsize / 2**30
        if est_gb > max_gb:
            print(f"skip {stem} {dtype}: estimated peak {est_gb:.1f} GB > {max_gb} GB")
            continue
        tnd = tn.astype(dtype)
        before = snapshot()
        times, vals = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            v = tnd.contract(all, output_inds=(), optimize=tree)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            vals.append(complex(v))
        after = snapshot()
        v = vals[-1]
        res = {
            "mode": "contract",
            "circuit": stem,
            "dtype": dtype,
            "threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0),
            "reps": reps,
            "times_s": times,
            "min_s": min(times),
            "tree_source": os.path.basename(pk),
            "tree_config": rec["config"],
            "tree_budget": rec["budget"],
            "log10_cost": rec["log10_cost"],
            "log2_max_size": rec["log2_max_size"],
            "nslices": rec["nslices"],
            "n_tensors": rec["n_tensors"],
            "est_peak_gb": est_gb,
            "complex_macs_per_s": rec["cost"] / min(times),
            "real_gflops_per_s_8_per_mac": 8 * rec["cost"] / min(times) / 1e9,
            "amplitude": [v.real, v.imag],
            "amplitude_all_reps_identical": all(x == vals[0] for x in vals),
            "porter_thomas_2n_abs2": (2.0**53) * abs(v) ** 2,
            "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "load_before": before,
            "load_after": after,
            "blas": blas_info(),
        "rlimit_as_bytes": qp.resource_info()["rlimit_as_bytes"],
        "cotengra_num_workers_env": os.environ.get("COTENGRA_NUM_WORKERS"),
            "versions": qp.versions(),
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": platform.node(),
        }
        print(json.dumps({k: res[k] for k in ("circuit", "dtype", "times_s", "min_s", "log10_cost", "log2_max_size",
                                                "nslices", "amplitude", "maxrss_gb")}), flush=True)
        print(f"   load before {before['load1']}  after {after['load1']}", flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")


def slices_mode(stem, dtypes, reps, work, out, max_gb):
    """Time `reps` single slices (slice i = 0..reps-1, identical cost) of the chosen sliced tree with
    cotengra's ContractionTree.contract_slice and extrapolate: total ~ min slice time x nslices.
    For trees too expensive to contract fully inside one bench-lock hold; no amplitude results."""
    d, pk = pick_tree(stem, work)
    tn, tree, rec = d["tn"], d["tree"], d["record"]
    nsl = int(tree.multiplicity)
    for dtype in dtypes:
        est_gb = tree.peak_size() * np.dtype(dtype).itemsize / 2**30
        if est_gb > max_gb:
            print(f"skip {stem} {dtype}: estimated peak {est_gb:.1f} GB > {max_gb} GB")
            continue
        arrays = [np.asarray(t.data).astype(dtype) for t in tn]  # tree inputs follow tn's tensor order
        before = snapshot()
        times = []
        for i in range(reps):
            t0 = time.perf_counter()
            tree.contract_slice(arrays, i)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        after = snapshot()
        res = {
            "mode": "slices",
            "circuit": stem,
            "dtype": dtype,
            "threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0),
            "slices_timed": list(range(reps)),
            "slice_times_s": times,
            "min_slice_s": min(times),
            "nslices": nsl,
            "extrapolated_total_s_min": min(times) * nsl,
            "extrapolated_total_s_mean": sum(times) / len(times) * nsl,
            "note": "NOT a full contraction: total extrapolated from single-slice times (all slices have equal cost)",
            "tree_source": os.path.basename(pk),
            "tree_config": rec["config"],
            "tree_budget": rec["budget"],
            "log10_cost": rec["log10_cost"],
            "log10_cost_per_slice": rec["log10_cost_per_slice"],
            "log2_max_size": rec["log2_max_size"],
            "est_peak_gb": est_gb,
            "complex_macs_per_s": (rec["cost"] / nsl) / min(times),
            "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "load_before": before,
            "load_after": after,
            "blas": blas_info(),
        "rlimit_as_bytes": qp.resource_info()["rlimit_as_bytes"],
        "cotengra_num_workers_env": os.environ.get("COTENGRA_NUM_WORKERS"),
            "versions": qp.versions(),
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": platform.node(),
        }
        print(json.dumps({k: res[k] for k in ("circuit", "dtype", "slice_times_s", "nslices",
                                                "extrapolated_total_s_min", "log10_cost", "log2_max_size")}), flush=True)
        print(f"   load before {before['load1']}  after {after['load1']}", flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")


def e2e_mode(stem, dtypes, out, optimize):
    """One fresh end-to-end amplitude: imports, parse, build, simplify, path search, contraction."""
    t_start = time.perf_counter()
    import cotengra  # noqa: F401
    import quimb.tensor  # noqa: F401

    before = snapshot()
    path = os.path.join(HERE, "circuits", f"{stem}.txt")
    circ, n, ops = qp.build_circuit(path)
    t_built = time.perf_counter()
    v = complex(circ.amplitude("0" * n, simplify_sequence="ADCRS", optimize=optimize, dtype=dtypes[0]))
    t_end = time.perf_counter()
    after = snapshot()
    # auto-hq caches its tree (ReusableHyperOptimizer, in memory): this returns the tree just used
    tree = circ.amplitude_rehearse("0" * n, simplify_sequence="ADCRS", optimize=optimize)["tree"]
    res = {
        "mode": "e2e",
        "circuit": stem,
        "dtype": dtypes[0],
        "threads_blas": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0),
        "optimize": optimize,
        "optimize_note": "quimb's default for Circuit.amplitude; cotengra AutoHQOptimizer, search workers = "
                         "OMP_NUM_THREADS (8)",
        "total_s_incl_imports": t_end - t_start,
        "build_s_incl_imports": t_built - t_start,
        "amplitude_call_s": t_end - t_built,
        "tree_log10_cost": math.log10(tree.contraction_cost()),
        "tree_log2_max_size": math.log2(tree.max_size()),
        "tree_nslices": int(tree.multiplicity),
        "amplitude": [v.real, v.imag],
        "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "load_before": before,
        "load_after": after,
        "blas": blas_info(),
        "rlimit_as_bytes": qp.resource_info()["rlimit_as_bytes"],
        "cotengra_num_workers_env": os.environ.get("COTENGRA_NUM_WORKERS"),
        "versions": qp.versions(),
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
    }
    print(json.dumps({k: res[k] for k in ("circuit", "dtype", "total_s_incl_imports", "amplitude_call_s",
                                            "tree_log10_cost", "tree_log2_max_size", "amplitude")}), flush=True)
    print(f"   load before {before['load1']}  after {after['load1']}", flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="timed quimb contractions")
    ap.add_argument("mode", choices=["contract", "slices", "e2e"])
    ap.add_argument("circuit", help="circuit stem, e.g. syc53_m10_s0")
    ap.add_argument("--dtypes", nargs="+", default=["complex64"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--work", default="/dev/shm/qsim/ext/tn-py/work")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-gb", type=float, default=8.0, help="refuse trees whose estimated peak exceeds this")
    ap.add_argument("--optimize", default="auto-hq", help="e2e: optimize= passed to Circuit.amplitude")
    args = ap.parse_args(argv)
    if args.mode == "contract":
        contract_mode(args.circuit, args.dtypes, args.reps, args.work, args.out, args.max_gb)
    elif args.mode == "slices":
        slices_mode(args.circuit, args.dtypes, args.reps, args.work, args.out, args.max_gb)
    else:
        e2e_mode(args.circuit, args.dtypes, args.out, args.optimize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
