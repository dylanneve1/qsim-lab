#!/usr/bin/env python3
"""Tests for the AlphaQubit-lite data layer (run: python test_aq.py <syc-root> [nd_tool]).

1. Layout: every d = 3 (d = 5) Sycamore experiment maps onto one canonical layout; events land on
   the right (round, stabilizer) cells (t = 0 and t = R only on-basis; detector count conserved).
2. dem_to_circuit: Stim's DEM sampler and Stim's circuit sampler on the converted circuit agree
   (per-detector marginals, observable rate, 50 random detector-pair and detector-observable
   correlations: |z| < 5 at 4e5 shots); the converted circuit's own DEM equals the input DEM's
   mechanism set (same symptom -> probability, combined).
3. FastSampler (nd_tool stream, if given) on the converted circuit agrees with Stim's DEM sampler
   in the same statistics.
"""
import subprocess, sys
import numpy as np
from aq_data import *


def zstat(a, b):
    """two-proportion z for Bernoulli means of 0/1 arrays a, b (independent samples)"""
    pa, pb = a.mean(0), b.mean(0)
    p = (a.sum(0) + b.sum(0)) / (len(a) + len(b))
    se = np.sqrt(np.maximum(p * (1 - p), 1e-12) * (1 / len(a) + 1 / len(b)))
    return (pa - pb) / se


def compare(x, y, rng, label):
    """x, y: (N, nd+1) samples incl. observable column; returns max |z| over marginals and products"""
    z = [np.abs(zstat(x, y)).max()]
    nd = x.shape[1]
    pairs = rng.integers(0, nd, size=(50, 2))
    for i, j in pairs:
        if i != j:
            z.append(abs(zstat((x[:, i] & x[:, j])[:, None], (y[:, i] & y[:, j])[:, None])[0]))
    zm = max(z)
    print(f"  {label}: max |z| = {zm:.2f} over {len(z)} statistics")
    assert zm < 5, label
    return zm


def test_layout(root):
    import stim
    for d in (3, 5):
        sig = set()
        for e in syc_experiments(root):
            if e["d"] != d or e["R"] < 3:  # R = 1 has no bulk round (not used: fits use R >= 3)
                continue
            txt = open(os.path.join(e["path"], "circuit_ideal.stim")).read()
            L = Layout(txt, d, e["R"])
            sig.add(L.signature())
            assert set(L.det_stab[L.det_round == 0]) == set(np.nonzero(L.onbasis)[0])
            assert set(L.det_stab[L.det_round == e["R"]]) == set(np.nonzero(L.onbasis)[0])
            assert L.nd == (d * d - 1) * (e["R"] - 1) + (d * d - 1)  # 2 half rounds + R-1 full
            dets, obs, _ = syc_load(e)
            g = to_grid(dets[:500], L)
            assert g.sum() == dets[:500].sum()
        assert len(sig) == 1, (d, len(sig))
        print(f"layout d={d}: all experiments share one canonical layout")


def test_dem_circuit(root, nd_tool=None, shots=400_000):
    import stim
    rng = np.random.default_rng(1)
    e = [x for x in syc_experiments(root) if x["d"] == 3 and x["R"] == 9][0]
    dem_text = open(os.path.join(e["path"], "pij_from_even_for_odd.dem")).read()
    dem = stim.DetectorErrorModel(dem_text)
    circ_text = dem_to_circuit(dem_text)
    circ = stim.Circuit(circ_text)
    assert circ.num_detectors == dem.num_detectors and circ.num_observables == dem.num_observables
    # mechanism sets agree
    def mech(dm):
        out = {}
        for inst in dm.flattened():
            if inst.type != "error":
                continue
            par = {}
            for t in inst.targets_copy():
                if not t.is_separator():
                    k = ("D" if t.is_relative_detector_id() else "L", t.val)
                    par[k] = par.get(k, 0) ^ 1
            key = tuple(sorted(k for k, v in par.items() if v))
            p = inst.args_copy()[0]
            q = out.get(key, 0.0)
            out[key] = q * (1 - p) + p * (1 - q)
        return out
    m1, m2 = mech(dem), mech(circ.detector_error_model())
    assert m1.keys() == m2.keys(), (len(m1), len(m2))
    assert max(abs(m1[k] - m2[k]) for k in m1) < 1e-9
    print(f"dem_to_circuit: {len(m1)} mechanisms reproduced exactly")
    d1, o1, _ = dem.compile_sampler(seed=5).sample(shots)
    ref = np.concatenate([d1, o1], 1).astype(np.uint8)
    d2, o2 = circ.compile_detector_sampler(seed=6).sample(shots, separate_observables=True)
    compare(ref, np.concatenate([d2, o2], 1).astype(np.uint8), rng, "stim circuit sampler vs stim DEM sampler")
    if nd_tool:
        open("/tmp/aq_test_dem.stim", "w").write(circ_text)
        nd = circ.num_detectors
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neural-decoder"))
        from nd_common import unpack_ptb64
        raw = subprocess.run([nd_tool, "stream", "/tmp/aq_test_dem.stim", "7", str(shots)], capture_output=True, check=True).stdout
        bits = unpack_ptb64(np.frombuffer(raw, dtype="<u8"), nd + 1)
        compare(ref, bits.astype(np.uint8), rng, "FastSampler (nd_tool) vs stim DEM sampler")


if __name__ == "__main__":
    root = sys.argv[1]
    test_layout(root)
    test_dem_circuit(root, sys.argv[2] if len(sys.argv) > 2 else None)
    print("ok")
