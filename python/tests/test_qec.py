"""Tests for qsimlab.qec against independent references (Stim, PyMatching).

Statistical tests follow research/qec/fast-sampler-audit.md: two-sample z-tests on
per-detector marginals and on DEM-correlated detector pairs, Bonferroni at a
1% family-wise error rate per circuit, plus a negative control that must be
rejected.
"""

import math
import sys
import time

import numpy as np
import pytest

import qsimlab as qs
import qsimlab.qec as qec
from qsimlab.errors import (
    MissingDependencyError,
    ParseError,
    UnsupportedOperationError,
)

# --------------------------------------------------------------------------- helpers


def _stim():
    return pytest.importorskip("stim")


def _bonferroni_z(tests: int, alpha: float = 0.01) -> float:
    """|z| threshold for a two-sided family-wise alpha over `tests` tests."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - alpha / (2 * max(tests, 1)))


def _two_sample_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z of p_a - p_b per column (pooled), for 0/1 arrays of shape (n, k)."""
    na, nb = len(a), len(b)
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    pool = (pa * na + pb * nb) / (na + nb)
    se = np.sqrt(np.maximum(pool * (1 - pool) * (1 / na + 1 / nb), 1e-300))
    z = (pa - pb) / se
    z[pool == 0] = 0.0
    return z


def _correlated_pairs(dem: qec.DetectorErrorModel, limit: int = 400):
    pairs = set()
    for _, d, _ in dem.errors:
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                pairs.add((d[i], d[j]))
    pairs = sorted(pairs)
    if len(pairs) > limit:
        idx = np.random.default_rng(0).choice(len(pairs), limit, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    return pairs


def _compare_with_stim(c: qs.Circuit, shots: int, seed: int):
    """Max |z| / threshold for marginals, pairs and events-per-shot."""
    stim = _stim()
    sc = stim.Circuit(c.to_stim())
    sd, so = sc.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    qd, qo = qec.sample_detectors(c, shots, seed=seed)
    dem = qec.detector_error_model(c)
    pairs = _correlated_pairs(dem)
    a = np.concatenate([qd, qo], axis=1).astype(np.float64)
    b = np.concatenate([sd, so], axis=1).astype(np.float64)
    zm = _two_sample_z(a, b)
    if pairs:
        i, j = np.array(pairs).T
        zp = _two_sample_z(qd[:, i] & qd[:, j], sd[:, i] & sd[:, j])
    else:
        zp = np.zeros(0)
    ea, eb = qd.sum(axis=1), sd.sum(axis=1)
    ze = (ea.mean() - eb.mean()) / math.sqrt(ea.var() / shots + eb.var() / shots + 1e-300)
    ntests = a.shape[1] + len(zp) + 1
    thr = _bonferroni_z(ntests)
    return {
        "marg": float(np.abs(zm).max()),
        "pair": float(np.abs(zp).max()) if len(zp) else 0.0,
        "events": abs(ze),
        "threshold": thr,
        "tests": ntests,
    }


# --------------------------------------------------------------------------- generators


@pytest.mark.parametrize(
    "make",
    [
        lambda: qec.surface_code_memory(3, 3, "Z"),
        lambda: qec.surface_code_memory(4, 2, "X"),
        lambda: qec.surface_code_memory(5, 1, "Z"),
        lambda: qec.repetition_code_memory(4, 3),
        lambda: qec.color_code_memory(5, 2, "X"),
        lambda: qec.color_code_memory(5, 2, "Z", flags=True),
        lambda: qec.color_code_memory(9, 1, schedule="global"),
    ],
)
def test_noiseless_memories_are_silent_and_deterministic(make):
    c = make()
    dets, obs = qec.sample_detectors(c, 256, seed=1)
    assert not dets.any() and not obs.any()
    dem = qec.detector_error_model(c)  # raises if anything is non-deterministic
    assert len(dem) == 0 and dem.num_detectors == len(c.detectors)


def test_generator_shapes_and_layouts():
    c, lay = qec.surface_code_memory(5, 4, "X", p=1e-3, return_layout=True)
    assert c.num_qubits == 49
    # 12 own-type per round, 12 other-type from round 1, 12 final
    assert len(c.detectors) == 12 * 4 + 12 * 3 + 12
    assert len(lay.memory_detectors) == 12 * 5
    assert lay.detector_coords.shape == (len(c.detectors), 3)
    assert set(lay.detector_basis) == {"X", "Z"}
    assert len(lay.data_qubits) == 25 and len(lay.ancilla_qubits) == 24
    r, rl = qec.repetition_code_memory(7, 5, return_layout=True)
    assert len(r.detectors) == 6 * 6 and rl.memory_detectors == list(range(36))
    cc, cl = qec.color_code_memory(7, 3, flags=True, return_layout=True)
    assert len(cl.flag_qubits) == 15 and cl.flag_detectors.sum() == 2 * 15 * 3
    assert cl.schedule is not None and len(cl.schedule) == 18


def test_noise_models_are_explicit_and_exportable():
    stim = _stim()
    for noise in qec.NOISE_MODELS:
        c = qec.surface_code_memory(3, 2, p=1e-3, noise=noise)
        s = c.to_stim()
        assert ("DEPOLARIZE1" in s) == (noise != "cnot")
        assert stim.Circuit(s).num_detectors == len(c.detectors)
    si = qec.surface_code_memory(3, 2, p=1e-3, noise="si1000")
    assert si.readout_error == pytest.approx(5e-3)
    ps = sorted({i.params[0] for i in si.instructions() if i.name == "depolarize1"})
    assert ps == pytest.approx([1e-4, 2e-3])
    cs = qec.color_code_memory(5, 2, p=1e-3, noise="si1000")
    ps = sorted({i.params[0] for i in cs.instructions() if i.name == "depolarize1"})
    assert ps == pytest.approx([1e-4, 2e-3])
    assert cs.readout_error == pytest.approx(5e-3)


def test_generator_argument_errors():
    with pytest.raises(ValueError):
        qec.surface_code_memory(3, 0)
    with pytest.raises(ValueError):
        qec.surface_code_memory(3, 1, "Y")
    with pytest.raises(ValueError):
        qec.surface_code_memory(3, 1, p=0.1, noise="biased")
    with pytest.raises(ValueError):
        qec.color_code_memory(4, 1)
    with pytest.raises(ValueError):
        qec.color_code_memory(5, 1, schedule="global")
    with pytest.raises(ValueError):  # collision: every plaquette uses step 1 twice
        qec.color_code_memory(3, 1, schedule=[[1, 1, 2, 3, 4, 5]] * 3)
    with pytest.raises(ParseError):
        qec._parse_schedule_text("1 2 x")


def test_schedule_file_roundtrip(tmp_path):
    rows, _ = qec.color_code_schedule(5, "kf")
    f = tmp_path / "s.sched"
    f.write_text("\n".join(" ".join(map(str, r)) + (" F" if i == 0 else "") for i, r in enumerate(rows)))
    a = qec.color_code_memory(5, 2, schedule=str(f), flags=[0])
    b = qec.color_code_memory(5, 2, schedule="kf", flags=[0])
    assert a == b
    assert a != qec.color_code_memory(5, 2)


# --------------------------------------------------------------------------- sampling


def test_seeds_threads_and_packing():
    c = qec.surface_code_memory(3, 3, p=5e-3)
    s = qec.DetectorSampler(c)
    a, ao = s.sample(5000, seed=11, threads=1)
    b, bo = s.sample(5000, seed=11, threads=3)
    assert (a == b).all() and (ao == bo).all()
    pa, po = s.sample(5000, seed=11, packed=True)
    assert pa.dtype == np.uint8 and pa.shape == (5000, 3)
    assert (np.unpackbits(pa, axis=1, count=24, bitorder="little").astype(bool) == a).all()
    assert (np.unpackbits(po, axis=1, count=1, bitorder="little").astype(bool) == ao).all()
    ta, to = s.sample(5000, seed=11, transposed=True, threads=2)
    assert ta.shape == (24, 625) and to.shape == (1, 625)
    assert (np.unpackbits(ta, axis=1, count=5000, bitorder="little").astype(bool) == a.T).all()
    assert (np.unpackbits(to, axis=1, count=5000, bitorder="little").astype(bool) == ao.T).all()
    tb, _ = s.sample(1003, seed=11, transposed=True)  # padding bits are zero
    assert not (np.unpackbits(tb, axis=1, bitorder="little")[:, 1003:]).any()
    c2, _ = s.sample(5000, seed=12)
    assert (c2 != a).any()
    assert s.sample(0, seed=1)[0].shape == (0, 24)


def test_packed_layout_matches_stim():
    stim = _stim()
    c = qec.repetition_code_memory(9, 2, p=0.05)
    d, _ = qec.sample_detectors(c, 300, seed=4)
    ours = np.packbits(d, axis=1, bitorder="little")
    sp = qec.sample_detectors(c, 300, seed=4, packed=True)[0]
    assert (ours == sp).all()
    # Stim's own bit_packed: same shape convention
    st = stim.Circuit(c.to_stim()).compile_detector_sampler().sample(3, bit_packed=True)
    assert st.shape[1] == sp.shape[1]


def test_rejects_non_clifford():
    c = qs.Circuit(2).h(0).t(0).cx(0, 1).measure_all()
    c.detector([0])
    with pytest.raises(UnsupportedOperationError):
        qec.sample_detectors(c, 10)
    with pytest.raises(UnsupportedOperationError):
        qec.detector_error_model(c)


def test_engines_agree_in_distribution():
    c = qec.color_code_memory(5, 2, p=4e-3, noise="uniform")
    n = 40_000
    a, ao = qec.DetectorSampler(c, "fast").sample(n, seed=1)
    b, bo = qec.DetectorSampler(c, "symphase").sample(n, seed=2)
    z = _two_sample_z(np.c_[a, ao].astype(float), np.c_[b, bo].astype(float))
    assert np.abs(z).max() < _bonferroni_z(z.size)
    assert qec.DetectorSampler(c, "symphase").engine == "symphase"
    with pytest.raises(ValueError):
        qec.DetectorSampler(c, "gpu")


CIRCUITS = {
    "surface3_Z_uniform": lambda: qec.surface_code_memory(3, 3, "Z", p=4e-3, noise="uniform"),
    "surface5_X_si1000": lambda: qec.surface_code_memory(5, 3, "X", p=2e-3, noise="si1000"),
    "repetition7_cnot": lambda: qec.repetition_code_memory(7, 4, p=0.02, noise="cnot"),
    "color5_kf_cnot": lambda: qec.color_code_memory(5, 3, p=4e-3, noise="cnot"),
    "color5_flags_uniform_X": lambda: qec.color_code_memory(
        5, 2, "X", flags=True, p=3e-3, noise="uniform"
    ),
}


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_detector_statistics_match_stim(name):
    c = CIRCUITS[name]()
    r = _compare_with_stim(c, 200_000, seed=17)
    thr = r["threshold"]
    assert r["marg"] < thr and r["pair"] < thr and r["events"] < thr, r


def test_negative_control_is_rejected():
    """Stim samples a circuit with 25% stronger noise: the test must notice."""
    stim = _stim()
    c = qec.surface_code_memory(3, 3, p=4e-3)
    hot = qec.surface_code_memory(3, 3, p=5e-3)
    n = 200_000
    qd, _ = qec.sample_detectors(c, n, seed=3)
    sd, _ = stim.Circuit(hot.to_stim()).compile_detector_sampler(seed=3).sample(
        n, separate_observables=True)
    ea, eb = qd.sum(axis=1), sd.sum(axis=1)
    ze = (ea.mean() - eb.mean()) / math.sqrt(ea.var() / n + eb.var() / n)
    assert abs(ze) > 10


# --------------------------------------------------------------------------- DEM


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_dem_equals_stim(name):
    stim = _stim()
    c = CIRCUITS[name]()
    ours = qec.detector_error_model(c)
    theirs = qec.DetectorErrorModel.from_stim(stim.Circuit(c.to_stim()).detector_error_model())
    assert ours.num_detectors == theirs.num_detectors
    assert set(ours.merged()) == set(theirs.merged())
    assert ours.approx_equal(theirs, rtol=1e-9, atol=1e-15)
    # our text parses in Stim and back to the same model
    back = qec.DetectorErrorModel.from_stim(ours.to_stim())
    assert back.approx_equal(ours, rtol=1e-12)


def test_dem_text_roundtrip_and_parser():
    text = """
        # comment
        error(0.125) D0 D1 ^ D1 D2 L0
        repeat 2 {
            error(0.01) D0 L1
            shift_detectors(0, 0, 1) 1
        }
        detector(1, 2, 3) D5
        logical_observable L2
    """
    dem = qec.DetectorErrorModel.from_stim_dem(text)
    assert dem.num_detectors == 8 and dem.num_observables == 3
    assert dem.errors[0] == qec.DemError(0.125, (0, 2), (0,))
    assert dem.errors[1].detectors == (0,) and dem.errors[2].detectors == (1,)
    again = qec.DetectorErrorModel.from_stim_dem(dem.to_stim_dem())
    assert again == dem
    H, L, p = dem.matrices()
    assert H.shape == (8, 3) and L.shape == (3, 3) and p[0] == 0.125
    with pytest.raises(ParseError):
        qec.DetectorErrorModel.from_stim_dem("error(0.1) X3")
    with pytest.raises(ParseError):
        qec.DetectorErrorModel.from_stim_dem("frobnicate D0")
    with pytest.raises(ValueError):
        qec.DetectorErrorModel(2, 1, [(0.1, [5], [])])


def test_dem_rejects_nondeterministic_detector():
    c = qs.Circuit(1).h(0).measure(0)
    c.detector([0])
    with pytest.raises(UnsupportedOperationError, match="not deterministic"):
        qec.detector_error_model(c)


# --------------------------------------------------------------------------- distance


def _stim_distance(c: qs.Circuit) -> int:
    stim = _stim()
    errs = stim.Circuit(c.to_stim()).search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    return len(errs)


@pytest.mark.parametrize(
    "make,expected",
    [
        (lambda: qec.surface_code_memory(3, 2, "Z", p=1e-3), 3),
        (lambda: qec.surface_code_memory(3, 3, "X", p=1e-3, noise="si1000"), 3),
        (lambda: qec.surface_code_memory(5, 2, "Z", p=1e-3), 5),
        (lambda: qec.surface_code_memory(5, 2, "X", p=1e-3, noise="cnot"), 5),
        (lambda: qec.repetition_code_memory(5, 3, p=1e-2), 5),
        (lambda: qec.color_code_memory(3, 2, p=1e-3), None),
        (lambda: qec.color_code_memory(5, 2, p=1e-3), None),
    ],
)
def test_distance_matches_stim(make, expected):
    c = make()
    r = qec.circuit_distance(c, timeout=120)
    assert r.complete and r.distance is not None
    if expected is not None:
        assert r.distance == expected
    assert r.distance == _stim_distance(c)
    # the example is a genuine undetectable logical of the model
    dets, obs = set(), set()
    for e in r.example:
        dets ^= set(e.detectors)
        obs ^= set(e.observables)
    assert not dets and 0 in obs and len(r.example) == r.distance
    assert r.count >= 1


def test_distance_sector_and_budgets():
    c, lay = qec.color_code_memory(7, 2, p=1e-3, return_layout=True)
    r = qec.circuit_distance(c, detectors=lay.memory_detectors)
    assert r.certified and r.complete and r.distance == 6  # K-F d = 7: d_circ = 6
    t0 = time.perf_counter()
    stopped = qec.circuit_distance(c, timeout=2.0)  # the full-DEM search takes minutes
    assert time.perf_counter() - t0 < 30
    assert stopped.timed_out and not stopped.complete and stopped.distance is None
    assert 2 <= stopped.lower_bound <= 6
    capped = qec.circuit_distance(c, max_weight=3)
    assert capped.distance is None and capped.complete and capped.lower_bound == 4
    lim = qec.circuit_distance(c, node_limit=10)
    assert lim.distance is None and lim.node_limit_hit and not lim.complete
    t = qec.circuit_distance(qec.surface_code_memory(7, 3, p=1e-3), timeout=0.0)
    assert t.timed_out and t.distance is None and t.lower_bound >= 1
    with pytest.raises(ValueError, match="no mechanisms"):
        qec.circuit_distance(qec.surface_code_memory(3, 1))


def test_global_schedule_raises_distance():
    """research/qec/colour-global.md: d = 9 K-F has d_circ 7, the global schedule 8."""
    kf, lk = qec.color_code_memory(9, 1, schedule="kf", p=1e-3, return_layout=True)
    gl, lg = qec.color_code_memory(9, 1, schedule="global", p=1e-3, return_layout=True)
    a = qec.circuit_distance(kf, detectors=lk.memory_detectors)
    b = qec.circuit_distance(gl, detectors=lg.memory_detectors)
    assert (a.distance, b.distance) == (7, 8) and a.certified and b.certified


# --------------------------------------------------------------------------- decoding


def test_decode_inputs_and_bposd():
    c = qec.surface_code_memory(3, 3, p=3e-3)
    dem = qec.detector_error_model(c)
    dets, obs = qec.sample_detectors(c, 4000, seed=9)
    dec = qec.make_decoder(dem, "bposd", osd_order=4)
    pa = dec.decode(dets)
    pb = dec.decode(np.packbits(dets, axis=1, bitorder="little"))
    pc = qec.decode(dem, dets.astype(np.uint8), "bposd", osd_order=4, threads=1)
    assert (pa == pb).all() and (pa == pc).all() and pa.shape == obs.shape
    assert (pa != obs).any(axis=1).mean() < 0.02
    with pytest.raises(ValueError):
        dec.decode(dets[:, :5])
    with pytest.raises(ValueError):
        qec.make_decoder(dem, "magic")


def test_logical_error_rate_reproducible_and_early_stop():
    c = qec.surface_code_memory(3, 3, p=6e-3)
    a = qec.logical_error_rate(c, 30_000, seed=5, threads=1, rounds=3)
    b = qec.logical_error_rate(c, 30_000, seed=5, threads=4, rounds=3)
    assert (a.errors, a.shots) == (b.errors, b.shots)
    assert a.ci[0] <= a.rate <= a.ci[1] and a.per_round < a.rate
    assert isinstance(a, qs.sim.Result) and a.engine == "fast" and a.seed == 5
    e = qec.logical_error_rate(c, 10_000_000, seed=5, max_errors=50)
    assert 50 <= e.errors and e.shots < 10_000_000 and e.shots % 65_536 == 0


def test_missing_optional_decoders(monkeypatch):
    dem = qec.detector_error_model(qec.repetition_code_memory(3, 1, p=0.01))
    monkeypatch.setitem(sys.modules, "pymatching", None)
    with pytest.raises(MissingDependencyError):
        qec.make_decoder(dem, "pymatching")
    monkeypatch.setitem(sys.modules, "tesseract_decoder", None)
    with pytest.raises(MissingDependencyError):
        qec.make_decoder(dem, "tesseract")


@pytest.mark.parametrize("d", [3, 5])
def test_pymatching_ler_sanity(d):
    """BP+OSD and PyMatching (on our graphlike decomposition) and PyMatching on
    Stim's own decomposed DEM give compatible logical error rates."""
    pymatching = pytest.importorskip("pymatching")
    stim = _stim()
    p = 3e-3
    c = qec.surface_code_memory(d, d, p=p)
    n = 100_000 if d == 3 else 20_000  # BP+OSD needs OSD on ~half the d = 5 shots
    dets, obs = qec.sample_detectors(c, n, seed=21)
    dem = qec.detector_error_model(c)
    ours_pm = qec.PyMatchingDecoder(dem)
    assert ours_pm.dropped == 0
    f_pm = int((ours_pm.decode(dets) != obs).any(axis=1).sum())
    sdem = stim.Circuit(c.to_stim()).detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(sdem)
    f_ref = int((m.decode_batch(dets.astype(np.uint8)) != obs).any(axis=1).sum())
    f_bp = int((qec.decode(dem, dets, "bposd") != obs).any(axis=1).sum())
    # same samples: the two matchings differ only in tie-breaking and weights
    assert abs(f_pm - f_ref) <= 4 * math.sqrt(f_ref + 1) + 5, (f_pm, f_ref)
    # BP+OSD within a factor 2 of matching at this distance
    assert f_bp <= 2 * f_ref + 10 and f_ref <= 2 * f_bp + 10, (f_bp, f_ref)


def test_ler_falls_with_distance():
    pytest.importorskip("pymatching")
    p = 2e-3
    r3 = qec.logical_error_rate(qec.surface_code_memory(3, 3, p=p), 100_000, seed=1,
                                decoder="pymatching")
    r5 = qec.logical_error_rate(qec.surface_code_memory(5, 5, p=p), 100_000, seed=1,
                                decoder="pymatching")
    assert r5.rate < r3.rate and r5.ci[1] < r3.ci[0]


def test_tesseract_adapter():
    pytest.importorskip("tesseract_decoder")
    _stim()
    c = qec.repetition_code_memory(5, 3, p=0.01)
    dets, obs = qec.sample_detectors(c, 500, seed=2)
    pred = qec.decode(qec.detector_error_model(c), dets, "tesseract")
    assert pred.shape == obs.shape and (pred != obs).any(axis=1).mean() < 0.05
