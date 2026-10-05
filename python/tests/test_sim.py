import math
import threading
import time

import numpy as np
import pytest

import qsimlab as qs
from qsimlab import Circuit, amplitudes, expectation, samples, simulate, statevector
from qsimlab.errors import (
    EngineAbortedError,
    ResourceLimitError,
    UnsupportedOperationError,
)

import reference as ref
from conftest import tv_distance

STATE_ENGINES = ["auto", "statevector", "sparse", "mps", "hsf"]


def assert_states_equal(a, b, atol=1e-9):
    np.testing.assert_allclose(a, b, atol=atol)


@pytest.mark.parametrize("engine", STATE_ENGINES)
def test_statevector_matches_numpy_reference(rng, engine):
    for trial in range(6):
        n = int(rng.integers(1, 6))
        c = ref.random_circuit(rng, n, 40)
        c.global_phase = float(rng.uniform(-3, 3))
        r = simulate(c, statevector(), engine=engine)
        assert r.state.dtype == np.complex128 and r.state.shape == (2**n,)
        assert_states_equal(r.state, ref.statevector(c))
        assert r.precision == "f64"


def test_statevector_f32():
    c = Circuit(4).h(0).cx(0, 1).rz(1, 0.3).ccx(0, 1, 3).t(2).h(2)
    r = simulate(c, statevector(), precision="f32")
    assert r.state.dtype == np.complex64 and r.precision == "f32"
    np.testing.assert_allclose(r.state, ref.statevector(c), atol=1e-6)


def test_little_endian_convention():
    psi = simulate(Circuit(3).x(0), statevector()).state
    assert abs(psi[1]) == pytest.approx(1)  # |001> = index 1: qubit 0 is the LSB
    amps = simulate(Circuit(3).x(2), amplitudes(["100", 4, "001"])).amplitudes
    np.testing.assert_allclose(np.abs(amps), [1, 1, 0])


@pytest.mark.parametrize("engine", STATE_ENGINES)
def test_amplitudes_match_reference(rng, engine):
    for trial in range(5):
        n = int(rng.integers(2, 7))
        c = ref.random_circuit(rng, n, 50)
        c.global_phase = 0.7
        xs = rng.integers(0, 2**n, size=6).tolist()
        r = simulate(c, amplitudes(xs), engine=engine)
        assert_states_equal(r.amplitudes, ref.statevector(c)[xs])


def test_amplitudes_beyond_state_vector_sizes():
    # 60-qubit GHZ: no 2^60 vector anywhere; the planner picks a cheap engine
    # (planned amplitudes are limited to components of <= 63 qubits)
    n = 60
    c = Circuit(n).h(0)
    for q in range(1, n):
        c.cx(q - 1, q)
    r = simulate(c, amplitudes([0, (1 << n) - 1, 1]), explain=True)
    np.testing.assert_allclose(r.amplitudes, [2**-0.5, 2**-0.5, 0], atol=1e-12)
    assert r.explanation is not None


def random_paulis(rng, n, k):
    out = []
    for _ in range(k):
        ops = {}
        for q in range(n):
            p = "IXYZ"[rng.integers(4)]
            if p != "I":
                ops[q] = p
        out.append(ops)
    return out


def sparse_str(ops):
    return " ".join(f"{p}{q}" for q, p in ops.items()) or "I"


@pytest.mark.parametrize("engine", ["auto", "statevector", "sparse", "mps", "hsf", "compressed"])
def test_pauli_expectations_match_reference(rng, engine):
    for trial in range(4):
        n = int(rng.integers(2, 6))
        c = ref.random_circuit(rng, n, 30)
        psi = ref.statevector(c)
        terms = random_paulis(rng, n, 6)
        r = simulate(c, expectation([sparse_str(t) for t in terms]), engine=engine)
        want = [ref.pauli_expectation(psi, n, t) for t in terms]
        np.testing.assert_allclose(r.values, want, atol=1e-9)


def test_pauli_string_forms_and_signs():
    c = Circuit(3).h(0).cx(0, 1).x(2)
    # dense strings: rightmost letter = qubit 0, so "IZZ" = Z0 Z1 and "ZII" = Z2
    v = simulate(c, expectation(["IZZ", "Z0 Z1", "Z0*Z1", "-Z2", "ZII", "+IIZ", "I", "", "X1X0"])).values
    np.testing.assert_allclose(v, [1, 1, 1, 1, -1, 0, 1, 1, 1], atol=1e-12)
    with pytest.raises(ValueError):
        simulate(c, expectation("XZ"))  # dense form must have n letters
    with pytest.raises(IndexError):
        simulate(c, expectation("X7"))


def test_clifford_expectation_on_many_qubits():
    n = 300
    c = Circuit(n).h(0)
    for q in range(1, n):
        c.cx(q - 1, q)
    v = simulate(c, expectation(["Z0 Z299", "X" * n, "Z5"]), engine="auto").values
    np.testing.assert_allclose(v, [1, 1, 0], atol=1e-12)
    v = simulate(c, expectation(["Z0 Z299"]), engine="tableau").values
    assert v[0] == pytest.approx(1)


def exact_probs(c):
    psi = ref.statevector(c)
    n = c.num_qubits
    p = np.abs(psi) ** 2
    return {format(i, f"0{n}b"): float(p[i]) for i in range(2**n) if p[i] > 1e-14}


@pytest.mark.parametrize("engine", ["auto", "statevector", "sparse", "mps", "hsf", "compressed"])
def test_sampling_distribution(rng, engine):
    shots = 20000
    for trial in range(3):
        n = int(rng.integers(2, 5))
        c = ref.random_circuit(rng, n, 25)
        probs = exact_probs(c)
        r = simulate(c.copy().measure_all(), samples(shots), engine=engine, seed=trial)
        assert r.bits.shape == (shots, n) and r.bits.dtype == np.uint8
        assert tv_distance(r.counts(), probs, shots) < 0.03


def test_sampling_without_measurements_samples_every_qubit():
    r = simulate(Circuit(3).x(1), samples(10))
    assert r.bits.shape == (10, 3) and r.measured_qubits == (0, 1, 2)
    assert r.counts() == {"010": 10}


def test_counts_key_order_and_partial_measurement():
    c = Circuit(3).x(0).x(2).measure(2, 0)  # column 0 = qubit 2, column 1 = qubit 0
    r = simulate(c, samples(4))
    assert r.measured_qubits == (2, 0)
    assert r.counts() == {"11": 4}
    c = Circuit(3).x(0).measure(0, 1)
    assert simulate(c, samples(3)).counts() == {"01": 3}
    assert simulate(c, samples(3)).counts(as_int=True) == {1: 3}


def test_seed_reproducibility():
    c = ref.random_circuit(np.random.default_rng(3), 4, 30).measure_all()
    a = simulate(c, samples(500), seed=99)
    b = simulate(c, samples(500), seed=99)
    assert a.seed == 99 and np.array_equal(a.bits, b.bits)
    d = simulate(c, samples(500))
    e = simulate(c, samples(500), seed=d.seed)
    assert np.array_equal(d.bits, e.bits)


def test_teleportation_with_conditionals():
    theta = 1.1
    c = Circuit(3).ry(0, theta)
    c.h(1).cx(1, 2).cx(0, 1).h(0).measure(0, 1)
    c.x(2, c_if=1).z(2, c_if=0).measure(2)
    for engine in ["auto", "statevector"]:
        r = simulate(c, samples(20000), engine=engine, seed=5)
        p1 = r.bits[:, 2].mean()
        assert p1 == pytest.approx(math.sin(theta / 2) ** 2, abs=0.015), engine


def test_mid_circuit_reset_and_measure():
    c = Circuit(1).x(0).measure(0).reset(0).h(0).measure(0)
    r = simulate(c, samples(4000), seed=1)
    assert (r.bits[:, 0] == 1).all()
    assert r.bits[:, 1].mean() == pytest.approx(0.5, abs=0.04)


def test_noise_channels_flip_rates():
    p = 0.2
    c = Circuit(3).x_error(0, p).z_error(1, p).depolarize1(2, p).measure_all()
    for engine in ["auto", "statevector", "tableau", "symphase"]:
        r = simulate(c, samples(40000), engine=engine, seed=2)
        rates = r.bits.mean(axis=0)
        assert rates[0] == pytest.approx(p, abs=0.012), engine
        assert rates[1] == pytest.approx(0, abs=1e-12), engine
        assert rates[2] == pytest.approx(2 * p / 3, abs=0.012), engine


def test_noise_model_and_readout_error():
    c = Circuit(2).h(0).cx(0, 1).measure_all()
    nm = qs.NoiseModel(readout=0.1)
    for engine in ["auto", "statevector", "symphase"]:
        r = simulate(c, samples(40000), noise=nm, engine=engine, seed=4)
        disagree = (r.bits[:, 0] != r.bits[:, 1]).mean()
        assert disagree == pytest.approx(2 * 0.1 * 0.9, abs=0.012), engine
    # non-Clifford circuit with gate noise: state vector shot by shot
    c2 = Circuit(1).t(0).measure(0)
    r = simulate(c2, samples(20000), noise=qs.NoiseModel(p1=0.3), seed=1)
    assert r.engine == "statevector" and r.bits.mean() == pytest.approx(0.2, abs=0.015)
    with pytest.raises(UnsupportedOperationError):
        simulate(Circuit(1).h(0), statevector(), noise=nm)


def test_symphase_auto_on_noisy_clifford():
    c = Circuit(4).h(0).cx(0, 1).depolarize2(0, 1, 0.01).cx(1, 2).cx(2, 3).measure_all()
    r = simulate(c, samples(5000), explain=True)
    assert r.engine == "symphase"
    assert any("symphase" in n for n in r.explanation.notes)


def test_requests_needing_unitary_circuits_refuse_noise():
    c = Circuit(2).h(0).measure(0).x(1, c_if=0)
    for req in [statevector(), amplitudes([0]), expectation("Z0")]:
        with pytest.raises(UnsupportedOperationError):
            simulate(c, req)
    # terminal measurements are ignored for these requests
    v = simulate(Circuit(1).x(0).measure(0), expectation("Z0")).values
    assert v[0] == pytest.approx(-1)


def test_forced_engine_limits():
    with pytest.raises(UnsupportedOperationError):
        simulate(Circuit(1).t(0), expectation("Z0"), engine="tableau")
    with pytest.raises(UnsupportedOperationError):
        simulate(Circuit(1).h(0), amplitudes([0]), engine="compressed")
    with pytest.raises(UnsupportedOperationError):
        simulate(Circuit(1).h(0).measure(0).reset(0), samples(1), engine="mps")
    with pytest.raises(ValueError):
        simulate(Circuit(1), statevector(), engine="warp-drive")
    with pytest.raises(ValueError):
        simulate(Circuit(1), statevector(), precision="f16")


def test_budget_limits():
    c = Circuit(20).h(0)
    with pytest.raises(ResourceLimitError) as e:
        simulate(c, statevector(), budget="1MB")
    assert isinstance(e.value, MemoryError)
    assert e.value.needed == 16 * 2**20 and e.value.limit == 1_000_000
    r = simulate(c, statevector(), budget=qs.Budget(memory="64MiB"))
    assert r.state.shape == (2**20,)


def test_mps_refuses_to_truncate():
    rng = np.random.default_rng(1)
    c = ref.random_circuit(rng, 12, 400, gates=["h", "t", "cx", "rx", "cz"])
    with pytest.raises(EngineAbortedError):
        simulate(c, amplitudes([0]), engine="mps", budget="100KB")


def test_explain_and_plan():
    c = Circuit(16)
    for q in range(16):
        c.h(q)
    for q in range(15):
        c.cx(q, q + 1).t(q + 1)
    e = qs.plan(c.copy().measure_all(), samples(1000))
    assert e.engine in qs.ENGINES and e.ranked
    assert all(t >= 0 for _, t in e.ranked)
    assert e.features["qubits"] == 16
    assert "planner choice" in str(e)
    r = simulate(c, expectation("Z3 Z9"), explain=True)
    assert r.explanation.engine is not None
    assert r.components and r.engine != ""


def test_threads_parameter_and_default():
    c = ref.random_circuit(np.random.default_rng(2), 10, 200)
    a = simulate(c, statevector(), threads=1).state
    b = simulate(c, statevector(), threads=2).state
    np.testing.assert_allclose(a, b, atol=1e-12)
    old = qs.get_num_threads()
    qs.set_num_threads(1)
    assert qs.get_num_threads() == 1
    qs.set_num_threads(0)
    assert qs.get_num_threads() >= 1
    del old


def test_gil_is_released():
    c = Circuit(22)
    for q in range(22):
        c.h(q)
    for layer in range(4):
        for q in range(21):
            c.cx(q, q + 1).rz(q, 0.1 * layer)
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.002)

    t = threading.Thread(target=ticker)
    t.start()
    t0 = time.perf_counter()
    simulate(c, statevector(), engine="statevector", threads=1)
    dt = time.perf_counter() - t0
    stop.set()
    t.join()
    during = [x for x in ticks if t0 < x < t0 + dt]
    # with the GIL held the ticker could not run at all during the call
    assert dt < 0.05 or len(during) >= 3, (dt, len(during))


def test_repeat_blocks_use_the_repeat_pass():
    layer = Circuit(4)
    for q in range(4):
        layer.rx(q, 0.05)
    for q in range(3):
        layer.cz(q, q + 1)
    c = Circuit(4).repeat(layer, 200)
    a = simulate(c, statevector()).state
    np.testing.assert_allclose(a, ref.statevector(c), atol=1e-9)
    v1 = simulate(c, expectation("Z0 Z3"), repeat=True).values
    v2 = simulate(c, expectation("Z0 Z3"), repeat=False).values
    np.testing.assert_allclose(v1, v2, atol=1e-9)


def test_type_errors():
    with pytest.raises(TypeError):
        simulate("not a circuit", statevector())
    with pytest.raises(TypeError):
        simulate(Circuit(1), "statevector")
    with pytest.raises(ValueError):
        samples(-1)


def test_concurrent_python_threads():
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(8)
    circuits = [ref.random_circuit(rng, 8, 120) for _ in range(8)]
    want = [ref.statevector(c) for c in circuits]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda c: simulate(c, statevector(), threads=1).state, circuits))
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=1e-9)


def test_plan_for_noisy_and_state_requests():
    noisy = Circuit(3).h(0).cx(0, 1).depolarize1(1, 0.1).measure_all()
    e = qs.plan(noisy, samples(100))
    assert e.engine == "symphase"
    e = qs.plan(Circuit(5).h(0).t(0), statevector())
    assert e.features["qubits"] == 5


def test_samples_engines_with_f32():
    c = Circuit(3).h(0).cx(0, 1).t(1).h(2).measure_all()
    r = simulate(c, samples(4000), engine="statevector", precision="f32", seed=1)
    assert r.precision == "f32"
    assert tv_distance(r.counts(), exact_probs(c), 4000) < 0.05
