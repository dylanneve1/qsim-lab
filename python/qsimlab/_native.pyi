"""Type stubs for the private PyO3 extension ``qsimlab._native``.

User code should use the public modules (``qsimlab.circuit``, ``qsimlab.sim``,
...); these stubs exist for type checkers and for phase-2 module authors.
"""

from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__version__: str
GATES: Dict[str, Tuple[int, int]]
GATE_ALIASES: Dict[str, str]
ENGINES: List[Tuple[str, str, str]]
MAX_STATE_BYTES: int

qec: ModuleType
shor: ModuleType
analysis: ModuleType

def set_num_threads(n: int) -> None: ...
def get_num_threads() -> int: ...

_Cond = Union[None, int, Tuple[int, Union[bool, int]]]

class CircuitCore:
    def __init__(self, num_qubits: int) -> None: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_ops(self) -> int: ...
    @property
    def num_measurements(self) -> int: ...
    global_phase: float
    readout_error: float
    has_repeats: bool
    @property
    def detectors(self) -> List[List[int]]: ...
    @property
    def observables(self) -> List[List[int]]: ...
    def append_gate(
        self, name: str, qubits: Sequence[int], params: Sequence[float] = ..., c_if: _Cond = ...
    ) -> None: ...
    def append_measure(self, qubits: Sequence[int]) -> int: ...
    def append_reset(self, qubits: Sequence[int]) -> None: ...
    def append_noise(self, kind: str, qubits: Sequence[int], p: float) -> None: ...
    def add_detector(self, meas: Sequence[int]) -> int: ...
    def add_observable(self, index: int, meas: Sequence[int]) -> None: ...
    def extend(
        self, other: "CircuitCore", qubits: Optional[Sequence[int]] = ..., reps: int = ...
    ) -> None: ...
    def instructions(
        self,
    ) -> List[Tuple[str, Tuple[int, ...], Tuple[float, ...], Optional[Tuple[int, bool]]]]: ...
    def stats(self) -> Dict[str, Any]: ...
    def draw(self) -> str: ...
    def to_qasm(self) -> str: ...
    @staticmethod
    def from_qasm(source: str) -> "CircuitCore": ...
    def to_stim(self) -> str: ...
    @staticmethod
    def from_stim(source: str) -> "CircuitCore": ...
    def copy(self) -> "CircuitCore": ...
    def inverse(self) -> "CircuitCore": ...
    def remove_final_measurements(self) -> "CircuitCore": ...
    def __eq__(self, other: object) -> bool: ...

def run(
    circuit: CircuitCore,
    kind: str,
    payload: Any,
    engine: str = ...,
    precision: str = ...,
    seed: Optional[int] = ...,
    memory: Union[int, str, None] = ...,
    threads: Optional[int] = ...,
    noise: Optional[Tuple[float, float, float, float]] = ...,
    repeat: Optional[bool] = ...,
    explain: bool = ...,
) -> Dict[str, Any]: ...
def plan(
    circuit: CircuitCore, kind: str, payload: Any, memory: Union[int, str, None] = ...
) -> Dict[str, Any]: ...

# ---------------------------------------------------------------------------
# qsimlab._native.qec (python/src/qec.rs; public wrapper: qsimlab.qec)

class _QecModule:
    CHUNK_SHOTS: int
    def surface_code_memory(
        self, d: int, rounds: int, basis: str = ..., p: float = ..., noise: str = ...
    ) -> Tuple[CircuitCore, Dict[str, Any]]: ...
    def repetition_code_memory(
        self, d: int, rounds: int, p: float = ..., noise: str = ...
    ) -> Tuple[CircuitCore, Dict[str, Any]]: ...
    def color_code_memory(
        self,
        d: int,
        rounds: int,
        basis: str = ...,
        schedule: Union[None, str, Sequence[Sequence[int]]] = ...,
        flags: Sequence[bool] = ...,
        p: float = ...,
        noise: str = ...,
    ) -> Tuple[CircuitCore, Dict[str, Any], List[List[int]]]: ...
    def color_code_plaquettes(
        self, d: int
    ) -> Tuple[List[Tuple[int, int, int, List[int]]], List[bool]]: ...
    def detector_error_model(
        self, circuit: CircuitCore
    ) -> Tuple[int, int, List[Tuple[float, List[int], List[int]]]]: ...
    def sample_decode_count(
        self,
        sampler: "DetectorSamplerCore",
        decoder: "BpOsdCore",
        shots: int,
        seed: int,
        max_errors: Optional[int] = ...,
        threads: Optional[int] = ...,
    ) -> Dict[str, Any]: ...
    def min_weight_logical(
        self,
        num_detectors: int,
        errors: Sequence[Tuple[float, Sequence[int], Sequence[int]]],
        observable: int = ...,
        keep: Optional[Sequence[int]] = ...,
        max_weight: int = ...,
        count_cap: int = ...,
        node_limit: Optional[int] = ...,
        timeout: Optional[float] = ...,
    ) -> Dict[str, Any]: ...
    DetectorSamplerCore: type
    BpOsdCore: type

class DetectorSamplerCore:
    def __init__(self, circuit: CircuitCore, engine: str = ...) -> None: ...
    num_detectors: int
    num_observables: int
    engine: str
    note: str
    compile_time: float
    rows: int
    def sample(
        self,
        shots: int,
        seed: int,
        packed: bool = ...,
        threads: Optional[int] = ...,
        transposed: bool = ...,
    ) -> Tuple[Any, Any, float]: ...
    def _bench(self, shots: int, packed: bool) -> Tuple[float, float]: ...

class BpOsdCore:
    def __init__(
        self,
        num_detectors: int,
        num_observables: int,
        errors: Sequence[Tuple[float, Sequence[int], Sequence[int]]],
        max_iter: int = ...,
        ms_scale: float = ...,
        osd_order: int = ...,
    ) -> None: ...
    num_detectors: int
    num_observables: int
    num_mechanisms: int
    def decode_packed(self, syndromes: Any, threads: Optional[int] = ...) -> Tuple[Any, int, int]: ...
