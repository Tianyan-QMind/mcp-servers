"""MCP tools for cqlib quantum information: simulation, Pauli, Hamiltonian, metrics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal, cast

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from tianyan_mcp.circuit_tools import (
    circuit_tool_errors,
    dump_circuit,
    load_circuit,
    render_png_base64,
)
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.qinfo_models import (
    ClassicalData,
    ClassicalRuntimeValue,
    HamiltonianResult,
    HamiltonianTerm,
    PauliAnalysisResult,
    PlotStateResult,
    SimulationResult,
    StateMetricsResult,
)

LOGGER = logging.getLogger(__name__)
CircuitText = Annotated[str, Field(min_length=1, max_length=20_000)]
PauliStr = Annotated[str, Field(min_length=1, max_length=128)]
QubitIndex = Annotated[int, Field(strict=True, ge=0, le=127)]
MaxQubits = Annotated[int, Field(strict=True, ge=1, le=20)]
Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
BlochVector = Annotated[list[float], Field(min_length=3, max_length=3)]
StatePlotKind = Literal["bloch_vector", "bloch_multivector", "city", "paulivec"]
StateSimulator = Literal["statevector", "density_matrix"]
NoiseErrors = Annotated[list[dict[str, object]] | None, Field(default=None, max_length=64)]
HamiltonianTerms = Annotated[list[dict[str, object]], Field(min_length=1, max_length=256)]
AdditionalHamiltonians = Annotated[
    list[HamiltonianTerms] | None, Field(default=None, max_length=16)
]
TrotterSteps = Annotated[int, Field(strict=True, ge=1, le=100_000)]
Measurements = Annotated[list[dict[str, object]] | None, Field(default=None, max_length=256)]
StateArray = Annotated[
    list[list[float]] | list[list[list[float]]] | None,
    Field(default=None, max_length=2**20),
]
StateMetric = Literal[
    "purity",
    "fidelity",
    "trace_distance",
    "entropy",
    "linear_entropy",
    "renyi_entropy",
    "entanglement_entropy",
    "negativity",
    "logarithmic_negativity",
    "concurrence",
    "entanglement_of_formation",
    "partial_transpose",
]
MetricSubsystem = Annotated[list[QubitIndex] | None, Field(default=None, max_length=20)]
ReadoutErrors = Annotated[list[dict[str, object]] | None, Field(default=None, max_length=64)]
ReadoutQubits = Annotated[list[QubitIndex] | None, Field(default=None, max_length=20)]
RuntimeKind = Literal["bit", "bool", "uint", "bit_vec"]
RuntimeValue = bool | int | str | None


def _probs_to_dict(probs: list[float], num_qubits: int) -> dict[str, float]:
    result: dict[str, float] = {}
    width = num_qubits
    for idx, p in enumerate(probs):
        if abs(p) > 1e-15:
            result[format(idx, f"0{width}b")] = float(p)
    return result


def _parse_hamiltonian_terms(
    raw_terms: list[dict[str, object]],
) -> list[tuple[Any, complex]]:
    from cqlib.qis import PauliString

    parsed: list[tuple[Any, complex]] = []
    for term in raw_terms:
        pauli = PauliString.from_str(str(term["pauli"]))
        coefficient = complex(term["coefficient"])  # type: ignore[arg-type]
        parsed.append((pauli, coefficient))
    return parsed


def _complex_matrix_to_json(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    return [[{"real": float(c.real), "imag": float(c.imag)} for c in row] for row in matrix]


def _complex_list_to_json(arr: np.ndarray) -> list[dict[str, float]]:
    return [{"real": float(c.real), "imag": float(c.imag)} for c in arr]


def _classical_type_kind(ty: Any) -> RuntimeKind:
    name = str(ty)
    if "bit_vec" in name:
        return "bit_vec"
    if "uint" in name:
        return "uint"
    if "bool" in name:
        return "bool"
    return "bit"


def _runtime_value_json(runtime: Any, ty: Any) -> tuple[RuntimeKind, RuntimeValue]:
    if runtime is None:
        return _classical_type_kind(ty), None
    kind = cast(RuntimeKind, str(runtime.kind))
    if kind == "bit":
        return kind, bool(runtime.as_bit())
    if kind == "bool":
        return kind, bool(runtime.as_bool())
    if kind == "uint":
        return kind, int(runtime.as_uint())
    return kind, str(runtime.to_bitstring())


def _extract_classical_data(circuit: Any, classical: Any) -> ClassicalData:
    """Read all immutable measurement values and mutable vars out of a run state."""
    from cqlib.circuit import ClassicalValue, ClassicalVar

    measure_qubits = [
        [int(q.index) for q in op.qubits]
        for op in circuit.operations
        if op.name in ("measure_bit", "measure_bits")
    ]

    values: list[ClassicalRuntimeValue] = []
    for idx, ty in enumerate(circuit.classical_values):
        kind, value = _runtime_value_json(classical.value(ClassicalValue(circuit.id, idx, ty)), ty)
        qubits = measure_qubits[idx] if idx < len(measure_qubits) else None
        values.append(ClassicalRuntimeValue(index=idx, kind=kind, value=value, qubits=qubits))

    variables: list[ClassicalRuntimeValue] = []
    for idx, ty in enumerate(circuit.classical_vars):
        kind, value = _runtime_value_json(classical.var(ClassicalVar(circuit.id, idx, ty)), ty)
        variables.append(ClassicalRuntimeValue(index=idx, kind=kind, value=value))

    return ClassicalData(values=values, vars=variables)


def _required_qubits(circuit: Any) -> int:
    """Highest qubit index touched by the circuit, plus one (0 for an empty circuit)."""
    max_index = -1
    for op in circuit.operations:
        for qubit in op.qubits:
            max_index = max(max_index, int(qubit.index))
    return max_index + 1


def _readout_error_spec(err: dict[str, object], num_qubits: int) -> tuple[int, float, float]:
    qubit = err.get("qubit")
    p_0_given_1 = err.get("p_0_given_1")
    p_1_given_0 = err.get("p_1_given_0")
    if not isinstance(qubit, int) or isinstance(qubit, bool) or not 0 <= qubit <= 127:
        raise ServiceError(
            "INVALID_NOISE",
            "readout_errors 每项需要整数 qubit（0-127）",
            False,
        )
    probabilities = (("p_0_given_1", p_0_given_1), ("p_1_given_0", p_1_given_0))
    for label, prob in probabilities:
        if (
            not isinstance(prob, (int, float))
            or isinstance(prob, bool)
            or not 0.0 <= float(prob) <= 1.0
        ):
            raise ServiceError(
                "INVALID_NOISE",
                f"readout_errors 的 {label} 必须是 [0, 1] 内的数值",
                False,
            )
    if qubit >= num_qubits:
        raise ServiceError(
            "INVALID_NOISE",
            f"readout_errors 的 qubit={qubit} 超出线路比特数 {num_qubits}",
            False,
        )
    return qubit, float(p_0_given_1), float(p_1_given_0)  # type: ignore[arg-type]


def render_state_plot(
    plot_fn: Callable[..., str],
    kind: StatePlotKind,
    target: Any,
    image_format: Literal["svg", "png"],
) -> PlotStateResult:
    """Render a cqlib state-plot call as an inline SVG string or base64 PNG."""
    if image_format == "png":
        content = render_png_base64(lambda path: plot_fn(target, output_path=path))
        return PlotStateResult(
            kind=kind,
            image_format="png",
            content=content,
            encoding="base64",
            media_type="image/png",
        )
    return PlotStateResult(
        kind=kind,
        image_format="svg",
        content=str(plot_fn(target)),
        encoding="utf-8",
        media_type="image/svg+xml",
    )


def register_quantum_info_tools(mcp: FastMCP) -> None:
    async def simulate_circuit(
        circuit: CircuitText,
        simulator: Literal[
            "statevector", "density_matrix", "density_matrix_noise", "stabilizer"
        ] = "statevector",
        format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        noise_errors: NoiseErrors = None,
        readout_errors: ReadoutErrors = None,
        readout_qubits: ReadoutQubits = None,
        additional_circuit: CircuitText | None = None,
        additional_format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        visualize: Literal["none", "bloch", "city", "paulivec"] = "none",
        visualize_format: Literal["svg", "png"] = "svg",
    ) -> SimulationResult:
        """Simulate a circuit; returns state data, probabilities, optional visualization.

        simulator: statevector / density_matrix / density_matrix_noise / stabilizer
        (Clifford only; classical_data has per-shot measurement values).
        noise_errors: [{gate, qubit, noise_type, probability}], noise_type in
        bit_flip|phase_flip|depolarizing|amplitude_damping|phase_damping.
        readout_errors: [{qubit, p_0_given_1, p_1_given_0}]; readout_qubits defaults
        to qubits with readout errors. Noise args require density_matrix_noise.
        additional_circuit is applied afterwards (stepwise simulation; appended
        stabilizer measurements are not in classical_data).
        visualize=bloch|city|paulivec renders statevector/density_matrix states;
        visualize_format=png returns base64 in state_png_base64.
        """
        with circuit_tool_errors():
            from cqlib import NoiseModel, ReadoutError, SingleQubitNoise, StandardGate
            from cqlib.qis import (
                DensityMatrix,
                DensityMatrixNoise,
                StabilizerState,
                Statevector,
            )

            if simulator != "density_matrix_noise" and (
                noise_errors or readout_errors or readout_qubits is not None
            ):
                raise ServiceError(
                    "INVALID_NOISE",
                    "噪声/读出参数仅支持 simulator=density_matrix_noise",
                    False,
                )

            parsed = load_circuit(circuit, format)
            num_qubits = parsed.num_qubits

            appended = None
            if additional_circuit is not None:
                appended = load_circuit(additional_circuit, additional_format)
                required_qubits = _required_qubits(appended)
                if required_qubits > num_qubits:
                    raise ServiceError(
                        "INVALID_CIRCUIT",
                        f"追加线路需要 {required_qubits} 个比特，超过主线路的 {num_qubits} 个",
                        False,
                    )

            sv: Statevector | None = None
            dm: DensityMatrix | None = None
            sim: DensityMatrixNoise | None = None
            state: Any = None
            stab_classical: Any = None
            configured_readout_qubits: set[int] = set()
            state_svg = None
            state_png_base64: str | None = None
            state_vector = None
            density_matrix_data = None
            readout_probabilities = None
            classical_data = None

            if simulator == "statevector":
                sv = Statevector.from_circuit(parsed)
                state = sv
            elif simulator == "density_matrix":
                dm = DensityMatrix.from_circuit(parsed)
                state = dm
            elif simulator == "density_matrix_noise":
                noise_model = NoiseModel()
                if noise_errors:
                    for err in noise_errors:
                        gate_name = str(err.get("gate", "X"))
                        qubit = int(err.get("qubit", 0))  # type: ignore[arg-type]
                        noise_type = str(err.get("noise_type", "bit_flip"))
                        prob = float(err.get("probability", 0.0))  # type: ignore[arg-type]
                        gate = getattr(StandardGate, gate_name)
                        if noise_type == "bit_flip":
                            noise = SingleQubitNoise.bit_flip(prob)
                        elif noise_type == "phase_flip":
                            noise = SingleQubitNoise.phase_flip(prob)
                        elif noise_type == "depolarizing":
                            noise = SingleQubitNoise.depolarizing(prob)
                        elif noise_type == "amplitude_damping":
                            noise = SingleQubitNoise.amplitude_damping(prob)
                        elif noise_type == "phase_damping":
                            noise = SingleQubitNoise.phase_damping(prob)
                        else:
                            raise ServiceError(
                                "INVALID_NOISE",
                                f"不支持的噪声类型: {noise_type}",
                                False,
                            )
                        noise_model.add_single_qubit_error(gate, qubit, noise)
                if readout_errors:
                    for err in readout_errors:
                        qubit, p_0_given_1, p_1_given_0 = _readout_error_spec(err, num_qubits)
                        noise_model.add_readout_error(
                            qubit,
                            ReadoutError(p_0_given_1=p_0_given_1, p_1_given_0=p_1_given_0),
                        )
                        configured_readout_qubits.add(qubit)
                sim = DensityMatrixNoise.from_circuit(parsed, noise_model)
                state = sim
            else:
                stab_result = StabilizerState.run_circuit(parsed)
                # ``stab_result.state`` returns a fresh copy on each access, so
                # keep one reference for both apply_circuit and probability reads.
                state = stab_result.state
                stab_classical = stab_result.classical

            if appended is not None:
                try:
                    state.apply_circuit(appended)
                except Exception as exc:
                    raise ServiceError(
                        "INVALID_CIRCUIT",
                        f"追加线路无法作用到当前量子态: {exc}",
                        False,
                    ) from None

            if simulator == "statevector":
                probs = list(state.probabilities())
                state_vector = _complex_list_to_json(np.asarray(state.data))
            elif simulator == "density_matrix":
                probs = list(state.probabilities())
                density_matrix_data = _complex_matrix_to_json(np.asarray(state.data))
            elif simulator == "density_matrix_noise":
                probs = list(state.probabilities())
                density_matrix_data = _complex_matrix_to_json(np.asarray(state.state))
                effective_readout_qubits = (
                    list(readout_qubits)
                    if readout_qubits is not None
                    else sorted(configured_readout_qubits)
                )
                for qubit in effective_readout_qubits:
                    if qubit >= num_qubits:
                        raise ServiceError(
                            "INVALID_NOISE",
                            f"readout_qubits 的 qubit={qubit} 超出线路比特数 {num_qubits}",
                            False,
                        )
                if effective_readout_qubits:
                    readout_probs = list(state.probabilities_with_readout(effective_readout_qubits))
                    readout_probabilities = _probs_to_dict(readout_probs, num_qubits)
            else:
                probs = list(state.probabilities())
                classical_data = _extract_classical_data(parsed, stab_classical)

            viz_state: Statevector | DensityMatrix | None = (
                state if simulator in ("statevector", "density_matrix") else None
            )

            if visualize != "none" and viz_state is not None:
                from cqlib.visualization import (
                    plot_bloch_multivector,
                    plot_state_city,
                    plot_state_paulivec,
                )

                if visualize == "bloch":
                    plot_fn = plot_bloch_multivector
                elif visualize == "city":
                    plot_fn = plot_state_city
                else:
                    plot_fn = plot_state_paulivec

                if visualize_format == "png":
                    state_png_base64 = render_png_base64(
                        lambda path: plot_fn(viz_state, output_path=path)
                    )
                else:
                    state_svg = str(plot_fn(viz_state))

            return SimulationResult(
                simulator=simulator,
                num_qubits=num_qubits,
                probabilities=_probs_to_dict(probs, num_qubits),
                state_vector=state_vector,
                density_matrix=density_matrix_data,
                readout_probabilities=readout_probabilities,
                classical_data=classical_data,
                state_svg=state_svg,
                state_png_base64=state_png_base64,
            )

    async def plot_bloch_vector(
        vector: BlochVector,
        image_format: Literal["svg", "png"] = "svg",
    ) -> PlotStateResult:
        """Plot one Bloch vector as an SVG string or base64-encoded PNG.

        vector is [x, y, z] Cartesian coordinates on the Bloch sphere; no circuit or
        simulation is involved. image_format=svg returns the SVG inline;
        image_format=png returns base64-encoded PNG, staged in a temporary file that
        is deleted immediately afterwards so no file is kept on the server.
        """
        with circuit_tool_errors():
            from cqlib.visualization import plot_bloch_vector as cqlib_plot_bloch_vector

            return render_state_plot(
                cqlib_plot_bloch_vector, "bloch_vector", list(vector), image_format
            )

    async def plot_state(
        kind: StatePlotKind,
        circuit: CircuitText | None = None,
        format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        vector: BlochVector | None = None,
        simulator: StateSimulator = "statevector",
        image_format: Literal["svg", "png"] = "svg",
    ) -> PlotStateResult:
        """Plot a quantum state using any cqlib state-visualization function.

        kind=bloch_vector plots one Cartesian [x, y, z] Bloch vector and needs only
        vector. kind=bloch_multivector (one Bloch sphere per qubit), kind=city
        (density-matrix real and imaginary parts), and kind=paulivec (Pauli-basis
        expectation values) need circuit, which is simulated with simulator first.
        image_format=svg returns the SVG string inline; image_format=png returns
        base64-encoded PNG. PNG is staged in a temporary file and deleted immediately
        afterwards; no files are kept on the server.
        """
        with circuit_tool_errors():
            if kind == "bloch_vector":
                if vector is None:
                    raise ServiceError(
                        "MISSING_VECTOR", "bloch_vector 需要 vector=[x, y, z]", False
                    )
                from cqlib.visualization import (
                    plot_bloch_vector as cqlib_plot_bloch_vector,
                )

                return render_state_plot(cqlib_plot_bloch_vector, kind, list(vector), image_format)

            if circuit is None:
                raise ServiceError("MISSING_CIRCUIT", f"{kind} 需要 circuit 与 format", False)

            from cqlib.qis import DensityMatrix, Statevector
            from cqlib.visualization import (
                plot_bloch_multivector,
                plot_state_city,
                plot_state_paulivec,
            )

            parsed = load_circuit(circuit, format)
            state = (
                Statevector.from_circuit(parsed)
                if simulator == "statevector"
                else DensityMatrix.from_circuit(parsed)
            )
            if kind == "bloch_multivector":
                return render_state_plot(plot_bloch_multivector, kind, state, image_format)
            if kind == "city":
                return render_state_plot(plot_state_city, kind, state, image_format)
            return render_state_plot(plot_state_paulivec, kind, state, image_format)

    async def analyze_pauli_string(
        pauli: PauliStr,
        operation: Literal["info", "matrix", "multiply", "commutes"] = "info",
        other: PauliStr | None = None,
    ) -> PauliAnalysisResult:
        """Inspect, multiply, or check commutation of Pauli strings.

        info: returns the canonical string, phase, support (non-identity qubits).
        matrix: returns the 2^n x 2^n complex matrix (only for small n).
        multiply: returns pauli * other as a new Pauli string (must match qubit count).
        commutes: returns whether pauli and other commute.
        Pauli string syntax: big-endian, e.g. \"XYZ\" has X on qubit 2, Z on qubit 0.
        Optional phase prefix: \"+ZZ\", \"-iZZ\".
        """
        with circuit_tool_errors():
            from cqlib.qis import PauliString

            ps = PauliString.from_str(pauli)
            matrix = None
            commutes = None
            product = None
            if operation == "matrix":
                matrix = _complex_matrix_to_json(np.asarray(ps.to_matrix()))
            elif operation in {"multiply", "commutes"}:
                if other is None:
                    raise ServiceError(
                        "MISSING_OTHER",
                        "multiply/commutes 操作需要 other 参数",
                        False,
                    )
                ps2 = PauliString.from_str(other)
                if operation == "multiply":
                    result = ps * ps2
                    product = str(result)
                else:
                    commutes = bool(ps.commutes_with(ps2))
            return PauliAnalysisResult(
                pauli_string=str(ps),
                phase=str(ps.phase),
                support=list(ps.support()),
                matrix=matrix,
                commutes_with=commutes,
                product=product,
            )

    async def analyze_hamiltonian(
        terms: HamiltonianTerms,
        operation: Literal[
            "info",
            "matrix",
            "expectation",
            "expectation_statevector",
            "expectation_density_matrix",
            "variance",
            "trotter_circuit",
        ] = "info",
        measurements: Measurements = None,
        circuit: CircuitText | None = None,
        circuit_format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        simplify: bool = True,
        additional_terms: AdditionalHamiltonians = None,
        scale: float | None = None,
        trotter_time: float = 1.0,
        trotter_steps: TrotterSteps = 1,
        trotter_mode: Literal["first_order", "second_order", "randomized"] = "first_order",
        random_seed: int | None = None,
        output_format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
    ) -> HamiltonianResult:
        """Build a Hamiltonian from (pauli, coefficient) terms and analyze it.

        terms is a list of {pauli: str, coefficient: float}. additional_terms is a
        list of extra Hamiltonians (each itself a terms list) added via H + Hk; all
        Hamiltonians must have the same qubit count. scale multiplies every term by
        a real factor before simplification. operation=info returns the term list
        and commutation status. matrix returns the full 2^n x 2^n matrix (small n
        only). expectation computes <H> from measurement probabilities:
        measurements is a list of {basis: pauli_string, probs: {bitstring: probability}};
        bit strings are little-endian (rightmost char = qubit 0).
        expectation_statevector simulates `circuit` into a pure statevector and
        computes <H>; expectation_density_matrix simulates it into a density
        matrix (e.g. mixed states) and computes <H>. variance computes Var(H)
        from a simulated statevector; non-Hermitian observables raise an error.
        trotter_circuit builds the Trotter-Suzuki circuit for e^(-i H
        trotter_time) using trotter_steps steps; trotter_mode=randomized requires
        random_seed, and the circuit is serialized in output_format.
        """
        with circuit_tool_errors():
            from cqlib.qis import (
                DensityMatrix,
                Hamiltonian,
                PauliString,
                Statevector,
                TrotterMode,
            )

            h = Hamiltonian.from_list(_parse_hamiltonian_terms(terms))
            if additional_terms is not None:
                for raw_extra in additional_terms:
                    extra = Hamiltonian.from_list(_parse_hamiltonian_terms(raw_extra))
                    try:
                        h = h + extra
                    except ValueError as exc:
                        raise ServiceError(
                            "HAMILTONIAN_MISMATCH",
                            f"哈密顿量相加失败（比特数必须一致）: {exc}",
                            False,
                        ) from None

            if simplify:
                h.simplify()
            if scale is not None:
                h.scale(scale)

            matrix = None
            expectation = None
            variance = None
            trotter_circuit = None
            if operation == "matrix":
                matrix = _complex_matrix_to_json(np.asarray(h.to_matrix()))
            elif operation == "expectation":
                if measurements is None:
                    raise ServiceError(
                        "MISSING_MEASUREMENTS",
                        "expectation 需要 measurements 参数",
                        False,
                    )
                meas_list: list[tuple[PauliString, dict[str, float]]] = []
                for m in measurements:
                    basis = PauliString.from_str(str(m["basis"]))
                    probs_dict: dict[str, float] = {
                        str(k): float(v)  # type: ignore[arg-type]
                        for k, v in m["probs"].items()  # type: ignore[union-attr]
                    }
                    meas_list.append((basis, probs_dict))
                try:
                    expectation = float(h.expectation_probs(meas_list))
                except ValueError as exc:
                    raise ServiceError(
                        "EXPECTATION_FAILED",
                        f"期望值计算失败（测量基可能无法覆盖全部哈密顿量项）: {exc}",
                        False,
                    ) from None
            elif operation in {
                "expectation_statevector",
                "expectation_density_matrix",
                "variance",
            }:
                if circuit is None:
                    raise ServiceError(
                        "MISSING_CIRCUIT",
                        f"{operation} 需要 circuit 参数用于仿真",
                        False,
                    )
                parsed = load_circuit(circuit, circuit_format)
                try:
                    if operation == "expectation_density_matrix":
                        dm = DensityMatrix.from_circuit(parsed)
                        expectation = float(h.expectation_density_matrix(dm))
                    else:
                        sv = Statevector.from_circuit(parsed)
                        if operation == "variance":
                            variance = float(h.variance_statevector(sv))
                        else:
                            expectation = float(h.expectation_statevector(sv))
                except ValueError as exc:
                    raise ServiceError(
                        "EXPECTATION_FAILED",
                        f"期望值/方差计算失败: {exc}",
                        False,
                    ) from None
            elif operation == "trotter_circuit":
                if trotter_mode == "first_order":
                    mode = TrotterMode.first_order()
                elif trotter_mode == "second_order":
                    mode = TrotterMode.second_order()
                else:
                    if random_seed is None:
                        raise ServiceError(
                            "MISSING_RANDOM_SEED",
                            "trotter_mode=randomized 时必须提供 random_seed",
                            False,
                        )
                    mode = TrotterMode.randomized(random_seed)
                try:
                    built = h.to_trotter_circuit(trotter_time, trotter_steps, mode)
                    trotter_circuit = dump_circuit(built, output_format)
                except ValueError as exc:
                    raise ServiceError(
                        "TROTTER_FAILED",
                        f"Trotter 线路生成失败: {exc}",
                        False,
                    ) from None

            out_terms = [
                HamiltonianTerm(pauli=str(ps), coefficient=float(coef.real)) for ps, coef in h.terms
            ]
            return HamiltonianResult(
                num_qubits=int(h.num_qubits),
                terms=out_terms,
                num_terms=int(h.num_terms),
                all_terms_commute=bool(h.all_terms_commute()),
                matrix=matrix,
                expectation=expectation,
                variance=variance,
                trotter_circuit=trotter_circuit,
            )

    async def calculate_state_metrics(
        metric: StateMetric,
        state_a: StateArray = None,
        state_b: StateArray = None,
        state_a_is_pure: bool = False,
        state_b_is_pure: bool = False,
        subsystem: MetricSubsystem = None,
        alpha: float | None = None,
    ) -> StateMetricsResult:
        """Compute quantum-state distance, entropy and entanglement metrics.

        States use [real, imag] pairs: a state vector is a 1D list of pairs,
        a density matrix is a 2D list of pairs. Set state_*_is_pure=true when
        the corresponding input is a state vector.

        Scalar metrics return ``value``: purity, fidelity, trace_distance,
        entropy (von Neumann), linear_entropy, renyi_entropy (needs alpha > 0),
        entanglement_entropy (pure state_a + subsystem), negativity,
        logarithmic_negativity, concurrence, entanglement_of_formation.
        partial_transpose returns the transposed density matrix in
        ``density_matrix`` instead of a scalar.

        subsystem is required by entanglement_entropy, negativity,
        logarithmic_negativity and partial_transpose.
        """
        with circuit_tool_errors():
            from cqlib.qis import DensityMatrix, Statevector
            from cqlib.qis.entropy import (
                concurrence,
                entanglement_entropy_pure,
                entanglement_of_formation,
                linear_entropy,
                negativity,
                renyi_entropy,
            )
            from cqlib.qis.metrics import (
                entropy,
                logarithmic_negativity,
                partial_transpose,
                purity_mixed,
                purity_pure,
                state_fidelity_mixed,
                state_fidelity_pure,
                state_fidelity_pure_mixed,
                trace_distance_mixed,
                trace_distance_pure,
            )

            subsystem_metrics = {
                "entanglement_entropy",
                "negativity",
                "logarithmic_negativity",
                "partial_transpose",
            }

            def _to_state(
                arr: list[list[float]] | list[list[list[float]]], is_pure: bool
            ) -> Statevector | DensityMatrix:
                try:
                    if is_pure:
                        pairs: list[list[float]] = arr  # type: ignore[assignment]
                        flat = [complex(r, i) for r, i in pairs]
                        n = int(np.log2(len(flat)))
                        return Statevector.from_state(n, np.array(flat, dtype=complex))
                    rows: list[list[list[float]]] = arr  # type: ignore[assignment]
                    matrix = np.array(
                        [[complex(r, i) for r, i in row] for row in rows],
                        dtype=complex,
                    )
                    n = matrix.shape[0].bit_length() - 1
                    return DensityMatrix.from_density_matrix(n, matrix.flatten())
                except ValueError as exc:
                    raise ServiceError("INVALID_STATE", f"量子态无效: {exc}", False) from None

            def _as_density_matrix(state: Statevector | DensityMatrix) -> DensityMatrix:
                if isinstance(state, Statevector):
                    return DensityMatrix.from_state(state.num_qubits, np.asarray(state.data))
                return state

            two_state_metrics = {"fidelity", "trace_distance"}
            if metric in two_state_metrics:
                if state_a is None or state_b is None:
                    raise ServiceError("MISSING_STATE", f"{metric} 需要 state_a 和 state_b", False)
            elif state_a is None:
                raise ServiceError("MISSING_STATE", f"{metric} 需要 state_a", False)
            if metric in subsystem_metrics and not subsystem:
                raise ServiceError("MISSING_SUBSYSTEM", f"{metric} 需要 subsystem 参数", False)
            if metric == "renyi_entropy":
                if alpha is None:
                    raise ServiceError("MISSING_ALPHA", "renyi_entropy 需要 alpha 参数", False)
                if alpha <= 0.0:
                    raise ServiceError(
                        "INVALID_ALPHA", "renyi_entropy 的 alpha 必须严格大于 0", False
                    )
            if metric == "entanglement_entropy" and not state_a_is_pure:
                raise ServiceError(
                    "INVALID_STATE",
                    "entanglement_entropy 需要纯态 state_a（state_a_is_pure=true）",
                    False,
                )

            value: float | None = None
            density_matrix: list[list[dict[str, float]]] | None = None
            try:
                if metric == "purity":
                    state = _to_state(state_a, state_a_is_pure)  # type: ignore[arg-type]
                    if state_a_is_pure:
                        value = float(purity_pure(cast(Statevector, state)))
                    else:
                        value = float(purity_mixed(cast(DensityMatrix, state)))
                elif metric in {"entropy", "linear_entropy", "renyi_entropy"}:
                    dm = _as_density_matrix(
                        _to_state(state_a, state_a_is_pure)  # type: ignore[arg-type]
                    )
                    if metric == "entropy":
                        value = float(entropy(dm))
                    elif metric == "linear_entropy":
                        value = float(linear_entropy(dm))
                    else:
                        value = float(renyi_entropy(dm, cast(float, alpha)))
                elif metric == "entanglement_entropy":
                    sv = cast(Statevector, _to_state(state_a, True))  # type: ignore[arg-type]
                    value = float(entanglement_entropy_pure(sv, cast(list[int], subsystem)))
                elif metric in {
                    "negativity",
                    "logarithmic_negativity",
                    "partial_transpose",
                }:
                    dm = _as_density_matrix(
                        _to_state(state_a, state_a_is_pure)  # type: ignore[arg-type]
                    )
                    qubits = cast(list[int], subsystem)
                    if metric == "negativity":
                        value = float(negativity(dm, qubits))
                    elif metric == "logarithmic_negativity":
                        value = float(logarithmic_negativity(dm, qubits))
                    else:
                        transposed = partial_transpose(dm, qubits)
                        density_matrix = _complex_matrix_to_json(np.asarray(transposed.data))
                elif metric in {"concurrence", "entanglement_of_formation"}:
                    dm = _as_density_matrix(
                        _to_state(state_a, state_a_is_pure)  # type: ignore[arg-type]
                    )
                    if metric == "concurrence":
                        value = float(concurrence(dm))
                    else:
                        value = float(entanglement_of_formation(dm))
                elif metric == "fidelity":
                    a = _to_state(state_a, state_a_is_pure)  # type: ignore[arg-type]
                    b = _to_state(state_b, state_b_is_pure)  # type: ignore[arg-type]
                    if state_a_is_pure and state_b_is_pure:
                        value = float(
                            state_fidelity_pure(cast(Statevector, a), cast(Statevector, b))
                        )
                    elif not state_a_is_pure and not state_b_is_pure:
                        value = float(
                            state_fidelity_mixed(cast(DensityMatrix, a), cast(DensityMatrix, b))
                        )
                    else:
                        pure = cast(Statevector, a if state_a_is_pure else b)
                        mixed = cast(DensityMatrix, b if state_a_is_pure else a)
                        value = float(state_fidelity_pure_mixed(pure, mixed))
                else:  # trace_distance
                    a = _to_state(state_a, state_a_is_pure)  # type: ignore[arg-type]
                    b = _to_state(state_b, state_b_is_pure)  # type: ignore[arg-type]
                    if state_a_is_pure and state_b_is_pure:
                        value = float(
                            trace_distance_pure(cast(Statevector, a), cast(Statevector, b))
                        )
                    elif not state_a_is_pure and not state_b_is_pure:
                        value = float(
                            trace_distance_mixed(cast(DensityMatrix, a), cast(DensityMatrix, b))
                        )
                    else:
                        # cqlib has no trace_distance_pure_mixed: promote the pure
                        # side to |psi><psi| and take the mixed-mixed code path.
                        value = float(
                            trace_distance_mixed(_as_density_matrix(a), _as_density_matrix(b))
                        )
            except (ValueError, IndexError) as exc:
                # Native kernels raise IndexError for out-of-range qubit indices.
                raise ServiceError("METRICS_FAILED", f"{metric} 计算失败: {exc}", False) from None

            return StateMetricsResult(metric=metric, value=value, density_matrix=density_matrix)

    tools = (
        simulate_circuit,
        plot_bloch_vector,
        plot_state,
        analyze_pauli_string,
        analyze_hamiltonian,
        calculate_state_metrics,
    )
    for tool in tools:
        mcp.tool(
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=True,
            )
        )(tool)
