"""MCP tools for cqlib compilation: commutation, device compile, decompose, optimize."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from tianyan_mcp.circuit_tools import circuit_tool_errors, dump_circuit, load_circuit
from tianyan_mcp.compile_models import (
    CommutationResult,
    CompileForDeviceResult,
    CompileWithTargetResult,
    DecomposeResult,
    NativeOptimizationStats,
    NativeOptimizationSummary,
    OptimizeResult,
)
from tianyan_mcp.errors import ServiceError

LOGGER = logging.getLogger(__name__)
CircuitText = Annotated[str, Field(min_length=1, max_length=20_000)]
GateName = Annotated[str, Field(min_length=1, max_length=32)]
QubitIndex = Annotated[int, Field(strict=True, ge=0, le=127)]
DeviceQubits = Annotated[int, Field(strict=True, ge=1, le=127)]
OptionalDeviceQubits = Annotated[int | None, Field(default=None, strict=True, ge=1, le=127)]
TargetBasis = Annotated[list[str] | None, Field(default=None, max_length=32)]
Seed = Annotated[int | None, Field(default=None, ge=0, le=2**31 - 1)]
Basis = Annotated[list[str] | None, Field(default=None, max_length=32)]
TargetKind = Literal["logical", "basis", "device", "topology_basis"]
ControlCount = Annotated[int | None, Field(default=None, strict=True, ge=1, le=8)]
DecomposeTransform = Literal[
    "expand_definitions",
    "decompose_unitaries",
    "decompose_mc_gates",
    "decompose_mc_gates_for_device",
    "circuit_decompose",
]

# Native capability whitelist for the internally built linear device. Only gates
# acting on at most two qubits can be stored as a native capability, so CCX and
# other multi-controlled gates must not be added here.
DEFAULT_NATIVE_GATES = [
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "rx",
    "ry",
    "rz",
    "cx",
    "cy",
    "cz",
    "swap",
]

# Subset of StandardGate names that are safe and common for commutation checks.
SUPPORTED_GATES = {
    "I",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "SDG",
    "T",
    "TDG",
    "RX",
    "RY",
    "RZ",
    "Phase",
    "U",
    "CX",
    "CY",
    "CZ",
    "SWAP",
    "CCX",
    "RXX",
    "RYY",
    "RZZ",
    "RZX",
    "RXY",
    "FSIM",
}


def _make_gate(name: str, param: float | None):
    from cqlib import StandardGate

    if name not in SUPPORTED_GATES:
        raise ServiceError("UNSUPPORTED_GATE", f"不支持的门: {name}", False)
    gate = getattr(StandardGate, name)
    if param is not None:
        return gate(param)
    return gate


def _layout_to_dict(layout: Any) -> dict[int, int]:
    """Convert a cqlib Layout into a JSON-friendly logical to physical mapping.

    Layout is not iterable, so the mapping is read from its ``l2p_map`` property.
    """

    return {int(k.index): int(v.index) for k, v in layout.l2p_map.items()}


def _compile_target_args_error(message: str) -> ServiceError:
    return ServiceError("INVALID_TARGET_ARGS", message, False)


def _append_mc_gate(circuit: Any, mc_controls: int) -> None:
    """Append a multi-controlled X gate on the lowest ``mc_controls + 1`` qubits.

    None of the accepted text formats can express a multi-controlled gate (QCIS
    cannot represent it, and QASM2/QASM3 only reach ``ccx``, which parses as the
    standard gate ``CCX``), so MC decomposition transforms need the gate built in
    memory. Missing qubits are added so the gate fits the circuit.
    """

    from cqlib import Qubit
    from cqlib.circuit.gates import MCGate, StandardGate

    required = mc_controls + 1
    existing = {int(qubit.index) for qubit in circuit.qubits}
    missing = [Qubit(index) for index in range(required) if index not in existing]
    if missing:
        circuit.add_qubits(missing)
    circuit.append_mc_gate(
        MCGate(mc_controls, StandardGate.X), [Qubit(index) for index in range(required)]
    )


def _resolve_target_args(
    target_kind: TargetKind,
    basis: list[str] | None,
    num_qubits: int | None,
    native_gates: list[str] | None,
    seed: int | None,
) -> tuple[list[str], int]:
    """Validate the target parameters and return the resolved basis and qubit count.

    Each target form accepts a fixed set of parameters; anything else is rejected
    instead of being silently ignored.
    """

    basis_names = basis or []
    device_qubits = 0
    if target_kind in ("device", "topology_basis"):
        if num_qubits is None:
            raise _compile_target_args_error(f"target_kind={target_kind} 需要提供 num_qubits")
        device_qubits = num_qubits
    if target_kind in ("basis", "topology_basis") and not basis_names:
        raise _compile_target_args_error(f"target_kind={target_kind} 需要提供非空 basis")
    allowed: dict[TargetKind, set[str]] = {
        "logical": set(),
        "basis": {"basis"},
        "device": {"num_qubits", "native_gates", "seed"},
        "topology_basis": {"basis", "num_qubits", "native_gates", "seed"},
    }
    provided = {
        "basis": basis,
        "num_qubits": num_qubits,
        "native_gates": native_gates,
        "seed": seed,
    }
    unexpected = sorted(
        name
        for name, value in provided.items()
        if value is not None and name not in allowed[target_kind]
    )
    if unexpected:
        raise _compile_target_args_error(
            f"target_kind={target_kind} 不支持参数: {', '.join(unexpected)}"
        )
    return basis_names, device_qubits


def _native_side_summary(summary: Any) -> NativeOptimizationSummary:
    """Convert one cqlib NativeOptimizationSummary into a JSON-friendly model."""

    predicted_log_error = summary.predicted_log_error
    return NativeOptimizationSummary(
        native_total_ops=int(summary.native_total_ops),
        native_two_qubit_ops=int(summary.native_two_qubit_ops),
        native_two_qubit_depth=int(summary.native_two_qubit_depth),
        total_native_depth=int(summary.total_native_depth),
        predicted_log_error=None if predicted_log_error is None else float(predicted_log_error),
        imputed_error_count=int(summary.imputed_error_count),
        unavailable_error_count=int(summary.unavailable_error_count),
    )


def _native_optimization_stats(result: Any) -> NativeOptimizationStats:
    """Build before/after native-gate statistics from a NativeOptimizationResult.

    The installed cqlib exposes the documented summary as ``result.before`` and
    ``result.after`` (statistics before and after the optimization rounds).
    """

    return NativeOptimizationStats(
        before=_native_side_summary(result.before),
        after=_native_side_summary(result.after),
        rounds=int(result.rounds),
        restored_best=bool(result.restored_best),
    )


def register_compile_tools(mcp: FastMCP) -> None:
    async def check_gate_commutation(
        left_gate: GateName,
        right_gate: GateName,
        qubit: QubitIndex,
        left_param: float | None = None,
        right_param: float | None = None,
        method: Literal["check", "algebraic"] = "check",
    ) -> CommutationResult:
        """Check whether two single-qubit gates commute when applied to the same qubit.

        Returns commutes=true when the gates can be safely reordered. exact=true means
        strict commutation; exact=false (with a phase) means they differ only by a
        global phase, which is physically equivalent. method=algebraic is faster but
        covers fewer cases.
        """
        with circuit_tool_errors():
            from cqlib import Qubit
            from cqlib.circuit import ValueOperation
            from cqlib.compile.commutation import (
                algebraic_commutation,
                check_commutation,
            )

            lhs = ValueOperation.from_standard_gate(
                _make_gate(left_gate, left_param), [Qubit(qubit)]
            )
            rhs = ValueOperation.from_standard_gate(
                _make_gate(right_gate, right_param), [Qubit(qubit)]
            )
            checker = check_commutation if method == "check" else algebraic_commutation
            proof = checker(lhs, rhs)
            if proof is None:
                return CommutationResult(commutes=False, exact=False, phase=None, method=method)
            exact = bool(proof.is_exact())
            phase = None if exact else str(proof.phase)
            return CommutationResult(commutes=True, exact=exact, phase=phase, method=method)

    async def compile_for_device(
        circuit: CircuitText,
        num_qubits: DeviceQubits,
        format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        mode: Literal["normal", "enhanced"] = "normal",
        target_basis: TargetBasis = None,
        seed: Seed = None,
    ) -> CompileForDeviceResult:
        """Compile a circuit for a linear-topology device with the given qubit count.

        The output circuit respects the device connectivity and native gate set.
        target_basis optionally restricts the output gate set. mode=enhanced applies
        stronger optimization. Returns the compiled QCIS circuit and layout metadata.
        """
        with circuit_tool_errors():
            from cqlib import Device, Instruction
            from cqlib.compile import CompileMode
            from cqlib.compile import compile as cqlib_compile
            from cqlib.ir import qcis

            parsed = load_circuit(circuit, format)
            device = Device.line("mcp-device", num_qubits)
            native_names = target_basis or DEFAULT_NATIVE_GATES
            device.native_gates = [Instruction.from_name(name) for name in native_names]
            compile_mode = CompileMode.enhanced() if mode == "enhanced" else CompileMode.normal()
            result = cqlib_compile(
                parsed,
                device=device,
                mode=compile_mode,
                target_basis=target_basis,
                seed=seed,
            )
            compiled = result.circuit
            initial_layout = None
            final_layout = None
            if result.device_metadata is not None:
                initial_layout = _layout_to_dict(result.device_metadata.initial_layout)
                final_layout = _layout_to_dict(result.device_metadata.final_layout)
            return CompileForDeviceResult(
                circuit=qcis.dumps(compiled),
                changed=bool(result.changed),
                mode=mode,
                initial_layout=initial_layout,
                final_layout=final_layout,
            )

    async def compile_with_target(
        circuit: CircuitText,
        target_kind: TargetKind = "logical",
        format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        mode: Literal["normal", "enhanced"] = "normal",
        basis: TargetBasis = None,
        num_qubits: OptionalDeviceQubits = None,
        native_gates: TargetBasis = None,
        seed: Seed = None,
    ) -> CompileWithTargetResult:
        """Compile a circuit for an explicit target form (cqlib CompileTarget).

        target_kind selects the target and which parameters apply:
        logical keeps logical-qubit space (no basis/num_qubits/native_gates/seed);
        basis lowers to a non-empty basis without regard for topology (basis only);
        device routes, lowers and validates for a linear device of num_qubits qubits
        (num_qubits required, native_gates defaults to a standard 1-2 qubit set);
        topology_basis routes on the device topology and lowers to basis
        (num_qubits and basis required). mode=enhanced applies stronger optimization.
        Returns the compiled QCIS circuit, whether it changed, and the logical to
        physical layouts for device-based targets. INVALID_TARGET_ARGS reports an
        unusable argument combination; COMPILE_FAILED reports a rejected or
        impossible compilation.
        """
        with circuit_tool_errors():
            from cqlib import Device, Instruction
            from cqlib.circuit import CircuitError
            from cqlib.compile import (
                CompileMode,
                CompilerConfigError,
                CompilerInternalError,
                CompilerTransformError,
                CompileTarget,
            )
            from cqlib.compile import compile as cqlib_compile
            from cqlib.ir import qcis

            basis_names, device_qubits = _resolve_target_args(
                target_kind, basis, num_qubits, native_gates, seed
            )
            parsed = load_circuit(circuit, format)
            compile_mode = CompileMode.enhanced() if mode == "enhanced" else CompileMode.normal()
            try:
                if target_kind == "logical":
                    target = CompileTarget.logical()
                elif target_kind == "basis":
                    target = CompileTarget.basis(basis_names)
                else:
                    device = Device.line("mcp-compile-target", device_qubits)
                    if target_kind == "device":
                        device.native_gates = [
                            Instruction.from_name(name)
                            for name in (native_gates or DEFAULT_NATIVE_GATES)
                        ]
                        target = CompileTarget.device(device, seed=seed)
                    else:
                        if native_gates is not None:
                            device.native_gates = [
                                Instruction.from_name(name) for name in native_gates
                            ]
                        target = CompileTarget.topology_basis(device, basis_names, seed=seed)
                result = cqlib_compile(parsed, mode=compile_mode, target=target)
            except CompilerConfigError as exc:
                raise ServiceError("COMPILE_FAILED", f"编译目标配置无效: {exc}", False) from None
            except CompilerTransformError as exc:
                raise ServiceError("COMPILE_FAILED", f"编译变换失败: {exc}", False) from None
            except CircuitError as exc:
                raise ServiceError("INVALID_CIRCUIT", f"线路结构校验失败: {exc}", False) from None
            except CompilerInternalError:
                raise ServiceError("INTERNAL_ERROR", "服务内部错误", False) from None
            initial_layout = None
            final_layout = None
            if result.device_metadata is not None:
                initial_layout = _layout_to_dict(result.device_metadata.initial_layout)
                final_layout = _layout_to_dict(result.device_metadata.final_layout)
            return CompileWithTargetResult(
                circuit=qcis.dumps(result.circuit),
                changed=bool(result.changed),
                mode=mode,
                target_kind=target_kind,
                initial_layout=initial_layout,
                final_layout=final_layout,
            )

    async def decompose_circuit(
        circuit: CircuitText,
        transform: DecomposeTransform = "expand_definitions",
        format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        two_qubit_basis: Literal["cz", "cx", "cy", "rzz", "pauli_rotations"] | None = None,
        mc_controls: ControlCount = None,
        num_qubits: OptionalDeviceQubits = None,
    ) -> DecomposeResult:
        """Decompose a circuit's gates into more primitive gates.

        expand_definitions expands circuit-gate definitions. decompose_unitaries
        synthesizes unitary matrices into standard gates (optionally restricted to a
        two-qubit basis). decompose_mc_gates breaks multi-controlled gates into
        single/two-qubit gates. decompose_mc_gates_for_device does the same while
        enforcing that the decomposition fits a linear device of num_qubits usable
        qubits (defaults to the circuit's own qubit count). circuit_decompose uses
        Circuit.decompose() to recursively expand circuit-defined gates. The input
        circuit is not modified.

        mc_controls appends a multi-controlled X gate on the lowest mc_controls + 1
        qubits before decomposing, and is used by the two decompose_mc_gates forms;
        no accepted text format can express a multi-controlled gate on its own.
        Pass num_qubits with decompose_mc_gates_for_device to set the device capacity.
        """
        with circuit_tool_errors():
            from cqlib import Device
            from cqlib.circuit import CircuitError
            from cqlib.compile import CompilerConfigError
            from cqlib.compile.transform.decompose import (
                TwoQubitUnitaryDecomposeBasis,
                UnitaryDecomposeConfig,
                decompose_mc_gates,
                decompose_mc_gates_for_device,
                decompose_unitaries,
                expand_definitions,
            )
            from cqlib.ir import qcis

            parsed = load_circuit(circuit, format)
            if mc_controls is not None:
                _append_mc_gate(parsed, mc_controls)
            try:
                if transform == "circuit_decompose":
                    decomposed = parsed.decompose()
                    changed = dump_circuit(decomposed, format) != dump_circuit(parsed, format)
                else:
                    result = None
                    if transform == "expand_definitions":
                        result = expand_definitions(parsed)
                    elif transform == "decompose_unitaries":
                        config = None
                        if two_qubit_basis is not None:
                            basis_map = {
                                "cz": TwoQubitUnitaryDecomposeBasis.cz,
                                "cx": TwoQubitUnitaryDecomposeBasis.cx,
                                "cy": TwoQubitUnitaryDecomposeBasis.cy,
                                "rzz": TwoQubitUnitaryDecomposeBasis.rzz,
                                "pauli_rotations": TwoQubitUnitaryDecomposeBasis.pauli_rotations,
                            }
                            config = UnitaryDecomposeConfig(
                                two_qubit_basis=basis_map[two_qubit_basis]()
                            )
                        result = decompose_unitaries(parsed, config=config)
                    elif transform == "decompose_mc_gates":
                        result = decompose_mc_gates(parsed)
                    else:
                        capacity = num_qubits if num_qubits is not None else parsed.num_qubits
                        device = Device.line("mcp-decompose-device", capacity)
                        result = decompose_mc_gates_for_device(parsed, device)
                    decomposed = result.circuit
                    changed = bool(result.changed)
            except CompilerConfigError as exc:
                raise ServiceError("DECOMPOSE_FAILED", f"分解失败: {exc}", False) from None
            except CircuitError as exc:
                raise ServiceError("INVALID_CIRCUIT", f"线路结构校验失败: {exc}", False) from None
            try:
                output = qcis.dumps(decomposed)
            except ValueError as exc:
                raise ServiceError(
                    "DECOMPOSE_FAILED", f"分解结果无法表示为 QCIS 线路: {exc}", False
                ) from None
            return DecomposeResult(
                circuit=output,
                changed=changed,
                transform=transform,
            )

    async def optimize_circuit(
        circuit: CircuitText,
        optimization_kind: Literal[
            "canonicalize",
            "commutative_cancellation",
            "optimize_one_qubit_runs",
            "native_optimization",
        ] = "canonicalize",
        format: Literal["qasm2", "qasm3", "qcis"] = "qcis",
        basis: Basis = None,
        native_policy: Literal["entangler_first", "balanced_depth"] | None = None,
    ) -> OptimizeResult:
        """Apply a circuit optimization pass and return the optimized circuit.

        canonicalize cleans up no-ops and merges adjacent gates.
        commutative_cancellation cancels adjacent inverse gates using commutation.
        optimize_one_qubit_runs folds consecutive single-qubit gates into one U gate
        (or into the given basis). native_optimization optimizes for a device's native
        gates using the selected policy. The input circuit is not modified.

        Only native_optimization populates summary: before/after native-gate
        statistics (total ops, two-qubit ops, two-qubit depth, total depth,
        predicted log error, imputed/unavailable error counts), the number of
        optimization rounds and whether the best circuit was restored. Other
        passes return summary=null.
        """
        with circuit_tool_errors():
            from cqlib.compile.transform import (
                CommutativeCancellation,
                NativeOptimizer,
                NativeQualityPolicy,
                OptimizeOneQubitRuns,
                canonicalize_circuit,
            )
            from cqlib.ir import qcis

            parsed = load_circuit(circuit, format)
            stats: NativeOptimizationStats | None = None
            if optimization_kind == "canonicalize":
                result = canonicalize_circuit(parsed)
            elif optimization_kind == "commutative_cancellation":
                result = CommutativeCancellation().run(parsed)
            elif optimization_kind == "optimize_one_qubit_runs":
                if basis:
                    result = OptimizeOneQubitRuns.basis(basis).run(parsed)
                else:
                    result = OptimizeOneQubitRuns.logical().run(parsed)
            else:
                from cqlib import Device, Instruction

                device = Device.line("mcp-opt", parsed.num_qubits)
                device.native_gates = [
                    Instruction.from_name(n)
                    for n in ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz"]
                ]
                if native_policy == "balanced_depth":
                    policy = NativeQualityPolicy.balanced_depth()
                else:
                    policy = NativeQualityPolicy.entangler_first()
                result = NativeOptimizer(device, quality_policy=policy).run(parsed)
                stats = _native_optimization_stats(result)
            return OptimizeResult(
                circuit=qcis.dumps(result.circuit),
                changed=bool(result.changed),
                pass_name=optimization_kind,
                summary=stats,
            )

    tools = (
        check_gate_commutation,
        compile_for_device,
        compile_with_target,
        decompose_circuit,
        optimize_circuit,
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
