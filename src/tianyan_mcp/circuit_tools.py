"""MCP tools for cqlib circuit analysis: conversion, depth, matrix, drawing, plotting."""

from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from tianyan_mcp.circuit_models import (
    CircuitAnalysisResult,
    CircuitEquivalenceResult,
    CircuitFormat,
    CircuitMatrixResult,
    CircuitOperation,
    CircuitSymbolicMatrixResult,
    ComplexNumber,
    ConvertCircuitFormatResult,
    DrawCircuitResult,
    PlotMeasurementResult,
    SymbolicComplexNumber,
)
from tianyan_mcp.errors import ServiceError

LOGGER = logging.getLogger(__name__)
CircuitText = Annotated[str, Field(min_length=1, max_length=20_000)]
ShotCount = Annotated[int, Field(strict=True, ge=1, le=1_000_000)]
QubitIndex = Annotated[int, Field(strict=True, ge=0, le=127)]
MaxQubits = Annotated[int, Field(strict=True, ge=1, le=20)]


@contextmanager
def circuit_tool_errors() -> Generator[None]:
    try:
        yield
    except ServiceError as exc:
        LOGGER.warning("event=circuit_tool_error code=%s", exc.code)
        raise RuntimeError(exc.as_json()) from None
    except Exception:
        LOGGER.error("event=circuit_tool_error code=INTERNAL_ERROR", exc_info=True)
        raise RuntimeError(
            ServiceError("INTERNAL_ERROR", "服务内部错误", False).as_json()
        ) from None


#: Built-in gate libraries cqlib resolves in-memory. Any other ``include``
#: target is resolved against the local filesystem by the Rust QASM front-end,
#: which turns the circuit input into an arbitrary file read / path traversal
#: primitive (and an unbounded read on devices such as ``/dev/zero``).
_ALLOWED_INCLUDES: dict[CircuitFormat, frozenset[str]] = {
    "qasm2": frozenset({"qelib1.inc"}),
    "qasm3": frozenset({"stdgates.inc"}),
    "qcis": frozenset(),
}

_INCLUDE_RE = re.compile(r'(?<![A-Za-z0-9_])include\s+"((?:[^"\\]|\\.)*)"')
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _mask_comments_preserving_strings(text: str) -> str:
    """Blank out ``//`` and ``/* */`` comments while keeping string literals.

    String contents are left byte-for-byte intact so include filenames inside
    them remain matchable; comments can no longer hide an include directive
    and comment text can no longer trigger false positives.
    """
    masked = list(text)
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = text.find("\n", i + 2)
            end = n if end == -1 else end
            for j in range(i, end):
                masked[j] = " "
            i = end
        elif ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            for j in range(i, end):
                masked[j] = "\n" if text[j] == "\n" else " "
            i = end
        elif ch == '"':
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
        else:
            i += 1
    return "".join(masked)


def reject_unsafe_includes(text: str, fmt: CircuitFormat) -> None:
    """Reject ``include`` directives that would make the parser read disk.

    Only the built-in gate library for the source format is allowed; this
    must run before cqlib parsing so the Rust front-end never touches the
    filesystem for attacker-controlled paths.
    """
    allowed = _ALLOWED_INCLUDES[fmt]
    for match in _INCLUDE_RE.finditer(_mask_comments_preserving_strings(text)):
        name = match.group(1)
        if name in allowed:
            continue
        if name:
            shown = _CONTROL_CHARS_RE.sub("?", name)[:100]
            detail = f"禁止通过 include 引用外部文件: {shown}"
        else:
            detail = "include 文件名不能为空"
        allowed_names = ", ".join(sorted(allowed)) if allowed else "无"
        raise ServiceError(
            "INVALID_CIRCUIT",
            f"无法解析 {fmt} 线路: include 仅允许内置门库文件（{allowed_names}）；{detail}",
            False,
        )


def load_circuit(text: str, fmt: CircuitFormat) -> Any:
    from cqlib.ir import qasm2, qasm3, qcis

    reject_unsafe_includes(text, fmt)
    module = {"qasm2": qasm2, "qasm3": qasm3, "qcis": qcis}[fmt]
    try:
        return module.loads(text)
    except Exception as exc:
        raise ServiceError("INVALID_CIRCUIT", f"无法解析 {fmt} 线路: {exc}", False) from None


def dump_circuit(circuit: Any, fmt: CircuitFormat) -> str:
    from cqlib.ir import qasm2, qasm3, qcis

    module = {"qasm2": qasm2, "qasm3": qasm3, "qcis": qcis}[fmt]
    return module.dumps(circuit)


def _matrix_to_json(matrix: np.ndarray) -> list[list[ComplexNumber]]:
    return [
        [ComplexNumber(real=float(cell.real), imag=float(cell.imag)) for cell in row]
        for row in matrix
    ]


def _symbolic_matrix_to_json(
    rows: list[list[Any]],
) -> list[list[SymbolicComplexNumber]]:
    return [
        [
            SymbolicComplexNumber(
                real=str(cell.real),
                imag=str(cell.imag),
            )
            for cell in row
        ]
        for row in rows
    ]


def _serialize_operation(op: Any) -> CircuitOperation:
    qubits = [int(q.index) for q in op.qubits]
    params: list[float | str] = []
    for param in op.params:
        if isinstance(param, (int, float)):
            params.append(float(param))
        elif param.is_constant():
            params.append(float(param.evaluate()))
        else:
            params.append(str(param))
    return CircuitOperation(
        name=str(op.name),
        qubits=qubits,
        params=params,
        instruction_type=str(op.instruction_type),
    )


@contextmanager
def temporary_media_path(suffix: str) -> Generator[str]:
    """Yield a temporary file path that is always removed afterwards.

    cqlib rendering functions can only write raster formats (PNG) to a file
    path, so the server stages the file there, streams it back to the client,
    and deletes it immediately. No rendered file is kept on the server.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="tianyan-mcp-")
    os.close(fd)
    os.remove(path)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOGGER.warning("event=temp_media_cleanup_failed path=%s error=%s", path, exc)


def render_png_base64(render: Callable[[str], object]) -> str:
    """Render via a cqlib output_path callback and return PNG bytes as base64."""
    with temporary_media_path(".png") as path:
        render(path)
        with open(path, "rb") as handle:
            data = handle.read()
    return base64.b64encode(data).decode("ascii")


def register_circuit_tools(mcp: FastMCP) -> None:
    async def convert_circuit_format(
        circuit: CircuitText,
        source_format: CircuitFormat,
        target_format: CircuitFormat,
    ) -> ConvertCircuitFormatResult:
        """Convert a quantum circuit text between OpenQASM 2.0, OpenQASM 3.0, and QCIS.

        The source is parsed into a cqlib Circuit and re-serialized in the target format.
        QCIS is the native format of Tianyan backends. CONVERSION_FAILED reports a
        construct the target format cannot represent (for example QCIS supports
        measurement but cannot represent classical registers or storing measurement
        results into classical variables).
        """
        with circuit_tool_errors():
            parsed = load_circuit(circuit, source_format)
            try:
                output = dump_circuit(parsed, target_format)
            except ValueError as exc:
                raise ServiceError(
                    "CONVERSION_FAILED",
                    f"线路无法转换为 {target_format} 格式: {exc}",
                    False,
                ) from None
            return ConvertCircuitFormatResult(
                source_format=source_format,
                target_format=target_format,
                circuit=output,
            )

    async def analyze_circuit(
        circuit: CircuitText,
        format: CircuitFormat = "qcis",
        recurse: bool = False,
    ) -> CircuitAnalysisResult:
        """Return circuit depth, gate counts, layers, and operations.

        ``depth`` follows ``recurse`` (default False; circuits with control flow
        should set recurse=True). ``depth_recursive`` always uses recurse=True.
        ``layers`` groups operation node IDs by ASAP dependency layer.
        ``operations`` lists every gate with name, qubits, params, and type.
        """
        with circuit_tool_errors():
            parsed = load_circuit(circuit, format)
            depth = int(parsed.depth(recurse=recurse))
            depth_recursive = int(parsed.depth(recurse=True))
            dag = parsed.dag()
            dag_depth = int(dag.depth())
            counts = dict(dag.operation_count_by_name())
            counts_recursive = dict(dag.operation_count_by_name_recursive())
            total = int(dag.num_ops)
            layers = [list(layer) for layer in dag.layers()]
            operations = [_serialize_operation(op) for op in parsed.operations]
            return CircuitAnalysisResult(
                depth=depth,
                depth_recursive=depth_recursive,
                dag_depth=dag_depth,
                total_operations=total,
                operation_counts=counts,
                operation_counts_recursive=counts_recursive,
                layers=layers,
                operations=operations,
            )

    async def circuit_to_matrix(
        circuit: CircuitText,
        format: CircuitFormat = "qcis",
        parameters: dict[str, float] | None = None,
    ) -> CircuitMatrixResult:
        """Convert a unitary circuit to its 2^n x 2^n complex matrix.

        Symbolic parameters are bound from `parameters` (name -> value) before
        conversion. Circuits with measurement, reset, or unresolved symbols fail.
        Matrix size grows exponentially; keep qubit count small.
        """
        with circuit_tool_errors():
            parsed = load_circuit(circuit, format)
            if parameters:
                parsed = parsed.assign_parameters(parameters)
            matrix = np.asarray(parsed.to_matrix())
            num_qubits = int(np.log2(matrix.shape[0]))
            return CircuitMatrixResult(
                num_qubits=num_qubits,
                matrix=_matrix_to_json(matrix),
            )

    async def circuit_to_symbolic_matrix(
        circuit: CircuitText,
        format: CircuitFormat = "qcis",
    ) -> CircuitSymbolicMatrixResult:
        """Convert a unitary circuit to its symbolic matrix (expressions kept).

        Unlike ``circuit_to_matrix``, symbolic parameters are preserved as
        expression strings in each matrix element. Only suitable for small
        circuits because storage grows as O(4^n). Circuits with measurement or
        reset (non-unitary ops) will fail.
        """
        with circuit_tool_errors():
            parsed = load_circuit(circuit, format)
            symbolic = parsed.to_symbolic_matrix()
            shape = symbolic.shape
            num_qubits = int(np.log2(shape[0]))
            symbols = list(symbolic.symbols)
            return CircuitSymbolicMatrixResult(
                num_qubits=num_qubits,
                symbols=symbols,
                matrix=_symbolic_matrix_to_json(symbolic.rows()),
            )

    async def draw_circuit(
        circuit: CircuitText,
        format: CircuitFormat = "qcis",
        kind: Literal["text", "svg", "png"] = "text",
        fold: int | None = Field(default=None, ge=1, le=100_000),
        initial_state: bool = False,
        show_params: bool = True,
        reverse_bits: bool = False,
        decompose_circuit_gates: bool = False,
    ) -> DrawCircuitResult:
        """Render a circuit as ASCII text, an SVG string, or a PNG image.

        text output is terminal friendly; svg is a full vector drawing; png is
        returned as a base64 string (media_type=image/png, encoding=base64).
        fold limits the number of gate columns before the figure wraps (svg/png
        only; ignored for text). PNG is staged in a temporary file while
        rendering and deleted immediately afterwards; no files are kept on the
        server.
        """
        with circuit_tool_errors():
            from cqlib.visualization import draw_figure, draw_text

            parsed = load_circuit(circuit, format)
            if kind == "text":
                content = draw_text(
                    parsed,
                    initial_state=initial_state,
                    show_params=show_params,
                    reverse_bits=reverse_bits,
                    decompose_circuit_gates=decompose_circuit_gates,
                )
                return DrawCircuitResult(
                    kind="text",
                    content=str(content),
                    encoding="utf-8",
                    media_type="text/plain",
                )
            if kind == "svg":
                content = draw_figure(
                    parsed,
                    fold=fold,
                    initial_state=initial_state,
                    show_params=show_params,
                    reverse_bits=reverse_bits,
                    decompose_circuit_gates=decompose_circuit_gates,
                )
                return DrawCircuitResult(
                    kind="svg",
                    content=str(content),
                    encoding="utf-8",
                    media_type="image/svg+xml",
                )
            png_base64 = render_png_base64(
                lambda path: draw_figure(
                    parsed,
                    fold=fold,
                    initial_state=initial_state,
                    show_params=show_params,
                    reverse_bits=reverse_bits,
                    decompose_circuit_gates=decompose_circuit_gates,
                    output_path=path,
                )
            )
            return DrawCircuitResult(
                kind="png",
                content=png_base64,
                encoding="base64",
                media_type="image/png",
            )

    async def plot_measurement_result(
        counts: dict[str, int],
        qubits: list[QubitIndex],
        shots: ShotCount,
        num_qubits: MaxQubits,
        kind: Literal["histogram", "distribution"] = "histogram",
        image_format: Literal["svg", "png"] = "svg",
        number_to_keep: int | None = Field(default=None, ge=1, le=1024),
        sort: Literal["asc", "desc"] = "asc",
        title: str | None = Field(default=None, max_length=200),
    ) -> PlotMeasurementResult:
        """Plot measurement counts as a histogram or probability distribution.

        counts maps bit strings to shot counts. qubits lists measured qubit indices.
        num_qubits is the bit-string width. image_format=svg returns the SVG string
        directly; image_format=png returns base64-encoded PNG. PNG is staged in a
        temporary file while rendering and deleted immediately afterwards; no files
        are kept on the server.
        """
        with circuit_tool_errors():
            from cqlib import ExecutionResult
            from cqlib.visualization import plot_distribution, plot_histogram

            result = ExecutionResult.from_counts(
                task_id="mcp-plot",
                qubits=qubits,
                shots=shots,
                num_qubits=num_qubits,
                counts=counts,
            )
            plot_fn = plot_histogram if kind == "histogram" else plot_distribution
            if image_format == "svg":
                svg = plot_fn(
                    result,
                    number_to_keep=number_to_keep,
                    sort=sort,
                    title=title,
                )
                return PlotMeasurementResult(
                    kind=kind,
                    image_format="svg",
                    content=str(svg),
                    encoding="utf-8",
                    media_type="image/svg+xml",
                )
            png_base64 = render_png_base64(
                lambda path: plot_fn(
                    result,
                    number_to_keep=number_to_keep,
                    sort=sort,
                    title=title,
                    output_path=path,
                )
            )
            return PlotMeasurementResult(
                kind=kind,
                image_format="png",
                content=png_base64,
                encoding="base64",
                media_type="image/png",
            )

    async def check_circuit_equivalence(
        circuit_a: CircuitText,
        circuit_b: CircuitText,
        format: CircuitFormat = "qcis",
        parameters: dict[str, float] | None = None,
        atol: float = Field(default=1e-8, gt=0, le=1.0),
    ) -> CircuitEquivalenceResult:
        """Check whether two unitary circuits are equivalent up to a global phase.

        Both circuits are converted to matrices (after binding symbolic parameters) and
        compared. A non-zero global phase is ignored when the matrices match after
        normalization. Circuits with measurement or reset are not comparable.
        """
        with circuit_tool_errors():
            a = load_circuit(circuit_a, format)
            b = load_circuit(circuit_b, format)
            if parameters:
                a = a.assign_parameters(parameters)
                b = b.assign_parameters(parameters)
            mat_a = np.asarray(a.to_matrix())
            mat_b = np.asarray(b.to_matrix())
            if mat_a.shape != mat_b.shape:
                return CircuitEquivalenceResult(
                    equivalent=False,
                    max_abs_diff=float("inf"),
                    global_phase_ignored=False,
                )
            # Ignore global phase: find best phase factor between the two matrices.
            ratio = mat_b / mat_a
            flat = ratio[np.isfinite(ratio) & (np.abs(ratio) > 1e-12)]
            if flat.size == 0:
                phase = 1.0 + 0j
            else:
                phase = flat[0]
            adjusted = mat_b / phase
            max_diff = float(np.max(np.abs(mat_a - adjusted)))
            equivalent = max_diff <= atol
            return CircuitEquivalenceResult(
                equivalent=equivalent,
                max_abs_diff=max_diff,
                global_phase_ignored=True,
            )

    tools = (
        convert_circuit_format,
        analyze_circuit,
        circuit_to_matrix,
        circuit_to_symbolic_matrix,
        draw_circuit,
        plot_measurement_result,
        check_circuit_equivalence,
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
