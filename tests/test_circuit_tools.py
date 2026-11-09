"""Tests for cqlib circuit-analysis MCP tools (matrix, symbolic matrix, etc.)."""

from __future__ import annotations

import base64
import glob
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from tianyan_mcp.circuit_tools import (
    load_circuit,
    register_circuit_tools,
    render_png_base64,
    temporary_media_path,
)
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.quantum_info_tools import register_quantum_info_tools

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def mcp() -> FastMCP:
    server = FastMCP("test-circuit-tools")
    register_circuit_tools(server)
    return server


@pytest.fixture()
def qmcp() -> FastMCP:
    server = FastMCP("test-quantum-info-tools")
    register_quantum_info_tools(server)
    return server


def _staged_pngs() -> set[str]:
    pattern = os.path.join(tempfile.gettempdir(), "tianyan-mcp-*.png")
    return set(glob.glob(pattern))


@pytest.mark.asyncio
async def test_circuit_to_symbolic_matrix_with_parameter(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "circuit_to_symbolic_matrix",
        {"circuit": "RY Q0 theta", "format": "qcis"},
    )
    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["num_qubits"] == 1
    assert structured["symbols"] == ["theta"]
    matrix = structured["matrix"]
    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    # RY(theta) matrix: [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    top_left = matrix[0][0]
    assert top_left["real"] == "cos(theta/2)"
    assert top_left["imag"] == "0"
    top_right = matrix[0][1]
    assert top_right["real"] == "-sin(theta/2)"
    assert top_right["imag"] == "0"


@pytest.mark.asyncio
async def test_circuit_to_symbolic_matrix_without_parameter(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "circuit_to_symbolic_matrix",
        {"circuit": "H Q0", "format": "qcis"},
    )
    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["num_qubits"] == 1
    assert structured["symbols"] == []
    matrix = structured["matrix"]
    assert matrix[0][0]["real"].startswith("0.70710678118654")
    assert matrix[0][0]["imag"] == "0"


@pytest.mark.asyncio
async def test_circuit_to_symbolic_matrix_registered(mcp: FastMCP) -> None:
    tools = await mcp.list_tools()
    names = {tool.name for tool in tools}
    assert "circuit_to_symbolic_matrix" in names
    tool = next(t for t in tools if t.name == "circuit_to_symbolic_matrix")
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False


QASM2_BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'


@pytest.mark.asyncio
async def test_convert_circuit_format_qasm2_to_qcis(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "convert_circuit_format",
        {
            "circuit": QASM2_BELL,
            "source_format": "qasm2",
            "target_format": "qcis",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["source_format"] == "qasm2"
    assert structured["target_format"] == "qcis"
    assert "H Q0" in structured["circuit"]
    assert "CX Q0 Q1" in structured["circuit"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_format", "source"),
    [
        # creg declaration alone
        (
            "qasm2",
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n',
        ),
        # measurement result stored into a classical register
        (
            "qasm2",
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n',
        ),
        # QASM 3 bit declaration with measurement assignment
        (
            "qasm3",
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\nc = measure q;\n',
        ),
    ],
)
async def test_convert_circuit_format_classical_storage_reports_readable_error(
    mcp: FastMCP, source_format: str, source: str
) -> None:
    with pytest.raises(ToolError, match=r"CONVERSION_FAILED.*classical") as exc_info:
        await mcp.call_tool(
            "convert_circuit_format",
            {
                "circuit": source,
                "source_format": source_format,
                "target_format": "qcis",
            },
        )

    assert "INTERNAL_ERROR" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_convert_circuit_format_bare_measurement_supported(mcp: FastMCP) -> None:
    source = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nh q[0];\nmeasure q;\n'
    result = await mcp.call_tool(
        "convert_circuit_format",
        {"circuit": source, "source_format": "qasm3", "target_format": "qcis"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert "M Q0" in structured["circuit"]
    assert "M Q1" in structured["circuit"]


QASM3_UNSAFE_INCLUDES = [
    "/etc/shadow",
    "/dev/zero",
    "/run/secrets/tianyan",
    "/.env",
    "/app/.env",
    "/proc/self/environ",
    "../../../../etc/passwd",
    "pyproject.toml",
    "README.md",
    r"C:\Windows\win.ini",
    "file:///etc/hosts",
    "",
    # only the qasm2 builtin is valid for qasm2; qasm3 only accepts stdgates.inc
    "qelib1.inc",
    "stdgates.inc.bak",
    "./stdgates.inc",
    "/inc/stdgates.inc",
]


@pytest.mark.parametrize("included", QASM3_UNSAFE_INCLUDES)
def test_load_circuit_qasm3_rejects_external_include(included: str) -> None:
    source = f'include "{included}";'
    with pytest.raises(ServiceError) as exc_info:
        load_circuit(source, "qasm3")
    assert exc_info.value.code == "INVALID_CIRCUIT"
    message = exc_info.value.message
    assert "include" in message
    assert "stdgates.inc" in message
    # the raw attacker-controlled path must not be handed to the parser at all
    assert included or "空" in message


@pytest.mark.parametrize(
    "source",
    [
        'OPENQASM 2.0;\ninclude "stdgates.inc";\nqreg q[1];',
        'OPENQASM 2.0;\ninclude "/etc/passwd";\nqreg q[1];',
        'OPENQASM 2.0;\ninclude "../../qelib1.inc";\nqreg q[1];',
    ],
)
def test_load_circuit_qasm2_only_allows_builtin_qelib(source: str) -> None:
    with pytest.raises(ServiceError) as exc_info:
        load_circuit(source, "qasm2")
    assert exc_info.value.code == "INVALID_CIRCUIT"
    assert "qelib1.inc" in exc_info.value.message


def test_load_circuit_qcis_never_allows_include() -> None:
    with pytest.raises(ServiceError) as exc_info:
        load_circuit('include "/etc/shadow";', "qcis")
    assert exc_info.value.code == "INVALID_CIRCUIT"


def test_load_circuit_allows_builtin_includes() -> None:
    qasm2_circuit = load_circuit(QASM2_BELL, "qasm2")
    assert len(qasm2_circuit.operations) == 2
    qasm3_source = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nh q[0];'
    qasm3_circuit = load_circuit(qasm3_source, "qasm3")
    assert len(qasm3_circuit.operations) == 1


@pytest.mark.parametrize(
    "source",
    [
        # include payload hidden in a line comment must not trigger a false positive
        'OPENQASM 3.0;\n// include "/etc/shadow";\ninclude "stdgates.inc";\nqubit[1] q;\nh q[0];',
        # ...or in a block comment
        '/* include "/etc/shadow"; */\nOPENQASM 3.0;\ninclude "stdgates.inc";\n'
        "qubit[1] q;\nh q[0];",
    ],
)
def test_load_circuit_ignores_include_inside_comments(source: str) -> None:
    circuit = load_circuit(source, "qasm3")
    assert len(circuit.operations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["convert_circuit_format", "analyze_circuit"])
async def test_tools_reject_dev_zero_include_without_touching_filesystem(
    mcp: FastMCP, tool_name: str
) -> None:
    # Before the guard this drove the Rust parser into an unbounded read of
    # /dev/zero, exhausting container memory and getting the process OOM-killed.
    arguments: dict[str, Any] = {
        "circuit": 'include "/dev/zero";',
        "format": "qasm3",
    }
    if tool_name == "convert_circuit_format":
        arguments = {
            "circuit": 'include "/dev/zero";',
            "source_format": "qasm3",
            "target_format": "qcis",
        }
    with pytest.raises(ToolError, match="INVALID_CIRCUIT.*include"):
        await mcp.call_tool(tool_name, arguments)


def test_render_png_base64_returns_content_and_removes_staged_file() -> None:
    staged: list[str] = []

    def render(path: str) -> None:
        staged.append(path)
        Path(path).write_bytes(PNG_MAGIC + b"payload")

    encoded = render_png_base64(render)

    assert base64.b64decode(encoded) == PNG_MAGIC + b"payload"
    assert len(staged) == 1
    assert not Path(staged[0]).exists()


def test_temporary_media_path_cleans_up_when_rendering_fails() -> None:
    staged_path: str | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with temporary_media_path(".png") as path:
            staged_path = path
            assert not Path(path).exists()
            Path(path).write_bytes(b"partial")
            raise RuntimeError("boom")

    assert staged_path is not None
    assert not Path(staged_path).exists()


@pytest.mark.asyncio
async def test_draw_circuit_png_returns_base64_without_keeping_file(
    mcp: FastMCP,
) -> None:
    before = _staged_pngs()
    result = await mcp.call_tool(
        "draw_circuit",
        {"circuit": "H Q0", "format": "qcis", "kind": "png"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == "png"
    assert structured["encoding"] == "base64"
    assert structured["media_type"] == "image/png"
    assert base64.b64decode(structured["content"]).startswith(PNG_MAGIC)
    assert _staged_pngs() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "media_type", "prefix"),
    [
        ("text", "text/plain", ""),
        ("svg", "image/svg+xml", "<svg"),
    ],
)
async def test_draw_circuit_text_and_svg_still_returned_inline(
    mcp: FastMCP,
    kind: str,
    media_type: str,
    prefix: str,
) -> None:
    result = await mcp.call_tool(
        "draw_circuit",
        {"circuit": "H Q0", "format": "qcis", "kind": kind},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == kind
    assert structured["encoding"] == "utf-8"
    assert structured["media_type"] == media_type
    assert structured["content"].lstrip().startswith(prefix)


@pytest.mark.asyncio
async def test_draw_circuit_svg_accepts_fold(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "draw_circuit",
        {
            "circuit": "H Q0\nCX Q0 Q1\nH Q0",
            "format": "qcis",
            "kind": "svg",
            "fold": 1,
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["media_type"] == "image/svg+xml"
    assert structured["content"].lstrip().startswith("<svg")


@pytest.mark.asyncio
async def test_draw_circuit_rejects_non_positive_fold(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="fold"):
        await mcp.call_tool(
            "draw_circuit",
            {"circuit": "H Q0", "format": "qcis", "kind": "svg", "fold": 0},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["histogram", "distribution"])
async def test_plot_measurement_result_png_returns_base64(
    mcp: FastMCP,
    kind: str,
) -> None:
    before = _staged_pngs()
    result = await mcp.call_tool(
        "plot_measurement_result",
        {
            "counts": {"00": 512, "11": 488},
            "qubits": [0, 1],
            "shots": 1000,
            "num_qubits": 2,
            "kind": kind,
            "image_format": "png",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == kind
    assert structured["image_format"] == "png"
    assert structured["encoding"] == "base64"
    assert structured["media_type"] == "image/png"
    assert base64.b64decode(structured["content"]).startswith(PNG_MAGIC)
    assert _staged_pngs() == before


@pytest.mark.asyncio
async def test_plot_measurement_result_defaults_to_inline_svg(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "plot_measurement_result",
        {
            "counts": {"00": 512, "11": 488},
            "qubits": [0, 1],
            "shots": 1000,
            "num_qubits": 2,
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["image_format"] == "svg"
    assert structured["encoding"] == "utf-8"
    assert structured["media_type"] == "image/svg+xml"
    assert structured["content"].lstrip().startswith("<svg")


@pytest.mark.asyncio
@pytest.mark.parametrize("visualize", ["bloch", "city", "paulivec"])
async def test_simulate_circuit_visualize_png_returns_base64(
    qmcp: FastMCP,
    visualize: str,
) -> None:
    before = _staged_pngs()
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": "H Q0",
            "format": "qcis",
            "visualize": visualize,
            "visualize_format": "png",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["state_svg"] is None
    assert structured["state_png_base64"] is not None
    assert base64.b64decode(structured["state_png_base64"]).startswith(PNG_MAGIC)
    assert _staged_pngs() == before


@pytest.mark.asyncio
async def test_simulate_circuit_visualize_defaults_to_inline_svg(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {"circuit": "H Q0", "format": "qcis", "visualize": "bloch"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["state_png_base64"] is None
    assert structured["state_svg"] is not None
    assert structured["state_svg"].lstrip().startswith("<svg")


QASM2_TWO_QUBITS_H0 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n'
QASM2_TWO_QUBITS_X1 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nx q[1];\n'


@pytest.mark.asyncio
async def test_simulate_circuit_statevector_stepwise_appends_circuit(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": QASM2_TWO_QUBITS_H0,
            "format": "qasm2",
            "additional_circuit": QASM2_TWO_QUBITS_X1,
            "additional_format": "qasm2",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["num_qubits"] == 2
    assert set(structured["probabilities"]) == {"10", "11"}
    assert structured["probabilities"]["10"] == pytest.approx(0.5)
    assert structured["probabilities"]["11"] == pytest.approx(0.5)
    assert structured["classical_data"] is None
    assert structured["readout_probabilities"] is None


@pytest.mark.asyncio
async def test_simulate_circuit_stabilizer_stepwise_appends_circuit(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": QASM2_TWO_QUBITS_H0,
            "format": "qasm2",
            "simulator": "stabilizer",
            "additional_circuit": QASM2_TWO_QUBITS_X1,
            "additional_format": "qasm2",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert set(structured["probabilities"]) == {"10", "11"}
    assert structured["classical_data"] == {"values": [], "vars": []}


@pytest.mark.asyncio
async def test_simulate_circuit_rejects_additional_circuit_exceeding_qubits(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT.*超过主线路"):
        await qmcp.call_tool(
            "simulate_circuit",
            {"circuit": "H Q0", "additional_circuit": "X Q2"},
        )


@pytest.mark.asyncio
async def test_simulate_circuit_rejects_unparseable_additional_circuit(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT"):
        await qmcp.call_tool(
            "simulate_circuit",
            {"circuit": "H Q0", "additional_circuit": "NOPE Q0"},
        )


@pytest.mark.asyncio
async def test_simulate_circuit_rejects_non_clifford_addition_on_stabilizer(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT.*追加线路无法作用"):
        await qmcp.call_tool(
            "simulate_circuit",
            {
                "circuit": "H Q0",
                "simulator": "stabilizer",
                "additional_circuit": "T Q0",
            },
        )


@pytest.mark.asyncio
async def test_simulate_circuit_stabilizer_reports_measurement_values(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": "X Q0\nX Q1\nM Q0\nM Q1",
            "simulator": "stabilizer",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    classical_data = structured["classical_data"]
    assert classical_data["vars"] == []
    values = classical_data["values"]
    assert len(values) == 2
    assert values[0] == {"index": 0, "kind": "bit", "value": True, "qubits": [0]}
    assert values[1] == {"index": 1, "kind": "bit", "value": True, "qubits": [1]}


@pytest.mark.asyncio
async def test_simulate_circuit_stabilizer_measurement_matches_probabilities(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {"circuit": "H Q0\nM Q0", "simulator": "stabilizer"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    values = structured["classical_data"]["values"]
    assert len(values) == 1
    assert values[0]["qubits"] == [0]
    measured = values[0]["value"]
    assert measured in (False, True)
    # the projected state must agree with the per-shot measurement outcome
    assert set(structured["probabilities"]) == {str(int(measured))}


@pytest.mark.asyncio
async def test_simulate_circuit_stabilizer_without_measurement_has_empty_classical(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {"circuit": "H Q0", "simulator": "stabilizer"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["classical_data"] == {"values": [], "vars": []}


QASM2_WITH_CREG = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "x q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


@pytest.mark.asyncio
async def test_simulate_circuit_stabilizer_reports_classical_vars_from_qasm2(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": QASM2_WITH_CREG,
            "format": "qasm2",
            "simulator": "stabilizer",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    classical_data = structured["classical_data"]
    assert len(classical_data["values"]) == 2
    assert classical_data["values"][0]["value"] is True
    variables = classical_data["vars"]
    assert len(variables) == 1
    assert variables[0] == {
        "index": 0,
        "kind": "bit_vec",
        "value": "01",
        "qubits": None,
    }


@pytest.mark.asyncio
async def test_simulate_circuit_density_matrix_noise_readout_probabilities(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": "X Q0",
            "simulator": "density_matrix_noise",
            "readout_errors": [{"qubit": 0, "p_0_given_1": 0.3, "p_1_given_0": 0.0}],
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    # ideal probabilities before readout error
    assert structured["probabilities"] == {"1": pytest.approx(1.0)}
    readout = structured["readout_probabilities"]
    assert readout.keys() == {"0", "1"}
    assert readout["0"] == pytest.approx(0.3)
    assert readout["1"] == pytest.approx(0.7)
    assert structured["density_matrix"] is not None


@pytest.mark.asyncio
async def test_simulate_circuit_readout_false_positive_on_zero_state(qmcp: FastMCP) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            # H followed by H cancels out and leaves |0>
            "circuit": "H Q0\nH Q0",
            "simulator": "density_matrix_noise",
            "readout_errors": [{"qubit": 0, "p_0_given_1": 0.0, "p_1_given_0": 0.2}],
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    readout = structured["readout_probabilities"]
    assert readout["0"] == pytest.approx(0.8)
    assert readout["1"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_simulate_circuit_explicit_readout_qubits_default_to_ideal(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": "H Q0",
            "simulator": "density_matrix_noise",
            "readout_qubits": [0],
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    readout = structured["readout_probabilities"]
    assert readout["0"] == pytest.approx(0.5)
    assert readout["1"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_simulate_circuit_density_matrix_returns_matrix(qmcp: FastMCP) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {"circuit": "X Q0", "simulator": "density_matrix"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["probabilities"] == {"1": pytest.approx(1.0)}
    assert structured["state_vector"] is None
    assert structured["density_matrix"] is not None
    assert structured["classical_data"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "noise_type",
    ["bit_flip", "phase_flip", "depolarizing", "amplitude_damping", "phase_damping"],
)
async def test_simulate_circuit_density_matrix_noise_accepts_gate_noise(
    qmcp: FastMCP,
    noise_type: str,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {
            "circuit": "X Q0",
            "simulator": "density_matrix_noise",
            "noise_errors": [
                {"gate": "X", "qubit": 0, "noise_type": noise_type, "probability": 0.1}
            ],
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["density_matrix"] is not None
    assert structured["readout_probabilities"] is None


@pytest.mark.asyncio
async def test_simulate_circuit_rejects_unknown_gate_noise_type(qmcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_NOISE"):
        await qmcp.call_tool(
            "simulate_circuit",
            {
                "circuit": "X Q0",
                "simulator": "density_matrix_noise",
                "noise_errors": [
                    {"gate": "X", "qubit": 0, "noise_type": "nope", "probability": 0.1}
                ],
            },
        )


@pytest.mark.asyncio
async def test_simulate_circuit_noise_without_readout_config_omits_field(
    qmcp: FastMCP,
) -> None:
    result = await qmcp.call_tool(
        "simulate_circuit",
        {"circuit": "H Q0", "simulator": "density_matrix_noise"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["readout_probabilities"] is None
    assert structured["classical_data"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "readout_error",
    [
        {"qubit": 0, "p_0_given_1": 1.5, "p_1_given_0": 0.0},
        {"qubit": 0, "p_0_given_1": 0.0, "p_1_given_0": -0.1},
        {"qubit": 0, "p_0_given_1": 0.1},
        {"qubit": "0", "p_0_given_1": 0.1, "p_1_given_0": 0.1},
    ],
)
async def test_simulate_circuit_rejects_invalid_readout_error_spec(
    qmcp: FastMCP,
    readout_error: dict[str, Any],
) -> None:
    with pytest.raises(ToolError, match="INVALID_NOISE"):
        await qmcp.call_tool(
            "simulate_circuit",
            {
                "circuit": "X Q0",
                "simulator": "density_matrix_noise",
                "readout_errors": [readout_error],
            },
        )


@pytest.mark.asyncio
async def test_simulate_circuit_rejects_readout_qubit_out_of_range(qmcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_NOISE.*超出线路比特数"):
        await qmcp.call_tool(
            "simulate_circuit",
            {
                "circuit": "X Q0",
                "simulator": "density_matrix_noise",
                "readout_errors": [{"qubit": 2, "p_0_given_1": 0.1, "p_1_given_0": 0.1}],
            },
        )
    with pytest.raises(ToolError, match="INVALID_NOISE.*超出线路比特数"):
        await qmcp.call_tool(
            "simulate_circuit",
            {
                "circuit": "X Q0",
                "simulator": "density_matrix_noise",
                "readout_qubits": [1],
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_args",
    [
        {
            "noise_errors": [
                {"gate": "X", "qubit": 0, "noise_type": "bit_flip", "probability": 0.01}
            ]
        },
        {"readout_errors": [{"qubit": 0, "p_0_given_1": 0.1, "p_1_given_0": 0.1}]},
        {"readout_qubits": [0]},
    ],
)
async def test_simulate_circuit_rejects_noise_args_for_other_simulators(
    qmcp: FastMCP,
    extra_args: dict[str, Any],
) -> None:
    with pytest.raises(ToolError, match="INVALID_NOISE"):
        await qmcp.call_tool(
            "simulate_circuit",
            {"circuit": "H Q0", "simulator": "statevector", **extra_args},
        )


@pytest.mark.asyncio
async def test_plot_bloch_vector_returns_inline_svg(qmcp: FastMCP) -> None:
    result = await qmcp.call_tool("plot_bloch_vector", {"vector": [0, 0, 1]})

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == "bloch_vector"
    assert structured["image_format"] == "svg"
    assert structured["encoding"] == "utf-8"
    assert structured["media_type"] == "image/svg+xml"
    assert structured["content"].lstrip().startswith("<svg")


@pytest.mark.asyncio
async def test_plot_bloch_vector_png_returns_base64_without_keeping_file(
    qmcp: FastMCP,
) -> None:
    before = _staged_pngs()
    result = await qmcp.call_tool(
        "plot_bloch_vector",
        {"vector": [0.0, 0.0, 1.0], "image_format": "png"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["image_format"] == "png"
    assert structured["encoding"] == "base64"
    assert structured["media_type"] == "image/png"
    assert base64.b64decode(structured["content"]).startswith(PNG_MAGIC)
    assert _staged_pngs() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0, 0], [0, 0, 1, 0]])
async def test_plot_bloch_vector_rejects_vector_without_three_components(
    qmcp: FastMCP,
    vector: list[int],
) -> None:
    with pytest.raises(ToolError, match=r"(at least|at most) 3 items"):
        await qmcp.call_tool("plot_bloch_vector", {"vector": vector})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["bloch_multivector", "city", "paulivec"])
async def test_plot_state_renders_state_kinds_from_circuit(
    qmcp: FastMCP,
    kind: str,
) -> None:
    result = await qmcp.call_tool(
        "plot_state",
        {"kind": kind, "circuit": "H Q0", "format": "qcis"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == kind
    assert structured["image_format"] == "svg"
    assert structured["media_type"] == "image/svg+xml"
    assert structured["content"].lstrip().startswith("<svg")


@pytest.mark.asyncio
async def test_plot_state_bloch_vector_kind_uses_vector(qmcp: FastMCP) -> None:
    result = await qmcp.call_tool(
        "plot_state",
        {"kind": "bloch_vector", "vector": [0, 0, 1]},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == "bloch_vector"
    assert structured["content"].lstrip().startswith("<svg")


@pytest.mark.asyncio
async def test_plot_state_accepts_density_matrix_simulator(qmcp: FastMCP) -> None:
    result = await qmcp.call_tool(
        "plot_state",
        {"kind": "city", "circuit": "H Q0", "simulator": "density_matrix"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["kind"] == "city"
    assert structured["content"].lstrip().startswith("<svg")


@pytest.mark.asyncio
async def test_plot_state_png_returns_base64_without_keeping_file(
    qmcp: FastMCP,
) -> None:
    before = _staged_pngs()
    result = await qmcp.call_tool(
        "plot_state",
        {"kind": "paulivec", "circuit": "H Q0", "image_format": "png"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["image_format"] == "png"
    assert structured["encoding"] == "base64"
    assert structured["media_type"] == "image/png"
    assert base64.b64decode(structured["content"]).startswith(PNG_MAGIC)
    assert _staged_pngs() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["bloch_multivector", "city", "paulivec"])
async def test_plot_state_requires_circuit_for_state_kinds(
    qmcp: FastMCP,
    kind: str,
) -> None:
    with pytest.raises(ToolError, match="MISSING_CIRCUIT"):
        await qmcp.call_tool("plot_state", {"kind": kind})


@pytest.mark.asyncio
async def test_plot_state_requires_vector_for_bloch_vector_kind(qmcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="MISSING_VECTOR"):
        await qmcp.call_tool("plot_state", {"kind": "bloch_vector"})


@pytest.mark.asyncio
async def test_plot_state_rejects_unparseable_circuit(qmcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT"):
        await qmcp.call_tool("plot_state", {"kind": "city", "circuit": "NOPE Q9"})


@pytest.mark.asyncio
async def test_plot_state_tools_are_registered_read_only(qmcp: FastMCP) -> None:
    tools = await qmcp.list_tools()
    by_name = {tool.name: tool for tool in tools}

    for name in ("plot_state", "plot_bloch_vector"):
        assert name in by_name
        annotations = by_name[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is True


@pytest.mark.asyncio
async def test_analyze_hamiltonian_scale_multiplies_coefficients(qmcp: FastMCP) -> None:
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
            "scale": 2.0,
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["terms"] == [{"pauli": "+ZZ", "coefficient": 1.0}]


@pytest.mark.asyncio
async def test_analyze_hamiltonian_add_additional_terms(qmcp: FastMCP) -> None:
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
            "additional_terms": [[{"pauli": "XX", "coefficient": 0.3}]],
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["num_terms"] == 2
    paulis = {term["pauli"] for term in structured["terms"]}
    assert paulis == {"+ZZ", "+XX"}


@pytest.mark.asyncio
async def test_analyze_hamiltonian_add_merges_like_terms(qmcp: FastMCP) -> None:
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
            "additional_terms": [[{"pauli": "ZZ", "coefficient": 0.3}]],
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["terms"] == [{"pauli": "+ZZ", "coefficient": 0.8}]


@pytest.mark.asyncio
async def test_analyze_hamiltonian_add_rejects_qubit_mismatch(qmcp: FastMCP) -> None:
    with pytest.raises(ToolError, match=r"HAMILTONIAN_MISMATCH.*比特数"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "additional_terms": [[{"pauli": "Z", "coefficient": 1.0}]],
            },
        )


@pytest.mark.asyncio
async def test_analyze_hamiltonian_trotter_first_order_returns_qcis(
    qmcp: FastMCP,
) -> None:
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [
                {"pauli": "ZZ", "coefficient": 0.5},
                {"pauli": "XX", "coefficient": 0.3},
            ],
            "operation": "trotter_circuit",
            "trotter_time": 1.0,
            "trotter_steps": 2,
        },
    )
    structured = cast(dict[str, Any], result[1])
    circuit = structured["trotter_circuit"]
    assert isinstance(circuit, str)
    assert "RZ" in circuit
    assert circuit.count("RZ") == 4  # 2 terms x 2 steps (first order)


@pytest.mark.asyncio
async def test_analyze_hamiltonian_trotter_second_order_qasm2(qmcp: FastMCP) -> None:
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [
                {"pauli": "ZZ", "coefficient": 0.5},
                {"pauli": "XX", "coefficient": 0.3},
            ],
            "operation": "trotter_circuit",
            "trotter_steps": 1,
            "trotter_mode": "second_order",
            "output_format": "qasm2",
        },
    )
    structured = cast(dict[str, Any], result[1])
    circuit = structured["trotter_circuit"]
    assert "OPENQASM 2.0" in circuit
    # second order: forward + reverse sweeps -> 4 rotations for 2 terms
    assert circuit.count("rz(") == 4


@pytest.mark.asyncio
async def test_analyze_hamiltonian_trotter_randomized_requires_seed(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match=r"MISSING_RANDOM_SEED"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "operation": "trotter_circuit",
                "trotter_mode": "randomized",
            },
        )


@pytest.mark.asyncio
async def test_analyze_hamiltonian_trotter_randomized_with_seed(
    qmcp: FastMCP,
) -> None:
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [
                {"pauli": "ZZ", "coefficient": 0.5},
                {"pauli": "XX", "coefficient": 0.3},
            ],
            "operation": "trotter_circuit",
            "trotter_steps": 2,
            "trotter_mode": "randomized",
            "random_seed": 42,
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert "RZ" in structured["trotter_circuit"]


@pytest.mark.asyncio
async def test_analyze_hamiltonian_trotter_rejects_non_positive_steps(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="trotter_steps"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "operation": "trotter_circuit",
                "trotter_steps": 0,
            },
        )


@pytest.mark.asyncio
async def test_analyze_hamiltonian_trotter_rejects_empty_after_zero_scale(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="TROTTER_FAILED"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "operation": "trotter_circuit",
                "scale": 0.0,
            },
        )


BELL_QCIS = "H Q0\nCX Q0 Q1"
BELL_HAMILTONIAN_TERMS = [
    {"pauli": "ZZ", "coefficient": 0.5},
    {"pauli": "XX", "coefficient": 0.3},
]


@pytest.mark.asyncio
async def test_analyze_hamiltonian_expectation_statevector_bell(qmcp: FastMCP) -> None:
    # Bell state: <ZZ>=1 and <XX>=1 -> <H> = 0.8
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": BELL_HAMILTONIAN_TERMS,
            "operation": "expectation_statevector",
            "circuit": BELL_QCIS,
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["expectation"] == pytest.approx(0.8)
    assert structured["variance"] is None
    assert structured["matrix"] is None


@pytest.mark.asyncio
async def test_analyze_hamiltonian_expectation_statevector_zero_on_plus(
    qmcp: FastMCP,
) -> None:
    # |+> has <Z>=0.
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [{"pauli": "Z", "coefficient": 1.0}],
            "operation": "expectation_statevector",
            "circuit": "H Q0",
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["expectation"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_analyze_hamiltonian_expectation_density_matrix_bell(
    qmcp: FastMCP,
) -> None:
    # A pure state simulated as a density matrix gives the same expectation.
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": BELL_HAMILTONIAN_TERMS,
            "operation": "expectation_density_matrix",
            "circuit": BELL_QCIS,
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["expectation"] == pytest.approx(0.8)
    assert structured["variance"] is None


@pytest.mark.asyncio
async def test_analyze_hamiltonian_variance_statevector_bell(qmcp: FastMCP) -> None:
    # Bell is an eigenstate of H = 0.5 ZZ + 0.3 XX, so Var(H) = 0.
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": BELL_HAMILTONIAN_TERMS,
            "operation": "variance",
            "circuit": BELL_QCIS,
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["variance"] == pytest.approx(0.0)
    assert structured["expectation"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["expectation_statevector", "expectation_density_matrix", "variance"],
)
async def test_analyze_hamiltonian_simulated_ops_require_circuit(
    qmcp: FastMCP,
    operation: str,
) -> None:
    with pytest.raises(ToolError, match="MISSING_CIRCUIT"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "operation": operation,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["expectation_statevector", "expectation_density_matrix"],
)
async def test_analyze_hamiltonian_simulated_expectation_rejects_bad_circuit(
    qmcp: FastMCP,
    operation: str,
) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "operation": operation,
                "circuit": "NOPE Q9",
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["expectation_statevector", "expectation_density_matrix", "variance"],
)
async def test_analyze_hamiltonian_simulated_ops_reject_qubit_mismatch(
    qmcp: FastMCP,
    operation: str,
) -> None:
    with pytest.raises(ToolError, match=r"EXPECTATION_FAILED.*Qubit count mismatch"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "Z", "coefficient": 1.0}],
                "operation": operation,
                "circuit": BELL_QCIS,
            },
        )


@pytest.mark.asyncio
async def test_analyze_hamiltonian_expectation_from_probs(qmcp: FastMCP) -> None:
    # Little-endian bit strings; "00"/"11" are both +1 eigenstates of ZZ.
    result: Any = await qmcp.call_tool(
        "analyze_hamiltonian",
        {
            "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
            "operation": "expectation",
            "measurements": [
                {"basis": "ZZ", "probs": {"00": 0.5, "11": 0.5}},
            ],
        },
    )
    structured = cast(dict[str, Any], result[1])
    assert structured["expectation"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_analyze_hamiltonian_expectation_requires_measurements(
    qmcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="MISSING_MEASUREMENTS"):
        await qmcp.call_tool(
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 0.5}],
                "operation": "expectation",
            },
        )


# --- calculate_state_metrics -------------------------------------------------

_SQRT1_2 = 1.0 / np.sqrt(2.0)
# State arrays use [real, imag] pairs: pure = 1D pairs, mixed = 2D pairs.
BELL_PURE: list[list[float]] = [
    [_SQRT1_2, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [_SQRT1_2, 0.0],
]
BELL_MIXED: list[list[list[float]]] = [
    [[0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
]
ZERO_PURE_1Q: list[list[float]] = [[1.0, 0.0], [0.0, 0.0]]
MAX_MIXED_1Q: list[list[list[float]]] = [
    [[0.5, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.5, 0.0]],
]
# Trace = 2 -> cqlib rejects construction.
INVALID_TRACE_DM: list[list[list[float]]] = [
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [1.0, 0.0]],
]


async def _state_metrics(qmcp: FastMCP, **kwargs: Any) -> dict[str, Any]:
    result: Any = await qmcp.call_tool("calculate_state_metrics", kwargs)
    return cast(dict[str, Any], result[1])


@pytest.mark.asyncio
async def test_state_metrics_purity_supports_pure_and_mixed_input(
    qmcp: FastMCP,
) -> None:
    pure = await _state_metrics(qmcp, metric="purity", state_a=BELL_PURE, state_a_is_pure=True)
    assert pure["value"] == pytest.approx(1.0)
    assert pure["density_matrix"] is None

    mixed = await _state_metrics(qmcp, metric="purity", state_a=BELL_MIXED, state_a_is_pure=False)
    assert mixed["value"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_state_metrics_fidelity_covers_all_input_combinations(
    qmcp: FastMCP,
) -> None:
    pure_pure = await _state_metrics(
        qmcp,
        metric="fidelity",
        state_a=BELL_PURE,
        state_b=BELL_PURE,
        state_a_is_pure=True,
        state_b_is_pure=True,
    )
    assert pure_pure["value"] == pytest.approx(1.0)

    mixed_mixed = await _state_metrics(
        qmcp,
        metric="fidelity",
        state_a=BELL_MIXED,
        state_b=BELL_MIXED,
    )
    assert mixed_mixed["value"] == pytest.approx(1.0)

    pure_mixed = await _state_metrics(
        qmcp,
        metric="fidelity",
        state_a=BELL_PURE,
        state_b=BELL_MIXED,
        state_a_is_pure=True,
    )
    assert pure_mixed["value"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_state_metrics_trace_distance_pure_pure_and_mixed_mixed(
    qmcp: FastMCP,
) -> None:
    pure = await _state_metrics(
        qmcp,
        metric="trace_distance",
        state_a=ZERO_PURE_1Q,
        state_b=[[0.0, 0.0], [1.0, 0.0]],
        state_a_is_pure=True,
        state_b_is_pure=True,
    )
    assert pure["value"] == pytest.approx(1.0)

    mixed = await _state_metrics(
        qmcp,
        metric="trace_distance",
        state_a=MAX_MIXED_1Q,
        state_b=[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
    )
    assert mixed["value"] == pytest.approx(0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("pure_on_side", ["a", "b"])
async def test_state_metrics_trace_distance_pure_mixed_pair(
    qmcp: FastMCP,
    pure_on_side: str,
) -> None:
    # |0> vs I/2: trace distance is 0.5. cqlib has no trace_distance_pure_mixed,
    # the tool must convert the pure side to a density matrix instead of
    # passing a Statevector into trace_distance_mixed.
    kwargs: dict[str, Any] = {"metric": "trace_distance"}
    if pure_on_side == "a":
        kwargs.update(
            state_a=ZERO_PURE_1Q,
            state_a_is_pure=True,
            state_b=MAX_MIXED_1Q,
        )
    else:
        kwargs.update(
            state_a=MAX_MIXED_1Q,
            state_b=ZERO_PURE_1Q,
            state_b_is_pure=True,
        )
    result = await _state_metrics(qmcp, **kwargs)
    assert result["value"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_state_metrics_entropy(qmcp: FastMCP) -> None:
    pure_state = await _state_metrics(qmcp, metric="entropy", state_a=BELL_MIXED)
    assert pure_state["value"] == pytest.approx(0.0, abs=1e-10)

    mixed = await _state_metrics(qmcp, metric="entropy", state_a=MAX_MIXED_1Q)
    assert mixed["value"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_state_metrics_linear_entropy(qmcp: FastMCP) -> None:
    mixed = await _state_metrics(qmcp, metric="linear_entropy", state_a=MAX_MIXED_1Q)
    assert mixed["value"] == pytest.approx(0.5)

    pure_state = await _state_metrics(qmcp, metric="linear_entropy", state_a=BELL_MIXED)
    assert pure_state["value"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.asyncio
async def test_state_metrics_renyi_entropy(qmcp: FastMCP) -> None:
    result = await _state_metrics(qmcp, metric="renyi_entropy", state_a=MAX_MIXED_1Q, alpha=2.0)
    assert result["value"] == pytest.approx(1.0)

    with pytest.raises(ToolError, match="MISSING_ALPHA"):
        await _state_metrics(qmcp, metric="renyi_entropy", state_a=MAX_MIXED_1Q)

    with pytest.raises(ToolError, match="INVALID_ALPHA"):
        await _state_metrics(qmcp, metric="renyi_entropy", state_a=MAX_MIXED_1Q, alpha=0.0)


@pytest.mark.asyncio
async def test_state_metrics_entanglement_entropy_pure_bell(qmcp: FastMCP) -> None:
    result = await _state_metrics(
        qmcp,
        metric="entanglement_entropy",
        state_a=BELL_PURE,
        state_a_is_pure=True,
        subsystem=[0],
    )
    assert result["value"] == pytest.approx(1.0)

    # cqlib computes this only for pure states.
    with pytest.raises(ToolError, match="INVALID_STATE"):
        await _state_metrics(
            qmcp,
            metric="entanglement_entropy",
            state_a=BELL_MIXED,
            subsystem=[0],
        )

    with pytest.raises(ToolError, match="MISSING_SUBSYSTEM"):
        await _state_metrics(
            qmcp,
            metric="entanglement_entropy",
            state_a=BELL_PURE,
            state_a_is_pure=True,
        )


@pytest.mark.asyncio
async def test_state_metrics_negativity_bell(qmcp: FastMCP) -> None:
    result = await _state_metrics(qmcp, metric="negativity", state_a=BELL_MIXED, subsystem=[0])
    assert result["value"] == pytest.approx(0.5)

    with pytest.raises(ToolError, match="MISSING_SUBSYSTEM"):
        await _state_metrics(qmcp, metric="negativity", state_a=BELL_MIXED)

    with pytest.raises(ToolError, match="METRICS_FAILED"):
        await _state_metrics(qmcp, metric="negativity", state_a=BELL_MIXED, subsystem=[9])


@pytest.mark.asyncio
async def test_state_metrics_logarithmic_negativity_bell(qmcp: FastMCP) -> None:
    result = await _state_metrics(
        qmcp, metric="logarithmic_negativity", state_a=BELL_MIXED, subsystem=[0]
    )
    assert result["value"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_state_metrics_concurrence_and_entanglement_of_formation_bell(
    qmcp: FastMCP,
) -> None:
    concurrence = await _state_metrics(qmcp, metric="concurrence", state_a=BELL_MIXED)
    assert concurrence["value"] == pytest.approx(1.0)

    formation = await _state_metrics(qmcp, metric="entanglement_of_formation", state_a=BELL_MIXED)
    assert formation["value"] == pytest.approx(1.0)

    # Both are only defined for exactly two qubits.
    with pytest.raises(ToolError, match="METRICS_FAILED"):
        await _state_metrics(qmcp, metric="concurrence", state_a=MAX_MIXED_1Q)


@pytest.mark.asyncio
async def test_state_metrics_partial_transpose_returns_matrix(
    qmcp: FastMCP,
) -> None:
    result = await _state_metrics(
        qmcp, metric="partial_transpose", state_a=BELL_MIXED, subsystem=[0]
    )
    assert result["value"] is None
    matrix = result["density_matrix"]
    assert matrix is not None
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)

    flat = np.array(
        [[[cell["real"], cell["imag"]] for cell in row] for row in matrix],
        dtype=float,
    )
    dm = flat[..., 0] + 1j * flat[..., 1]
    assert np.trace(dm).real == pytest.approx(1.0)
    # Partial transpose of a Bell state has eigenvalues {-0.5, 0.5, 0.5, 0.5}.
    eigenvalues = np.linalg.eigvalsh((dm + dm.conj().T) / 2)
    assert sorted(eigenvalues.tolist()) == pytest.approx([-0.5, 0.5, 0.5, 0.5], abs=1e-10)

    with pytest.raises(ToolError, match="MISSING_SUBSYSTEM"):
        await _state_metrics(qmcp, metric="partial_transpose", state_a=BELL_MIXED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "purity"},
        {"metric": "entropy"},
        {"metric": "linear_entropy"},
        {"metric": "fidelity", "state_b": BELL_MIXED},
        {"metric": "trace_distance", "state_b": BELL_MIXED},
        {"metric": "concurrence"},
    ],
)
async def test_state_metrics_requires_state_a(qmcp: FastMCP, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ToolError, match="MISSING_STATE"):
        await _state_metrics(qmcp, **kwargs)


@pytest.mark.asyncio
async def test_state_metrics_rejects_invalid_state(qmcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_STATE"):
        await _state_metrics(qmcp, metric="entropy", state_a=INVALID_TRACE_DM)
