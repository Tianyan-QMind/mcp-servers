"""Tests for cqlib compilation MCP tools (explicit CompileTarget and device compile)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from tianyan_mcp.compile_tools import register_compile_tools

TWO_QUBIT_QCIS = "H Q0\nCX Q0 Q1"
SUB_CIRCUIT_QASM3 = """OPENQASM 3.0;
include "stdgates.inc";
gate my_bell a, b { h a; cx a, b; }
qubit[2] q;
my_bell q[0], q[1];
"""


@pytest.fixture()
def mcp() -> FastMCP:
    server = FastMCP("test-compile-tools")
    register_compile_tools(server)
    return server


@pytest.mark.asyncio
async def test_compile_tools_registered_read_only(mcp: FastMCP) -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for name in ("compile_with_target", "compile_for_device"):
        assert name in tools
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False


@pytest.mark.asyncio
async def test_compile_with_target_logical_keeps_logical_space(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_with_target",
        {"circuit": "H Q0", "target_kind": "logical"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["target_kind"] == "logical"
    assert structured["changed"] is False
    assert structured["initial_layout"] is None
    assert structured["final_layout"] is None
    assert structured["circuit"]


@pytest.mark.asyncio
async def test_compile_with_target_basis_lowers_without_topology(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_with_target",
        {"circuit": "H Q0", "target_kind": "basis", "basis": ["rz", "rx", "ry", "cz"]},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["target_kind"] == "basis"
    assert structured["changed"] is True
    assert structured["initial_layout"] is None
    assert structured["final_layout"] is None


@pytest.mark.asyncio
async def test_compile_with_target_device_returns_layouts(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_with_target",
        {"circuit": TWO_QUBIT_QCIS, "target_kind": "device", "num_qubits": 2},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["target_kind"] == "device"
    initial_layout = structured["initial_layout"]
    assert initial_layout is not None
    assert {int(k): v for k, v in initial_layout.items()} == {0: 0, 1: 1}
    assert structured["final_layout"] is not None


@pytest.mark.asyncio
async def test_compile_with_target_topology_basis_returns_layouts(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_with_target",
        {
            "circuit": TWO_QUBIT_QCIS,
            "target_kind": "topology_basis",
            "num_qubits": 2,
            "basis": ["cz", "rz", "rx", "ry"],
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["target_kind"] == "topology_basis"
    assert structured["initial_layout"] is not None
    assert structured["final_layout"] is not None
    assert structured["circuit"]


@pytest.mark.asyncio
async def test_compile_with_target_topology_basis_accepts_native_gates(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_with_target",
        {
            "circuit": TWO_QUBIT_QCIS,
            "target_kind": "topology_basis",
            "num_qubits": 2,
            "basis": ["cz", "rz", "rx", "ry"],
            "native_gates": ["h", "x", "rz", "cz"],
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["target_kind"] == "topology_basis"
    assert structured["circuit"]


@pytest.mark.asyncio
async def test_compile_with_target_basis_requires_basis(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "basis"},
        )


@pytest.mark.asyncio
async def test_compile_with_target_device_requires_num_qubits(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "device"},
        )


@pytest.mark.asyncio
async def test_compile_with_target_topology_basis_requires_basis_and_num_qubits(
    mcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "topology_basis", "num_qubits": 2},
        )


@pytest.mark.asyncio
async def test_compile_with_target_logical_rejects_basis(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "logical", "basis": ["rz"]},
        )


@pytest.mark.asyncio
async def test_compile_with_target_basis_rejects_device_arguments(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "basis", "basis": ["rz"], "num_qubits": 2},
        )


@pytest.mark.asyncio
async def test_compile_with_target_rejects_unparseable_circuit(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "NOPE Q9", "target_kind": "logical"},
        )


@pytest.mark.asyncio
async def test_compile_with_target_reports_unknown_basis_gate(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="COMPILE_FAILED"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "basis", "basis": ["nope"]},
        )


@pytest.mark.asyncio
async def test_compile_with_target_reports_unlowerable_basis(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="COMPILE_FAILED"):
        await mcp.call_tool(
            "compile_with_target",
            {"circuit": "H Q0", "target_kind": "basis", "basis": ["ccx"]},
        )


@pytest.mark.asyncio
async def test_compile_for_device_default_basis_compiles(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_for_device",
        {"circuit": TWO_QUBIT_QCIS, "num_qubits": 2},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["mode"] == "normal"
    assert structured["circuit"]


@pytest.mark.asyncio
async def test_compile_for_device_returns_initial_layout(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "compile_for_device",
        {"circuit": TWO_QUBIT_QCIS, "num_qubits": 2},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    initial_layout = structured["initial_layout"]
    assert initial_layout is not None
    assert {int(k): v for k, v in initial_layout.items()} == {0: 0, 1: 1}


@pytest.mark.asyncio
async def test_decompose_circuit_expands_definitions_by_default(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "decompose_circuit",
        {"circuit": SUB_CIRCUIT_QASM3, "format": "qasm3"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["transform"] == "expand_definitions"
    assert structured["changed"] is True
    assert structured["circuit"] == "H Q0\nCX Q0 Q1\n"


@pytest.mark.asyncio
async def test_decompose_circuit_circuit_decompose_expands_sub_circuit(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "decompose_circuit",
        {"circuit": SUB_CIRCUIT_QASM3, "format": "qasm3", "transform": "circuit_decompose"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["transform"] == "circuit_decompose"
    assert structured["changed"] is True
    assert structured["circuit"] == "H Q0\nCX Q0 Q1\n"


@pytest.mark.asyncio
async def test_decompose_circuit_circuit_decompose_reports_unchanged_circuit(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "decompose_circuit",
        {"circuit": TWO_QUBIT_QCIS, "transform": "circuit_decompose"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["changed"] is False
    assert structured["circuit"] == "H Q0\nCX Q0 Q1\n"


@pytest.mark.asyncio
async def test_decompose_circuit_mc_controls_makes_mc_decomposition_reachable(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "decompose_circuit",
        {"circuit": "H Q0", "transform": "decompose_mc_gates", "mc_controls": 2},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["changed"] is True
    assert structured["circuit"] == "H Q0\nCCX Q0 Q1 Q2\n"


@pytest.mark.asyncio
async def test_decompose_circuit_mc_gates_for_device_decomposes_within_capacity(
    mcp: FastMCP,
) -> None:
    result = await mcp.call_tool(
        "decompose_circuit",
        {
            "circuit": "H Q0",
            "transform": "decompose_mc_gates_for_device",
            "mc_controls": 3,
            "num_qubits": 4,
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["transform"] == "decompose_mc_gates_for_device"
    assert structured["changed"] is True
    assert "T Q2" in structured["circuit"]


@pytest.mark.asyncio
async def test_decompose_circuit_mc_gates_for_device_reports_capacity_overflow(
    mcp: FastMCP,
) -> None:
    with pytest.raises(ToolError, match="DECOMPOSE_FAILED"):
        await mcp.call_tool(
            "decompose_circuit",
            {
                "circuit": "H Q0",
                "transform": "decompose_mc_gates_for_device",
                "mc_controls": 3,
                "num_qubits": 3,
            },
        )


@pytest.mark.asyncio
async def test_decompose_circuit_rejects_out_of_range_mc_controls(mcp: FastMCP) -> None:
    with pytest.raises(ToolError):
        await mcp.call_tool(
            "decompose_circuit",
            {"circuit": "H Q0", "transform": "decompose_mc_gates", "mc_controls": 0},
        )


@pytest.mark.asyncio
async def test_decompose_circuit_reports_unparseable_circuit(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT"):
        await mcp.call_tool("decompose_circuit", {"circuit": "NOPE Q9"})


@pytest.mark.asyncio
async def test_decompose_circuit_reports_result_not_representable_in_qcis(mcp: FastMCP) -> None:
    qasm2_with_measure = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q -> c;\n'
    )

    with pytest.raises(ToolError, match="DECOMPOSE_FAILED"):
        await mcp.call_tool(
            "decompose_circuit",
            {"circuit": qasm2_with_measure, "format": "qasm2", "transform": "circuit_decompose"},
        )


NATIVE_SUMMARY_KEYS = (
    "native_total_ops",
    "native_two_qubit_ops",
    "native_two_qubit_depth",
    "total_native_depth",
    "predicted_log_error",
    "imputed_error_count",
    "unavailable_error_count",
)


@pytest.mark.asyncio
async def test_optimize_circuit_native_optimization_returns_summary(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "optimize_circuit",
        {"circuit": TWO_QUBIT_QCIS, "optimization_kind": "native_optimization"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["pass_name"] == "native_optimization"
    assert structured["circuit"]
    summary = structured["summary"]
    assert summary is not None
    assert isinstance(summary["rounds"], int)
    assert summary["rounds"] >= 1
    assert isinstance(summary["restored_best"], bool)
    for side in ("before", "after"):
        stats = summary[side]
        for key in NATIVE_SUMMARY_KEYS:
            assert key in stats
        assert isinstance(stats["native_total_ops"], int)
        assert isinstance(stats["native_two_qubit_ops"], int)
        assert isinstance(stats["native_two_qubit_depth"], int)
        assert isinstance(stats["total_native_depth"], int)
        assert stats["predicted_log_error"] is None or isinstance(
            stats["predicted_log_error"], float
        )
    assert summary["after"]["native_two_qubit_ops"] >= 0


@pytest.mark.asyncio
async def test_optimize_circuit_native_optimization_balanced_depth_returns_summary(
    mcp: FastMCP,
) -> None:
    result = await mcp.call_tool(
        "optimize_circuit",
        {
            "circuit": TWO_QUBIT_QCIS,
            "optimization_kind": "native_optimization",
            "native_policy": "balanced_depth",
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["summary"] is not None
    assert structured["summary"]["after"]["native_total_ops"] >= 1


@pytest.mark.asyncio
async def test_optimize_circuit_non_native_pass_omits_summary(mcp: FastMCP) -> None:
    result = await mcp.call_tool(
        "optimize_circuit",
        {"circuit": "X Q0\nX Q0\nH Q0", "optimization_kind": "commutative_cancellation"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["changed"] is True
    assert structured["summary"] is None
