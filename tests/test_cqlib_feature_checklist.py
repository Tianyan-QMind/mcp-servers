"""End-to-end tests for the cqlib MCP feature checklist.

Covers every feature documented in ``cqlib MCP 功能清单与函数调用方式.md``:
circuit conversion / analysis / matrix / drawing / equivalence, compilation
(commutation, device compile, decompose, optimize) and quantum information
(simulation, Pauli, Hamiltonian, state metrics, visualization).

Tools are invoked through FastMCP ``call_tool`` so JSON-schema validation and
error serialization are exercised exactly as an MCP client sees them.
"""

from __future__ import annotations

import base64
import math
from typing import Any, cast

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from tianyan_mcp.circuit_tools import register_circuit_tools
from tianyan_mcp.compile_tools import register_compile_tools
from tianyan_mcp.quantum_info_tools import register_quantum_info_tools

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
INV_SQRT2 = 1.0 / math.sqrt(2.0)

QCIS_BELL = "H Q0\nCX Q0 Q1\n"
QASM2_BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'
# Classical-register measurement assignment: QCIS cannot represent "store".
QASM2_MEASURE_TO_CREG = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n'
)


@pytest.fixture()
def mcp() -> FastMCP:
    server = FastMCP("test-cqlib-checklist")
    register_circuit_tools(server)
    register_compile_tools(server)
    register_quantum_info_tools(server)
    return server


async def call(mcp: FastMCP, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await mcp.call_tool(name, arguments)
    assert isinstance(result, tuple)
    return cast(dict[str, Any], result[1])


# ---------------------------------------------------------------------------
# 线路 1. OpenQASM2/3 与 QCIS 互转
# ---------------------------------------------------------------------------


async def test_convert_qasm2_to_qcis(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "convert_circuit_format",
        {"circuit": QASM2_BELL, "source_format": "qasm2", "target_format": "qcis"},
    )
    assert result["source_format"] == "qasm2"
    assert result["target_format"] == "qcis"
    assert "H Q0" in result["circuit"]
    assert "CX Q0 Q1" in result["circuit"]


async def test_convert_qcis_to_qasm3_roundtrip(mcp: FastMCP) -> None:
    qasm3 = await call(
        mcp,
        "convert_circuit_format",
        {"circuit": QCIS_BELL, "source_format": "qcis", "target_format": "qasm3"},
    )
    assert "OPENQASM 3" in qasm3["circuit"]
    back = await call(
        mcp,
        "convert_circuit_format",
        {
            "circuit": qasm3["circuit"],
            "source_format": "qasm3",
            "target_format": "qcis",
        },
    )
    assert "CX Q0 Q1" in back["circuit"]


async def test_convert_invalid_source_is_invalid_circuit(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_CIRCUIT"):
        await call(
            mcp,
            "convert_circuit_format",
            {
                "circuit": "NOT_A_GATE Q0\n",
                "source_format": "qcis",
                "target_format": "qasm2",
            },
        )


async def test_convert_qcis_cannot_represent_creg_store(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="CONVERSION_FAILED"):
        await call(
            mcp,
            "convert_circuit_format",
            {
                "circuit": QASM2_MEASURE_TO_CREG,
                "source_format": "qasm2",
                "target_format": "qcis",
            },
        )


# ---------------------------------------------------------------------------
# 线路 2. 深度 / 门数统计（含 DAG 与展开统计）
# ---------------------------------------------------------------------------


async def test_analyze_circuit_depth_counts_layers_operations(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "analyze_circuit",
        {"circuit": "H Q0\nCX Q0 Q1\nX Q1\n", "format": "qcis"},
    )
    # H -> CX -> X on the critical path: depth 3 by every definition.
    assert result["depth"] == 3
    assert result["depth_recursive"] == 3
    assert result["dag_depth"] == 3
    assert result["total_operations"] == 3
    assert result["operation_counts"] == {"H": 1, "CX": 1, "X": 1}
    assert result["operation_counts_recursive"] == {"H": 1, "CX": 1, "X": 1}
    assert len(result["layers"]) == 3
    operations = result["operations"]
    assert [op["name"] for op in operations] == ["H", "CX", "X"]
    assert operations[1]["qubits"] == [0, 1]
    assert operations[0]["params"] == []
    assert operations[0]["instruction_type"] == "standard"


async def test_analyze_circuit_counts_parametric_gate(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "analyze_circuit",
        {"circuit": "RY Q0 0.5\nRZ Q1 0.25\n", "format": "qcis"},
    )
    assert result["operation_counts"] == {"RY": 1, "RZ": 1}
    params = result["operations"][0]["params"]
    assert params == [pytest.approx(0.5)]


# ---------------------------------------------------------------------------
# 线路 3. 线路转矩阵（数值 / 符号 / 参数绑定）
# ---------------------------------------------------------------------------


async def test_circuit_to_matrix_hadamard(mcp: FastMCP) -> None:
    result = await call(mcp, "circuit_to_matrix", {"circuit": "H Q0", "format": "qcis"})
    assert result["num_qubits"] == 1
    matrix = result["matrix"]
    assert matrix[0][0]["real"] == pytest.approx(INV_SQRT2)
    assert matrix[1][0]["real"] == pytest.approx(INV_SQRT2)
    assert matrix[1][1]["real"] == pytest.approx(-INV_SQRT2)
    assert all(cell["imag"] == pytest.approx(0.0) for row in matrix for cell in row)


async def test_circuit_to_matrix_binds_parameters(mcp: FastMCP) -> None:
    # RY(pi) = [[0, -1], [1, 0]]
    result = await call(
        mcp,
        "circuit_to_matrix",
        {"circuit": "RY Q0 theta", "format": "qcis", "parameters": {"theta": math.pi}},
    )
    matrix = result["matrix"]
    assert matrix[0][0]["real"] == pytest.approx(0.0, abs=1e-10)
    assert matrix[0][1]["real"] == pytest.approx(-1.0, abs=1e-10)
    assert matrix[1][0]["real"] == pytest.approx(1.0, abs=1e-10)
    assert matrix[1][1]["real"] == pytest.approx(0.0, abs=1e-10)


async def test_symbolic_matrix_keeps_expression(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "circuit_to_symbolic_matrix",
        {"circuit": "RY Q0 theta", "format": "qcis"},
    )
    assert result["num_qubits"] == 1
    assert result["symbols"] == ["theta"]
    assert result["matrix"][0][0]["real"] == "cos(theta/2)"
    assert result["matrix"][0][1]["real"] == "-sin(theta/2)"


# ---------------------------------------------------------------------------
# 线路 4. 线路可视化（text / svg / png）
# ---------------------------------------------------------------------------


async def test_draw_circuit_text_svg_png(mcp: FastMCP) -> None:
    text = await call(mcp, "draw_circuit", {"circuit": QCIS_BELL, "kind": "text"})
    assert text["media_type"] == "text/plain"
    assert "Q0" in text["content"]
    assert "Q1" in text["content"]

    svg = await call(mcp, "draw_circuit", {"circuit": QCIS_BELL, "kind": "svg"})
    assert svg["content"].startswith("<svg")
    assert svg["media_type"] == "image/svg+xml"

    png = await call(mcp, "draw_circuit", {"circuit": QCIS_BELL, "kind": "png"})
    assert png["encoding"] == "base64"
    assert png["media_type"] == "image/png"
    assert base64.b64decode(png["content"]).startswith(PNG_MAGIC)


# ---------------------------------------------------------------------------
# 线路 5. 结果可视化（ExecutionResult.from_counts + plot）
# ---------------------------------------------------------------------------


async def test_plot_measurement_histogram_and_distribution(mcp: FastMCP) -> None:
    args = {
        "counts": {"00": 512, "11": 488},
        "qubits": [0, 1],
        "shots": 1000,
        "num_qubits": 2,
    }
    histogram = await call(mcp, "plot_measurement_result", {**args, "kind": "histogram"})
    assert histogram["content"].startswith("<svg")
    distribution = await call(
        mcp,
        "plot_measurement_result",
        {**args, "kind": "distribution", "image_format": "png"},
    )
    assert distribution["encoding"] == "base64"
    assert base64.b64decode(distribution["content"]).startswith(PNG_MAGIC)


# ---------------------------------------------------------------------------
# 线路 7. 等价线路判断（含全局相位）
# ---------------------------------------------------------------------------


async def test_equivalence_hxh_equals_z(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "check_circuit_equivalence",
        {"circuit_a": "H Q0\nX Q0\nH Q0\n", "circuit_b": "Z Q0\n"},
    )
    assert result["equivalent"] is True
    assert result["max_abs_diff"] == pytest.approx(0.0, abs=1e-10)
    assert result["global_phase_ignored"] is True


async def test_equivalence_different_gates_not_equivalent(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "check_circuit_equivalence",
        {"circuit_a": "H Q0\n", "circuit_b": "X Q0\n"},
    )
    assert result["equivalent"] is False
    assert result["max_abs_diff"] > 0.1


async def test_equivalence_different_qubit_counts(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "check_circuit_equivalence",
        {"circuit_a": "H Q0\n", "circuit_b": QCIS_BELL},
    )
    assert result["equivalent"] is False
    assert result["max_abs_diff"] == float("inf")


# ---------------------------------------------------------------------------
# 编译 1. 对易门检查
# ---------------------------------------------------------------------------


async def test_commutation_rz_and_x_do_not_commute(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "check_gate_commutation",
        {"left_gate": "RZ", "left_param": 0.2, "right_gate": "X", "qubit": 0},
    )
    assert result["commutes"] is False
    assert result["phase"] is None
    assert result["method"] == "check"


async def test_commutation_z_and_rz_commute_algebraic(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "check_gate_commutation",
        {
            "left_gate": "Z",
            "right_gate": "RZ",
            "right_param": 0.3,
            "qubit": 0,
            "method": "algebraic",
        },
    )
    assert result["commutes"] is True
    assert result["exact"] is True
    assert result["method"] == "algebraic"


async def test_commutation_unsupported_gate_rejected(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="UNSUPPORTED_GATE"):
        await call(
            mcp,
            "check_gate_commutation",
            {"left_gate": "NOT_A_GATE", "right_gate": "X", "qubit": 0},
        )


# ---------------------------------------------------------------------------
# 编译 2. 针对设备的编译
# ---------------------------------------------------------------------------


async def test_compile_for_device_routes_nonadjacent_cx(mcp: FastMCP) -> None:
    # Logical Q0-Q2 are not adjacent on a 3-qubit line: routing must remap.
    result = await call(
        mcp,
        "compile_for_device",
        {"circuit": "H Q0\nCX Q0 Q2\n", "num_qubits": 3, "seed": 42},
    )
    assert result["changed"] is True
    assert result["mode"] == "normal"
    assert result["initial_layout"] is not None
    assert set(result["initial_layout"]) == {"0", "2"}
    assert result["final_layout"] is not None
    assert "CX" in result["circuit"]


async def test_compile_for_device_seed_is_reproducible(mcp: FastMCP) -> None:
    args = {"circuit": "H Q0\nCX Q0 Q2\n", "num_qubits": 4, "seed": 7}
    first = await call(mcp, "compile_for_device", args)
    second = await call(mcp, "compile_for_device", args)
    assert first["circuit"] == second["circuit"]


# ---------------------------------------------------------------------------
# 编译 2b. 显式 CompileTarget
# ---------------------------------------------------------------------------


async def test_compile_with_target_logical_basis_and_topology(mcp: FastMCP) -> None:
    logical = await call(
        mcp,
        "compile_with_target",
        {"circuit": QCIS_BELL, "target_kind": "logical"},
    )
    assert logical["target_kind"] == "logical"
    assert "CX Q0 Q1" in logical["circuit"]

    basis = await call(
        mcp,
        "compile_with_target",
        {"circuit": QCIS_BELL, "target_kind": "basis", "basis": ["cx", "h", "rz"]},
    )
    # H and CX are already in the target basis: circuit is preserved, possibly
    # reported as unchanged.
    assert "H Q0" in basis["circuit"]
    assert "CX Q0 Q1" in basis["circuit"]

    topology = await call(
        mcp,
        "compile_with_target",
        {
            "circuit": QCIS_BELL,
            "target_kind": "topology_basis",
            "basis": ["h", "rz", "cz"],
            "num_qubits": 2,
        },
    )
    # CX = H CZ H, so lowering CX to {H, RZ, CZ} surrounds CZ with H on Q1.
    assert topology["changed"] is True
    assert "CZ" in topology["circuit"]
    assert topology["initial_layout"] == {"0": 0, "1": 1}


async def test_compile_with_target_validates_argument_combinations(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await call(
            mcp,
            "compile_with_target",
            {"circuit": QCIS_BELL, "target_kind": "logical", "basis": ["cz"]},
        )
    with pytest.raises(ToolError, match="INVALID_TARGET_ARGS"):
        await call(
            mcp,
            "compile_with_target",
            {"circuit": QCIS_BELL, "target_kind": "device"},
        )


async def test_compile_with_target_infeasible_basis_fails(mcp: FastMCP) -> None:
    # {RZ, CZ} alone cannot synthesize H: the compiler rejects the plan.
    with pytest.raises(ToolError, match="COMPILE_FAILED"):
        await call(
            mcp,
            "compile_with_target",
            {
                "circuit": QCIS_BELL,
                "target_kind": "topology_basis",
                "basis": ["rz", "cz"],
                "num_qubits": 2,
            },
        )


# ---------------------------------------------------------------------------
# 编译 3. 量子门分解
# ---------------------------------------------------------------------------


async def test_decompose_mc_gate_to_basic_gates(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "decompose_circuit",
        {"circuit": "H Q0\n", "transform": "decompose_mc_gates", "mc_controls": 3},
    )
    assert result["changed"] is True
    assert "CCX" not in result["circuit"]
    assert "CX" in result["circuit"]


async def test_decompose_two_controls_is_standard_ccx(mcp: FastMCP) -> None:
    # CCX is itself a cqlib standard gate, so the generic MC gate normalizes to it.
    result = await call(
        mcp,
        "decompose_circuit",
        {"circuit": "H Q0\n", "transform": "decompose_mc_gates", "mc_controls": 2},
    )
    assert result["changed"] is True
    assert "CCX Q0 Q1 Q2" in result["circuit"]


async def test_decompose_unitaries_with_cz_basis(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "decompose_circuit",
        {
            "circuit": "H Q0\n",
            "transform": "decompose_unitaries",
            "two_qubit_basis": "cz",
        },
    )
    # H is already a standard gate: nothing to synthesize.
    assert result["changed"] is False
    assert "H Q0" in result["circuit"]


async def test_circuit_decompose_transform(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "decompose_circuit",
        {"circuit": "H Q0\n", "transform": "circuit_decompose"},
    )
    assert result["transform"] == "circuit_decompose"
    assert "H Q0" in result["circuit"]


# ---------------------------------------------------------------------------
# 编译 4. 量子门优化
# ---------------------------------------------------------------------------


async def test_commutative_cancellation_removes_xx_pair(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "optimize_circuit",
        {
            "circuit": "X Q0\nX Q0\nH Q1\n",
            "optimization_kind": "commutative_cancellation",
        },
    )
    assert result["changed"] is True
    assert result["circuit"].strip() == "H Q1"
    assert result["summary"] is None


async def test_canonicalize_and_one_qubit_runs(mcp: FastMCP) -> None:
    canonical = await call(
        mcp, "optimize_circuit", {"circuit": QCIS_BELL, "optimization_kind": "canonicalize"}
    )
    assert "CX Q0 Q1" in canonical["circuit"]

    folded = await call(
        mcp,
        "optimize_circuit",
        {
            "circuit": "H Q0\nT Q0\nS Q0\n",
            "optimization_kind": "optimize_one_qubit_runs",
        },
    )
    assert folded["changed"] is True


async def test_native_optimization_reports_statistics(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "optimize_circuit",
        {
            "circuit": QCIS_BELL,
            "optimization_kind": "native_optimization",
            "native_policy": "balanced_depth",
        },
    )
    summary = result["summary"]
    assert summary is not None
    assert summary["before"]["native_two_qubit_ops"] == 1
    assert summary["after"]["native_two_qubit_depth"] >= 1
    assert summary["rounds"] >= 1
    assert isinstance(summary["restored_best"], bool)


# ---------------------------------------------------------------------------
# 量子信息 1. 仿真：状态向量 / 密度矩阵 / 带噪声 / 稳定子
# ---------------------------------------------------------------------------


async def test_simulate_statevector_bell(mcp: FastMCP) -> None:
    result = await call(mcp, "simulate_circuit", {"circuit": QCIS_BELL})
    assert result["simulator"] == "statevector"
    assert result["num_qubits"] == 2
    assert set(result["probabilities"]) == {"00", "11"}
    assert result["probabilities"]["00"] == pytest.approx(0.5, abs=1e-12)
    assert len(result["state_vector"]) == 4
    assert result["state_vector"][0]["real"] == pytest.approx(INV_SQRT2)
    assert result["state_vector"][3]["real"] == pytest.approx(INV_SQRT2)


async def test_simulate_density_matrix_bell(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "simulate_circuit",
        {"circuit": QCIS_BELL, "simulator": "density_matrix"},
    )
    matrix = result["density_matrix"]
    assert len(matrix) == 4
    # Bell density matrix: 0.5 on the four |00><00|, |00><11|, |11><00|, |11><11|
    assert matrix[0][0]["real"] == pytest.approx(0.5)
    assert matrix[0][3]["real"] == pytest.approx(0.5)
    assert matrix[3][0]["real"] == pytest.approx(0.5)
    assert matrix[3][3]["real"] == pytest.approx(0.5)
    assert result["state_vector"] is None


async def test_simulate_noise_and_readout_channels(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "simulate_circuit",
        {
            "circuit": "X Q0\n",
            "simulator": "density_matrix_noise",
            "noise_errors": [
                {"gate": "X", "qubit": 0, "noise_type": "bit_flip", "probability": 0.1}
            ],
            "readout_errors": [{"qubit": 0, "p_0_given_1": 0.05, "p_1_given_0": 0.05}],
        },
    )
    # X then 10% bit flip: 10% in |0>, 90% in |1>.
    assert result["probabilities"]["0"] == pytest.approx(0.1, abs=1e-12)
    assert result["probabilities"]["1"] == pytest.approx(0.9, abs=1e-12)
    # Readout: P(0) = 0.1*0.95 + 0.9*0.05 = 0.14; P(1) = 0.86.
    assert result["readout_probabilities"]["0"] == pytest.approx(0.14, abs=1e-12)
    assert result["readout_probabilities"]["1"] == pytest.approx(0.86, abs=1e-12)


async def test_simulate_noise_args_rejected_for_statevector(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="INVALID_NOISE"):
        await call(
            mcp,
            "simulate_circuit",
            {
                "circuit": "X Q0\n",
                "noise_errors": [
                    {"gate": "X", "qubit": 0, "noise_type": "bit_flip", "probability": 0.1}
                ],
            },
        )


async def test_simulate_stabilizer_records_measurement(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "simulate_circuit",
        {"circuit": "X Q0\nM Q0\n", "simulator": "stabilizer"},
    )
    assert result["probabilities"] == {"1": pytest.approx(1.0)}
    classical = result["classical_data"]
    assert classical["vars"] == []
    value = classical["values"][0]
    assert value["kind"] == "bit"
    assert value["value"] is True
    assert value["qubits"] == [0]


async def test_simulate_applies_additional_circuit(mcp: FastMCP) -> None:
    # H then H returns the qubit to |0>.
    result = await call(
        mcp,
        "simulate_circuit",
        {
            "circuit": "H Q0\n",
            "additional_circuit": "H Q0\n",
        },
    )
    assert result["probabilities"] == {"0": pytest.approx(1.0, abs=1e-12)}


async def test_simulate_visualization_svg_and_png(mcp: FastMCP) -> None:
    svg = await call(
        mcp,
        "simulate_circuit",
        {"circuit": "H Q0\n", "visualize": "bloch"},
    )
    assert svg["state_svg"].startswith("<svg")
    png = await call(
        mcp,
        "simulate_circuit",
        {"circuit": "H Q0\n", "visualize": "city", "visualize_format": "png"},
    )
    assert base64.b64decode(png["state_png_base64"]).startswith(PNG_MAGIC)


# ---------------------------------------------------------------------------
# 量子信息 6. 量子态可视化（独立绘图入口）
# ---------------------------------------------------------------------------


async def test_plot_bloch_vector_svg_and_png(mcp: FastMCP) -> None:
    svg = await call(mcp, "plot_bloch_vector", {"vector": [0, 0, 1]})
    assert svg["kind"] == "bloch_vector"
    assert svg["content"].startswith("<svg")
    png = await call(mcp, "plot_bloch_vector", {"vector": [1, 0, 0], "image_format": "png"})
    assert base64.b64decode(png["content"]).startswith(PNG_MAGIC)


async def test_plot_state_kinds_for_bell(mcp: FastMCP) -> None:
    for kind in ("bloch_multivector", "city", "paulivec"):
        result = await call(
            mcp,
            "plot_state",
            {"kind": kind, "circuit": QCIS_BELL},
        )
        assert result["kind"] == kind
        assert result["content"].startswith("<svg")


async def test_plot_state_requires_inputs(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="MISSING_VECTOR"):
        await call(mcp, "plot_state", {"kind": "bloch_vector"})
    with pytest.raises(ToolError, match="MISSING_CIRCUIT"):
        await call(mcp, "plot_state", {"kind": "city"})


# ---------------------------------------------------------------------------
# 量子信息 2. Pauli 构造 / 乘法 / 对易 / 矩阵
# ---------------------------------------------------------------------------


async def test_pauli_info_phase_and_support(mcp: FastMCP) -> None:
    result = await call(mcp, "analyze_pauli_string", {"pauli": "-iXZ"})
    assert result["pauli_string"] == "-iXZ"
    assert result["phase"] == "-i"
    assert result["support"] == [0, 1]


async def test_pauli_multiply_x_y_is_i_z(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "analyze_pauli_string",
        {"pauli": "X", "operation": "multiply", "other": "Y"},
    )
    assert result["product"] == "+iZ"


async def test_pauli_commutation(mcp: FastMCP) -> None:
    anti = await call(
        mcp,
        "analyze_pauli_string",
        {"pauli": "X", "operation": "commutes", "other": "Z"},
    )
    assert anti["commutes_with"] is False
    commute = await call(
        mcp,
        "analyze_pauli_string",
        {"pauli": "XX", "operation": "commutes", "other": "ZZ"},
    )
    assert commute["commutes_with"] is True


async def test_pauli_matrix_z(mcp: FastMCP) -> None:
    result = await call(mcp, "analyze_pauli_string", {"pauli": "Z", "operation": "matrix"})
    matrix = result["matrix"]
    assert matrix[0][0]["real"] == pytest.approx(1.0)
    assert matrix[1][1]["real"] == pytest.approx(-1.0)


async def test_pauli_multiply_requires_other(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="MISSING_OTHER"):
        await call(
            mcp,
            "analyze_pauli_string",
            {"pauli": "X", "operation": "multiply"},
        )


# ---------------------------------------------------------------------------
# 量子信息 3/4. Hamiltonian 构造 / 化简 / 矩阵 / 期望值 / 方差 / Trotter
# ---------------------------------------------------------------------------


async def test_hamiltonian_info_simplifies_like_terms(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "analyze_hamiltonian",
        {
            "terms": [
                {"pauli": "ZZ", "coefficient": 0.5},
                {"pauli": "ZZ", "coefficient": 0.3},
            ]
        },
    )
    assert result["num_qubits"] == 2
    assert result["num_terms"] == 1
    assert result["terms"][0]["pauli"] == "+ZZ"
    assert result["terms"][0]["coefficient"] == pytest.approx(0.8)
    assert result["all_terms_commute"] is True


async def test_hamiltonian_additional_terms_scale_and_mismatch(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "analyze_hamiltonian",
        {
            "terms": [{"pauli": "Z", "coefficient": 0.5}],
            "additional_terms": [[{"pauli": "X", "coefficient": 0.25}]],
            "scale": 2.0,
        },
    )
    coefficients = {term["pauli"]: term["coefficient"] for term in result["terms"]}
    assert coefficients["+Z"] == pytest.approx(1.0)
    assert coefficients["+X"] == pytest.approx(0.5)

    with pytest.raises(ToolError, match="HAMILTONIAN_MISMATCH"):
        await call(
            mcp,
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "Z", "coefficient": 1.0}],
                "additional_terms": [[{"pauli": "ZZ", "coefficient": 1.0}]],
            },
        )


async def test_hamiltonian_matrix_z(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "analyze_hamiltonian",
        {"terms": [{"pauli": "Z", "coefficient": 1.0}], "operation": "matrix"},
    )
    matrix = result["matrix"]
    assert matrix[0][0]["real"] == pytest.approx(1.0)
    assert matrix[1][1]["real"] == pytest.approx(-1.0)


async def test_hamiltonian_expectation_from_measurement_probs(mcp: FastMCP) -> None:
    # Bit strings are little-endian: "01" means q0=1, q1=0 -> ZZ eigenvalue -1
    # and XX eigenvalue -1; <H> = 0.5*(-1) + 0.3*(-1) = -0.8.
    result = await call(
        mcp,
        "analyze_hamiltonian",
        {
            "terms": [
                {"pauli": "ZZ", "coefficient": 0.5},
                {"pauli": "XX", "coefficient": 0.3},
            ],
            "operation": "expectation",
            "measurements": [
                {"basis": "ZZ", "probs": {"01": 1.0}},
                {"basis": "XX", "probs": {"01": 1.0}},
            ],
        },
    )
    assert result["expectation"] == pytest.approx(-0.8, abs=1e-12)


async def test_hamiltonian_expectation_missing_basis_is_semantic_error(
    mcp: FastMCP,
) -> None:
    # H has ZZ and XX terms but only a ZZ measurement basis is supplied: cqlib
    # raises ValueError, which must surface as EXPECTATION_FAILED, not INTERNAL_ERROR.
    with pytest.raises(ToolError, match="EXPECTATION_FAILED"):
        await call(
            mcp,
            "analyze_hamiltonian",
            {
                "terms": [
                    {"pauli": "ZZ", "coefficient": 0.5},
                    {"pauli": "XX", "coefficient": 0.3},
                ],
                "operation": "expectation",
                "measurements": [{"basis": "ZZ", "probs": {"01": 1.0}}],
            },
        )


async def test_hamiltonian_expectation_requires_measurements(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="MISSING_MEASUREMENTS"):
        await call(
            mcp,
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 1.0}],
                "operation": "expectation",
            },
        )


async def test_hamiltonian_statevector_expectation_and_variance_bell(
    mcp: FastMCP,
) -> None:
    args = {"terms": [{"pauli": "ZZ", "coefficient": 1.0}]}
    expectation = await call(
        mcp,
        "analyze_hamiltonian",
        {**args, "operation": "expectation_statevector", "circuit": QCIS_BELL},
    )
    # Bell state is a +1 eigenstate of ZZ.
    assert expectation["expectation"] == pytest.approx(1.0, abs=1e-12)

    dm_expectation = await call(
        mcp,
        "analyze_hamiltonian",
        {**args, "operation": "expectation_density_matrix", "circuit": QCIS_BELL},
    )
    assert dm_expectation["expectation"] == pytest.approx(1.0, abs=1e-12)

    variance = await call(
        mcp,
        "analyze_hamiltonian",
        {**args, "operation": "variance", "circuit": QCIS_BELL},
    )
    assert variance["variance"] == pytest.approx(0.0, abs=1e-12)


async def test_hamiltonian_simulated_operation_requires_circuit(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="MISSING_CIRCUIT"):
        await call(
            mcp,
            "analyze_hamiltonian",
            {
                "terms": [{"pauli": "ZZ", "coefficient": 1.0}],
                "operation": "expectation_statevector",
            },
        )


async def test_hamiltonian_trotter_circuit_modes(mcp: FastMCP) -> None:
    terms = [
        {"pauli": "ZZ", "coefficient": 0.5},
        {"pauli": "XX", "coefficient": 0.3},
    ]
    first = await call(
        mcp,
        "analyze_hamiltonian",
        {
            "terms": terms,
            "operation": "trotter_circuit",
            "trotter_time": 0.5,
            "trotter_steps": 2,
        },
    )
    assert "RZ" in first["trotter_circuit"]

    qasm3 = await call(
        mcp,
        "analyze_hamiltonian",
        {
            "terms": terms,
            "operation": "trotter_circuit",
            "trotter_mode": "second_order",
            "output_format": "qasm3",
        },
    )
    assert "OPENQASM 3" in qasm3["trotter_circuit"]

    with pytest.raises(ToolError, match="MISSING_RANDOM_SEED"):
        await call(
            mcp,
            "analyze_hamiltonian",
            {"terms": terms, "operation": "trotter_circuit", "trotter_mode": "randomized"},
        )

    randomized = await call(
        mcp,
        "analyze_hamiltonian",
        {
            "terms": terms,
            "operation": "trotter_circuit",
            "trotter_mode": "randomized",
            "random_seed": 42,
        },
    )
    assert randomized["trotter_circuit"]


# ---------------------------------------------------------------------------
# 量子信息 5. state_metrics（保真度 / 迹距离 / 纯度 / 熵 / 纠缠）
# ---------------------------------------------------------------------------

PURE_ZERO = [[1.0, 0.0], [0.0, 0.0]]
PURE_ONE = [[0.0, 0.0], [1.0, 0.0]]
PURE_PLUS = [[INV_SQRT2, 0.0], [INV_SQRT2, 0.0]]
PURE_BELL = [
    [INV_SQRT2, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [INV_SQRT2, 0.0],
]
DM_ZERO = [
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0]],
]
DM_MAX_MIXED_1Q = [
    [[0.5, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.5, 0.0]],
]
DM_BELL = [
    [[0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
]


async def test_metric_purity_pure_and_mixed(mcp: FastMCP) -> None:
    pure = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "purity", "state_a": PURE_ZERO, "state_a_is_pure": True},
    )
    assert pure["value"] == pytest.approx(1.0)
    mixed = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "purity", "state_a": DM_MAX_MIXED_1Q},
    )
    assert mixed["value"] == pytest.approx(0.5)


async def test_metric_fidelity_all_input_combinations(mcp: FastMCP) -> None:
    pure_pure = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "fidelity",
            "state_a": PURE_ZERO,
            "state_a_is_pure": True,
            "state_b": PURE_PLUS,
            "state_b_is_pure": True,
        },
    )
    # |<0|+>|^2 = 0.5
    assert pure_pure["value"] == pytest.approx(0.5)

    mixed_mixed = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "fidelity",
            "state_a": DM_ZERO,
            "state_b": DM_ZERO,
        },
    )
    assert mixed_mixed["value"] == pytest.approx(1.0)

    pure_mixed = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "fidelity",
            "state_a": PURE_ZERO,
            "state_a_is_pure": True,
            "state_b": DM_MAX_MIXED_1Q,
        },
    )
    assert pure_mixed["value"] == pytest.approx(0.5)


async def test_metric_trace_distance_including_pure_mixed(mcp: FastMCP) -> None:
    orthogonal = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "trace_distance",
            "state_a": PURE_ZERO,
            "state_a_is_pure": True,
            "state_b": PURE_ONE,
            "state_b_is_pure": True,
        },
    )
    assert orthogonal["value"] == pytest.approx(1.0)

    identical = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "trace_distance",
            "state_a": DM_MAX_MIXED_1Q,
            "state_b": DM_MAX_MIXED_1Q,
        },
    )
    assert identical["value"] == pytest.approx(0.0)

    # cqlib has no native pure-vs-mixed trace distance: the server promotes the
    # pure side to a density matrix. |0> vs I/2 -> 0.5.
    cross = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "trace_distance",
            "state_a": PURE_ZERO,
            "state_a_is_pure": True,
            "state_b": DM_MAX_MIXED_1Q,
        },
    )
    assert cross["value"] == pytest.approx(0.5)


async def test_metric_entropies_on_maximally_mixed_state(mcp: FastMCP) -> None:
    # cqlib reports entropies in bits (log base 2): I/2 has entropy 1.
    entropy = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "entropy", "state_a": DM_MAX_MIXED_1Q},
    )
    assert entropy["value"] == pytest.approx(1.0, abs=1e-10)

    linear = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "linear_entropy", "state_a": DM_MAX_MIXED_1Q},
    )
    assert linear["value"] == pytest.approx(0.5)

    renyi = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "renyi_entropy",
            "state_a": DM_MAX_MIXED_1Q,
            "alpha": 2.0,
        },
    )
    assert renyi["value"] == pytest.approx(1.0, abs=1e-10)


async def test_metric_entanglement_of_bell_state(mcp: FastMCP) -> None:
    ent_entropy = await call(
        mcp,
        "calculate_state_metrics",
        {
            "metric": "entanglement_entropy",
            "state_a": PURE_BELL,
            "state_a_is_pure": True,
            "subsystem": [0],
        },
    )
    assert ent_entropy["value"] == pytest.approx(1.0, abs=1e-10)

    negativity = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "negativity", "state_a": DM_BELL, "subsystem": [0]},
    )
    assert negativity["value"] == pytest.approx(0.5, abs=1e-10)

    log_negativity = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "logarithmic_negativity", "state_a": DM_BELL, "subsystem": [0]},
    )
    assert log_negativity["value"] == pytest.approx(1.0, abs=1e-10)

    concurrence = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "concurrence", "state_a": DM_BELL},
    )
    assert concurrence["value"] == pytest.approx(1.0, abs=1e-10)

    eof = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "entanglement_of_formation", "state_a": DM_BELL},
    )
    assert eof["value"] == pytest.approx(1.0, abs=1e-10)


async def test_metric_partial_transpose_returns_matrix(mcp: FastMCP) -> None:
    result = await call(
        mcp,
        "calculate_state_metrics",
        {"metric": "partial_transpose", "state_a": DM_BELL, "subsystem": [0]},
    )
    assert result["value"] is None
    assert len(result["density_matrix"]) == 4
    assert len(result["density_matrix"][0]) == 4


async def test_metric_validates_arguments(mcp: FastMCP) -> None:
    with pytest.raises(ToolError, match="MISSING_STATE"):
        await call(mcp, "calculate_state_metrics", {"metric": "purity"})
    with pytest.raises(ToolError, match="MISSING_SUBSYSTEM"):
        await call(
            mcp,
            "calculate_state_metrics",
            {"metric": "negativity", "state_a": DM_BELL},
        )
    with pytest.raises(ToolError, match="MISSING_ALPHA"):
        await call(
            mcp,
            "calculate_state_metrics",
            {"metric": "renyi_entropy", "state_a": DM_MAX_MIXED_1Q},
        )
    with pytest.raises(ToolError, match="INVALID_ALPHA"):
        await call(
            mcp,
            "calculate_state_metrics",
            {"metric": "renyi_entropy", "state_a": DM_MAX_MIXED_1Q, "alpha": 0},
        )
    with pytest.raises(ToolError, match="INVALID_STATE"):
        await call(
            mcp,
            "calculate_state_metrics",
            {
                "metric": "entanglement_entropy",
                "state_a": DM_BELL,
                "subsystem": [0],
            },
        )


async def test_all_cqlib_tools_are_registered_read_only(mcp: FastMCP) -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    expected = {
        "convert_circuit_format",
        "analyze_circuit",
        "circuit_to_matrix",
        "circuit_to_symbolic_matrix",
        "draw_circuit",
        "plot_measurement_result",
        "check_circuit_equivalence",
        "check_gate_commutation",
        "compile_for_device",
        "compile_with_target",
        "decompose_circuit",
        "optimize_circuit",
        "simulate_circuit",
        "plot_bloch_vector",
        "plot_state",
        "analyze_pauli_string",
        "analyze_hamiltonian",
        "calculate_state_metrics",
    }
    assert expected <= set(tools)
    for name in expected:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False
