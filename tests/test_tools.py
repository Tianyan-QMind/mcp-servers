import json
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_models import (
    ExperimentBatchSubmissionResult,
    ExperimentResultQueryResult,
    ExperimentSubmissionResult,
)
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.models import BackendListResult, BackendType
from tianyan_mcp.server import create_mcp_server
from tianyan_mcp.service import BackendService
from tianyan_mcp.waveform_models import WaveformDiagramResult, WaveformGenerationResult
from tianyan_mcp.waveform_service import WaveformService

FETCHED_AT = datetime(2026, 7, 29, 8, 30, tzinfo=UTC)


class FakeService:
    async def list_available(
        self,
        backend_type: BackendType = BackendType.ALL,
    ) -> BackendListResult:
        return BackendListResult(
            fetched_at=FETCHED_AT,
            backend_type_filter=backend_type,
            total=0,
            backends=[],
        )

    async def get_backend(self, identifier: str) -> Any:
        raise ServiceError("RESOURCE_NOT_FOUND", f"missing: {identifier}", False)


class BrokenService(FakeService):
    async def list_available(
        self,
        backend_type: BackendType = BackendType.ALL,
    ) -> BackendListResult:
        raise ValueError("sensitive implementation detail")


class FakeExperimentService:
    async def submit(
        self,
        *,
        computer_code: str,
        circuit: str,
        name: str,
        lab_name: str | None,
        shots: int,
        allow_hardware: bool,
    ) -> ExperimentSubmissionResult:
        return ExperimentSubmissionResult(
            query_id="query-1",
            backend_code=computer_code,
            resource_type=BackendType.SIMULATOR,
            shots=shots,
            name=name,
            lab_name=lab_name,
        )

    async def submit_batch(
        self,
        *,
        computer_code: str,
        circuits: list[str],
        name: str,
        lab_name: str | None,
        shots: int,
        allow_hardware: bool,
    ) -> ExperimentBatchSubmissionResult:
        return ExperimentBatchSubmissionResult(
            query_ids=tuple(f"query-{index}" for index, _ in enumerate(circuits, start=1)),
            backend_code=computer_code,
            resource_type=BackendType.SIMULATOR,
            shots=shots,
            name=name,
            lab_name=lab_name,
        )


class BrokenExperimentService(FakeExperimentService):
    async def submit(
        self,
        *,
        computer_code: str,
        circuit: str,
        name: str,
        lab_name: str | None,
        shots: int,
        allow_hardware: bool,
    ) -> ExperimentSubmissionResult:
        raise ValueError("sensitive submission detail")


class FakeExperimentResultService:
    async def query(
        self, query_id: str, include_result_status: bool
    ) -> ExperimentResultQueryResult:
        return ExperimentResultQueryResult(
            query_id=query_id,
            status="completed",
            upstream_status=0,
            probability={"0": 1.0},
            experiment_task_id=query_id,
            result_status=((0,),) if include_result_status else None,
        )


class FakeWaveformService:
    async def generate_waveform(
        self,
        *,
        qc_code: str,
        circuit: str,
        circuit_name: str | None,
        is_verify: bool,
    ) -> WaveformGenerationResult:
        return WaveformGenerationResult(
            id="1823156489274675200",
            qc_code=qc_code,
            circuit_name=circuit_name or "example1700000000000",
            circuit_content=circuit,
            status="generating",
            upstream_status=0,
            measure_qubits=("Q0",),
        )

    async def get_waveform(self, task_id: str) -> WaveformDiagramResult:
        return WaveformDiagramResult(
            id=task_id,
            circuit_name="bell-state",
            status="completed",
            visible_url="https://files.example.com/waveform/xxx?token=tok",
        )


class BrokenWaveformService(FakeWaveformService):
    async def generate_waveform(
        self,
        *,
        qc_code: str,
        circuit: str,
        circuit_name: str | None,
        is_verify: bool,
    ) -> WaveformGenerationResult:
        raise ValueError("sensitive waveform detail")

    async def get_waveform(self, task_id: str) -> WaveformDiagramResult:
        raise ServiceError("WAVEFORM_TASK_NOT_FOUND", "未查询到波形线路数据", False)


def create_test_server(
    service: FakeService,
    experiment_service: FakeExperimentService | None = None,
    result_service: FakeExperimentResultService | None = None,
    waveform_service: FakeWaveformService | None = None,
) -> FastMCP:
    return create_mcp_server(
        cast(BackendService, service),
        cast(ExperimentService, experiment_service or FakeExperimentService()),
        cast(ExperimentResultService, result_service or FakeExperimentResultService()),
        cast(WaveformService, waveform_service or FakeWaveformService()),
    )


@pytest.mark.asyncio
async def test_tool_catalog_and_input_schema() -> None:
    mcp = create_test_server(FakeService())

    tools = await mcp.list_tools()
    by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {
        "list_available_quantum_backends",
        "get_quantum_backend",
        "submit_quantum_experiment",
        "submit_quantum_experiment_batch",
        "get_quantum_experiment_result",
        "estimate_quantum_expectation",
        "batch_estimate_quantum_expectation",
        "get_quantum_expectation_result",
        "generate_waveform_diagram",
        "get_waveform_diagram",
    }
    assert len(by_name) == 10
    backend_schema = by_name["list_available_quantum_backends"].inputSchema
    backend_schema_text = json.dumps(backend_schema, sort_keys=True)
    assert '"enum"' in backend_schema_text
    for value in ("all", "quantum_computer", "simulator"):
        assert f'"{value}"' in backend_schema_text
    identifier_schema = by_name["get_quantum_backend"].inputSchema["properties"]["identifier"]
    assert identifier_schema["minLength"] == 1
    assert identifier_schema["maxLength"] == 128
    detail_output_schema = json.dumps(
        by_name["get_quantum_backend"].outputSchema,
        sort_keys=True,
    )
    assert "image_url" not in detail_output_schema
    assert "imageUrl" not in detail_output_schema
    submit = by_name["submit_quantum_experiment"]
    assert submit.annotations is not None
    assert submit.annotations.readOnlyHint is False
    assert submit.annotations.destructiveHint is True
    assert submit.annotations.idempotentHint is False
    assert submit.annotations.openWorldHint is True
    properties = submit.inputSchema["properties"]
    assert set(properties) == {
        "computer_code",
        "circuit",
        "name",
        "lab_name",
        "shots",
        "allow_hardware",
    }
    assert set(submit.inputSchema["required"]) == {
        "computer_code",
        "circuit",
        "name",
        "shots",
    }
    lab_name_schema = json.dumps(properties["lab_name"], sort_keys=True)
    assert '"null"' in lab_name_schema
    assert properties["lab_name"]["default"] is None
    assert properties["shots"]["minimum"] == 1
    assert properties["shots"]["maximum"] == 1000
    assert properties["allow_hardware"]["default"] is False
    batch_submit = by_name["submit_quantum_experiment_batch"]
    batch_properties = batch_submit.inputSchema["properties"]
    assert set(batch_properties) == {
        "computer_code",
        "circuits",
        "name",
        "lab_name",
        "shots",
        "allow_hardware",
    }
    assert batch_properties["circuits"]["minItems"] == 2
    assert batch_properties["circuits"]["maxItems"] == 20
    result = by_name["get_quantum_experiment_result"]
    assert result.annotations is not None
    assert result.annotations.readOnlyHint is True
    assert result.annotations.destructiveHint is False
    assert result.annotations.idempotentHint is True
    assert result.annotations.openWorldHint is True
    for tool_name in ("estimate_quantum_expectation", "batch_estimate_quantum_expectation"):
        expectation = by_name[tool_name]
        assert expectation.annotations is not None
        assert expectation.annotations.readOnlyHint is False
        assert expectation.annotations.destructiveHint is True
        assert expectation.annotations.idempotentHint is False
        assert expectation.inputSchema["properties"]["dry_run"]["default"] is True
        assert expectation.inputSchema["properties"]["shots"]["maximum"] == 1000
    expectation_query_annotations = by_name["get_quantum_expectation_result"].annotations
    assert expectation_query_annotations is not None
    assert expectation_query_annotations.readOnlyHint is True
    result_properties = result.inputSchema["properties"]
    assert set(result_properties) == {"query_id", "include_result_status"}
    assert result_properties["query_id"]["minLength"] == 1
    assert result_properties["query_id"]["maxLength"] == 128
    assert result_properties["include_result_status"]["default"] is False
    generate_waveform = by_name["generate_waveform_diagram"]
    assert generate_waveform.annotations is not None
    assert generate_waveform.annotations.readOnlyHint is False
    assert generate_waveform.annotations.destructiveHint is False
    assert generate_waveform.annotations.idempotentHint is False
    assert generate_waveform.annotations.openWorldHint is True
    waveform_properties = generate_waveform.inputSchema["properties"]
    assert set(waveform_properties) == {"qc_code", "circuit", "circuit_name", "is_verify"}
    assert set(generate_waveform.inputSchema["required"]) == {"qc_code", "circuit"}
    assert waveform_properties["qc_code"]["maxLength"] == 64
    assert waveform_properties["qc_code"]["pattern"] == "^[A-Za-z0-9_-]+$"
    assert waveform_properties["circuit"]["maxLength"] == 20_000
    assert waveform_properties["circuit_name"]["default"] is None
    assert waveform_properties["is_verify"]["default"] is True
    waveform_query = by_name["get_waveform_diagram"]
    assert waveform_query.annotations is not None
    assert waveform_query.annotations.readOnlyHint is True
    assert waveform_query.annotations.destructiveHint is False
    assert waveform_query.annotations.idempotentHint is True
    assert set(waveform_query.inputSchema["properties"]) == {"task_id"}
    list_description = by_name["list_available_quantum_backends"].description
    assert list_description is not None
    assert "\u67e5\u8be2\u5f53\u524d\u8fd0\u884c\u4e2d" in list_description
    assert "\u6bcf\u6b21\u8c03\u7528\u90fd\u4f1a\u5b9e\u65f6\u8bfb\u53d6" in list_description
    detail_description = by_name["get_quantum_backend"].description
    assert detail_description is not None
    assert "\u5148\u5339\u914d id" in detail_description
    assert (
        "\u5ffd\u7565 code \u5927\u5c0f\u5199\u8fdb\u884c\u5b8c\u5168\u5339\u914d"
        in detail_description
    )


@pytest.mark.asyncio
async def test_direct_tool_call_has_structured_success() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool(
        "list_available_quantum_backends",
        {"backend_type": "all"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["total"] == 0


@pytest.mark.asyncio
async def test_submit_tool_call_has_structured_success() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool(
        "submit_quantum_experiment",
        {
            "computer_code": "SIMULATOR",
            "circuit": "H q[0]",
            "name": "experiment",
            "shots": 100,
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["query_id"] == "query-1"
    assert structured["lab_name"] is None


@pytest.mark.asyncio
async def test_batch_submit_tool_call_has_ordered_query_ids() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool(
        "submit_quantum_experiment_batch",
        {
            "computer_code": "tianyan_sw",
            "circuits": ["H Q1\nM Q1", "X Q1\nM Q1"],
            "name": "batch",
            "shots": 100,
        },
    )

    assert isinstance(result, tuple)
    assert cast(dict[str, Any], result[1])["query_ids"] == ["query-1", "query-2"]


@pytest.mark.asyncio
async def test_result_tool_call_has_structured_success() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool(
        "get_quantum_experiment_result",
        {"query_id": "123", "include_result_status": True},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["query_id"] == "123"
    assert structured["result_status"] == [[0]]


@pytest.mark.asyncio
async def test_direct_tool_call_raises_stable_domain_error() -> None:
    mcp = create_test_server(FakeService())

    with pytest.raises(ToolError, match="RESOURCE_NOT_FOUND"):
        await mcp.call_tool("get_quantum_backend", {"identifier": "missing"})


@pytest.mark.asyncio
async def test_direct_tool_call_normalises_unexpected_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mcp = create_test_server(BrokenService())

    with pytest.raises(ToolError, match="INTERNAL_ERROR") as captured:
        await mcp.call_tool(
            "list_available_quantum_backends",
            {"backend_type": "all"},
        )
    assert "sensitive implementation detail" not in str(captured.value)
    assert "sensitive implementation detail" not in caplog.text
    assert "\u670d\u52a1\u5185\u90e8\u9519\u8bef" in str(captured.value)


@pytest.mark.asyncio
async def test_submit_tool_normalises_unexpected_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mcp = create_test_server(FakeService(), BrokenExperimentService())

    with pytest.raises(ToolError, match="INTERNAL_ERROR") as captured:
        await mcp.call_tool(
            "submit_quantum_experiment",
            {
                "computer_code": "SIMULATOR",
                "circuit": "H q[0]",
                "name": "experiment",
                "lab_name": "lab",
                "shots": 100,
            },
        )
    assert "sensitive submission detail" not in str(captured.value)
    assert "sensitive submission detail" not in caplog.text
    assert "\u670d\u52a1\u5185\u90e8\u9519\u8bef" in str(captured.value)


@pytest.mark.asyncio
async def test_tool_catalog_declares_side_effect_annotations() -> None:
    mcp = create_test_server(FakeService())
    by_name = {tool.name: tool for tool in await mcp.list_tools()}

    for name in (
        "list_available_quantum_backends",
        "get_quantum_backend",
        "get_quantum_experiment_result",
    ):
        annotations = by_name[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is True
        assert annotations.openWorldHint is True

    submit_annotations = by_name["submit_quantum_experiment"].annotations
    assert submit_annotations is not None
    assert submit_annotations.readOnlyHint is False
    assert submit_annotations.destructiveHint is True
    assert submit_annotations.idempotentHint is False
    assert submit_annotations.openWorldHint is True
    batch_submit_annotations = by_name["submit_quantum_experiment_batch"].annotations
    assert batch_submit_annotations is not None
    assert batch_submit_annotations.readOnlyHint is False
    assert batch_submit_annotations.destructiveHint is True
    assert batch_submit_annotations.idempotentHint is False
    assert batch_submit_annotations.openWorldHint is True


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_hardware", ["true", 0, 1])
async def test_submit_tool_rejects_non_boolean_allow_hardware(allow_hardware: object) -> None:
    mcp = create_test_server(FakeService())

    with pytest.raises(ToolError):
        await mcp.call_tool(
            "submit_quantum_experiment",
            {
                "computer_code": "tianyan_sw",
                "circuit": "H Q1\nM Q1",
                "name": "experiment",
                "lab_name": "lab",
                "shots": 100,
                "allow_hardware": allow_hardware,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("include_result_status", ["true", 0, 1])
async def test_result_tool_rejects_non_boolean_include_result_status(
    include_result_status: object,
) -> None:
    mcp = create_test_server(FakeService())

    with pytest.raises(ToolError):
        await mcp.call_tool(
            "get_quantum_experiment_result",
            {
                "query_id": "123",
                "include_result_status": include_result_status,
            },
        )


@pytest.mark.asyncio
async def test_generate_waveform_tool_call_has_structured_success() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool(
        "generate_waveform_diagram",
        {
            "qc_code": "QPU-01",
            "circuit": "H Q0\nM Q0",
            "circuit_name": "bell-state",
            "is_verify": False,
        },
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["id"] == "1823156489274675200"
    assert structured["qc_code"] == "QPU-01"
    assert structured["status"] == "generating"
    assert structured["measure_qubits"] == ["Q0"]


@pytest.mark.asyncio
async def test_generate_waveform_tool_defaults_circuit_name_and_is_verify() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool(
        "generate_waveform_diagram",
        {"qc_code": "QPU-01", "circuit": "H Q0"},
    )

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["circuit_name"] == "example1700000000000"


@pytest.mark.asyncio
async def test_get_waveform_tool_call_has_structured_success() -> None:
    mcp = create_test_server(FakeService())

    result = await mcp.call_tool("get_waveform_diagram", {"task_id": "1823156489274675200"})

    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["id"] == "1823156489274675200"
    assert structured["status"] == "completed"
    assert structured["visible_url"].startswith("https://files.example.com/")


@pytest.mark.asyncio
async def test_get_waveform_tool_passes_stable_domain_error() -> None:
    mcp = create_test_server(FakeService(), waveform_service=BrokenWaveformService())

    with pytest.raises(ToolError, match="WAVEFORM_TASK_NOT_FOUND"):
        await mcp.call_tool("get_waveform_diagram", {"task_id": "123"})


@pytest.mark.asyncio
async def test_generate_waveform_tool_normalises_unexpected_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mcp = create_test_server(FakeService(), waveform_service=BrokenWaveformService())

    with pytest.raises(ToolError, match="INTERNAL_ERROR") as captured:
        await mcp.call_tool(
            "generate_waveform_diagram",
            {"qc_code": "QPU-01", "circuit": "H Q0"},
        )
    assert "sensitive waveform detail" not in str(captured.value)
    assert "sensitive waveform detail" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("is_verify", ["true", 0, 1])
async def test_generate_waveform_tool_rejects_non_boolean_is_verify(is_verify: object) -> None:
    mcp = create_test_server(FakeService())

    with pytest.raises(ToolError):
        await mcp.call_tool(
            "generate_waveform_diagram",
            {"qc_code": "QPU-01", "circuit": "H Q0", "is_verify": is_verify},
        )


@pytest.mark.asyncio
async def test_generate_waveform_tool_rejects_qc_code_failing_pattern() -> None:
    mcp = create_test_server(FakeService())

    with pytest.raises(ToolError):
        await mcp.call_tool(
            "generate_waveform_diagram",
            {"qc_code": "QPU 01", "circuit": "H Q0"},
        )
