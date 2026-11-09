from types import SimpleNamespace
from typing import cast

import pytest

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.waveform_models import (
    UpstreamWaveformDiagram,
    UpstreamWaveformSubmission,
    UpstreamWaveformTask,
)
from tianyan_mcp.waveform_service import WaveformService


class FakeWaveformGenerator:
    def __init__(self, task: UpstreamWaveformTask | None = None) -> None:
        self.task = task or UpstreamWaveformTask.model_validate(
            {
                "id": 1823156489274675200,
                "circuitName": "bell-state",
                "circuitContent": "H Q0\nM Q0",
                "status": 0,
                "createTime": "2026-09-28 15:00:00",
                "visibleUrl": None,
                "measureQubitList": "Q0, Q1",
            }
        )
        self.request: UpstreamWaveformSubmission | None = None

    async def generate(self, request: UpstreamWaveformSubmission) -> UpstreamWaveformTask:
        self.request = request
        return self.task


class FailingGenerator(FakeWaveformGenerator):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def generate(self, request: UpstreamWaveformSubmission) -> UpstreamWaveformTask:
        raise self._error


class FakeWaveformReader:
    def __init__(self, diagram: UpstreamWaveformDiagram) -> None:
        self.diagram = diagram
        self.requested_id: str | None = None

    async def get_diagram(self, task_id: str) -> UpstreamWaveformDiagram:
        self.requested_id = task_id
        return self.diagram


class FailingReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_diagram(self, task_id: str) -> UpstreamWaveformDiagram:
        raise self._error


def diagram(*, visible_url: object, status: int | None = None) -> UpstreamWaveformDiagram:
    payload: dict[str, object] = {
        "id": "1823156489274675200",
        "circuit_name": "bell-state",
        "visibleUrl": visible_url,
    }
    if status is not None:
        payload["status"] = status
    return UpstreamWaveformDiagram.model_validate(payload)


def service(
    generator: FakeWaveformGenerator | FailingGenerator | None = None,
    reader: FakeWaveformReader | FailingReader | None = None,
) -> WaveformService:
    return WaveformService(
        generator or FakeWaveformGenerator(), reader or FakeWaveformReader(diagram(visible_url=""))
    )


@pytest.mark.asyncio
async def test_generate_validates_inputs_and_maps_generating_task() -> None:
    generator = FakeWaveformGenerator()
    result = await service(generator).generate_waveform(
        qc_code="  QPU-01  ",
        circuit="H Q0\nM Q0",
        circuit_name="  bell-state  ",
        is_verify=True,
    )

    assert generator.request is not None
    assert generator.request.model_dump(by_alias=True) == {
        "circuit": "H Q0\nM Q0",
        "qcCode": "QPU-01",
        "circuitName": "bell-state",
        "isVerify": "true",
    }
    assert result.id == "1823156489274675200"
    assert result.qc_code == "QPU-01"
    assert result.circuit_name == "bell-state"
    assert result.status == "generating"
    assert result.upstream_status == 0
    assert result.create_time == "2026-09-28 15:00:00"
    assert result.visible_url is None
    assert result.measure_qubits == ("Q0", "Q1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "expected_name"),
    [(None, None), ("", None), ("   ", None), ("example 波形-1", "example 波形-1")],
)
async def test_generate_blank_name_becomes_none_and_valid_name_is_stripped(
    name: str | None, expected_name: str | None
) -> None:
    generator = FakeWaveformGenerator()
    await service(generator).generate_waveform(
        qc_code="QPU-01",
        circuit="H Q0",
        circuit_name=name,
        is_verify=False,
    )

    assert generator.request is not None
    assert generator.request.circuit_name == expected_name
    assert generator.request.is_verify == "false"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [0, 1, 2])
async def test_generate_maps_each_upstream_status(status: int) -> None:
    labels = {0: "generating", 1: "completed", 2: "failed"}
    task = UpstreamWaveformTask.model_validate(
        {
            "id": "1",
            "circuitName": "n",
            "circuitContent": "H Q0",
            "status": status,
            "measureQubitList": None,
        }
    )
    result = await service(FakeWaveformGenerator(task)).generate_waveform(
        qc_code="QPU-01",
        circuit="H Q0",
        circuit_name=None,
        is_verify=True,
    )

    assert result.status == labels[status]
    assert result.upstream_status == status
    assert result.measure_qubits == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("qc_code", "circuit", "circuit_name"),
    [
        ("", "H Q0", None),
        ("   ", "H Q0", None),
        ("QPU 01", "H Q0", None),
        ("QPU/01", "H Q0", None),
        ("Q" * 65, "H Q0", None),
        ("QPU-01", "", None),
        ("QPU-01", "   ", None),
        ("QPU-01", "H Q0\x00", None),
        ("QPU-01", "x" * 20_001, None),
        ("QPU-01", "H Q0", "bad/name"),
        ("QPU-01", "H Q0", "x" * 101),
    ],
    ids=[
        "empty-qc-code",
        "blank-qc-code",
        "qc-code-space",
        "qc-code-slash",
        "qc-code-too-long",
        "empty-circuit",
        "blank-circuit",
        "nul-circuit",
        "circuit-too-long",
        "invalid-name",
        "name-too-long",
    ],
)
async def test_generate_rejects_invalid_arguments(
    qc_code: str, circuit: str, circuit_name: str | None
) -> None:
    generator = FakeWaveformGenerator()
    with pytest.raises(ServiceError) as captured:
        await service(generator).generate_waveform(
            qc_code=qc_code,
            circuit=circuit,
            circuit_name=circuit_name,
            is_verify=True,
        )

    assert captured.value.code == "INVALID_WAVEFORM_ARGUMENT"
    assert generator.request is None


@pytest.mark.asyncio
async def test_generate_passes_service_error_through() -> None:
    error = ServiceError("WAVEFORM_SUBMISSION_REJECTED", "天衍云拒绝了波形生成任务", False)
    with pytest.raises(ServiceError, match="WAVEFORM_SUBMISSION_REJECTED"):
        await service(FailingGenerator(error)).generate_waveform(
            qc_code="QPU-01",
            circuit="H Q0",
            circuit_name=None,
            is_verify=True,
        )


@pytest.mark.asyncio
async def test_generate_normalises_unexpected_error() -> None:
    with pytest.raises(ServiceError) as captured:
        await service(FailingGenerator(RuntimeError("boom"))).generate_waveform(
            qc_code="QPU-01",
            circuit="H Q0",
            circuit_name=None,
            is_verify=True,
        )

    assert captured.value.code == "INTERNAL_ERROR"
    assert "boom" not in str(captured.value)


@pytest.mark.asyncio
async def test_get_waveform_maps_generating_and_completed() -> None:
    generating_reader = FakeWaveformReader(diagram(visible_url=""))
    generating = await service(reader=generating_reader).get_waveform("123")
    assert generating_reader.requested_id == "123"
    assert generating.status == "generating"
    assert generating.visible_url is None

    completed_reader = FakeWaveformReader(diagram(visible_url="https://files.example.com/x"))
    completed = await service(reader=completed_reader).get_waveform("123")
    assert completed.status == "completed"
    assert completed.visible_url == "https://files.example.com/x"
    assert completed.circuit_name == "bell-state"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [None, 0, 1])
async def test_get_waveform_empty_url_without_failed_status_is_generating(
    status: int | None,
) -> None:
    result = await service(
        reader=FakeWaveformReader(diagram(visible_url="", status=status))
    ).get_waveform("123")
    assert result.status == "generating"
    assert result.visible_url is None


@pytest.mark.asyncio
async def test_get_waveform_status_two_with_empty_url_marks_generation_failed() -> None:
    reader = FakeWaveformReader(diagram(visible_url="", status=2))
    with pytest.raises(ServiceError) as captured:
        await service(reader=reader).get_waveform("123")

    assert captured.value.code == "WAVEFORM_GENERATION_FAILED"
    assert captured.value.retryable is False


@pytest.mark.asyncio
async def test_get_waveform_visible_url_wins_over_failed_status() -> None:
    # Mirror the direct probe: a non-empty visibleUrl always means completed.
    reader = FakeWaveformReader(diagram(visible_url="https://files.example.com/x", status=2))
    result = await service(reader=reader).get_waveform("123")
    assert result.status == "completed"
    assert result.visible_url == "https://files.example.com/x"


@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", ["", "abc", "12.3", "-1", "1" * 33, "12 3"])
async def test_get_waveform_rejects_non_numeric_or_overlong_id(task_id: str) -> None:
    reader = FakeWaveformReader(diagram(visible_url=""))
    with pytest.raises(ServiceError) as captured:
        await service(reader=reader).get_waveform(task_id)

    assert captured.value.code == "INVALID_WAVEFORM_TASK_ID"
    assert reader.requested_id is None


@pytest.mark.asyncio
async def test_get_waveform_passes_service_error_through() -> None:
    error = ServiceError("WAVEFORM_TASK_NOT_FOUND", "未查询到波形线路数据", False)
    with pytest.raises(ServiceError, match="WAVEFORM_TASK_NOT_FOUND"):
        await service(reader=FailingReader(error)).get_waveform("123")


@pytest.mark.asyncio
async def test_get_waveform_normalises_unexpected_error() -> None:
    with pytest.raises(ServiceError) as captured:
        await service(reader=FailingReader(RuntimeError("boom"))).get_waveform("123")

    assert captured.value.code == "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_get_waveform_rejects_non_string_visible_url() -> None:
    broken = cast(
        UpstreamWaveformDiagram,
        SimpleNamespace(id="123", circuit_name="n", visible_url=42),
    )
    with pytest.raises(ServiceError) as captured:
        await service(reader=FakeWaveformReader(broken)).get_waveform("123")

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
