from __future__ import annotations

from typing import get_type_hints

import pytest

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_models import UpstreamExperimentResult
from tianyan_mcp.experiment_result_service import ExperimentResultService


def upstream_result(
    experiment_task_id: str = "123",
    *,
    probability: dict[str, float] | None = None,
    result_status: tuple[tuple[int, ...], ...] = ((0,), (1,)),
) -> UpstreamExperimentResult:
    return UpstreamExperimentResult(
        experimentTaskId=experiment_task_id,
        probability=probability if probability is not None else {"0": 0.6, "1": 0.4},
        resultStatus=result_status,
    )


class FakeReader:
    def __init__(
        self,
        *,
        status: int = 0,
        results: tuple[UpstreamExperimentResult, ...] = (),
        status_error: Exception | None = None,
        result_error: Exception | None = None,
    ) -> None:
        self.status = status
        self.results = results
        self.status_error = status_error
        self.result_error = result_error
        self.status_calls: list[str] = []
        self.result_calls: list[str] = []

    async def get_status(self, query_id: str) -> int:
        self.status_calls.append(query_id)
        if self.status_error is not None:
            raise self.status_error
        return self.status

    async def get_result(self, query_id: str) -> tuple[UpstreamExperimentResult, ...]:
        self.result_calls.append(query_id)
        if self.result_error is not None:
            raise self.result_error
        return self.results


@pytest.mark.asyncio
async def test_processing_returns_without_fetching_result() -> None:
    reader = FakeReader(status=1)
    service = ExperimentResultService(reader)

    result = await service.query(" 123 ", include_result_status=False)

    assert result.status == "processing"
    assert result.upstream_status == 1
    assert result.probability is None
    assert result.experiment_task_id is None
    assert result.result_status is None
    assert reader.status_calls == ["123"]
    assert reader.result_calls == []


@pytest.mark.asyncio
async def test_completed_trims_measurements_by_default() -> None:
    reader = FakeReader(status=0, results=(upstream_result("123"),))
    service = ExperimentResultService(reader)

    result = await service.query("123", include_result_status=False)

    assert result.status == "completed"
    assert result.probability == {"0": 0.6, "1": 0.4}
    assert result.experiment_task_id == "123"
    assert result.result_status is None
    assert reader.status_calls == ["123"]
    assert reader.result_calls == ["123"]


@pytest.mark.asyncio
async def test_completed_includes_measurements_when_requested() -> None:
    measurements = ((2,), (0,), (1,))
    reader = FakeReader(results=(upstream_result("123", result_status=measurements),))
    service = ExperimentResultService(reader)

    result = await service.query("123", include_result_status=True)

    assert result.result_status == measurements


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_id",
    ["", "   ", "abc", "12.3", "１２３", "1" * 129],
    ids=["empty", "whitespace", "non-numeric", "punctuation", "non-ascii-digits", "too-long"],
)
async def test_invalid_query_id_fails_before_any_reader_call(query_id: str) -> None:
    reader = FakeReader()
    service = ExperimentResultService(reader)

    with pytest.raises(ServiceError) as captured:
        await service.query(query_id, include_result_status=False)

    assert captured.value.code == "INVALID_EXPERIMENT_QUERY_ID"
    assert captured.value.retryable is False
    assert reader.status_calls == []
    assert reader.result_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [2, True, False], ids=["unknown", "boolean-true", "boolean-false"]
)
async def test_unknown_status_fails_without_fetching_result(status: int) -> None:
    reader = FakeReader(status=status)
    service = ExperimentResultService(reader)

    with pytest.raises(ServiceError) as captured:
        await service.query("123", include_result_status=False)

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert reader.status_calls == ["123"]
    assert reader.result_calls == []


@pytest.mark.asyncio
async def test_completed_accepts_single_result_with_different_upstream_task_id() -> None:
    reader = FakeReader(results=(upstream_result("upstream-task-999"),))
    service = ExperimentResultService(reader)

    result = await service.query("123", include_result_status=False)

    assert result.query_id == "123"
    assert result.experiment_task_id == "upstream-task-999"
    assert result.probability == {"0": 0.6, "1": 0.4}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [(), (upstream_result("123"), upstream_result("999"))],
    ids=["no-result", "multiple-results"],
)
async def test_completed_requires_exactly_one_result_item(
    results: tuple[UpstreamExperimentResult, ...],
) -> None:
    reader = FakeReader(results=results)
    service = ExperimentResultService(reader)

    with pytest.raises(ServiceError) as captured:
        await service.query("123", include_result_status=False)

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert captured.value.retryable is False
    assert reader.status_calls == ["123"]
    assert reader.result_calls == ["123"]


@pytest.mark.asyncio
async def test_reader_service_error_is_propagated_unchanged() -> None:
    upstream_error = ServiceError("EXPERIMENT_QUERY_UNAVAILABLE", "暂时不可用", True)
    reader = FakeReader(status_error=upstream_error)
    service = ExperimentResultService(reader)

    with pytest.raises(ServiceError) as captured:
        await service.query("123", include_result_status=False)

    assert captured.value is upstream_error
    assert captured.value.code == "EXPERIMENT_QUERY_UNAVAILABLE"
    assert captured.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_method", ["status", "result"], ids=["status", "result"])
async def test_unexpected_reader_exception_is_normalized(reader_method: str) -> None:
    unexpected = RuntimeError("not for callers")
    reader = (
        FakeReader(status_error=unexpected)
        if reader_method == "status"
        else FakeReader(result_error=unexpected, results=())
    )
    service = ExperimentResultService(reader)

    with pytest.raises(ServiceError) as captured:
        await service.query("123", include_result_status=False)

    assert captured.value.code == "INTERNAL_ERROR"
    assert captured.value.message == "服务内部错误"
    assert captured.value.retryable is False


def test_public_result_model_rejects_non_status_literals() -> None:
    from pydantic import ValidationError

    from tianyan_mcp.experiment_models import ExperimentResultQueryResult

    with pytest.raises(ValidationError):
        ExperimentResultQueryResult.model_validate(
            {"query_id": "123", "status": "unknown", "upstream_status": 0}
        )


def test_service_query_signature_exposes_strict_boolean_contract() -> None:
    assert get_type_hints(ExperimentResultService.query)["include_result_status"] is bool
