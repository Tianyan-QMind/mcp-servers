from typing import cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.expectation_models import ExpectationManifest, ObservableTerm
from tianyan_mcp.expectation_service import ExpectationService
from tianyan_mcp.experiment_models import ExperimentResultQueryResult, ExperimentSubmissionResult
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.models import BackendType


class FakeSubmitter:
    def __init__(self, fail_at: int | None = None, unknown: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.unknown = unknown

    async def submit(
        self,
        *,
        computer_code: str,
        circuit: str,
        name: str,
        shots: int,
        lab_name: str | None,
        allow_hardware: bool,
    ) -> ExperimentSubmissionResult:
        self.calls.append(circuit)
        if len(self.calls) == self.fail_at:
            raise ServiceError(
                "SUBMISSION_OUTCOME_UNKNOWN" if self.unknown else "SUBMISSION_REJECTED",
                "提交失败",
                False,
            )
        return ExperimentSubmissionResult(
            query_id=str(len(self.calls)),
            backend_code=computer_code,
            resource_type=BackendType.SIMULATOR,
            shots=shots,
            name=name,
            lab_name=lab_name,
        )


class FakeReader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.processing: set[str] = set()
        self.errors: set[str] = set()

    async def query(
        self, query_id: str, include_result_status: bool
    ) -> ExperimentResultQueryResult:
        assert include_result_status
        self.calls.append(query_id)
        if query_id in self.errors:
            raise ServiceError("UPSTREAM_TIMEOUT", "查询超时", True)
        if query_id in self.processing:
            return ExperimentResultQueryResult(
                query_id=query_id, status="processing", upstream_status=1
            )
        return ExperimentResultQueryResult(
            query_id=query_id,
            status="completed",
            upstream_status=0,
            probability={"0" if query_id == "1" else "1": 1.0},
            result_status=((1,),),
        )


def service(submitter: FakeSubmitter, reader: FakeReader) -> ExpectationService:
    return ExpectationService(
        cast(ExperimentService, submitter), cast(ExperimentResultService, reader)
    )


async def submit(
    estimator: ExpectationService,
    *,
    circuits: list[str] | None = None,
    dry_run: bool = False,
    computer_code: str = "tianyan_sw",
):
    return await estimator.submit(
        computer_code=computer_code,
        circuits=circuits or ["H Q1", "X Q1"],
        observable=[ObservableTerm(pauli="Z1", coefficient=1)],
        name="test",
        lab_name=None,
        shots=10,
        allow_hardware=False,
        dry_run=dry_run,
    )


async def test_preview_does_not_write_and_shows_total_cost() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    result = await submit(service(writer, reader), dry_run=True)
    assert writer.calls == reader.calls == []
    assert result.job_count == 2
    assert result.total_shots == 20
    assert result.dry_run
    assert all(job.query_id is None for job in result.manifest.jobs)


async def test_batch_validation_finishes_before_first_write() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    with pytest.raises(ServiceError):
        await submit(service(writer, reader), circuits=["H Q1", "M Q1"])
    with pytest.raises(ServiceError):
        await submit(service(writer, reader), computer_code="tianyan_sa")
    assert writer.calls == []


@pytest.mark.parametrize("unknown", [True, False])
async def test_partial_submission_stops_without_retry(unknown: bool) -> None:
    writer, reader = FakeSubmitter(fail_at=2, unknown=unknown), FakeReader()
    result = await submit(service(writer, reader), circuits=["H Q1"] * 3)
    assert len(writer.calls) == 2
    assert [job.status for job in result.manifest.jobs] == [
        "submitted",
        "unknown" if unknown else "rejected",
        "not_submitted",
    ]
    assert result.manifest.jobs[0].query_id == "1"
    assert result.manifest.jobs[1].query_id is None


async def test_query_roundtrips_manifest_and_preserves_batch_order() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator)
    manifest = ExpectationManifest.model_validate_json(submitted.manifest.model_dump_json())
    result = await estimator.query(manifest)
    assert [item.value for item in result.results] == [1.0, -1.0]
    assert [item.circuit_index for item in result.results] == [0, 1]
    assert reader.calls == ["1", "2"]
    assert len(writer.calls) == 2


async def test_query_errors_do_not_discard_other_circuits() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator)
    reader.errors.add("1")
    result = await estimator.query(submitted.manifest)
    assert result.results[0].status == "query_error"
    assert result.results[0].value is None
    assert result.results[1].value == -1.0


async def test_processing_does_not_invent_value_or_poll() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator)
    reader.processing.add("2")
    result = await estimator.query(submitted.manifest)
    assert result.results[0].value == 1.0
    assert result.results[1].status == "processing"
    assert result.results[1].value is None
    assert reader.calls == ["1", "2"]


async def test_duplicate_ids_and_incomplete_manifests_rejected_before_query() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator)
    data = submitted.manifest.model_dump()
    data["jobs"][1]["query_id"] = "1"
    with pytest.raises((ValidationError, ServiceError)):
        await estimator.query(ExpectationManifest.model_validate(data))
    data = submitted.manifest.model_dump()
    data["jobs"] = data["jobs"][:1]
    with pytest.raises((ValidationError, ServiceError)):
        await estimator.query(ExpectationManifest.model_validate(data))
    assert reader.calls == []


async def test_unexpected_write_exception_is_unknown_and_never_retried() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    with patch.object(writer, "submit", side_effect=RuntimeError("private-detail")) as mocked:
        result = await submit(service(writer, reader))
    assert mocked.await_count == 1
    assert [job.status for job in result.manifest.jobs] == ["unknown", "not_submitted"]
    assert "private-detail" not in result.model_dump_json()


async def test_partial_query_retains_finished_results_and_incomplete_statuses() -> None:
    writer, reader = FakeSubmitter(fail_at=2), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator, circuits=["H Q1"] * 3)
    result = await estimator.query(submitted.manifest)
    assert [item.status for item in result.results] == [
        "completed",
        "incomplete_submission",
        "incomplete_submission",
    ]
    assert result.results[0].value == 1.0
    assert reader.calls == ["1"]


@pytest.mark.parametrize("header", [None, ((9,),), ((1, 1),)])
async def test_invalid_completed_results_never_produce_values(
    header: tuple[tuple[int, ...], ...] | None,
) -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator, circuits=["H Q1"])
    invalid = ExperimentResultQueryResult(
        query_id="1",
        status="completed",
        upstream_status=0,
        probability={"0": 1.0},
        result_status=header,
    )
    with patch.object(reader, "query", return_value=invalid):
        result = await estimator.query(submitted.manifest)
    assert result.results[0].status == "invalid_result"
    assert result.results[0].value is None


async def test_constant_observable_needs_no_cloud_calls() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await estimator.submit(
        computer_code="tianyan_sw",
        circuits=["I Q1", "H Q1"],
        observable=[ObservableTerm(pauli="I", coefficient=0.5)],
        name="test",
        lab_name=None,
        shots=10,
        dry_run=False,
    )
    assert submitted.job_count == submitted.total_shots == 0
    result = await estimator.query(submitted.manifest)
    assert [item.value for item in result.results] == [0.5, 0.5]
    assert writer.calls == reader.calls == []


async def test_copy_bypassed_manifest_is_revalidated() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    estimator = service(writer, reader)
    submitted = await submit(estimator)
    invalid = submitted.manifest.model_copy(update={"circuit_count": 9})
    with pytest.raises(ServiceError, match="期望值任务清单无效"):
        await estimator.query(invalid)
    assert reader.calls == []
