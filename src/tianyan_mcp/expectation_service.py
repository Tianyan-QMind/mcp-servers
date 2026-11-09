"""Sequential submission and one-shot queries using existing identity-aware services."""

import re

from tianyan_mcp.backend_policy import resolve_submission_backend
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.expectation_calculation import calculate_expectation, prepare_measurements
from tianyan_mcp.expectation_models import (
    CircuitExpectation,
    ExpectationManifest,
    ExpectationResults,
    ExpectationSubmission,
    ManifestJob,
    Measurement,
    ObservableTerm,
)
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService


class ExpectationService:
    def __init__(self, submitter: ExperimentService, reader: ExperimentResultService) -> None:
        self._submitter = submitter
        self._reader = reader

    async def submit(
        self,
        *,
        computer_code: str,
        circuits: list[str],
        observable: list[ObservableTerm],
        name: str,
        lab_name: str | None,
        shots: int,
        allow_hardware: bool = False,
        dry_run: bool = True,
    ) -> ExpectationSubmission:
        target = resolve_submission_backend(computer_code)
        if target.canonical_code == "tianyan_sa":
            raise ServiceError("UNSUPPORTED_BACKEND", "单振幅后端不支持期望值估计", False)
        try:
            plan = prepare_measurements(circuits, observable)
        except ValueError as exc:
            raise ServiceError(
                "INVALID_EXPECTATION_ARGUMENT", "观测量、制备线路或展开任务数量无效", False
            ) from exc
        if type(shots) is not int or type(dry_run) is not bool or type(allow_hardware) is not bool:
            raise ServiceError("INVALID_EXPECTATION_ARGUMENT", "shots 和布尔开关类型无效", False)
        for circuit in circuits:
            ExperimentService.validate_submission(circuit, name, lab_name, shots)
        for job in plan.jobs:
            ExperimentService.validate_submission(job.circuit, name, lab_name, shots)
        jobs = [
            ManifestJob(circuit_index=job.circuit_index, pauli=job.pauli, qubits=job.qubits)
            for job in plan.jobs
        ]
        if not dry_run:
            seen_ids: set[str] = set()
            for index, prepared in enumerate(plan.jobs):
                try:
                    submitted = await self._submitter.submit(
                        computer_code=target.canonical_code,
                        circuit=prepared.circuit,
                        name=name,
                        lab_name=lab_name,
                        shots=shots,
                        allow_hardware=allow_hardware,
                    )
                    query_id = submitted.query_id
                    if re.fullmatch(r"[0-9]{1,128}", query_id) is None or query_id in seen_ids:
                        raise ServiceError(
                            "SUBMISSION_OUTCOME_UNKNOWN", "提交响应缺少有效的唯一任务 ID", False
                        )
                    seen_ids.add(query_id)
                    jobs[index] = ManifestJob(
                        circuit_index=prepared.circuit_index,
                        pauli=prepared.pauli,
                        qubits=prepared.qubits,
                        status="submitted",
                        query_id=query_id,
                    )
                except ServiceError as exc:
                    jobs[index] = ManifestJob(
                        circuit_index=prepared.circuit_index,
                        pauli=prepared.pauli,
                        qubits=prepared.qubits,
                        status="unknown"
                        if exc.code == "SUBMISSION_OUTCOME_UNKNOWN"
                        else "rejected",
                        error_code=exc.code,
                    )
                    break
                except Exception:
                    # An unexpected exception during a write cannot prove non-submission.
                    jobs[index] = ManifestJob(
                        circuit_index=prepared.circuit_index,
                        pauli=prepared.pauli,
                        qubits=prepared.qubits,
                        status="unknown",
                        error_code="SUBMISSION_OUTCOME_UNKNOWN",
                    )
                    break
        return ExpectationSubmission(
            dry_run=dry_run,
            computer_code=target.canonical_code,
            shots_per_job=shots,
            job_count=len(jobs),
            total_shots=len(jobs) * shots,
            measurements=plan.jobs,
            manifest=ExpectationManifest(
                circuit_count=len(circuits), observable=plan.observable, jobs=tuple(jobs)
            ),
        )

    async def query(self, manifest: ExpectationManifest) -> ExpectationResults:
        try:
            checked = ExpectationManifest.model_validate(manifest.model_dump())
        except ValueError as exc:
            raise ServiceError("INVALID_EXPECTATION_MANIFEST", "期望值任务清单无效", False) from exc
        results = [
            await self._query_circuit(checked, index) for index in range(checked.circuit_count)
        ]
        return ExpectationResults(results=tuple(results))

    async def _query_circuit(self, manifest: ExpectationManifest, index: int) -> CircuitExpectation:
        jobs = [job for job in manifest.jobs if job.circuit_index == index]
        query_ids = tuple(job.query_id for job in jobs if job.query_id is not None)
        measurements: list[Measurement] = []
        processing = False
        error_code: str | None = None
        invalid_result = False
        for job in jobs:
            if job.query_id is None:
                continue
            try:
                result = await self._reader.query(job.query_id, include_result_status=True)
                if result.status == "processing":
                    processing = True
                    continue
                if not result.result_status or result.probability is None:
                    invalid_result = True
                    continue
                measurements.append(
                    Measurement(
                        pauli=job.pauli,
                        qubits=job.qubits,
                        header=result.result_status[0],
                        probability=result.probability,
                    )
                )
            except ServiceError as exc:
                error_code = exc.code
            except Exception:
                error_code = "INTERNAL_ERROR"
        if error_code is not None:
            return CircuitExpectation(
                circuit_index=index,
                status="query_error",
                query_ids=query_ids,
                error_code=error_code,
            )
        if invalid_result:
            return CircuitExpectation(
                circuit_index=index,
                status="invalid_result",
                query_ids=query_ids,
                error_code="INVALID_MEASUREMENT_RESULT",
            )
        if any(job.status != "submitted" for job in jobs):
            return CircuitExpectation(
                circuit_index=index, status="incomplete_submission", query_ids=query_ids
            )
        if processing:
            return CircuitExpectation(circuit_index=index, status="processing", query_ids=query_ids)
        try:
            value = calculate_expectation(manifest.observable, measurements)
        except ValueError:
            return CircuitExpectation(
                circuit_index=index,
                status="invalid_result",
                query_ids=query_ids,
                error_code="INVALID_MEASUREMENT_RESULT",
            )
        except Exception:
            return CircuitExpectation(
                circuit_index=index,
                status="query_error",
                query_ids=query_ids,
                error_code="INTERNAL_ERROR",
            )
        return CircuitExpectation(
            circuit_index=index, status="completed", value=value, query_ids=query_ids
        )
