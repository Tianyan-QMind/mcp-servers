from __future__ import annotations

import re
from typing import Protocol

from tianyan_mcp.backend_policy import resolve_submission_backend
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_models import (
    ExperimentBatchSubmissionResult,
    ExperimentSubmissionResult,
    UpstreamExperimentSubmission,
)
from tianyan_mcp.service import BackendService

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\- \u4e00-\u9fff]+$")


class ExperimentSubmitter(Protocol):
    async def submit(self, request: UpstreamExperimentSubmission) -> str: ...

    async def submit_batch(self, request: UpstreamExperimentSubmission) -> tuple[str, ...]: ...


class ExperimentService:
    def __init__(
        self,
        backend_service: BackendService,
        experiment_client: ExperimentSubmitter,
        *,
        allow_hardware_submission: bool,
    ) -> None:
        self._backend_service = backend_service
        self._experiment_client = experiment_client
        self._allow_hardware_submission = allow_hardware_submission

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
        self.validate_submission(circuit, name, lab_name, shots)
        target = resolve_submission_backend(computer_code)
        detail = await self._backend_service.get_backend(target.lookup_code)
        if not detail.available:
            raise ServiceError("BACKEND_UNAVAILABLE", "指定后端当前不可用", True)
        if detail.resource_type is not target.resource_type:
            raise ServiceError("BACKEND_POLICY_MISMATCH", "后端类型与提交策略不一致", False)
        if target.is_hardware and not self._allow_hardware_submission:
            raise ServiceError("HARDWARE_SUBMISSION_DISABLED", "服务端未开放真实硬件提交", False)
        if target.is_hardware and not allow_hardware:
            raise ServiceError(
                "HARDWARE_CONFIRMATION_REQUIRED",
                "真实硬件提交必须显式设置 allow_hardware=true",
                False,
            )

        clean_name = name.strip()
        clean_lab_name = lab_name.strip() if lab_name is not None else None
        request = UpstreamExperimentSubmission(
            shots=shots,
            circuit=(circuit,),
            name=clean_name,
            lab_name=clean_lab_name,
            computerCode=target.canonical_code,
            quantumState=0 if target.canonical_code == "tianyan_sa" else None,
        )
        query_id = await self._experiment_client.submit(request)
        return ExperimentSubmissionResult(
            query_id=query_id,
            backend_code=target.canonical_code,
            resource_type=target.resource_type,
            shots=shots,
            name=clean_name,
            lab_name=clean_lab_name,
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
        if not 2 <= len(circuits) <= 20:
            raise ServiceError(
                "INVALID_EXPERIMENT_ARGUMENT", "circuits 必须包含 2 到 20 条线路", False
            )
        for circuit in circuits:
            self.validate_submission(circuit, name, lab_name, shots)

        target = resolve_submission_backend(computer_code)
        detail = await self._backend_service.get_backend(target.lookup_code)
        if not detail.available:
            raise ServiceError("BACKEND_UNAVAILABLE", "指定后端当前不可用", True)
        if detail.resource_type is not target.resource_type:
            raise ServiceError("BACKEND_POLICY_MISMATCH", "后端类型与提交策略不一致", False)
        if target.is_hardware and not self._allow_hardware_submission:
            raise ServiceError("HARDWARE_SUBMISSION_DISABLED", "服务端未开放真实硬件提交", False)
        if target.is_hardware and not allow_hardware:
            raise ServiceError(
                "HARDWARE_CONFIRMATION_REQUIRED",
                "真实硬件提交必须显式设置 allow_hardware=true",
                False,
            )

        clean_name = name.strip()
        clean_lab_name = lab_name.strip() if lab_name is not None else None
        request = UpstreamExperimentSubmission(
            shots=shots,
            circuit=tuple(circuits),
            name=clean_name,
            lab_name=clean_lab_name,
            computerCode=target.canonical_code,
            quantumState=0 if target.canonical_code == "tianyan_sa" else None,
        )
        query_ids = await self._experiment_client.submit_batch(request)
        return ExperimentBatchSubmissionResult(
            query_ids=query_ids,
            backend_code=target.canonical_code,
            resource_type=target.resource_type,
            shots=shots,
            name=clean_name,
            lab_name=clean_lab_name,
        )

    @staticmethod
    def validate_submission(circuit: str, name: str, lab_name: str | None, shots: int) -> None:
        if isinstance(shots, bool) or not 1 <= shots <= 1000:
            raise ServiceError("INVALID_EXPERIMENT_ARGUMENT", "shots 必须在 1 到 1000 之间", False)
        if not circuit.strip() or len(circuit) > 20_000 or "\x00" in circuit:
            raise ServiceError("INVALID_EXPERIMENT_ARGUMENT", "QCIS 线路内容无效", False)
        names = [("name", name)]
        if lab_name is not None:
            names.append(("lab_name", lab_name))
        for field_name, value in names:
            cleaned = value.strip()
            if not 1 <= len(cleaned) <= 100 or NAME_PATTERN.fullmatch(cleaned) is None:
                raise ServiceError(
                    "INVALID_EXPERIMENT_ARGUMENT",
                    f"{field_name} 格式无效",
                    False,
                )
