from __future__ import annotations

import httpx

from tianyan_mcp.backend_policy import resolve_submission_backend
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import TianyanExperimentClient
from tianyan_mcp.experiment_models import (
    UpstreamExperimentResult,
    UpstreamExperimentSubmission,
)
from tianyan_mcp.identity import TenantPrincipal, current_tenant
from tianyan_mcp.tenant_sdk_auth import TenantSdkAuthManager
from tianyan_mcp.tenant_store import AuditOutcome


class _TenantAccessTokenProvider:
    def __init__(self, manager: TenantSdkAuthManager, principal: TenantPrincipal) -> None:
        self._manager = manager
        self._principal = principal

    async def get_access_token(self) -> str:
        return await self._manager.get_access_token(self._principal)

    def invalidate(self) -> None:
        self._manager.invalidate(self._principal.tenant_id)


class TenantExperimentClient:
    def __init__(
        self,
        manager: TenantSdkAuthManager,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._manager = manager
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def submit(self, request: UpstreamExperimentSubmission) -> str:
        if resolve_submission_backend(request.computer_code).is_hardware:
            raise ServiceError(
                "HARDWARE_SUBMISSION_DISABLED",
                "hardware submission is disabled in multi-tenant mode",
                False,
            )
        principal = current_tenant()
        try:
            query_id = await self._client_for_current_principal().submit(request)
        except ServiceError as error:
            outcome = (
                AuditOutcome.UNKNOWN
                if error.code == "SUBMISSION_OUTCOME_UNKNOWN"
                else AuditOutcome.DENIED
            )
            await self._record_submission_best_effort(principal, outcome)
            raise
        await self._record_submission_best_effort(principal, AuditOutcome.SUCCESS)
        return query_id

    async def submit_batch(self, request: UpstreamExperimentSubmission) -> tuple[str, ...]:
        if resolve_submission_backend(request.computer_code).is_hardware:
            raise ServiceError(
                "HARDWARE_SUBMISSION_DISABLED",
                "hardware submission is disabled in multi-tenant mode",
                False,
            )
        principal = current_tenant()
        try:
            query_ids = await self._client_for_current_principal().submit_batch(request)
        except ServiceError as error:
            outcome = (
                AuditOutcome.UNKNOWN
                if error.code == "SUBMISSION_OUTCOME_UNKNOWN"
                else AuditOutcome.DENIED
            )
            await self._record_submission_best_effort(principal, outcome)
            raise
        await self._record_submission_best_effort(principal, AuditOutcome.SUCCESS)
        return query_ids

    async def get_status(self, query_id: str) -> int:
        return await self._client_for_current_principal().get_status(query_id)

    async def get_result(self, query_id: str) -> tuple[UpstreamExperimentResult, ...]:
        return await self._client_for_current_principal().get_result(query_id)

    def _client_for_current_principal(self) -> TianyanExperimentClient:
        provider = _TenantAccessTokenProvider(self._manager, current_tenant())
        return TianyanExperimentClient(
            provider,
            self._timeout_seconds,
            transport=self._transport,
        )

    async def _record_submission_best_effort(
        self, principal: TenantPrincipal, outcome: AuditOutcome
    ) -> None:
        try:
            await self._manager.record_submission(principal, outcome)
        except Exception:
            # The upstream POST outcome is already known; audit failure must not
            # misrepresent a successful submission as a failed submission.
            return
