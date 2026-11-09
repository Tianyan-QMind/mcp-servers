from __future__ import annotations

import logging

import httpx

from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import TianyanExperimentClient
from tianyan_mcp.experiment_models import (
    UpstreamExperimentResult,
    UpstreamExperimentSubmission,
)
from tianyan_mcp.identity import DirectSdkPrincipal, current_direct_sdk

LOGGER = logging.getLogger(__name__)


class _DirectAccessTokenProvider:
    def __init__(self, manager: DirectSdkAuthManager, principal: DirectSdkPrincipal) -> None:
        self._manager = manager
        self._principal = principal

    async def get_access_token(self) -> str:
        return await self._manager.get_access_token(self._principal)

    def invalidate(self) -> None:
        self._manager.invalidate(self._principal.sdk_fingerprint)


class DirectSdkExperimentClient:
    def __init__(
        self,
        manager: DirectSdkAuthManager,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._manager = manager
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def submit(self, request: UpstreamExperimentSubmission) -> str:
        principal = current_direct_sdk()
        try:
            query_id = await self._client_for_current_principal().submit(request)
        except ServiceError as error:
            result = "unknown" if error.code == "SUBMISSION_OUTCOME_UNKNOWN" else "denied"
            self._log(principal, result)
            raise
        self._log(principal, "success")
        return query_id

    async def submit_batch(self, request: UpstreamExperimentSubmission) -> tuple[str, ...]:
        principal = current_direct_sdk()
        try:
            query_ids = await self._client_for_current_principal().submit_batch(request)
        except ServiceError as error:
            result = "unknown" if error.code == "SUBMISSION_OUTCOME_UNKNOWN" else "denied"
            self._log(principal, result)
            raise
        self._log(principal, "success")
        return query_ids

    async def get_status(self, query_id: str) -> int:
        return await self._client_for_current_principal().get_status(query_id)

    async def get_result(self, query_id: str) -> tuple[UpstreamExperimentResult, ...]:
        return await self._client_for_current_principal().get_result(query_id)

    def _client_for_current_principal(self) -> TianyanExperimentClient:
        provider = _DirectAccessTokenProvider(self._manager, current_direct_sdk())
        return TianyanExperimentClient(
            provider,
            self._timeout_seconds,
            transport=self._transport,
        )

    @staticmethod
    def _log(principal: DirectSdkPrincipal, result: str) -> None:
        LOGGER.info(
            "event=experiment_submit request_id=%s sdk_fingerprint=%s result=%s",
            principal.request_id,
            principal.sdk_fingerprint.hex()[:12],
            result,
        )
