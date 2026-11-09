from __future__ import annotations

import logging

import httpx

from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import DirectSdkPrincipal, current_direct_sdk
from tianyan_mcp.waveform_client import TianyanWaveformClient
from tianyan_mcp.waveform_models import (
    UpstreamWaveformDiagram,
    UpstreamWaveformSubmission,
    UpstreamWaveformTask,
)

LOGGER = logging.getLogger(__name__)


class _DirectAccessTokenProvider:
    def __init__(self, manager: DirectSdkAuthManager, principal: DirectSdkPrincipal) -> None:
        self._manager = manager
        self._principal = principal

    async def get_access_token(self) -> str:
        return await self._manager.get_access_token(self._principal)

    def invalidate(self) -> None:
        self._manager.invalidate(self._principal.sdk_fingerprint)


class DirectSdkWaveformClient:
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

    async def generate(self, request: UpstreamWaveformSubmission) -> UpstreamWaveformTask:
        principal = current_direct_sdk()
        try:
            task = await self._client_for_current_principal().generate(request)
        except ServiceError as error:
            result = "unknown" if error.code == "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN" else "denied"
            self._log(principal, result)
            raise
        self._log(principal, "success")
        return task

    async def get_diagram(self, task_id: str) -> UpstreamWaveformDiagram:
        return await self._client_for_current_principal().get_diagram(task_id)

    def _client_for_current_principal(self) -> TianyanWaveformClient:
        provider = _DirectAccessTokenProvider(self._manager, current_direct_sdk())
        return TianyanWaveformClient(
            provider,
            self._timeout_seconds,
            transport=self._transport,
        )

    @staticmethod
    def _log(principal: DirectSdkPrincipal, result: str) -> None:
        LOGGER.info(
            "event=waveform_generate request_id=%s sdk_fingerprint=%s result=%s",
            principal.request_id,
            principal.sdk_fingerprint.hex()[:12],
            result,
        )
