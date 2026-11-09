from __future__ import annotations

import httpx

from tianyan_mcp.identity import TenantPrincipal, current_tenant
from tianyan_mcp.tenant_sdk_auth import TenantSdkAuthManager
from tianyan_mcp.waveform_client import TianyanWaveformClient
from tianyan_mcp.waveform_models import (
    UpstreamWaveformDiagram,
    UpstreamWaveformSubmission,
    UpstreamWaveformTask,
)


class _TenantAccessTokenProvider:
    def __init__(self, manager: TenantSdkAuthManager, principal: TenantPrincipal) -> None:
        self._manager = manager
        self._principal = principal

    async def get_access_token(self) -> str:
        return await self._manager.get_access_token(self._principal)

    def invalidate(self) -> None:
        self._manager.invalidate(self._principal.tenant_id)


class TenantWaveformClient:
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

    async def generate(self, request: UpstreamWaveformSubmission) -> UpstreamWaveformTask:
        return await self._client_for_current_principal().generate(request)

    async def get_diagram(self, task_id: str) -> UpstreamWaveformDiagram:
        return await self._client_for_current_principal().get_diagram(task_id)

    def _client_for_current_principal(self) -> TianyanWaveformClient:
        provider = _TenantAccessTokenProvider(self._manager, current_tenant())
        return TianyanWaveformClient(
            provider,
            self._timeout_seconds,
            transport=self._transport,
        )
