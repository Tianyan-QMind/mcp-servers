from __future__ import annotations

from tianyan_mcp.client import TianyanApiClient
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.mapping import normalise_backend, summary_from_detail
from tianyan_mcp.models import (
    BackendDetail,
    BackendListResult,
    BackendType,
    UpstreamBackend,
)


class BackendService:
    def __init__(self, client: TianyanApiClient) -> None:
        self._client = client

    async def list_available(
        self,
        backend_type: BackendType = BackendType.ALL,
    ) -> BackendListResult:
        snapshot = await self._client.fetch_backends()
        details = [
            detail
            for raw in snapshot.backends
            if (detail := normalise_backend(raw, snapshot.fetched_at)) is not None
            and detail.available
            and (backend_type is BackendType.ALL or detail.resource_type is backend_type)
        ]
        summaries = [summary_from_detail(detail) for detail in details]
        return BackendListResult(
            fetched_at=snapshot.fetched_at,
            backend_type_filter=backend_type,
            total=len(summaries),
            backends=summaries,
        )

    async def get_backend(self, identifier: str) -> BackendDetail:
        value = identifier.strip()
        snapshot = await self._client.fetch_backends()
        matched = _find(snapshot.backends, value)
        if matched is None:
            raise ServiceError(
                "RESOURCE_NOT_FOUND",
                f"未找到 id 或 code 为 {value} 的量子计算资源",
                False,
            )
        detail = normalise_backend(matched, snapshot.fetched_at)
        if detail is None:
            raise ServiceError(
                "RESOURCE_TYPE_UNSUPPORTED",
                "指定资源不属于当前 MVP 支持范围",
                False,
            )
        return detail


def _find(
    backends: list[UpstreamBackend],
    identifier: str,
) -> UpstreamBackend | None:
    for backend in backends:
        if backend.id == identifier:
            return backend
    folded = identifier.casefold()
    for backend in backends:
        if backend.code is not None and backend.code.casefold() == folded:
            return backend
    return None
