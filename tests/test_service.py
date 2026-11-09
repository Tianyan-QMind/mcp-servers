from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from tianyan_mcp.client import TianyanApiClient
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.models import BackendSnapshot, BackendType, UpstreamBackend
from tianyan_mcp.service import BackendService

FETCHED_AT = datetime(2026, 7, 29, 8, 30, tzinfo=UTC)


def make_service(payload: dict[str, Any]) -> tuple[BackendService, AsyncMock]:
    client = AsyncMock(spec=TianyanApiClient)
    client.fetch_backends.return_value = BackendSnapshot(
        fetched_at=FETCHED_AT,
        backends=[UpstreamBackend.model_validate(item) for item in payload["data"]],
    )
    return BackendService(cast(TianyanApiClient, client)), client


@pytest.mark.asyncio
async def test_list_returns_only_running_known_backends(
    upstream_payload: dict[str, Any],
) -> None:
    service, _ = make_service(upstream_payload)

    result = await service.list_available()

    assert [backend.code for backend in result.backends] == [
        "tianyan-287",
        "tianyan_s",
    ]
    assert result.total == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_type", "expected"),
    [
        (BackendType.QUANTUM_COMPUTER, ["tianyan-287"]),
        (BackendType.SIMULATOR, ["tianyan_s"]),
    ],
)
async def test_list_filters_backend_type(
    upstream_payload: dict[str, Any],
    backend_type: BackendType,
    expected: list[str],
) -> None:
    service, _ = make_service(upstream_payload)

    result = await service.list_available(backend_type)

    assert [backend.code for backend in result.backends] == expected


@pytest.mark.asyncio
async def test_list_calls_upstream_every_time(
    upstream_payload: dict[str, Any],
) -> None:
    service, client = make_service(upstream_payload)

    await service.list_available()
    await service.list_available()

    assert client.fetch_backends.await_count == 2


@pytest.mark.asyncio
async def test_get_prefers_exact_id() -> None:
    payload = {
        "data": [
            {"id": "target", "code": "first", "labels": "1", "status": 0},
            {"id": "second", "code": "target", "labels": "1", "status": 0},
        ]
    }
    service, _ = make_service(payload)

    detail = await service.get_backend("target")

    assert detail.code == "first"


@pytest.mark.asyncio
async def test_get_matches_code_case_insensitively(
    upstream_payload: dict[str, Any],
) -> None:
    service, _ = make_service(upstream_payload)

    detail = await service.get_backend("TIANYAN-287")

    assert detail.id == "1968635345322422274"


@pytest.mark.asyncio
async def test_get_returns_unavailable_backend(
    upstream_payload: dict[str, Any],
) -> None:
    service, _ = make_service(upstream_payload)

    detail = await service.get_backend("tianyan504")

    assert detail.operational_status == "upgrading"
    assert detail.available is False


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["天衍-287", "tianyan-"])
async def test_get_does_not_match_name_or_partial_code(
    upstream_payload: dict[str, Any],
    identifier: str,
) -> None:
    service, _ = make_service(upstream_payload)

    with pytest.raises(ServiceError) as captured:
        await service.get_backend(identifier)

    assert captured.value.code == "RESOURCE_NOT_FOUND"


@pytest.mark.asyncio
async def test_get_rejects_unknown_resource_type() -> None:
    payload = {
        "data": [
            {
                "id": "unknown",
                "code": "unknown",
                "labels": "99",
                "status": 0,
            }
        ]
    }
    service, _ = make_service(payload)

    with pytest.raises(ServiceError) as captured:
        await service.get_backend("unknown")

    assert captured.value.code == "RESOURCE_TYPE_UNSUPPORTED"
