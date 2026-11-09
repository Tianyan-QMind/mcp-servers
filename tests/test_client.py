from typing import Any

import httpx
import pytest
import respx

from tianyan_mcp.client import TIANYAN_QUERY, TIANYAN_URL, TianyanApiClient
from tianyan_mcp.errors import ServiceError


@pytest.mark.asyncio
@respx.mock
async def test_fetches_fixed_endpoint(upstream_payload: dict[str, Any]) -> None:
    route = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        return_value=httpx.Response(200, json=upstream_payload)
    )

    snapshot = await TianyanApiClient(timeout_seconds=1).fetch_backends()

    assert route.call_count == 1
    assert len(snapshot.backends) == 3


@pytest.mark.asyncio
@respx.mock
async def test_retries_503_once(upstream_payload: dict[str, Any]) -> None:
    route = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=upstream_payload),
        ]
    )

    snapshot = await TianyanApiClient(timeout_seconds=1).fetch_backends()

    assert len(snapshot.backends) == 3
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_does_not_retry_400() -> None:
    route = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(return_value=httpx.Response(400))

    with pytest.raises(ServiceError, match="UPSTREAM_REJECTED"):
        await TianyanApiClient(timeout_seconds=1).fetch_backends()

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_rejects_non_json() -> None:
    respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        return_value=httpx.Response(200, text="<html>bad</html>")
    )
    with pytest.raises(ServiceError, match="UPSTREAM_INVALID_RESPONSE"):
        await TianyanApiClient(timeout_seconds=1).fetch_backends()


@pytest.mark.asyncio
@respx.mock
async def test_accepts_empty_data() -> None:
    respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        return_value=httpx.Response(200, json={"code": 200, "msg": None, "data": []})
    )
    snapshot = await TianyanApiClient(timeout_seconds=1).fetch_backends()
    assert snapshot.backends == []


@pytest.mark.asyncio
@respx.mock
async def test_timeout_twice_returns_retryable_error() -> None:
    request = httpx.Request("GET", TIANYAN_URL)
    route = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        side_effect=httpx.ReadTimeout("timeout", request=request)
    )

    with pytest.raises(ServiceError) as captured:
        await TianyanApiClient(timeout_seconds=1).fetch_backends()

    assert captured.value.code == "UPSTREAM_TIMEOUT"
    assert captured.value.retryable is True
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_does_not_retry_500() -> None:
    route = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(return_value=httpx.Response(500))

    with pytest.raises(ServiceError) as captured:
        await TianyanApiClient(timeout_seconds=1).fetch_backends()

    assert captured.value.code == "UPSTREAM_REJECTED"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"code": 500, "msg": "failed", "data": []},
        {"code": 200, "msg": None, "data": {}},
        {"code": 200, "msg": None, "data": [{"name": "missing critical fields"}]},
    ],
)
async def test_rejects_invalid_business_payload(payload: dict[str, Any]) -> None:
    respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        return_value=httpx.Response(200, json=payload)
    )

    with pytest.raises(ServiceError) as captured:
        await TianyanApiClient(timeout_seconds=1).fetch_backends()

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
