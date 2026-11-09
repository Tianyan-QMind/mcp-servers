from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from pydantic import SecretBytes, SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tianyan_mcp.config import Settings
from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_security import DirectSdkSecurityMiddleware, FixedWindowRateLimiter
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import IdentityMode

_FINGERPRINT_KEY = b"f" * 32
_VALID_OPEN_ID = "alice-open-id"


def rate_limit_settings(
    *,
    enabled: bool = True,
    window_seconds: float = 60.0,
    max_requests: int = 3,
) -> Settings:
    return Settings(
        identity_mode=IdentityMode.DIRECT_SDK,
        direct_sdk_fingerprint_key=SecretBytes(_FINGERPRINT_KEY),
        direct_sdk_rate_limit_enabled=enabled,
        direct_sdk_rate_limit_window_seconds=window_seconds,
        direct_sdk_rate_limit_max_requests=max_requests,
    )


class LoginClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        value = open_id.get_secret_value()
        self.calls.append(value)
        await asyncio.sleep(0)
        if value == _VALID_OPEN_ID:
            return SecretStr("alice-token"), 720.0
        raise ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)


async def endpoint(_: Request) -> JSONResponse:
    return JSONResponse({"reached": True})


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_app(
    settings: Settings,
    login: LoginClient,
    clock: list[float] | None = None,
) -> tuple[DirectSdkSecurityMiddleware, LoginClient]:
    manager = DirectSdkAuthManager(login)
    inner = Starlette(
        routes=[
            Route("/mcp", endpoint, methods=["POST"]),
            Route("/healthz", health, methods=["GET"]),
        ]
    )
    app = DirectSdkSecurityMiddleware(
        inner,
        settings,
        manager,
        clock=(lambda: clock[0]) if clock is not None else time.monotonic,
    )
    return app, login


async def client_post(
    app: DirectSdkSecurityMiddleware,
    *,
    credential: str = _VALID_OPEN_ID,
    forwarded_for: str | None = None,
    include_auth: bool = True,
    path: str = "/mcp",
) -> httpx.Response:
    headers = {}
    if include_auth:
        headers["Authorization"] = f"Bearer {credential}"
    if forwarded_for is not None:
        headers["x-forwarded-for"] = forwarded_for
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        if path == "/healthz":
            return await client.get(path)
        return await client.post(path, headers=headers)


def test_limiter_allows_quota_then_blocks_until_window_resets() -> None:
    now = [100.0]
    limiter = FixedWindowRateLimiter(window_seconds=60.0, max_requests=3, clock=lambda: now[0])

    allowed = [limiter.take("peer") for _ in range(3)]
    assert [item[0] for item in allowed] == [True, True, True]

    ok, retry_after = limiter.take("peer")
    assert ok is False
    assert retry_after == 60

    now[0] = 159.0
    ok, retry_after = limiter.take("peer")
    assert ok is False
    assert retry_after == 1

    now[0] = 160.0
    ok, retry_after = limiter.take("peer")
    assert ok is True
    assert retry_after == 0


def test_limiter_tracks_keys_independently_and_is_lru_bounded() -> None:
    limiter = FixedWindowRateLimiter(
        window_seconds=60.0, max_requests=2, max_keys=2, clock=lambda: 0.0
    )

    assert limiter.take("a")[0] is True
    assert limiter.take("b")[0] is True
    # Touching "a" again promotes it; inserting "c" must then evict "b".
    assert limiter.take("a")[0] is True
    assert limiter.take("c")[0] is True
    assert len(limiter) == 2
    assert limiter.take("a")[0] is False
    # "b" was LRU-evicted, so it starts with a fresh window.
    assert limiter.take("b")[0] is True


@pytest.mark.parametrize(
    ("window_seconds", "max_requests"),
    [(0.0, 1), (-1.0, 1), (60.0, 0), (60.0, -1)],
)
def test_limiter_rejects_invalid_parameters(window_seconds: float, max_requests: int) -> None:
    with pytest.raises(ValueError):
        FixedWindowRateLimiter(window_seconds, max_requests)


@pytest.mark.asyncio
async def test_requests_beyond_quota_get_429_before_reaching_inner_app() -> None:
    app, login = build_app(rate_limit_settings(), LoginClient())

    responses = [await client_post(app) for _ in range(4)]

    assert [response.status_code for response in responses[:3]] == [200, 200, 200]
    blocked = responses[3]
    assert blocked.status_code == 429
    assert blocked.json() == {"error": "rate_limited", "retryable": "true"}
    assert int(blocked.headers["retry-after"]) > 0
    assert login.calls == [_VALID_OPEN_ID]


@pytest.mark.asyncio
async def test_rate_limit_partitions_by_credential_not_peer() -> None:
    app, login = build_app(rate_limit_settings(max_requests=2), LoginClient())

    # Distinct credentials own independent buckets even from the same peer:
    # each forged credential gets its full quota before a local 401 (no 429).
    forged = [
        await client_post(app, credential=f"forged-{index}")
        for index in range(4)
    ]

    assert [response.status_code for response in forged] == [401, 401, 401, 401]
    assert sorted(login.calls) == ["forged-0", "forged-1", "forged-2", "forged-3"]


@pytest.mark.asyncio
async def test_same_credential_quota_is_shared_across_forwarded_headers() -> None:
    app, login = build_app(rate_limit_settings(max_requests=2), LoginClient())

    responses = [
        await client_post(app, forwarded_for=f"10.0.0.{index}") for index in range(3)
    ]

    # X-Forwarded-For no longer partitions buckets: the same credential is
    # capped regardless of the spoofable header value.
    assert [response.status_code for response in responses] == [200, 200, 429]
    assert login.calls == [_VALID_OPEN_ID]


@pytest.mark.asyncio
async def test_distinct_credentials_get_independent_buckets() -> None:
    app, _ = build_app(rate_limit_settings(max_requests=2), LoginClient())

    assert (await client_post(app, credential="forged-a")).status_code == 401
    assert (await client_post(app, credential="forged-b")).status_code == 401
    assert (await client_post(app, credential="forged-a")).status_code == 401
    # Third call for "forged-a" exhausts only its own bucket.
    assert (await client_post(app, credential="forged-a")).status_code == 429
    # "forged-b" still has its own quota untouched.
    assert (await client_post(app, credential="forged-b")).status_code == 401


@pytest.mark.asyncio
async def test_missing_credential_is_rejected_without_consuming_quota() -> None:
    app, login = build_app(rate_limit_settings(max_requests=1), LoginClient())

    unauthenticated = [await client_post(app, include_auth=False) for _ in range(5)]

    assert {response.status_code for response in unauthenticated} == {401}
    # The credential bucket was never touched; the valid request still passes.
    assert (await client_post(app)).status_code == 200
    assert login.calls == [_VALID_OPEN_ID]


@pytest.mark.asyncio
async def test_rate_limit_window_reset_lets_traffic_through_again() -> None:
    now = [100.0]
    app, _ = build_app(rate_limit_settings(window_seconds=60.0, max_requests=2), LoginClient(), now)

    assert (await client_post(app)).status_code == 200
    assert (await client_post(app)).status_code == 200
    assert (await client_post(app)).status_code == 429

    now[0] = 160.0
    assert (await client_post(app)).status_code == 200


@pytest.mark.asyncio
async def test_healthz_is_exempt_from_rate_limiting() -> None:
    app, _ = build_app(rate_limit_settings(max_requests=1), LoginClient())

    assert (await client_post(app)).status_code == 200
    assert (await client_post(app)).status_code == 429
    for _ in range(5):
        assert (await client_post(app, path="/healthz")).status_code == 200


@pytest.mark.asyncio
async def test_disabled_rate_limit_lets_every_request_through() -> None:
    app, login = build_app(rate_limit_settings(enabled=False, max_requests=1), LoginClient())

    responses = [await client_post(app) for _ in range(20)]

    assert {response.status_code for response in responses} == {200}
    assert login.calls == [_VALID_OPEN_ID]
