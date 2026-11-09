from __future__ import annotations

import asyncio
import hmac
import logging

import httpx
import pytest
from pydantic import SecretBytes, SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tianyan_mcp.config import Settings
from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_security import DirectSdkSecurityMiddleware
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import IdentityMode

_FINGERPRINT_KEY = b"f" * 32
_VALID_OPEN_ID = "alice-open-id"
_FORGED_OPEN_ID = "mallory-open-id"
_DOWN_OPEN_ID = "downstream-open-id"


def direct_settings() -> Settings:
    return Settings(
        identity_mode=IdentityMode.DIRECT_SDK,
        direct_sdk_fingerprint_key=SecretBytes(_FINGERPRINT_KEY),
    )


class BoundaryLoginClient:
    def __init__(
        self,
        *,
        valid: frozenset[str] = frozenset({_VALID_OPEN_ID}),
        unavailable: frozenset[str] = frozenset({_DOWN_OPEN_ID}),
    ) -> None:
        self.calls: list[str] = []
        self._valid = valid
        self._unavailable = unavailable

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        value = open_id.get_secret_value()
        self.calls.append(value)
        await asyncio.sleep(0)
        if value in self._unavailable:
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录服务不可用", True)
        if value in self._valid:
            return SecretStr(f"{value}-token"), 720.0
        raise ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)


async def endpoint(_: Request) -> JSONResponse:
    return JSONResponse({"reached": True})


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_app(
    login: BoundaryLoginClient,
    *,
    verify_open_id: bool = True,
    enforcement: str = "enforce",
) -> tuple[DirectSdkSecurityMiddleware, BoundaryLoginClient]:
    manager = DirectSdkAuthManager(login)
    inner = Starlette(
        routes=[
            Route("/mcp", endpoint, methods=["POST"]),
            Route("/healthz", health, methods=["GET"]),
        ]
    )
    app = DirectSdkSecurityMiddleware(
        inner,
        direct_settings(),
        manager,
        verify_open_id=verify_open_id,
        enforcement=enforcement,  # type: ignore[arg-type]
    )
    return app, login


async def post(
    app: DirectSdkSecurityMiddleware, *, credential: str | None = _FORGED_OPEN_ID
) -> httpx.Response:
    headers = {"Authorization": f"Bearer {credential}"} if credential is not None else None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        return await client.post("/mcp", headers=headers)


@pytest.mark.asyncio
async def test_missing_and_forged_credentials_have_byte_identical_401() -> None:
    app, login = build_app(BoundaryLoginClient())

    missing = await post(app, credential=None)
    forged = await post(app, credential=_FORGED_OPEN_ID)

    assert missing.status_code == 401
    assert forged.status_code == 401
    assert missing.content == forged.content
    assert forged.json() == {"error": "invalid_sdk_credential"}
    assert forged.headers["www-authenticate"] == "Bearer"
    assert login.calls == [_FORGED_OPEN_ID]


@pytest.mark.asyncio
async def test_forged_credential_is_verified_upstream_only_five_times_per_window() -> None:
    app, login = build_app(BoundaryLoginClient())

    responses = [await post(app) for _ in range(7)]

    assert {response.status_code for response in responses} == {401}
    assert login.calls == [_FORGED_OPEN_ID] * 5


@pytest.mark.asyncio
async def test_sdk_auth_unavailable_fails_closed_with_503() -> None:
    app, login = build_app(BoundaryLoginClient())

    first = await post(app, credential=_DOWN_OPEN_ID)
    second = await post(app, credential=_DOWN_OPEN_ID)

    assert first.status_code == 503
    assert first.json() == {"error": "sdk_auth_unavailable", "retryable": "true"}
    assert first.headers["retry-after"] == "5"
    assert first.headers["www-authenticate"] == "Bearer"
    assert second.status_code == 503
    # Upstream outages must not consume the failure budget.
    assert login.calls == [_DOWN_OPEN_ID, _DOWN_OPEN_ID]


@pytest.mark.asyncio
async def test_valid_credential_passes_once_and_reuses_positive_cache() -> None:
    app, login = build_app(BoundaryLoginClient())

    first = await post(app, credential=_VALID_OPEN_ID)
    second = await post(app, credential=_VALID_OPEN_ID)

    assert first.status_code == 200
    assert first.json() == {"reached": True}
    assert second.status_code == 200
    assert login.calls == [_VALID_OPEN_ID]


@pytest.mark.asyncio
async def test_healthz_stays_available_while_upstream_login_is_down() -> None:
    manager = DirectSdkAuthManager(BoundaryLoginClient())
    inner = Starlette(routes=[Route("/healthz", health, methods=["GET"])])
    app = DirectSdkSecurityMiddleware(inner, direct_settings(), manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_shadow_mode_logs_rejection_but_allows_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    app, login = build_app(BoundaryLoginClient(), enforcement="shadow")

    response = await post(app)

    assert response.status_code == 200
    assert response.json() == {"reached": True}
    assert login.calls == [_FORGED_OPEN_ID]
    fingerprint_preview = hmac.digest(_FINGERPRINT_KEY, _FORGED_OPEN_ID.encode("utf-8"), "sha256")[
        :8
    ].hex()
    assert "shadow" in caplog.text.lower()
    assert fingerprint_preview in caplog.text
    assert _FORGED_OPEN_ID not in caplog.text


@pytest.mark.asyncio
async def test_shadow_mode_also_passes_through_upstream_outage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    app, _ = build_app(BoundaryLoginClient(), enforcement="shadow")

    response = await post(app, credential=_DOWN_OPEN_ID)

    assert response.status_code == 200
    assert "shadow" in caplog.text.lower()
    assert _DOWN_OPEN_ID not in caplog.text


@pytest.mark.asyncio
async def test_disabled_verification_passes_without_contacting_upstream() -> None:
    app, login = build_app(BoundaryLoginClient(), verify_open_id=False)

    response = await post(app)

    assert response.status_code == 200
    assert login.calls == []


@pytest.mark.asyncio
async def test_malformed_credential_is_rejected_before_upstream_verification() -> None:
    app, login = build_app(BoundaryLoginClient())

    response = await post(app, credential="")

    assert response.status_code == 401
    assert response.json() == {"error": "invalid_sdk_credential"}
    assert login.calls == []


@pytest.mark.asyncio
async def test_enforce_rejection_log_never_contains_open_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    app, _ = build_app(BoundaryLoginClient())

    response = await post(app)

    assert response.status_code == 401
    assert _FORGED_OPEN_ID not in caplog.text


def test_unknown_enforcement_mode_is_rejected_at_construction() -> None:
    manager = DirectSdkAuthManager(BoundaryLoginClient())
    inner = Starlette(routes=[Route("/mcp", endpoint, methods=["POST"])])

    with pytest.raises(ValueError):
        DirectSdkSecurityMiddleware(
            inner,
            direct_settings(),
            manager,
            enforcement="invalid",  # type: ignore[arg-type]
        )
