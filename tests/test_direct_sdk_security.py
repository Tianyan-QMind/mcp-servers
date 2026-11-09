from __future__ import annotations

import asyncio
import hmac
import json
import logging
from typing import Any

import httpx
import pytest
from pydantic import SecretBytes, SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tianyan_mcp.config import Settings
from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_security import DirectSdkSecurityMiddleware
from tianyan_mcp.identity import IdentityMode, current_direct_sdk

_FINGERPRINT_KEY = b"f" * 32
_ORIGIN = "https://agent.example.com"


def direct_sdk_settings() -> Settings:
    return Settings(
        identity_mode=IdentityMode.DIRECT_SDK,
        direct_sdk_fingerprint_key=SecretBytes(_FINGERPRINT_KEY),
        allowed_origins=(_ORIGIN,),
    )


class AlwaysSuccessfulLoginClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        self.calls.append(open_id.get_secret_value())
        return SecretStr(f"token-{len(self.calls)}"), 720.0


def wrap_with_direct_sdk_security(inner: ASGIApp) -> DirectSdkSecurityMiddleware:
    manager = DirectSdkAuthManager(AlwaysSuccessfulLoginClient())
    return DirectSdkSecurityMiddleware(inner, direct_sdk_settings(), manager)


def direct_sdk_app(
    *,
    captured_scope: dict[str, Any] | None = None,
    seen: dict[str, Any] | None = None,
) -> DirectSdkSecurityMiddleware:
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def endpoint(request: Request) -> JSONResponse:
        principal = current_direct_sdk()
        if captured_scope is not None:
            captured_scope.update(request.scope)
        if seen is not None:
            seen.update(
                {
                    "open_id": principal.sdk_open_id.get_secret_value(),
                    "fingerprint": principal.sdk_fingerprint,
                }
            )
        await asyncio.sleep(0)
        return JSONResponse({"reached": True})

    inner = Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/mcp", endpoint, methods=["POST"]),
        ]
    )
    return wrap_with_direct_sdk_security(inner)


async def request(
    app: ASGIApp,
    *,
    path: str = "/mcp",
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
    method: str = "POST",
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions),
        base_url="http://127.0.0.1",
    ) as client:
        return await client.request(method, path, headers=headers)


async def raw_http_request(
    app: ASGIApp, headers: list[tuple[bytes, bytes]]
) -> tuple[int, dict[bytes, list[bytes]], bytes]:
    messages: list[Message] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"127.0.0.1"), *headers],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 80),
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    response_headers: dict[bytes, list[bytes]] = {}
    for name, value in start["headers"]:
        response_headers.setdefault(name, []).append(value)
    return start["status"], response_headers, body


@pytest.mark.asyncio
async def test_valid_request_scopes_fingerprint_without_retaining_credential(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    captured_scope: dict[str, Any] = {}
    seen: dict[str, Any] = {}

    response = await request(
        direct_sdk_app(captured_scope=captured_scope, seen=seen),
        headers={"Authorization": "Bearer alice-open-id"},
    )

    assert response.status_code == 200
    assert seen == {
        "open_id": "alice-open-id",
        "fingerprint": hmac.digest(b"f" * 32, b"alice-open-id", "sha256"),
    }
    assert "direct_sdk" not in captured_scope
    assert "authorization" not in response.text.lower()
    assert "alice-open-id" not in caplog.text
    with pytest.raises(RuntimeError, match="direct SDK context"):
        current_direct_sdk()


@pytest.mark.asyncio
async def test_only_exact_healthz_bypasses_authentication() -> None:
    app = direct_sdk_app()

    health = await request(app, path="/healthz", method="GET")
    near_health = await request(app, path="/healthz/", method="GET")

    assert health.status_code == 200
    assert near_health.status_code == 401


@pytest.mark.asyncio
async def test_non_http_scope_bypasses_authentication() -> None:
    calls: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(scope["type"])

    app = wrap_with_direct_sdk_security(inner)

    await app({"type": "lifespan"}, _empty_receive, _discard_send)

    assert calls == ["lifespan"]


@pytest.mark.asyncio
async def test_host_then_origin_then_authentication_boundary_order() -> None:
    app = direct_sdk_app()

    host = await request(
        app,
        headers={"Host": "evil.example", "Origin": "https://evil.example"},
    )
    origin = await request(app, headers={"Origin": "https://evil.example"})
    authentication = await request(app, headers={"Origin": _ORIGIN})

    assert (host.status_code, host.json()) == (421, {"error": "host_not_allowed"})
    assert (origin.status_code, origin.json()) == (403, {"error": "origin_forbidden"})
    assert (authentication.status_code, authentication.json()) == (
        401,
        {"error": "invalid_sdk_credential"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Bearer alice"), (b"authorization", b"Bearer bob")],
        [(b"authorization", b"bearer alice")],
        [(b"authorization", b"Bearer")],
        [(b"authorization", b"Bearer ")],
        [(b"authorization", b"Bearer alice ")],
        [(b"authorization", b"Bearer  alice")],
        [(b"authorization", b"Bearer alice\x7f")],
        [(b"authorization", b"Bearer " + b"a" * 4097)],
    ],
)
async def test_malformed_credentials_have_a_uniform_authentication_response(
    headers: list[tuple[bytes, bytes]],
) -> None:
    status, response_headers, body = await raw_http_request(direct_sdk_app(), headers)

    assert status == 401
    assert response_headers[b"www-authenticate"] == [b"Bearer"]
    assert json.loads(body) == {"error": "invalid_sdk_credential"}
    assert b"authorization" not in body.lower()


@pytest.mark.asyncio
async def test_4096_character_credential_is_accepted() -> None:
    credential = "a" * 4096
    seen: dict[str, Any] = {}

    response = await request(
        direct_sdk_app(seen=seen), headers={"Authorization": f"Bearer {credential}"}
    )

    assert response.status_code == 200
    assert seen["open_id"] == credential


@pytest.mark.asyncio
async def test_concurrent_requests_keep_alice_and_bob_principals_isolated() -> None:
    seen: list[str] = []

    async def endpoint(_: Request) -> JSONResponse:
        principal = current_direct_sdk()
        await asyncio.sleep(0)
        seen.append(principal.sdk_open_id.get_secret_value())
        return JSONResponse({"open_id": principal.sdk_open_id.get_secret_value()})

    app = wrap_with_direct_sdk_security(
        Starlette(routes=[Route("/mcp", endpoint, methods=["POST"])])
    )

    alice, bob = await asyncio.gather(
        request(app, headers={"Authorization": "Bearer alice-open-id"}),
        request(app, headers={"Authorization": "Bearer bob-open-id"}),
    )

    assert {alice.json()["open_id"], bob.json()["open_id"]} == {"alice-open-id", "bob-open-id"}
    assert set(seen) == {"alice-open-id", "bob-open-id"}


@pytest.mark.asyncio
async def test_context_is_cleaned_after_downstream_exception() -> None:
    async def crashing_endpoint(_: Request) -> JSONResponse:
        assert current_direct_sdk().sdk_open_id.get_secret_value() == "alice-open-id"
        raise RuntimeError("downstream failure")

    app = wrap_with_direct_sdk_security(
        Starlette(routes=[Route("/mcp", crashing_endpoint, methods=["POST"])])
    )

    response = await request(
        app,
        headers={"Authorization": "Bearer alice-open-id"},
        raise_app_exceptions=False,
    )

    assert response.status_code == 500
    with pytest.raises(RuntimeError, match="direct SDK context"):
        current_direct_sdk()


@pytest.mark.asyncio
async def test_context_is_cleaned_after_downstream_cancellation() -> None:
    started = asyncio.Event()
    cancelled_with_principal: list[bool] = []

    async def blocking_app(scope: Scope, receive: Receive, send: Send) -> None:
        assert current_direct_sdk().sdk_open_id.get_secret_value() == "alice-open-id"
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_with_principal.append(
                current_direct_sdk().sdk_open_id.get_secret_value() == "alice-open-id"
            )
            raise

    task = asyncio.create_task(
        raw_http_request(
            wrap_with_direct_sdk_security(blocking_app),
            [(b"authorization", b"Bearer alice-open-id")],
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_with_principal == [True]
    with pytest.raises(RuntimeError, match="direct SDK context"):
        current_direct_sdk()


async def _empty_receive() -> dict[str, object]:
    return {"type": "lifespan.startup"}


async def _discard_send(_: Message) -> None:
    return None
