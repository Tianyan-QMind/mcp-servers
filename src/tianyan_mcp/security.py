from __future__ import annotations

import hmac
import json
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

from tianyan_mcp.config import Settings


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self._app = app
        self._api_key = settings.api_key_value()
        self._origins = set(settings.allowed_origins)
        self._hosts = {host.casefold() for host in settings.allowed_hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/healthz":
            await self._app(scope, receive, send)
            return
        if not scope.get("path", "").startswith("/mcp"):
            await self._app(scope, receive, send)
            return

        boundary_error = validate_host_and_origin(scope, self._origins, self._hosts)
        if boundary_error is not None:
            await send_json(send, *boundary_error)
            return

        headers = headers_for(scope)
        authorization = _one(headers, b"authorization")
        if authorization is None:
            await send_json(
                send,
                401,
                {"error": "authentication_required"},
                [(b"www-authenticate", b"Bearer")],
            )
            return
        value = authorization.decode("latin-1")
        if not value.startswith("Bearer ") or not hmac.compare_digest(
            value[7:].strip(), self._api_key
        ):
            await send_json(
                send,
                401,
                {"error": "invalid_token"},
                [(b"www-authenticate", b"Bearer")],
            )
            return
        await self._app(scope, receive, send)


def headers_for(scope: Scope) -> dict[bytes, list[bytes]]:
    result: dict[bytes, list[bytes]] = {}
    for key, value in scope.get("headers", []):
        result.setdefault(key.lower(), []).append(value)
    return result


def _one(headers: dict[bytes, list[bytes]], name: bytes) -> bytes | None:
    values = headers.get(name, [])
    return values[0] if len(values) == 1 else None


def validate_host_and_origin(
    scope: Scope, origins: set[str], hosts: set[str]
) -> tuple[int, dict[str, str]] | None:
    headers = headers_for(scope)
    hostname = _hostname(_one(headers, b"host"))
    if hostname is None or hostname.casefold() not in hosts:
        return 421, {"error": "host_not_allowed"}

    origin_values = headers.get(b"origin", [])
    if len(origin_values) > 1 or (
        len(origin_values) == 1 and origin_values[0].decode("latin-1") not in origins
    ):
        return 403, {"error": "origin_forbidden"}
    return None


def _hostname(host: bytes | None) -> str | None:
    if host is None:
        return None
    try:
        return urlsplit("//" + host.decode("latin-1")).hostname
    except ValueError:
        return None


async def send_json(
    send: Send,
    status: int,
    payload: dict[str, str],
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    headers.extend(extra_headers or [])
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body})
