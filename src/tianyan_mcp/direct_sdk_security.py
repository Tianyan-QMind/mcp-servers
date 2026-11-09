from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
import unicodedata
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import SecretStr
from starlette.types import ASGIApp, Receive, Scope, Send

from tianyan_mcp.config import Settings
from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import DirectSdkPrincipal, direct_sdk_scope
from tianyan_mcp.security import headers_for, send_json, validate_host_and_origin

_MAX_CREDENTIAL_LENGTH = 4096
_MAX_RATE_LIMIT_KEYS = 10_000
#: Length of the ``sdk:<sha256>`` rate-limit key prefix written to logs.
_RATE_KEY_LOG_LENGTH = 20
_LOGGER = logging.getLogger(__name__)
AuthEnforcement = Literal["enforce", "shadow"]


@dataclass(slots=True)
class _RateWindow:
    started_at: float
    count: int = 0


class FixedWindowRateLimiter:
    """Per-key fixed-window counter bounded by an LRU eviction."""

    def __init__(
        self,
        window_seconds: float,
        max_requests: int,
        *,
        max_keys: int = _MAX_RATE_LIMIT_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._max_keys = max_keys
        self._clock = clock
        self._windows: OrderedDict[str, _RateWindow] = OrderedDict()

    def take(self, key: str) -> tuple[bool, int]:
        """Register one request. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        window = self._windows.get(key)
        if window is not None and now - window.started_at >= self._window_seconds:
            self._windows.pop(key, None)
            window = None
        if window is None:
            window = _RateWindow(now)
            self._windows[key] = window
        if window.count >= self._max_requests:
            retry_after = max(1, math.ceil(self._window_seconds - (now - window.started_at)))
            return False, retry_after
        window.count += 1
        self._windows.move_to_end(key)
        if len(self._windows) > self._max_keys:
            self._windows.popitem(last=False)
        return True, 0

    def __len__(self) -> int:
        return len(self._windows)


class DirectSdkSecurityMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        settings: Settings,
        auth_manager: DirectSdkAuthManager,
        *,
        verify_open_id: bool = True,
        enforcement: AuthEnforcement = "enforce",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if enforcement not in ("enforce", "shadow"):
            raise ValueError("enforcement must be 'enforce' or 'shadow'")
        if settings.direct_sdk_fingerprint_key is None:
            raise RuntimeError("direct SDK fingerprint key is unavailable")
        self._app = app
        self._auth_manager = auth_manager
        self._verify_open_id = verify_open_id
        self._enforcement: AuthEnforcement = enforcement
        self._fingerprint_key = settings.direct_sdk_fingerprint_key.get_secret_value()
        self._origins = set(settings.allowed_origins)
        self._hosts = {host.casefold() for host in settings.allowed_hosts}
        self._rate_limiter: FixedWindowRateLimiter | None = (
            FixedWindowRateLimiter(
                settings.direct_sdk_rate_limit_window_seconds,
                settings.direct_sdk_rate_limit_max_requests,
                clock=clock,
            )
            if settings.direct_sdk_rate_limit_enabled
            else None
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/healthz":
            await self._app(scope, receive, send)
            return
        boundary_error = validate_host_and_origin(scope, self._origins, self._hosts)
        if boundary_error is not None:
            await send_json(send, *boundary_error)
            return
        open_id = _bearer_open_id(scope)
        if open_id is None:
            await _authentication_error(send)
            return
        if self._rate_limiter is not None:
            client_key = _sdk_rate_key(open_id)
            allowed, retry_after = self._rate_limiter.take(client_key)
            if not allowed:
                request_id = str(uuid.uuid4())
                _LOGGER.warning(
                    "event=direct_sdk_rate_limited client=%s retry_after=%s request_id=%s",
                    client_key[: _RATE_KEY_LOG_LENGTH],
                    retry_after,
                    request_id,
                )
                await send_json(
                    send,
                    429,
                    {"error": "rate_limited", "retryable": "true"},
                    [(b"retry-after", str(retry_after).encode("ascii"))],
                )
                return
        principal = DirectSdkPrincipal(
            sdk_open_id=SecretStr(open_id),
            sdk_fingerprint=hmac.digest(self._fingerprint_key, open_id.encode("utf-8"), "sha256"),
            request_id=str(uuid.uuid4()),
        )
        if self._verify_open_id:
            try:
                await self._auth_manager.verify(principal)
            except ServiceError as exc:
                if not await self._handle_verification_failure(send, principal, exc):
                    return
        with direct_sdk_scope(principal):
            await self._app(scope, receive, send)

    async def _handle_verification_failure(
        self, send: Send, principal: DirectSdkPrincipal, exc: ServiceError
    ) -> bool:
        """Return True when the request may proceed (shadow mode)."""
        fingerprint_preview = principal.sdk_fingerprint[:8].hex()
        if self._enforcement == "shadow":
            _LOGGER.warning(
                "event=direct_sdk_openid_verify_failed_shadow code=%s retryable=%s "
                "fingerprint=%s request_id=%s",
                exc.code,
                exc.retryable,
                fingerprint_preview,
                principal.request_id,
            )
            return True
        if exc.code == "SDK_AUTH_FAILED":
            _LOGGER.warning(
                "event=direct_sdk_openid_rejected fingerprint=%s request_id=%s",
                fingerprint_preview,
                principal.request_id,
            )
            await _authentication_error(send)
            return False
        _LOGGER.warning(
            "event=direct_sdk_openid_verify_unavailable fingerprint=%s request_id=%s",
            fingerprint_preview,
            principal.request_id,
            exc_info=exc,
        )
        await send_json(
            send,
            503,
            {"error": "sdk_auth_unavailable", "retryable": "true"},
            [(b"retry-after", b"5"), (b"www-authenticate", b"Bearer")],
        )
        return False


def _sdk_rate_key(open_id: str) -> str:
    """Derive the per-credential rate-limit bucket key from the Bearer Open ID.

    The raw credential is never used as a map key or written to logs; its
    SHA-256 digest partitions buckets so distinct SDK credentials have
    independent quotas regardless of source IP.
    """
    digest = hashlib.sha256(open_id.encode("utf-8")).hexdigest()
    return f"sdk:{digest}"


def _bearer_open_id(scope: Scope) -> str | None:
    values = headers_for(scope).get(b"authorization", [])
    if len(values) != 1:
        return None
    value = values[0].decode("latin-1")
    if not value.startswith("Bearer "):
        return None
    credential = value[7:]
    if (
        not credential
        or credential != credential.strip()
        or len(credential) > _MAX_CREDENTIAL_LENGTH
        or any(unicodedata.category(character).startswith("C") for character in credential)
    ):
        return None
    return credential


async def _authentication_error(send: Send) -> None:
    await send_json(
        send,
        401,
        {"error": "invalid_sdk_credential"},
        [(b"www-authenticate", b"Bearer")],
    )
