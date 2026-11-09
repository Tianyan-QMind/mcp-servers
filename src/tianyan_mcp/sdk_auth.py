from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import httpx
from pydantic import SecretStr

from tianyan_mcp.errors import ServiceError

SDK_LOGIN_URL = "https://qc.zdxlz.com/qccp-auth/oauth2/sdk/opnId"
REFRESH_BUFFER_SECONDS = 60.0
logging.getLogger("httpx").setLevel(logging.WARNING)


class AccessTokenProvider(Protocol):
    async def get_access_token(self) -> str: ...

    def invalidate(self) -> None: ...


@dataclass(slots=True)
class _CachedToken:
    value: SecretStr
    expires_at: float


class TianyanSdkLoginClient:
    def __init__(
        self,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._transport = transport

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    SDK_LOGIN_URL,
                    params={"openId": open_id.get_secret_value()},
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录服务不可用", True) from exc
        if not 200 <= response.status_code < 300:
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录服务响应异常", True)
        try:
            raw: object = response.json()
        except ValueError as exc:
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录响应无效", True) from exc
        if not isinstance(raw, dict):
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录响应无效", True)
        payload = cast(dict[str, object], raw)
        code = payload.get("code")
        if type(code) is not int:
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录响应无效", True)
        if code != 0 or payload.get("success") is not True:
            raise ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录响应无效", True)
        payload_data = cast(dict[str, object], data)
        token = payload_data.get("access_token")
        expires_in = payload_data.get("expires_in")
        if not isinstance(token, str) or not token.strip():
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录响应缺少 Token", True)
        if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)):
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK Token 有效期无效", True)
        if not math.isfinite(expires_in) or expires_in <= 0:
            raise ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK Token 有效期无效", True)
        return SecretStr(token.strip()), float(expires_in)


class TianyanSdkAuthProvider:
    def __init__(
        self,
        open_id: SecretStr | None,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._open_id = open_id
        self._login_client = TianyanSdkLoginClient(timeout_seconds, transport=transport)
        self._clock = clock
        self._cached: _CachedToken | None = None
        self._generation = 0
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._cached = None
        self._generation += 1

    async def get_access_token(self) -> str:
        cached = self._valid_cached()
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._valid_cached()
            if cached is not None:
                return cached
            generation = self._generation
            token, expires_in = await self._login_once()
            if generation == self._generation:
                self._cached = _CachedToken(token, self._clock() + expires_in)
            return token.get_secret_value()

    def _valid_cached(self) -> str | None:
        if self._cached is None:
            return None
        if self._cached.expires_at - self._clock() <= REFRESH_BUFFER_SECONDS:
            return None
        return self._cached.value.get_secret_value()

    async def _login_once(self) -> tuple[SecretStr, float]:
        if self._open_id is None:
            raise ServiceError("SDK_AUTH_FAILED", "未配置天衍云 SDK 密钥", False)
        return await self._login_client.login(self._open_id)
