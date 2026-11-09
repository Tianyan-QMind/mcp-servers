from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import DirectSdkPrincipal
from tianyan_mcp.sdk_auth import REFRESH_BUFFER_SECONDS

_MAX_TOKEN_ENTRIES = 1024
_MAX_VERIFY_ENTRIES = 4096
_REJECT_TTL_SECONDS = 60.0
_MAX_VERIFY_ATTEMPTS = 5
_FAILED_LOGIN = ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)


class SdkLoginClient(Protocol):
    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]: ...


@dataclass(slots=True)
class _DirectToken:
    value: SecretStr
    expires_at: float


@dataclass(slots=True)
class _VerifyWindow:
    started_at: float
    failures: int = 0


@dataclass(slots=True)
class _LockSlot:
    lock: asyncio.Lock
    users: int = 0
    attempt_generation: int = 0
    last_error: ServiceError | None = None


class DirectSdkAuthManager:
    def __init__(
        self,
        login_client: SdkLoginClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        reject_ttl_seconds: float = _REJECT_TTL_SECONDS,
        max_verify_attempts: int = _MAX_VERIFY_ATTEMPTS,
    ) -> None:
        if reject_ttl_seconds <= 0:
            raise ValueError("reject_ttl_seconds must be positive")
        if max_verify_attempts < 1:
            raise ValueError("max_verify_attempts must be >= 1")
        self._login_client = login_client
        self._clock = clock
        self._reject_ttl = reject_ttl_seconds
        self._max_verify_attempts = max_verify_attempts
        self._tokens: OrderedDict[bytes, _DirectToken] = OrderedDict()
        self._verify_windows: OrderedDict[bytes, _VerifyWindow] = OrderedDict()
        self._locks: dict[bytes, _LockSlot] = {}
        self._owner_guard = threading.Lock()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._owner_thread_id: int | None = None

    async def get_access_token(self, principal: DirectSdkPrincipal) -> str:
        self._bind_or_check_owner()
        cache_key = principal.sdk_fingerprint
        cached = self._valid_cached(cache_key)
        if cached is not None:
            return cached
        slot = self._retain_slot(cache_key)
        try:
            async with slot.lock:
                cached = self._valid_cached(cache_key)
                if cached is not None:
                    return cached
                token, expires_in = await self._login_client.login(principal.sdk_open_id)
                self._store(cache_key, _DirectToken(token, self._clock() + expires_in))
                return token.get_secret_value()
        finally:
            self._release_slot(cache_key, slot)

    async def verify(self, principal: DirectSdkPrincipal) -> None:
        """Validate the caller's SDK Open ID against Tianyan before serving it.

        A successful login prewarms the positive token cache; forged Open IDs
        get a fixed-window failure budget so credential-stuffing floods cannot
        turn this verification into an upstream amplifier. Callers queued on
        the per-fingerprint single-flight lock reuse the leading attempt's
        outcome instead of each replaying a login.
        """
        self._bind_or_check_owner()
        cache_key = principal.sdk_fingerprint
        if self._valid_cached(cache_key) is not None:
            return
        slot = self._retain_slot(cache_key)
        seen_generation = slot.attempt_generation
        try:
            async with slot.lock:
                if self._valid_cached(cache_key) is not None:
                    return
                now = self._clock()
                window = self._verify_windows.get(cache_key)
                if window is not None and now - window.started_at >= self._reject_ttl:
                    self._verify_windows.pop(cache_key, None)
                    window = None
                if window is not None and window.failures >= self._max_verify_attempts:
                    raise _FAILED_LOGIN
                if slot.attempt_generation != seen_generation:
                    # A leading queued attempt already landed while we waited;
                    # its success is covered by the positive-cache check above.
                    raise slot.last_error or _FAILED_LOGIN
                try:
                    token, expires_in = await self._login_client.login(principal.sdk_open_id)
                except ServiceError as exc:
                    slot.attempt_generation += 1
                    slot.last_error = exc
                    if exc.code == "SDK_AUTH_FAILED":
                        if window is None:
                            window = _VerifyWindow(now)
                            self._verify_windows[cache_key] = window
                        window.failures += 1
                        self._verify_windows.move_to_end(cache_key)
                        while len(self._verify_windows) > _MAX_VERIFY_ENTRIES:
                            self._verify_windows.popitem(last=False)
                    raise
                slot.attempt_generation += 1
                slot.last_error = None
                self._verify_windows.pop(cache_key, None)
                self._store(cache_key, _DirectToken(token, self._clock() + expires_in))
        finally:
            self._release_slot(cache_key, slot)

    def invalidate(self, fingerprint: bytes) -> None:
        with self._owner_guard:
            if self._owner_loop is not None:
                self._check_owner()
            self._tokens.pop(fingerprint, None)
            self._verify_windows.pop(fingerprint, None)

    def _retain_slot(self, cache_key: bytes) -> _LockSlot:
        slot = self._locks.get(cache_key)
        if slot is None:
            slot = _LockSlot(asyncio.Lock())
            self._locks[cache_key] = slot
        slot.users += 1
        return slot

    def _release_slot(self, cache_key: bytes, slot: _LockSlot) -> None:
        slot.users -= 1
        if slot.users == 0 and self._locks.get(cache_key) is slot:
            self._locks.pop(cache_key, None)

    def _bind_or_check_owner(self) -> None:
        with self._owner_guard:
            if self._owner_loop is None:
                self._owner_loop = asyncio.get_running_loop()
                self._owner_thread_id = threading.get_ident()
                return
            self._check_owner()

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "DirectSdkAuthManager must be used from its owner event loop and thread"
            )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "DirectSdkAuthManager must be used from its owner event loop and thread"
            ) from exc
        if current_loop is not self._owner_loop:
            raise RuntimeError(
                "DirectSdkAuthManager must be used from its owner event loop and thread"
            )

    def _valid_cached(self, cache_key: bytes) -> str | None:
        cached = self._tokens.get(cache_key)
        if cached is None:
            return None
        if cached.expires_at - self._clock() <= REFRESH_BUFFER_SECONDS:
            self._tokens.pop(cache_key, None)
            return None
        self._tokens.move_to_end(cache_key)
        return cached.value.get_secret_value()

    def _store(self, cache_key: bytes, token: _DirectToken) -> None:
        self._tokens[cache_key] = token
        self._tokens.move_to_end(cache_key)
        while len(self._tokens) > _MAX_TOKEN_ENTRIES:
            self._tokens.popitem(last=False)
