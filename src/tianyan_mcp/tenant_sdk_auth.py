from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr

from tianyan_mcp.credential_crypto import CredentialCipher, CredentialDecryptionError
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import TenantPrincipal, current_audit_request_id
from tianyan_mcp.sdk_auth import REFRESH_BUFFER_SECONDS
from tianyan_mcp.tenant_store import AuditEventType, AuditOutcome, TenantStore


class SdkLoginClient(Protocol):
    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]: ...


@dataclass(slots=True)
class _TenantToken:
    value: SecretStr
    expires_at: float
    credential_revision: int


@dataclass(slots=True)
class _LockSlot:
    lock: asyncio.Lock
    users: int = 0


class TenantSdkAuthManager:
    def __init__(
        self,
        store: TenantStore,
        cipher: CredentialCipher,
        login_client: SdkLoginClient,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._cipher = cipher
        self._login_client = login_client
        self._clock = clock
        self._tokens: dict[tuple[str, int], _TenantToken] = {}
        self._locks: dict[tuple[str, int], _LockSlot] = {}
        self._generations: dict[str, int] = {}

    async def get_access_token(self, principal: TenantPrincipal) -> str:
        credential = await asyncio.to_thread(self._store.get_credential, principal.tenant_id)
        if credential is None:
            raise ServiceError(
                "SDK_CREDENTIAL_NOT_CONFIGURED", "SDK credential is not configured", False
            )
        cache_key = (principal.tenant_id, credential.credential_revision)
        self._retire_stale_tenant_tokens(principal.tenant_id, credential.credential_revision)
        cached = self._valid_cached(cache_key)
        if cached is not None:
            return cached
        slot = self._locks.get(cache_key)
        if slot is None:
            slot = _LockSlot(asyncio.Lock())
            self._locks[cache_key] = slot
        slot.users += 1
        try:
            async with slot.lock:
                cached = self._valid_cached(cache_key)
                if cached is not None:
                    return cached
                generation = self._generations.get(principal.tenant_id, 0)
                try:
                    open_id = self._cipher.decrypt(principal.tenant_id, credential.encrypted)
                except CredentialDecryptionError as error:
                    await self._record_sdk_failure(principal)
                    raise ServiceError(
                        "SDK_AUTH_FAILED", "SDK credential is unavailable", False
                    ) from error
                try:
                    token, expires_in = await self._login_client.login(open_id)
                except ServiceError:
                    await self._record_sdk_failure(principal)
                    raise
                if generation == self._generations.get(principal.tenant_id, 0):
                    self._tokens[cache_key] = _TenantToken(
                        token,
                        self._clock() + expires_in,
                        credential.credential_revision,
                    )
                return token.get_secret_value()
        finally:
            slot.users -= 1
            if slot.users == 0 and self._locks.get(cache_key) is slot:
                self._locks.pop(cache_key, None)

    def invalidate(self, tenant_id: str) -> None:
        self._generations[tenant_id] = self._generations.get(tenant_id, 0) + 1
        for cache_key in tuple(self._tokens):
            if cache_key[0] == tenant_id:
                self._tokens.pop(cache_key, None)

    async def record_submission(self, principal: TenantPrincipal, outcome: AuditOutcome) -> None:
        await asyncio.to_thread(
            self._store.append_audit,
            principal.tenant_id,
            principal.key_id,
            AuditEventType.EXPERIMENT_SUBMITTED,
            outcome,
            current_audit_request_id(),
        )

    async def _record_sdk_failure(self, principal: TenantPrincipal) -> None:
        await asyncio.to_thread(
            self._store.append_audit,
            principal.tenant_id,
            principal.key_id,
            AuditEventType.SDK_AUTH_FAILED,
            AuditOutcome.FAILURE,
            current_audit_request_id(),
        )

    def _valid_cached(self, cache_key: tuple[str, int]) -> str | None:
        cached = self._tokens.get(cache_key)
        if cached is None:
            return None
        if cached.expires_at - self._clock() <= REFRESH_BUFFER_SECONDS:
            self._tokens.pop(cache_key, None)
            return None
        return cached.value.get_secret_value()

    def _retire_stale_tenant_tokens(self, tenant_id: str, credential_revision: int) -> None:
        for cache_key in tuple(self._tokens):
            if cache_key[0] == tenant_id and cache_key[1] != credential_revision:
                self._tokens.pop(cache_key, None)
