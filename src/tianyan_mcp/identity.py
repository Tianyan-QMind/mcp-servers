from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

from pydantic import SecretStr


class IdentityMode(StrEnum):
    SINGLE_USER = "single_user"
    MULTI_TENANT = "multi_tenant"
    DIRECT_SDK = "direct_sdk"


@dataclass(frozen=True, slots=True)
class TenantPrincipal:
    tenant_id: str
    key_id: str
    auth_revision: int


@dataclass(frozen=True, slots=True)
class DirectSdkPrincipal:
    sdk_open_id: SecretStr
    sdk_fingerprint: bytes
    request_id: str


_TENANT: ContextVar[TenantPrincipal | None] = ContextVar("tianyan_tenant", default=None)
_AUDIT_REQUEST_ID: ContextVar[str | None] = ContextVar("tianyan_audit_request_id", default=None)
_DIRECT_SDK: ContextVar[DirectSdkPrincipal | None] = ContextVar("tianyan_direct_sdk", default=None)


@contextmanager
def tenant_scope(principal: TenantPrincipal, *, request_id: str | None = None) -> Generator[None]:
    token = _TENANT.set(principal)
    request_token = _AUDIT_REQUEST_ID.set(request_id or str(uuid.uuid4()))
    try:
        yield
    finally:
        _AUDIT_REQUEST_ID.reset(request_token)
        _TENANT.reset(token)


def current_tenant() -> TenantPrincipal:
    principal = _TENANT.get()
    if principal is None:
        raise RuntimeError("tenant context is unavailable")
    return principal


def current_audit_request_id() -> str:
    request_id = _AUDIT_REQUEST_ID.get()
    if request_id is None:
        raise RuntimeError("audit request context is unavailable")
    return request_id


@contextmanager
def direct_sdk_scope(principal: DirectSdkPrincipal) -> Generator[None]:
    token = _DIRECT_SDK.set(principal)
    try:
        yield
    finally:
        _DIRECT_SDK.reset(token)


def current_direct_sdk() -> DirectSdkPrincipal:
    principal = _DIRECT_SDK.get()
    if principal is None:
        raise RuntimeError("direct SDK context is unavailable")
    return principal
