from __future__ import annotations

import asyncio
import uuid

from starlette.types import ASGIApp, Receive, Scope, Send

from tianyan_mcp.config import Settings
from tianyan_mcp.identity import TenantPrincipal, tenant_scope
from tianyan_mcp.security import headers_for, send_json, validate_host_and_origin
from tianyan_mcp.tenant_store import AuditEventType, AuditOutcome, TenantDisabledError, TenantStore

_MAX_CREDENTIAL_LENGTH = 128


class TenantSecurityMiddleware:
    """Authenticate multi-tenant HTTP requests without retaining caller credentials."""

    def __init__(self, app: ASGIApp, settings: Settings, store: TenantStore) -> None:
        self._app = app
        self._store = store
        self._origins = set(settings.allowed_origins)
        self._hosts = {host.casefold() for host in settings.allowed_hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/healthz":
            await self._app(scope, receive, send)
            return

        boundary_error = validate_host_and_origin(scope, self._origins, self._hosts)
        if boundary_error is not None:
            await send_json(send, *boundary_error)
            return

        credential = _bearer_credential(scope)
        if credential is None:
            await self._record_unidentified_auth_failure()
            await _authentication_error(send)
            return

        request_id = str(uuid.uuid4())
        principal = await asyncio.to_thread(self._authenticate, credential, request_id)
        if principal is None:
            await _authentication_error(send)
            return

        with tenant_scope(principal, request_id=request_id):
            await self._app(scope, receive, send)

    def _authenticate(self, credential: str, request_id: str) -> TenantPrincipal | None:
        try:
            principal = self._store.authenticate(credential)
        except TenantDisabledError as error:
            principal = error.principal
            outcome = AuditOutcome.FAILURE
        else:
            outcome = AuditOutcome.SUCCESS if principal else AuditOutcome.FAILURE
        self._store.append_audit(
            principal.tenant_id if principal else None,
            principal.key_id if principal else None,
            AuditEventType.MCP_AUTHENTICATED
            if outcome is AuditOutcome.SUCCESS
            else AuditEventType.MCP_AUTH_FAILED,
            outcome,
            request_id,
        )
        return principal if outcome is AuditOutcome.SUCCESS else None

    async def _record_unidentified_auth_failure(self) -> None:
        try:
            await asyncio.to_thread(
                self._store.append_audit,
                None,
                None,
                AuditEventType.MCP_AUTH_FAILED,
                AuditOutcome.FAILURE,
                str(uuid.uuid4()),
            )
        except Exception:
            # Authentication never becomes permissive when auditing is unavailable.
            return


def _bearer_credential(scope: Scope) -> str | None:
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
    ):
        return None
    return credential


async def _authentication_error(send: Send) -> None:
    await send_json(
        send,
        401,
        {"error": "invalid_mcp_api_key"},
        [(b"www-authenticate", b"Bearer")],
    )
