from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tianyan_mcp.credential_crypto import CredentialCipher
from tianyan_mcp.identity import current_audit_request_id, current_tenant
from tianyan_mcp.tenant_store import AuditEventType, TenantStore

_MAX_BODY_BYTES = 65536


class CredentialPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sdk_open_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
    ]


def credential_routes(
    store: TenantStore, cipher: CredentialCipher, token_invalidator: Callable[[str], None]
) -> list[Route]:
    async def get_credential(_: Request) -> JSONResponse:
        try:
            principal = current_tenant()
            record = await asyncio.to_thread(store.get_credential, principal.tenant_id)
        except Exception:
            return _error(500, "credential_operation_failed")
        if record is None:
            return JSONResponse({"configured": False, "updated_at": None})
        return JSONResponse(
            {
                "configured": True,
                "updated_at": record.updated_at,
            }
        )

    async def put_credential(request: Request) -> JSONResponse:
        try:
            body = await _read_body(request)
            payload = CredentialPayload.model_validate(json.loads(body))
            principal = current_tenant()
            token_invalidator(principal.tenant_id)
            encrypted = cipher.encrypt(principal.tenant_id, SecretStr(payload.sdk_open_id))
            status = await asyncio.to_thread(
                store.put_credential,
                principal.tenant_id,
                encrypted,
                audit_event=AuditEventType.CREDENTIAL_UPDATED,
                audit_key_id=principal.key_id,
                request_id=current_audit_request_id(),
            )
            _best_effort_invalidate(token_invalidator, principal.tenant_id)
        except _BodyTooLarge:
            return _error(413, "credential_body_too_large")
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            return _error(400, "invalid_credential_payload")
        except Exception:
            return _error(500, "credential_operation_failed")
        return JSONResponse(
            {
                "configured": True,
                "credential_revision": status.credential_revision,
                "updated_at": status.updated_at,
            }
        )

    async def delete_credential(_: Request) -> JSONResponse:
        try:
            principal = current_tenant()
            token_invalidator(principal.tenant_id)
            status = await asyncio.to_thread(
                store.delete_credential,
                principal.tenant_id,
                audit_event=AuditEventType.CREDENTIAL_DELETED,
                audit_key_id=principal.key_id,
                request_id=current_audit_request_id(),
            )
            _best_effort_invalidate(token_invalidator, principal.tenant_id)
        except Exception:
            return _error(500, "credential_operation_failed")
        return JSONResponse(
            {
                "configured": False,
                "credential_revision": status.credential_revision,
                "updated_at": status.updated_at,
            }
        )

    return [
        Route("/account/sdk-credential", get_credential, methods=["GET"]),
        Route("/account/sdk-credential", put_credential, methods=["PUT"]),
        Route("/account/sdk-credential", delete_credential, methods=["DELETE"]),
    ]


class _BodyTooLarge(Exception):
    pass


async def _read_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                raise _BodyTooLarge
        except ValueError as error:
            raise _BodyTooLarge from error
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_BODY_BYTES:
            raise _BodyTooLarge
        body.extend(chunk)
    return bytes(body)


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


def _best_effort_invalidate(token_invalidator: Callable[[str], None], tenant_id: str) -> None:
    try:
        token_invalidator(tenant_id)
    except Exception:
        return
