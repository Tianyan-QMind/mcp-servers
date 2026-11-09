from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from pydantic import SecretBytes
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tianyan_mcp.config import Settings
from tianyan_mcp.credential_crypto import ApiKeyCodec
from tianyan_mcp.identity import IdentityMode, current_tenant
from tianyan_mcp.tenant_security import TenantSecurityMiddleware
from tianyan_mcp.tenant_store import TenantStore


@dataclass(frozen=True, slots=True)
class TenantCredentials:
    store: TenantStore
    alice_key: str
    bob_key: str
    alice_id: str
    bob_id: str


@pytest.fixture
def tenant_credentials(tmp_path: Path) -> TenantCredentials:
    codec = ApiKeyCodec(b"p" * 32)
    store = TenantStore(tmp_path / "tenants.sqlite", codec)
    store.initialize()
    alice_key = codec.generate()
    bob_key = codec.generate()
    alice = store.create_tenant("alice", alice_key)
    bob = store.create_tenant("bob", bob_key)
    return TenantCredentials(store, alice_key.plaintext, bob_key.plaintext, alice.id, bob.id)


@pytest.fixture
def multi_tenant_settings(tmp_path: Path) -> Settings:
    return Settings(
        identity_mode=IdentityMode.MULTI_TENANT,
        database_path=tmp_path / "tenants.sqlite",
        api_key_pepper=SecretBytes(b"p" * 32),
        credential_key_version=1,
        credential_keys={1: SecretBytes(b"k" * 32)},
        allowed_origins=("https://agent.example.com",),
    )


async def tenant_endpoint(_: Request) -> JSONResponse:
    principal = current_tenant()
    await asyncio.sleep(0)
    return JSONResponse({"tenant_id": principal.tenant_id, "key_id": principal.key_id})


async def health_endpoint(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def crashing_endpoint(_: Request) -> JSONResponse:
    assert current_tenant().tenant_id
    raise RuntimeError("downstream failure")


@pytest.fixture
def tenant_app(
    tenant_credentials: TenantCredentials, multi_tenant_settings: Settings
) -> TenantSecurityMiddleware:
    inner = Starlette(
        routes=[
            Route("/healthz", health_endpoint, methods=["GET"]),
            Route("/mcp", tenant_endpoint, methods=["POST"]),
            Route("/account/sdk-credential", tenant_endpoint, methods=["GET", "POST"]),
            Route("/crash", crashing_endpoint, methods=["POST"]),
        ]
    )
    return TenantSecurityMiddleware(inner, multi_tenant_settings, tenant_credentials.store)


async def request(
    app: TenantSecurityMiddleware,
    *,
    path: str = "/mcp",
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://127.0.0.1",
    ) as client:
        return await client.post(path, headers=headers)


@pytest.mark.asyncio
async def test_valid_key_sets_tenant_only_for_request(
    tenant_app: TenantSecurityMiddleware, tenant_credentials: TenantCredentials
) -> None:
    response = await request(
        tenant_app, headers={"Authorization": f"Bearer {tenant_credentials.alice_key}"}
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == tenant_credentials.alice_id
    with pytest.raises(RuntimeError, match="tenant context"):
        current_tenant()
    with sqlite3.connect(tenant_credentials.store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id "
            "FROM audit_events ORDER BY occurred_at DESC LIMIT 1"
        ).fetchone()
    assert event[:4] == (
        tenant_credentials.alice_id,
        tenant_credentials.store._codec.parse_key_id(  # pyright: ignore[reportPrivateUsage]
            tenant_credentials.alice_key
        ),
        "mcp_authenticated",
        "success",
    )
    assert event[4] is not None


@pytest.mark.asyncio
async def test_concurrent_keys_never_cross_tenants(
    tenant_app: TenantSecurityMiddleware, tenant_credentials: TenantCredentials
) -> None:
    alice, bob = await asyncio.gather(
        request(tenant_app, headers={"Authorization": f"Bearer {tenant_credentials.alice_key}"}),
        request(tenant_app, headers={"Authorization": f"Bearer {tenant_credentials.bob_key}"}),
    )

    assert {alice.json()["tenant_id"], bob.json()["tenant_id"]} == {
        tenant_credentials.alice_id,
        tenant_credentials.bob_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic text"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer " + "x" * 513},
        [("Authorization", "Bearer first"), ("Authorization", "Bearer second")],
        {"Authorization": "Bearer tymcp_0000000000000000_" + "x" * 43},
    ],
)
async def test_invalid_authorization_has_uniform_secret_free_response(
    tenant_app: TenantSecurityMiddleware,
    tenant_credentials: TenantCredentials,
    headers: dict[str, str] | list[tuple[str, str]],
) -> None:
    response = await request(tenant_app, headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"error": "invalid_mcp_api_key"}
    assert tenant_credentials.alice_key not in response.text
    assert "Authorization" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [{}, [("Authorization", "Bearer first"), ("Authorization", "Bearer second")]],
)
async def test_missing_or_duplicate_authorization_is_audited_without_identity(
    tenant_app: TenantSecurityMiddleware,
    tenant_credentials: TenantCredentials,
    headers: dict[str, str] | list[tuple[str, str]],
) -> None:
    response = await request(tenant_app, headers=headers)

    assert response.status_code == 401
    with sqlite3.connect(tenant_credentials.store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id FROM audit_events"
        ).fetchone()
    assert event[:4] == (None, None, "mcp_auth_failed", "failure")
    assert event[4] is not None


@pytest.mark.asyncio
async def test_invalid_authorization_audit_failure_remains_secret_free_and_closed(
    tenant_app: TenantSecurityMiddleware,
    tenant_credentials: TenantCredentials,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_audit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(tenant_credentials.store, "append_audit", fail_audit)
    response = await request(tenant_app)

    assert response.status_code == 401
    assert response.json() == {"error": "invalid_mcp_api_key"}
    assert "audit unavailable" not in response.text


@pytest.mark.asyncio
async def test_revoked_and_disabled_key_match_invalid_key_response(
    tenant_app: TenantSecurityMiddleware, tenant_credentials: TenantCredentials
) -> None:
    replacement = ApiKeyCodec(b"p" * 32).generate()
    tenant_credentials.store.rotate_key(tenant_credentials.alice_id, replacement)
    tenant_credentials.store.set_tenant_status(tenant_credentials.bob_id, enabled=False)

    invalid = await request(tenant_app, headers={"Authorization": "Bearer invalid"})
    revoked = await request(
        tenant_app, headers={"Authorization": f"Bearer {tenant_credentials.alice_key}"}
    )
    disabled = await request(
        tenant_app, headers={"Authorization": f"Bearer {tenant_credentials.bob_key}"}
    )

    assert revoked.status_code == disabled.status_code == invalid.status_code == 401
    assert revoked.json() == disabled.json() == invalid.json() == {"error": "invalid_mcp_api_key"}
    with sqlite3.connect(tenant_credentials.store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id "
            "FROM audit_events ORDER BY occurred_at DESC LIMIT 1"
        ).fetchone()
    assert event[:4] == (
        tenant_credentials.bob_id,
        tenant_credentials.store._codec.parse_key_id(  # pyright: ignore[reportPrivateUsage]
            tenant_credentials.bob_key
        ),
        "mcp_auth_failed",
        "failure",
    )
    assert event[4] is not None


@pytest.mark.asyncio
async def test_health_is_public_and_account_requires_authentication(
    tenant_app: TenantSecurityMiddleware, tenant_credentials: TenantCredentials
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=tenant_app), base_url="http://127.0.0.1"
    ) as client:
        health = await client.get("/healthz")
    account_missing_key = await request(tenant_app, path="/account/sdk-credential")
    account_valid_key = await request(
        tenant_app,
        path="/account/sdk-credential",
        headers={"Authorization": f"Bearer {tenant_credentials.alice_key}"},
    )

    assert health.status_code == 200
    assert account_missing_key.status_code == 401
    assert account_valid_key.status_code == 200
    assert account_valid_key.json()["tenant_id"] == tenant_credentials.alice_id


@pytest.mark.asyncio
async def test_host_and_origin_rejection_precede_authentication(
    tenant_app: TenantSecurityMiddleware, tenant_credentials: TenantCredentials
) -> None:
    bad_host = await request(
        tenant_app,
        headers={"Authorization": f"Bearer {tenant_credentials.alice_key}", "Host": "evil.example"},
    )
    bad_origin = await request(
        tenant_app,
        headers={
            "Authorization": f"Bearer {tenant_credentials.alice_key}",
            "Origin": "https://evil.example",
        },
    )

    assert bad_host.status_code == 421
    assert bad_host.json() == {"error": "host_not_allowed"}
    assert bad_origin.status_code == 403
    assert bad_origin.json() == {"error": "origin_forbidden"}


@pytest.mark.asyncio
async def test_key_is_not_in_scope_response_or_logs(
    tenant_app: TenantSecurityMiddleware,
    tenant_credentials: TenantCredentials,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    response = await request(
        tenant_app, headers={"Authorization": f"Bearer {tenant_credentials.alice_key}"}
    )

    assert response.status_code == 200
    assert tenant_credentials.alice_key not in response.text
    assert tenant_credentials.alice_key not in caplog.text


@pytest.mark.asyncio
async def test_context_is_cleaned_after_downstream_error(
    tenant_app: TenantSecurityMiddleware, tenant_credentials: TenantCredentials
) -> None:
    response = await request(
        tenant_app,
        path="/crash",
        headers={"Authorization": f"Bearer {tenant_credentials.alice_key}"},
    )

    assert response.status_code == 500
    with pytest.raises(RuntimeError, match="tenant context"):
        current_tenant()
