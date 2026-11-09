from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette

from tianyan_mcp.client_cli import main as client_main
from tianyan_mcp.credential_crypto import ApiKeyCodec, CredentialCipher
from tianyan_mcp.credential_routes import credential_routes
from tianyan_mcp.identity import TenantPrincipal, tenant_scope
from tianyan_mcp.tenant_store import TenantStore


@pytest.fixture
def store(tmp_path: Path) -> TenantStore:
    result = TenantStore(tmp_path / "tenants.sqlite", ApiKeyCodec(b"p" * 32))
    result.initialize()
    result.create_tenant("alice", ApiKeyCodec(b"p" * 32).generate())
    return result


@pytest.fixture
def alice(store: TenantStore) -> TenantPrincipal:
    return TenantPrincipal(store.list_tenants()[0].id, "0000000000000000", 1)


async def request(app: Starlette, principal: TenantPrincipal, **kwargs: object) -> httpx.Response:
    async def scoped(scope: object, receive: object, send: object) -> None:
        with tenant_scope(principal):
            await app(scope, receive, send)  # type: ignore[misc]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=scoped), base_url="http://test"
    ) as client:
        return await client.request(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_put_get_delete_encrypts_current_tenant_and_invalidates_only_it(
    store: TenantStore, alice: TenantPrincipal
) -> None:
    invalidated: list[str] = []
    cipher = CredentialCipher(active_version=1, keys={1: b"k" * 32})
    app = Starlette(routes=credential_routes(store, cipher, invalidated.append))

    put = await request(
        app,
        alice,
        method="PUT",
        url="/account/sdk-credential",
        json={"sdk_open_id": "  fake-open-id  "},
    )
    get = await request(app, alice, method="GET", url="/account/sdk-credential")
    delete = await request(app, alice, method="DELETE", url="/account/sdk-credential")

    assert put.status_code == 200
    assert get.json()["configured"] is True
    assert delete.json()["configured"] is False
    assert invalidated == [alice.tenant_id, alice.tenant_id, alice.tenant_id, alice.tenant_id]
    assert b"fake-open-id" not in store.path.read_bytes()
    with sqlite3.connect(store.path) as connection:
        events = connection.execute(
            "SELECT key_id, event_type, outcome, request_id FROM audit_events ORDER BY occurred_at"
        ).fetchall()
    assert [(event[0], event[1], event[2]) for event in events] == [
        (alice.key_id, "credential_updated", "success"),
        (alice.key_id, "credential_deleted", "success"),
    ]
    assert all(event[3] is not None for event in events)


@pytest.mark.asyncio
async def test_actual_get_contract_is_accepted_by_strict_status_cli(
    store: TenantStore,
    alice: TenantPrincipal,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def make_handler(body: dict[str, object]) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body, request=request)

        return httpx.MockTransport(handler)

    app = Starlette(
        routes=credential_routes(
            store, CredentialCipher(active_version=1, keys={1: b"k" * 32}), lambda _: None
        )
    )
    configured = await request(
        app,
        alice,
        method="PUT",
        url="/account/sdk-credential",
        json={"sdk_open_id": "fake-open-id"},
    )
    assert configured.status_code == 200
    configured_status = await request(app, alice, method="GET", url="/account/sdk-credential")
    deleted = await request(app, alice, method="DELETE", url="/account/sdk-credential")
    assert deleted.status_code == 200
    unconfigured_status = await request(app, alice, method="GET", url="/account/sdk-credential")
    monkeypatch.setattr("getpass.getpass", lambda: "fake-mcp-key")

    for status, expected_configured in ((configured_status, True), (unconfigured_status, False)):
        assert (
            client_main(
                ["status", "--server", "https://mcp.example.com"],
                transport=make_handler(status.json()),
            )
            == 0
        )
        assert f"configured: {str(expected_configured).lower()}" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [b"not-json", b'{"sdk_open_id":"value","extra":"no"}', b'{"sdk_open_id":"   "}'],
)
async def test_put_rejects_invalid_payload_without_leaking_it(
    store: TenantStore, alice: TenantPrincipal, payload: bytes
) -> None:
    app = Starlette(
        routes=credential_routes(
            store, CredentialCipher(active_version=1, keys={1: b"k" * 32}), lambda _: None
        )
    )
    response = await request(
        app, alice, method="PUT", url="/account/sdk-credential", content=payload
    )
    assert response.status_code == 400
    assert payload.decode("utf-8", errors="ignore") not in response.text


@pytest.mark.asyncio
async def test_put_rejects_large_body_before_storage(
    store: TenantStore, alice: TenantPrincipal
) -> None:
    app = Starlette(
        routes=credential_routes(
            store, CredentialCipher(active_version=1, keys={1: b"k" * 32}), lambda _: None
        )
    )
    response = await request(
        app, alice, method="PUT", url="/account/sdk-credential", content=b"x" * 70000
    )
    assert response.status_code == 413
    assert store.get_credential(alice.tenant_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["量" * 4096, "😀" * 4096], ids=["chinese", "emoji"])
async def test_put_accepts_maximum_unicode_open_id(
    store: TenantStore, alice: TenantPrincipal, value: str
) -> None:
    app = Starlette(
        routes=credential_routes(
            store, CredentialCipher(active_version=1, keys={1: b"k" * 32}), lambda _: None
        )
    )
    response = await request(
        app, alice, method="PUT", url="/account/sdk-credential", json={"sdk_open_id": value}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalidator_failure_before_write_leaves_credential_unchanged(
    store: TenantStore, alice: TenantPrincipal
) -> None:
    def fail(_: str) -> None:
        raise RuntimeError("invalidate failed")

    app = Starlette(
        routes=credential_routes(
            store, CredentialCipher(active_version=1, keys={1: b"k" * 32}), fail
        )
    )
    response = await request(
        app, alice, method="PUT", url="/account/sdk-credential", json={"sdk_open_id": "value"}
    )
    assert response.status_code == 500
    assert store.get_credential(alice.tenant_id) is None
