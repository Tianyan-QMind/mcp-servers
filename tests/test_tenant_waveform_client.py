from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from tianyan_mcp.credential_crypto import ApiKeyCodec, CredentialCipher
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import TenantPrincipal, tenant_scope
from tianyan_mcp.tenant_sdk_auth import TenantSdkAuthManager
from tianyan_mcp.tenant_store import TenantStore
from tianyan_mcp.tenant_waveform_client import TenantWaveformClient
from tianyan_mcp.waveform_client import DIAGRAM_URL, GENERATE_URL
from tianyan_mcp.waveform_models import UpstreamWaveformSubmission


class FakeLoginClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        self.calls.append(open_id.get_secret_value())
        return SecretStr(f"token-{open_id.get_secret_value()}"), 720.0


@pytest.fixture
def store(tmp_path: Path) -> TenantStore:
    result = TenantStore(tmp_path / "tenants.sqlite", ApiKeyCodec(b"p" * 32))
    result.initialize()
    return result


@pytest.fixture
def cipher() -> CredentialCipher:
    return CredentialCipher(active_version=1, keys={1: b"k" * 32})


def tenant_with_credential(
    store: TenantStore, cipher: CredentialCipher, name: str, open_id: str
) -> TenantPrincipal:
    generated = ApiKeyCodec(b"p" * 32).generate()
    tenant = store.create_tenant(name, generated)
    store.put_credential(tenant.id, cipher.encrypt(tenant.id, SecretStr(open_id)))
    return TenantPrincipal(tenant.id, generated.key_id, tenant.auth_revision)


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
    store: TenantStore,
    cipher: CredentialCipher,
    login: FakeLoginClient,
) -> TenantWaveformClient:
    manager = TenantSdkAuthManager(store, cipher, login)
    return TenantWaveformClient(manager, timeout_seconds=1, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_generate_and_query_use_the_current_tenant_token(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-waveform-open-id")
    bob = tenant_with_credential(store, cipher, "bob", "bob-waveform-open-id")
    login = FakeLoginClient()
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        token = request.headers["Authorization"]
        seen.append((url, token))
        if url == GENERATE_URL:
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "success": True,
                    "message": None,
                    "data": {
                        "id": "123",
                        "circuitName": "example1",
                        "circuitContent": "H Q0",
                        "status": 0,
                    },
                },
            )
        assert url.startswith(DIAGRAM_URL)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {"id": "123", "circuit_name": "n", "visibleUrl": ""},
            },
        )

    client = client_for(handler, store, cipher, login)

    with tenant_scope(alice):
        task = await client.generate(UpstreamWaveformSubmission(circuit="H Q0", qcCode="QPU-01"))
        generating = await client.get_diagram("123")
    with tenant_scope(bob):
        bob_task = await client.generate(
            UpstreamWaveformSubmission(circuit="X Q0", qcCode="QPU-01", isVerify="false")
        )

    assert task.id == bob_task.id == "123"
    assert generating.id == "123"
    assert generating.visible_url == ""
    assert seen == [
        (GENERATE_URL, "Bearer token-alice-waveform-open-id"),
        (DIAGRAM_URL + "?id=123", "Bearer token-alice-waveform-open-id"),
        (GENERATE_URL, "Bearer token-bob-waveform-open-id"),
    ]
    assert login.calls == ["alice-waveform-open-id", "bob-waveform-open-id"]


@pytest.mark.asyncio
async def test_generate_without_tenant_scope_raises(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    login = FakeLoginClient()
    client = client_for(lambda _: httpx.Response(500), store, cipher, login)

    with pytest.raises(RuntimeError, match="tenant context is unavailable"):
        await client.generate(UpstreamWaveformSubmission(circuit="H Q0", qcCode="QPU-01"))


@pytest.mark.asyncio
async def test_unauthorized_response_invalidates_cached_tenant_token(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-waveform-open-id")
    login = FakeLoginClient()
    statuses: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(DIAGRAM_URL)
        status = 401 if not statuses else 200
        statuses.append(status)
        if status == 401:
            return httpx.Response(401)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {"id": "123", "circuit_name": "n", "visibleUrl": ""},
            },
        )

    client = client_for(handler, store, cipher, login)

    with tenant_scope(alice):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await client.get_diagram("123")
        diagram = await client.get_diagram("123")

    assert login.calls == ["alice-waveform-open-id", "alice-waveform-open-id"]
    assert diagram.id == "123"
