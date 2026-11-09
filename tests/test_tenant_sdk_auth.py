from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from tianyan_mcp.credential_crypto import ApiKeyCodec, CredentialCipher, EncryptedCredential
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_models import UpstreamExperimentSubmission
from tianyan_mcp.identity import TenantPrincipal, tenant_scope
from tianyan_mcp.tenant_experiment_client import TenantExperimentClient
from tianyan_mcp.tenant_sdk_auth import TenantSdkAuthManager
from tianyan_mcp.tenant_store import TenantStore


@dataclass
class FakeLoginClient:
    calls: list[str] = field(default_factory=lambda: [])
    expires_in: float = 720.0
    started: asyncio.Event | None = None
    release: asyncio.Event | None = None

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        self.calls.append(open_id.get_secret_value())
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return SecretStr(f"token-{open_id.get_secret_value()}"), self.expires_in


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


@pytest.mark.asyncio
async def test_tokens_are_isolated_by_tenant_and_credential_revision(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    bob = tenant_with_credential(store, cipher, "bob", "bob-open-id")
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login, clock=lambda: 100.0)

    assert await manager.get_access_token(alice) == "token-alice-open-id"
    assert await manager.get_access_token(bob) == "token-bob-open-id"
    assert await manager.get_access_token(alice) == "token-alice-open-id"
    assert login.calls == ["alice-open-id", "bob-open-id"]


@pytest.mark.asyncio
async def test_concurrent_same_tenant_requests_share_one_login(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)

    tokens = await asyncio.gather(*(manager.get_access_token(alice) for _ in range(5)))

    assert tokens == ["token-alice-open-id"] * 5
    assert login.calls == ["alice-open-id"]


@pytest.mark.asyncio
async def test_concurrent_tenants_never_share_tokens(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    bob = tenant_with_credential(store, cipher, "bob", "bob-open-id")
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)

    alice_token, bob_token = await asyncio.gather(
        manager.get_access_token(alice), manager.get_access_token(bob)
    )

    assert {alice_token, bob_token} == {"token-alice-open-id", "token-bob-open-id"}
    assert set(login.calls) == {"alice-open-id", "bob-open-id"}


@pytest.mark.asyncio
async def test_missing_credential_fails_before_sdk_login(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    tenant = store.create_tenant("alice", ApiKeyCodec(b"p" * 32).generate())
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)

    with pytest.raises(ServiceError, match="SDK_CREDENTIAL_NOT_CONFIGURED"):
        await manager.get_access_token(TenantPrincipal(tenant.id, "key-alice", 1))
    assert login.calls == []


@pytest.mark.asyncio
async def test_corrupted_credential_is_secret_free_and_never_logs_in(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    generated = ApiKeyCodec(b"p" * 32).generate()
    tenant = store.create_tenant("alice", generated)
    store.put_credential(
        tenant.id,
        EncryptedCredential(ciphertext=b"corrupt-secret", nonce=b"n" * 12, key_version=1),
    )
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)

    with (
        tenant_scope(TenantPrincipal(tenant.id, generated.key_id, 1)),
        pytest.raises(ServiceError) as error,
    ):
        await manager.get_access_token(TenantPrincipal(tenant.id, generated.key_id, 1))
    assert error.value.code == "SDK_AUTH_FAILED"
    assert "corrupt-secret" not in str(error.value)
    assert login.calls == []
    with sqlite3.connect(store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id FROM audit_events"
        ).fetchone()
    assert event[:4] == (tenant.id, generated.key_id, "sdk_auth_failed", "failure")
    assert event[4] is not None


@pytest.mark.asyncio
async def test_revision_expiry_and_invalidate_only_refresh_affected_tenant(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    now = 100.0
    alice = tenant_with_credential(store, cipher, "alice", "alice-one")
    bob = tenant_with_credential(store, cipher, "bob", "bob-open-id")
    login = FakeLoginClient(expires_in=120.0)
    manager = TenantSdkAuthManager(store, cipher, login, clock=lambda: now)

    assert await manager.get_access_token(alice) == "token-alice-one"
    assert await manager.get_access_token(bob) == "token-bob-open-id"
    store.put_credential(alice.tenant_id, cipher.encrypt(alice.tenant_id, SecretStr("alice-two")))
    assert await manager.get_access_token(alice) == "token-alice-two"
    manager.invalidate(alice.tenant_id)
    assert await manager.get_access_token(alice) == "token-alice-two"
    now = 161.0
    assert await manager.get_access_token(bob) == "token-bob-open-id"
    assert login.calls == ["alice-one", "bob-open-id", "alice-two", "alice-two", "bob-open-id"]


@pytest.mark.asyncio
async def test_revision_refresh_retires_old_tenant_tokens_without_touching_other_tenant(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-one")
    bob = tenant_with_credential(store, cipher, "bob", "bob-open-id")
    manager = TenantSdkAuthManager(store, cipher, FakeLoginClient())

    await manager.get_access_token(alice)
    await manager.get_access_token(bob)
    store.put_credential(alice.tenant_id, cipher.encrypt(alice.tenant_id, SecretStr("alice-two")))
    await manager.get_access_token(alice)

    keys = set(manager._tokens)  # pyright: ignore[reportPrivateUsage]
    assert all(key[0] != alice.tenant_id or key[1] == 2 for key in keys)
    assert (bob.tenant_id, 1) in keys


@pytest.mark.asyncio
async def test_expired_token_is_removed_before_refresh_and_revisions_remain_bounded(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    now = 100.0
    alice = tenant_with_credential(store, cipher, "alice", "alice-one")
    manager = TenantSdkAuthManager(
        store, cipher, FakeLoginClient(expires_in=120.0), clock=lambda: now
    )

    await manager.get_access_token(alice)
    first_key = (alice.tenant_id, 1)
    assert manager._valid_cached(first_key) is not None  # pyright: ignore[reportPrivateUsage]
    now = 161.0
    assert manager._valid_cached(first_key) is None  # pyright: ignore[reportPrivateUsage]
    assert first_key not in manager._tokens  # pyright: ignore[reportPrivateUsage]

    for revision in range(2, 6):
        store.put_credential(
            alice.tenant_id,
            cipher.encrypt(alice.tenant_id, SecretStr(f"alice-{revision}")),
        )
        await manager.get_access_token(alice)
        keys = set(manager._tokens)  # pyright: ignore[reportPrivateUsage]
        assert keys == {(alice.tenant_id, revision)}


@pytest.mark.asyncio
async def test_cancelled_login_cleans_up_lock_for_next_request(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    started = asyncio.Event()
    release = asyncio.Event()
    login = FakeLoginClient(started=started, release=release)
    manager = TenantSdkAuthManager(store, cipher, login)

    pending = asyncio.create_task(manager.get_access_token(alice))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    login.release = None

    assert await manager.get_access_token(alice) == "token-alice-open-id"
    assert login.calls == ["alice-open-id", "alice-open-id"]


def submission(computer_code: str = "tianyan_sw") -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1",),
        name="tenant-test",
        lab_name="tenant-lab",
        computerCode=computer_code,
    )


def batch_submission() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1", "X Q1"),
        name="tenant-batch-test",
        computerCode="tianyan_sw",
    )


@pytest.mark.asyncio
async def test_tenant_experiment_client_uses_current_tenant_and_submits_once(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer token-alice-open-id"
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["tenant-query"]}},
        )

    client = TenantExperimentClient(
        manager, timeout_seconds=1, transport=httpx.MockTransport(handler)
    )
    with tenant_scope(alice):
        assert await client.submit(submission()) == "tenant-query"
    assert calls == 1
    with sqlite3.connect(store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id FROM audit_events"
        ).fetchone()
    assert event[:4] == (alice.tenant_id, alice.key_id, "experiment_submitted", "success")
    assert event[4] is not None


@pytest.mark.asyncio
async def test_tenant_batch_submission_audits_one_successful_upstream_request(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    manager = TenantSdkAuthManager(store, cipher, FakeLoginClient())
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer token-alice-open-id"
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["query-1", "query-2"]}},
        )

    client = TenantExperimentClient(
        manager, timeout_seconds=1, transport=httpx.MockTransport(handler)
    )
    with tenant_scope(alice):
        assert await client.submit_batch(batch_submission()) == ("query-1", "query-2")
    assert calls == 1
    with sqlite3.connect(store.path) as connection:
        events = connection.execute(
            "SELECT event_type, outcome FROM audit_events ORDER BY occurred_at"
        ).fetchall()
    assert events == [("experiment_submitted", "success")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_outcome"),
    [
        (httpx.Response(500), "unknown"),
        (httpx.Response(400), "denied"),
    ],
)
async def test_tenant_experiment_attempt_audits_non_success_outcomes(
    store: TenantStore,
    cipher: CredentialCipher,
    response: httpx.Response,
    expected_outcome: str,
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    manager = TenantSdkAuthManager(store, cipher, FakeLoginClient())
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    client = TenantExperimentClient(
        manager, timeout_seconds=1, transport=httpx.MockTransport(handler)
    )
    with tenant_scope(alice), pytest.raises(ServiceError):
        await client.submit(submission())
    assert calls == 1
    with sqlite3.connect(store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id FROM audit_events"
        ).fetchone()
    assert event[:4] == (
        alice.tenant_id,
        alice.key_id,
        "experiment_submitted",
        expected_outcome,
    )
    assert event[4] is not None


@pytest.mark.asyncio
async def test_submission_success_remains_success_when_best_effort_audit_fails(
    store: TenantStore, cipher: CredentialCipher, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    manager = TenantSdkAuthManager(store, cipher, FakeLoginClient())

    def fail_audit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(store, "append_audit", fail_audit)
    client = TenantExperimentClient(
        manager,
        timeout_seconds=1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={"code": 0, "success": True, "data": {"query_ids": ["submitted"]}},
            )
        ),
    )
    with tenant_scope(alice):
        assert await client.submit(submission()) == "submitted"


@pytest.mark.asyncio
async def test_tenant_experiment_client_rejects_hardware_before_login_or_post(
    store: TenantStore, cipher: CredentialCipher
) -> None:
    alice = tenant_with_credential(store, cipher, "alice", "alice-open-id")
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)
    client = TenantExperimentClient(
        manager,
        timeout_seconds=1,
        transport=httpx.MockTransport(lambda _: pytest.fail("must not submit")),
    )

    with tenant_scope(alice), pytest.raises(ServiceError, match="HARDWARE_SUBMISSION_DISABLED"):
        await client.submit(submission("tianyan176"))
    assert login.calls == []
