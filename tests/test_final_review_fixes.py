from __future__ import annotations

import base64
import getpass
import sqlite3
import uuid
from pathlib import Path

import httpx
import pytest
from pydantic import SecretBytes, SecretStr

from tianyan_mcp.client_cli import main as client_main
from tianyan_mcp.config import Settings
from tianyan_mcp.credential_crypto import ApiKeyCodec, CredentialCipher
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import TianyanExperimentClient
from tianyan_mcp.experiment_models import UpstreamExperimentSubmission
from tianyan_mcp.identity import IdentityMode
from tianyan_mcp.tenant_store import AuditEventType, AuditOutcome, TenantStore


def _multi_settings(path: Path, host: str = "127.0.0.1") -> Settings:
    return Settings(
        identity_mode=IdentityMode.MULTI_TENANT,
        database_path=path,
        api_key_pepper=SecretBytes(b"p" * 32),
        credential_key_version=1,
        credential_keys={1: SecretBytes(b"k" * 32)},
        host=host,
    )


def test_multi_tenant_rejects_relative_db_and_public_bind_host(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        _multi_settings(Path("relative.sqlite"))
    with pytest.raises(ValueError, match="loopback"):
        _multi_settings(tmp_path / "tenant.sqlite", ".".join(("0", "0", "0", "0")))
    env = {
        "TIANYAN_IDENTITY_MODE": "multi_tenant",
        "TIANYAN_DATABASE_PATH": "relative.sqlite",
        "TIANYAN_API_KEY_PEPPER": "p" * 32,
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION": "1",
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V1": base64.b64encode(b"k" * 32).decode(),
    }
    with pytest.raises(ValueError, match="absolute"):
        Settings.from_env(env)


def test_client_status_prints_only_valid_non_secret_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api_key = "tymcp_0000000000000000_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    open_id = "never-echo-this"
    monkeypatch.setattr(getpass, "getpass", lambda: api_key)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"configured": True, "updated_at": "2026-08-11T00:00:00+00:00"},
            request=request,
        )

    assert (
        client_main(
            ["status", "--server", "https://mcp.example.com"],
            transport=httpx.MockTransport(handler),
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "configured: true" in output
    assert "updated_at: 2026-08-11T00:00:00+00:00" in output
    assert api_key not in output and open_id not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_submit_auth_rejection_has_unknown_outcome_semantics(status: int) -> None:
    class Provider:
        invalidated = False

        async def get_access_token(self) -> str:
            return "token"

        def invalidate(self) -> None:
            self.invalidated = True

    provider = Provider()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    request = UpstreamExperimentSubmission(
        shots=1, circuit=("H q[0]",), name="test", lab_name="lab", computerCode="tianyan_sw"
    )
    client = TianyanExperimentClient(provider, 1, transport=httpx.MockTransport(handler))
    with pytest.raises(ServiceError, match="SUBMISSION_OUTCOME_UNKNOWN"):
        await client.submit(request)
    assert provider.invalidated is True and calls == 1


def test_credential_update_and_success_audit_share_transaction(tmp_path: Path) -> None:
    store = TenantStore(tmp_path / "tenant.sqlite", ApiKeyCodec(b"p" * 32))
    store.initialize()
    tenant = store.create_tenant("alice", ApiKeyCodec(b"p" * 32).generate())
    encrypted = CredentialCipher(active_version=1, keys={1: b"k" * 32}).encrypt(
        tenant.id, SecretStr("open-id")
    )
    request_id = str(uuid.uuid4())
    store.put_credential(
        tenant.id,
        encrypted,
        audit_event=AuditEventType.CREDENTIAL_UPDATED,
        request_id=request_id,
    )
    with sqlite3.connect(store.path) as connection:
        event = connection.execute(
            "SELECT tenant_id, event_type, outcome, request_id FROM audit_events"
        ).fetchone()
    assert event == (tenant.id, "credential_updated", AuditOutcome.SUCCESS.value, request_id)


def test_nginx_template_limits_only_credential_route() -> None:
    text = Path("docs/deployment/nginx-multi-tenant.conf.example").read_text(encoding="utf-8")
    assert "events {" in text
    assert "limit_req_zone" in text
    account = text.split("location = /account/sdk-credential", 1)[1].split("}", 1)[0]
    assert "limit_req" in account
    assert text.count("limit_req ") == 1
