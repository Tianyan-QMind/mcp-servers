from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import TextContent
from pydantic import SecretBytes, SecretStr

from tests.mcp_helper import running_mcp_client
from tianyan_mcp.app import TenantDependencies, build_tenant_dependencies, create_app
from tianyan_mcp.client import TianyanApiClient
from tianyan_mcp.config import Settings
from tianyan_mcp.credential_crypto import ApiKeyCodec
from tianyan_mcp.experiment_client import RESULT_URL, STATUS_URL
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.identity import IdentityMode
from tianyan_mcp.models import BackendSnapshot, UpstreamBackend
from tianyan_mcp.service import BackendService
from tianyan_mcp.tenant_experiment_client import TenantExperimentClient
from tianyan_mcp.tenant_sdk_auth import TenantSdkAuthManager
from tianyan_mcp.tenant_security import TenantSecurityMiddleware
from tianyan_mcp.tenant_waveform_client import TenantWaveformClient
from tianyan_mcp.waveform_service import WaveformService


@pytest.fixture
def multi_tenant_settings(tmp_path: Path) -> Settings:
    return Settings(
        identity_mode=IdentityMode.MULTI_TENANT,
        database_path=tmp_path / "tenants.sqlite",
        api_key_pepper=SecretBytes(b"p" * 32),
        credential_key_version=1,
        credential_keys={1: SecretBytes(b"k" * 32)},
    )


@pytest.mark.asyncio
async def test_multi_tenant_assembly_needs_no_global_sdk_open_id(
    multi_tenant_settings: Settings,
) -> None:
    app = create_app(multi_tenant_settings)

    assert isinstance(app, TenantSecurityMiddleware)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_multi_tenant_assembly_authenticates_account_routes(
    multi_tenant_settings: Settings,
) -> None:
    dependencies = build_tenant_dependencies(multi_tenant_settings)
    generated_key = ApiKeyCodec(b"p" * 32).generate()
    tenant = dependencies.store.create_tenant("alice", generated_key)
    app = create_app(multi_tenant_settings, tenant_dependencies=dependencies)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        configured = await client.get(
            "/account/sdk-credential",
            headers={"Authorization": f"Bearer {generated_key.plaintext}"},
        )
        missing_key = await client.get("/account/sdk-credential")
    dependencies.store.set_tenant_status(tenant.id, enabled=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        disabled = await client.get(
            "/account/sdk-credential",
            headers={"Authorization": f"Bearer {generated_key.plaintext}"},
        )

    assert configured.status_code == 200
    assert configured.json()["configured"] is False
    assert missing_key.status_code == disabled.status_code == 401
    assert missing_key.json() == disabled.json() == {"error": "invalid_mcp_api_key"}


@pytest.mark.asyncio
async def test_credential_change_shares_request_audit_id_with_mcp_auth(
    multi_tenant_settings: Settings,
) -> None:
    dependencies = build_tenant_dependencies(multi_tenant_settings)
    generated = ApiKeyCodec(b"p" * 32).generate()
    tenant = dependencies.store.create_tenant("alice", generated)
    app = create_app(multi_tenant_settings, tenant_dependencies=dependencies)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.put(
            "/account/sdk-credential",
            headers={"Authorization": f"Bearer {generated.plaintext}"},
            json={"sdk_open_id": "alice-open-id"},
        )

    assert response.status_code == 200
    with sqlite3.connect(dependencies.store.path) as connection:
        events = connection.execute(
            "SELECT key_id, event_type, outcome, request_id FROM audit_events "
            "WHERE tenant_id = ? ORDER BY occurred_at",
            (tenant.id,),
        ).fetchall()
    assert [(event[0], event[1], event[2]) for event in events] == [
        (generated.key_id, "mcp_authenticated", "success"),
        (generated.key_id, "credential_updated", "success"),
    ]
    assert events[0][3] == events[1][3]


@dataclass
class FakeLoginClient:
    open_ids: list[str] = field(default_factory=lambda: [])

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        value = open_id.get_secret_value()
        self.open_ids.append(value)
        return SecretStr(f"token-{value}"), 720.0


@pytest.mark.asyncio
async def test_two_keys_submit_with_isolated_sdk_identities_and_hardware_is_rejected(
    multi_tenant_settings: Settings,
) -> None:
    codec = ApiKeyCodec(b"p" * 32)
    initial_dependencies = build_tenant_dependencies(multi_tenant_settings)
    store = initial_dependencies.store
    cipher = initial_dependencies.cipher
    alice_key = codec.generate()
    bob_key = codec.generate()
    alice = store.create_tenant("alice", alice_key)
    bob = store.create_tenant("bob", bob_key)
    store.put_credential(alice.id, cipher.encrypt(alice.id, SecretStr("alice-open-id")))
    store.put_credential(bob.id, cipher.encrypt(bob.id, SecretStr("bob-open-id")))
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)
    observed_tokens: list[str] = []
    observed_bodies: list[dict[str, object]] = []

    def submit_handler(request: httpx.Request) -> httpx.Response:
        observed_tokens.append(request.headers["Authorization"])
        observed_bodies.append(cast(dict[str, object], json.loads(request.content)))
        query_id = (
            "alice-query"
            if request.headers["Authorization"].startswith("Bearer token-alice")
            else "bob-query"
        )
        return httpx.Response(
            200, json={"code": 0, "success": True, "data": {"query_ids": [query_id]}}
        )

    backend_client = AsyncMock(spec=TianyanApiClient)
    backend_client.fetch_backends.return_value = BackendSnapshot(
        fetched_at=datetime.now(UTC),
        backends=[
            UpstreamBackend(id="sim", code="tianyan_s", labels="3", status=0),
            UpstreamBackend(id="hardware", code="tianyan24", labels="1", status=0),
        ],
    )
    backend_service = BackendService(cast(TianyanApiClient, backend_client))
    experiment_client = TenantExperimentClient(
        manager,
        1.0,
        transport=httpx.MockTransport(submit_handler),
    )
    experiment_service = ExperimentService(
        backend_service,
        experiment_client,
        allow_hardware_submission=False,
    )
    experiment_result_service = ExperimentResultService(experiment_client)
    waveform_client = TenantWaveformClient(
        manager,
        1.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    waveform_service = WaveformService(waveform_client, waveform_client)
    dependencies = TenantDependencies(
        store,
        cipher,
        manager,
        backend_service,
        experiment_service,
        experiment_result_service,
        waveform_service,
    )

    async def call(
        key: str,
        computer_code: str = "tianyan_s",
        allow_hardware: bool = False,
        lab_name: str | None = "lab",
    ):
        arguments: dict[str, object] = {
            "computer_code": computer_code,
            "circuit": "H q[0]",
            "name": "test",
            "shots": 100,
            "allow_hardware": allow_hardware,
        }
        if lab_name is not None:
            arguments["lab_name"] = lab_name
        # Each live Streamable HTTP server receives a fresh MCP session manager while sharing
        # the same tenant dependencies, mirroring independent client connections.
        fresh_app = create_app(multi_tenant_settings, tenant_dependencies=dependencies)
        async with running_mcp_client(fresh_app, key) as session:
            return await session.call_tool("submit_quantum_experiment", arguments)

    alice_result, bob_result = await asyncio.gather(
        call(alice_key.plaintext, lab_name=None), call(bob_key.plaintext)
    )
    hardware = await call(alice_key.plaintext, "tianyan24", True)

    assert alice_result.structuredContent is not None
    assert bob_result.structuredContent is not None
    assert alice_result.structuredContent["query_id"] == "alice-query"
    assert bob_result.structuredContent["query_id"] == "bob-query"
    assert set(login.open_ids) == {"alice-open-id", "bob-open-id"}
    assert observed_tokens.count("Bearer token-alice-open-id") == 1
    assert observed_tokens.count("Bearer token-bob-open-id") == 1
    assert hardware.isError is True
    assert isinstance(hardware.content[0], TextContent)
    assert "HARDWARE_SUBMISSION_DISABLED" in hardware.content[0].text
    assert len(login.open_ids) == 2
    assert len(observed_tokens) == 2
    bodies_by_token = dict(zip(observed_tokens, observed_bodies, strict=True))
    assert "lab_name" not in bodies_by_token["Bearer token-alice-open-id"]
    assert bodies_by_token["Bearer token-bob-open-id"]["lab_name"] == "lab"

    fresh_app = create_app(multi_tenant_settings, tenant_dependencies=dependencies)
    async with running_mcp_client(fresh_app, alice_key.plaintext) as session:
        tools = await session.list_tools()
    tool_contract: list[tuple[str, bool | None, bool | None, bool | None, bool | None]] = []
    for tool in tools.tools:
        assert tool.annotations is not None
        annotations = tool.annotations
        tool_contract.append(
            (
                tool.name,
                annotations.readOnlyHint,
                annotations.destructiveHint,
                annotations.idempotentHint,
                annotations.openWorldHint,
            )
        )
    assert tool_contract == [
        ("list_available_quantum_backends", True, False, True, True),
        ("get_quantum_backend", True, False, True, True),
        ("submit_quantum_experiment", False, True, False, True),
        ("submit_quantum_experiment_batch", False, True, False, True),
        ("get_quantum_experiment_result", True, False, True, True),
        ("generate_waveform_diagram", False, False, False, True),
        ("get_waveform_diagram", True, False, True, True),
        ("estimate_quantum_expectation", False, True, False, True),
        ("batch_estimate_quantum_expectation", False, True, False, True),
        ("get_quantum_expectation_result", True, False, True, True),
    ]

    missing_key = codec.generate()
    store.create_tenant("missing", missing_key)
    missing_credential = await call(missing_key.plaintext)
    assert missing_credential.isError is True
    assert isinstance(missing_credential.content[0], TextContent)
    assert "SDK_CREDENTIAL_NOT_CONFIGURED" in missing_credential.content[0].text
    assert len(login.open_ids) == len(observed_tokens) == 2

    replacement_key = codec.generate()
    store.rotate_key(alice.id, replacement_key)
    old_key_app = create_app(multi_tenant_settings, tenant_dependencies=dependencies)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=old_key_app), base_url="http://127.0.0.1"
    ) as client:
        old_key_response = await client.post(
            "/mcp", headers={"Authorization": f"Bearer {alice_key.plaintext}"}
        )
    assert old_key_response.status_code == 401

    store.put_credential(alice.id, cipher.encrypt(alice.id, SecretStr("alice-replacement-open-id")))
    replacement = await call(replacement_key.plaintext)
    assert replacement.structuredContent is not None
    assert replacement.structuredContent["query_id"] == "alice-query"
    assert login.open_ids[-1] == "alice-replacement-open-id"
    assert observed_tokens[-1] == "Bearer token-alice-replacement-open-id"


@pytest.mark.asyncio
async def test_two_tenants_query_results_with_isolated_credentials_and_no_submission_audit(
    multi_tenant_settings: Settings,
) -> None:
    codec = ApiKeyCodec(b"p" * 32)
    initial = build_tenant_dependencies(multi_tenant_settings)
    store = initial.store
    cipher = initial.cipher
    alice_key = codec.generate()
    bob_key = codec.generate()
    alice = store.create_tenant("alice-query", alice_key)
    bob = store.create_tenant("bob-query", bob_key)
    store.put_credential(alice.id, cipher.encrypt(alice.id, SecretStr("alice-open-id")))
    store.put_credential(bob.id, cipher.encrypt(bob.id, SecretStr("bob-open-id")))
    login = FakeLoginClient()
    manager = TenantSdkAuthManager(store, cipher, login)
    requests: list[httpx.Request] = []

    def query_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        token = request.headers["Authorization"].removeprefix("Bearer token-")
        query_id = "101" if token == "alice-open-id" else "202"
        if str(request.url) == STATUS_URL:
            return httpx.Response(
                200,
                json={"code": 0, "success": True, "data": {"status": 0}},
            )
        assert str(request.url) == RESULT_URL
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {token: 1.0},
                            "experimentTaskId": query_id,
                        }
                    ]
                },
            },
        )

    backend_service = initial.backend_service
    experiment_client = TenantExperimentClient(
        manager,
        1.0,
        transport=httpx.MockTransport(query_handler),
    )
    waveform_client = TenantWaveformClient(
        manager,
        1.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    dependencies = TenantDependencies(
        store,
        cipher,
        manager,
        backend_service,
        ExperimentService(
            backend_service,
            experiment_client,
            allow_hardware_submission=False,
        ),
        ExperimentResultService(experiment_client),
        WaveformService(waveform_client, waveform_client),
    )

    async def query_as(key: str, query_id: str):
        fresh_app = create_app(multi_tenant_settings, tenant_dependencies=dependencies)
        async with running_mcp_client(fresh_app, key) as session:
            return await session.call_tool(
                "get_quantum_experiment_result",
                {"query_id": query_id, "include_result_status": False},
            )

    alice_result, bob_result = await asyncio.gather(
        query_as(alice_key.plaintext, "101"),
        query_as(bob_key.plaintext, "202"),
    )

    assert alice_result.structuredContent is not None
    assert bob_result.structuredContent is not None
    assert alice_result.structuredContent["probability"] == {"alice-open-id": 1.0}
    assert bob_result.structuredContent["probability"] == {"bob-open-id": 1.0}
    assert set(login.open_ids) == {"alice-open-id", "bob-open-id"}
    assert len(requests) == 4
    with sqlite3.connect(store.path) as connection:
        submission_events = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'experiment_submitted'"
        ).fetchone()
    assert submission_events == (0,)
