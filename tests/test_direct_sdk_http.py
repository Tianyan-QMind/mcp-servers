from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest
from mcp.types import TextContent, Tool
from pydantic import SecretBytes, SecretStr

from tests.mcp_helper import running_mcp_client
from tianyan_mcp.app import DirectSdkDependencies, create_app
from tianyan_mcp.client import TianyanApiClient
from tianyan_mcp.config import Settings
from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_experiment_client import DirectSdkExperimentClient
from tianyan_mcp.direct_sdk_security import DirectSdkSecurityMiddleware
from tianyan_mcp.direct_sdk_waveform_client import DirectSdkWaveformClient
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import RESULT_URL, STATUS_URL
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.identity import IdentityMode
from tianyan_mcp.models import BackendSnapshot, UpstreamBackend
from tianyan_mcp.service import BackendService
from tianyan_mcp.tenant_store import TenantStore
from tianyan_mcp.waveform_service import WaveformService


@pytest.fixture
def direct_settings() -> Callable[[bool], Settings]:
    def make(allow_hardware_submission: bool = False) -> Settings:
        return Settings(
            identity_mode=IdentityMode.DIRECT_SDK,
            direct_sdk_fingerprint_key=SecretBytes(b"f" * 32),
            host="127.0.0.1",
            allow_hardware_submission=allow_hardware_submission,
        )

    return make


@dataclass
class FakeBackendClient:
    calls: int = 0

    async def fetch_backends(self) -> BackendSnapshot:
        self.calls += 1
        return BackendSnapshot(
            fetched_at=datetime(2026, 8, 26, tzinfo=UTC),
            backends=[
                UpstreamBackend(id="sim", code="tianyan_s", labels="3", status=0),
                UpstreamBackend(id="hardware", code="tianyan24", labels="1", status=0),
            ],
        )


@dataclass
class FakeLoginClient:
    tokens: dict[str, str]
    calls: list[str] = field(default_factory=lambda: [])

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        value = open_id.get_secret_value()
        self.calls.append(value)
        token = self.tokens.get(value)
        if token is None:
            raise ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)
        await asyncio.sleep(0)
        return SecretStr(token), 720.0


@dataclass
class DirectHarness:
    dependencies: DirectSdkDependencies
    backend: FakeBackendClient
    login: FakeLoginClient
    posts: list[httpx.Request]


def make_harness(settings: Settings, tokens: dict[str, str]) -> DirectHarness:
    backend = FakeBackendClient()
    login = FakeLoginClient(tokens)
    posts: list[httpx.Request] = []

    def submit_handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        token = request.headers["Authorization"].removeprefix("Bearer ")
        query_id = "101" if token == "alice-token" else "202"
        if str(request.url) == STATUS_URL:
            return httpx.Response(
                200,
                json={"code": 0, "success": True, "data": {"status": 0}},
            )
        if str(request.url) == RESULT_URL:
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
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"query_ids": [f"query-{token}"]},
            },
        )

    backend_service = BackendService(cast(TianyanApiClient, backend))
    manager = DirectSdkAuthManager(login)
    experiment_client = DirectSdkExperimentClient(
        manager,
        settings.http_timeout_seconds,
        transport=httpx.MockTransport(submit_handler),
    )
    experiment_service = ExperimentService(
        backend_service,
        experiment_client,
        allow_hardware_submission=settings.allow_hardware_submission,
    )
    experiment_result_service = ExperimentResultService(experiment_client)
    waveform_client = DirectSdkWaveformClient(
        manager,
        settings.http_timeout_seconds,
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    waveform_service = WaveformService(waveform_client, waveform_client)
    return DirectHarness(
        DirectSdkDependencies(
            manager,
            backend_service,
            experiment_service,
            experiment_result_service,
            waveform_service,
        ),
        backend,
        login,
        posts,
    )


def submission_arguments(
    computer_code: str = "tianyan_s",
    *,
    allow_hardware: bool = False,
    lab_name: str | None = "integration",
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "computer_code": computer_code,
        "circuit": "H q[0]",
        "name": "direct sdk test",
        "shots": 100,
        "allow_hardware": allow_hardware,
    }
    if lab_name is not None:
        arguments["lab_name"] = lab_name
    return arguments


def assert_tool_contract(tools: list[Tool]) -> None:
    by_name = {tool.name: tool for tool in tools}
    assert list(by_name) == [
        "list_available_quantum_backends",
        "get_quantum_backend",
        "submit_quantum_experiment",
        "submit_quantum_experiment_batch",
        "get_quantum_experiment_result",
        "generate_waveform_diagram",
        "get_waveform_diagram",
        "estimate_quantum_expectation",
        "batch_estimate_quantum_expectation",
        "get_quantum_expectation_result",
    ]
    assert len(by_name) == 10

    expected_annotations = {
        "list_available_quantum_backends": (True, False, True, True),
        "get_quantum_backend": (True, False, True, True),
        "submit_quantum_experiment": (False, True, False, True),
        "submit_quantum_experiment_batch": (False, True, False, True),
        "get_quantum_experiment_result": (True, False, True, True),
        "generate_waveform_diagram": (False, False, False, True),
        "get_waveform_diagram": (True, False, True, True),
        "estimate_quantum_expectation": (False, True, False, True),
        "batch_estimate_quantum_expectation": (False, True, False, True),
        "get_quantum_expectation_result": (True, False, True, True),
    }
    for name, expected in expected_annotations.items():
        annotations = by_name[name].annotations
        assert annotations is not None
        assert (
            annotations.readOnlyHint,
            annotations.destructiveHint,
            annotations.idempotentHint,
            annotations.openWorldHint,
        ) == expected

    list_schema = by_name["list_available_quantum_backends"].inputSchema
    list_schema_text = json.dumps(list_schema, sort_keys=True)
    for value in ("all", "quantum_computer", "simulator"):
        assert f'"{value}"' in list_schema_text
    identifier = by_name["get_quantum_backend"].inputSchema["properties"]["identifier"]
    assert (identifier["minLength"], identifier["maxLength"]) == (1, 128)
    submit = by_name["submit_quantum_experiment"].inputSchema["properties"]
    assert set(submit) == {
        "computer_code",
        "circuit",
        "name",
        "lab_name",
        "shots",
        "allow_hardware",
    }
    assert (submit["shots"]["minimum"], submit["shots"]["maximum"]) == (1, 1000)
    assert submit["allow_hardware"]["default"] is False
    result = by_name["get_quantum_experiment_result"].inputSchema["properties"]
    assert set(result) == {"query_id", "include_result_status"}
    assert (result["query_id"]["minLength"], result["query_id"]["maxLength"]) == (1, 128)
    assert result["include_result_status"]["default"] is False


def test_create_app_selects_direct_branch_without_initializing_tenant_store(
    direct_settings: Callable[[bool], Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_initialize(_: TenantStore) -> None:
        raise AssertionError("direct SDK mode initialized the tenant database")

    monkeypatch.setattr(TenantStore, "initialize", fail_initialize)

    app = create_app(direct_settings(False))

    assert isinstance(app, DirectSdkSecurityMiddleware)


@pytest.mark.asyncio
async def test_direct_http_exposes_health_and_requires_sdk_header_only_for_protected_routes(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(False)
    harness = make_harness(settings, {"probe-open-id": "probe-token"})
    app = create_app(settings, direct_sdk_dependencies=harness.dependencies)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        health = await client.get("/healthz")
        missing_header = await client.post("/mcp")
        forged_header = await client.post("/mcp", headers={"Authorization": "Bearer fake-open-id"})
        account = await client.get(
            "/account/sdk-credential",
            headers={"Authorization": "Bearer probe-open-id"},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert missing_header.status_code == 401
    assert missing_header.json() == {"error": "invalid_sdk_credential"}
    assert forged_header.status_code == 401
    assert forged_header.json() == {"error": "invalid_sdk_credential"}
    assert account.status_code == 404


@pytest.mark.asyncio
async def test_direct_http_read_tools_prewarm_login_once_without_submit(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(False)
    harness = make_harness(settings, {"reader-open-id": "reader-token"})
    app = create_app(settings, direct_sdk_dependencies=harness.dependencies)

    async with running_mcp_client(app, "reader-open-id") as session:
        catalog = await session.list_tools()
        listed = await session.call_tool(
            "list_available_quantum_backends", {"backend_type": "simulator"}
        )
        detail = await session.call_tool("get_quantum_backend", {"identifier": "tianyan_s"})

    assert {tool.name for tool in catalog.tools} >= {
        "list_available_quantum_backends",
        "get_quantum_backend",
    }
    assert listed.structuredContent is not None
    assert listed.structuredContent["total"] == 1
    assert detail.structuredContent is not None
    assert detail.structuredContent["id"] == "sim"
    assert harness.backend.calls == 2
    assert harness.login.calls == ["reader-open-id"]
    assert harness.posts == []


@pytest.mark.asyncio
async def test_invalid_sdk_open_id_is_rejected_at_boundary_before_mcp_session(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(False)
    harness = make_harness(settings, {})
    app = create_app(settings, direct_sdk_dependencies=harness.dependencies)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        rejected = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer invalid-open-id"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )

    assert rejected.status_code == 401
    assert rejected.json() == {"error": "invalid_sdk_credential"}
    assert harness.login.calls == ["invalid-open-id"]
    assert harness.posts == []


@pytest.mark.asyncio
async def test_concurrent_direct_submissions_keep_sdk_identities_isolated(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(False)
    harness = make_harness(
        settings,
        {"alice-open-id": "alice-token", "bob-open-id": "bob-token"},
    )

    async def submit_as(open_id: str, lab_name: str | None = "integration"):
        fresh_app = create_app(settings, direct_sdk_dependencies=harness.dependencies)
        async with running_mcp_client(fresh_app, open_id) as session:
            return await session.call_tool(
                "submit_quantum_experiment", submission_arguments(lab_name=lab_name)
            )

    alice, bob = await asyncio.gather(
        submit_as("alice-open-id", lab_name=None), submit_as("bob-open-id")
    )

    assert alice.structuredContent is not None
    assert bob.structuredContent is not None
    assert alice.structuredContent["query_id"] == "query-alice-token"
    assert bob.structuredContent["query_id"] == "query-bob-token"
    assert set(harness.login.calls) == {"alice-open-id", "bob-open-id"}
    assert len(harness.login.calls) == 2
    assert {request.headers["Authorization"] for request in harness.posts} == {
        "Bearer alice-token",
        "Bearer bob-token",
    }
    assert len(harness.posts) == 2
    bodies_by_authorization = {
        request.headers["Authorization"]: json.loads(request.content) for request in harness.posts
    }
    assert "lab_name" not in bodies_by_authorization["Bearer alice-token"]
    assert bodies_by_authorization["Bearer bob-token"]["lab_name"] == "integration"


@pytest.mark.asyncio
async def test_direct_result_queries_keep_inbound_sdk_identities_isolated(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(False)
    harness = make_harness(
        settings,
        {"alice-open-id": "alice-token", "bob-open-id": "bob-token"},
    )

    async def query_as(open_id: str, query_id: str):
        fresh_app = create_app(settings, direct_sdk_dependencies=harness.dependencies)
        async with running_mcp_client(fresh_app, open_id) as session:
            return await session.call_tool(
                "get_quantum_experiment_result",
                {"query_id": query_id, "include_result_status": False},
            )

    alice, bob = await asyncio.gather(
        query_as("alice-open-id", "101"),
        query_as("bob-open-id", "202"),
    )

    assert alice.structuredContent is not None
    assert bob.structuredContent is not None
    assert alice.structuredContent["probability"] == {"alice-token": 1.0}
    assert bob.structuredContent["probability"] == {"bob-token": 1.0}
    assert set(harness.login.calls) == {"alice-open-id", "bob-open-id"}
    assert {request.headers["Authorization"] for request in harness.posts} == {
        "Bearer alice-token",
        "Bearer bob-token",
    }
    assert len(harness.posts) == 4


@pytest.mark.asyncio
async def test_disabled_service_flag_rejects_hardware_at_tool_layer_without_post(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(False)
    harness = make_harness(settings, {"hardware-open-id": "hardware-token"})
    app = create_app(settings, direct_sdk_dependencies=harness.dependencies)

    async with running_mcp_client(app, "hardware-open-id") as session:
        submitted = await session.call_tool(
            "submit_quantum_experiment",
            submission_arguments("tianyan24", allow_hardware=True),
        )

    assert submitted.isError is True
    assert isinstance(submitted.content[0], TextContent)
    assert "HARDWARE_SUBMISSION_DISABLED" in submitted.content[0].text
    assert harness.login.calls == ["hardware-open-id"]
    assert harness.posts == []


@pytest.mark.asyncio
async def test_enabled_service_flag_and_tool_confirmation_permit_one_hardware_post(
    direct_settings: Callable[[bool], Settings],
) -> None:
    settings = direct_settings(True)
    harness = make_harness(settings, {"hardware-open-id": "hardware-token"})
    app = create_app(settings, direct_sdk_dependencies=harness.dependencies)

    async with running_mcp_client(app, "hardware-open-id") as session:
        submitted = await session.call_tool(
            "submit_quantum_experiment",
            submission_arguments("tianyan24", allow_hardware=True),
        )

    assert submitted.structuredContent is not None
    assert submitted.structuredContent["query_id"] == "query-hardware-token"
    assert harness.login.calls == ["hardware-open-id"]
    assert len(harness.posts) == 1
    assert harness.posts[0].headers["Authorization"] == "Bearer hardware-token"
